"""
Two-stage license plate capture pipeline for 4K video.

Stage 1: Vehicle detection + tracking (ByteTrack via Ultralytics YOLO)
Stage 2: Plate detection on cropped vehicle regions
Stage 3: Best-frame-per-track selection (blur + size + confidence scoring)
Stage 4: OCR on final best crops only (optional)

Usage:
    python pipeline.py --video input.mp4 \
        --vehicle-model yolov8n.pt \
        --plate-model plate_detector.pt \
        --output-dir output \
        --ocr

Requires:
    pip install ultralytics opencv-python paddleocr paddlepaddle-gpu
    (or paddlepaddle for CPU-only; swap for easyocr if preferred)
"""

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# NOTE: paddleocr cannot run in the same process as torch (paddle <-> torch
# DLL/pybind conflict on Windows: "_gpuDeviceProperties already registered" or
# shm.dll WinError 127, whichever loads first). Detection + tracking below do
# not need paddle. For OCR we use easyocr (torch-based, no conflict) — see
# _run_ocr. If you prefer paddleocr, run it in a separate process.
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False

from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids


@dataclass
class TrackState:
    best_score: float = -1.0
    best_crop: Optional[np.ndarray] = None
    best_frame_idx: int = -1
    best_plate_bbox: Optional[tuple] = None
    best_vehicle_bbox: Optional[tuple] = None
    last_seen_frame: int = -1
    finalized: bool = False


def quality_score(plate_crop: np.ndarray, detection_conf: float,
                   min_width: int, min_blur: float) -> float:
    if plate_crop is None or plate_crop.size == 0:
        return -1.0
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    width = plate_crop.shape[1]
    if width < min_width or blur < min_blur:
        return -1.0
    return (0.4 * min(blur / 500, 1.0)) + (0.3 * min(width / 300, 1.0)) + (0.3 * detection_conf)


def clamp_bbox(x1, y1, x2, y2, w, h):
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


class PlateCapturePipeline:
    def __init__(self, vehicle_model_path: str, plate_model_path: str,
                 output_dir: str, conf_threshold: float = 0.4,
                 min_plate_width: int = 80, min_blur: float = 50.0,
                 max_missed_frames: int = 30, run_ocr: bool = False,
                 device: str = None):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.min_plate_width = min_plate_width
        self.min_blur = min_blur
        self.max_missed_frames = max_missed_frames
        self.run_ocr = run_ocr
        self.device = device

        os.makedirs(os.path.join(output_dir, "plates"), exist_ok=True)
        self.tracks: dict[int, TrackState] = {}
        self.results_rows = []

        self.ocr_engine = None
        if run_ocr:
            if not _EASYOCR_AVAILABLE:
                raise RuntimeError("easyocr is not installed; run: pip install easyocr")
            self.ocr_engine = easyocr.Reader(["en"], gpu=True, verbose=False)

    def _detect_plates_batched(self, vehicle_crops: list[np.ndarray]):
        """Run plate model on ALL vehicle crops from this frame in one forward pass.
        Returns a list, same length/order as vehicle_crops, of (bbox, conf) or None.
        Ultralytics accepts a list of arrays and batches them internally (each is
        letterboxed to imgsz, so differing crop sizes are fine)."""
        if not vehicle_crops:
            return []
        results = self.plate_model.predict(
            vehicle_crops, conf=self.conf_threshold, device=self.device,
            half=(self.device is not None and "cuda" in str(self.device)),
            verbose=False
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(None)
                continue
            best_idx = int(r.boxes.conf.argmax())
            box = r.boxes.xyxy[best_idx].cpu().numpy()
            conf = float(r.boxes.conf[best_idx].cpu().numpy())
            out.append((box, conf))
        return out

    def _finalize_track(self, track_id: int):
        state = self.tracks.get(track_id)
        if state is None or state.finalized or state.best_crop is None:
            return
        state.finalized = True
        fname = f"{track_id}_{state.best_score:.2f}.jpg"
        fpath = os.path.join(self.output_dir, "plates", fname)
        cv2.imwrite(fpath, state.best_crop)

        ocr_text, ocr_conf = "", 0.0
        if self.run_ocr and self.ocr_engine is not None:
            ocr_text, ocr_conf = self._run_ocr(state.best_crop)

        self.results_rows.append({
            "track_id": track_id,
            "best_frame_idx": state.best_frame_idx,
            "plate_crop_path": fpath,
            "quality_score": round(state.best_score, 3),
            "vehicle_bbox": state.best_vehicle_bbox,
            "plate_bbox": state.best_plate_bbox,
            "ocr_text": ocr_text,
            "ocr_confidence": round(ocr_conf, 3),
        })

    def _run_ocr(self, crop: np.ndarray):
        try:
            result = self.ocr_engine.readtext(crop)
            if not result:
                return "", 0.0
            texts, confs = [], []
            for box, text, conf in result:
                texts.append(text)
                confs.append(conf)
            return " ".join(texts), float(np.mean(confs))
        except Exception:
            return "", 0.0

    def process_video(self, video_path: str, vehicle_infer_scale: float = 1.0,
                      plate_every: int = 1,
                      on_progress=None, on_frame=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        plate_every = max(1, int(plate_every))
        h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        last_plate_boxes = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Stage 1: vehicle detection frame (optionally downscaled)
            if vehicle_infer_scale != 1.0:
                infer_frame = cv2.resize(frame, None, fx=vehicle_infer_scale, fy=vehicle_infer_scale)
            else:
                infer_frame = frame

            track_results = self.vehicle_model.track(
                infer_frame, persist=True, conf=self.conf_threshold,
                device=self.device, verbose=False, tracker="bytetrack.yaml"
            )
            r = track_results[0]

            boxes = []
            plate_boxes_full = []
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                # --- collect every vehicle crop in this frame first ---
                vehicle_crops, vehicle_meta = [], []
                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    if cls not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = box / vehicle_infer_scale
                    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w_full, h_full)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    vehicle_crops.append(frame[y1:y2, x1:x2])
                    vehicle_meta.append((int(track_id), (x1, y1, x2, y2)))

                # --- one batched plate-model forward pass (every plate_every frames) ---
                if frame_idx % plate_every == 0:
                    plate_results = self._detect_plates_batched(vehicle_crops)
                    plate_boxes_full = []

                    for vehicle_crop, (track_id, (x1, y1, x2, y2)), plate_result in zip(
                            vehicle_crops, vehicle_meta, plate_results):
                        if plate_result is None:
                            continue
                        (px1, py1, px2, py2), plate_conf = plate_result
                        px1, py1, px2, py2 = clamp_bbox(px1, py1, px2, py2,
                                                         vehicle_crop.shape[1], vehicle_crop.shape[0])
                        if px2 <= px1 or py2 <= py1:
                            continue
                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                        score = quality_score(plate_crop, plate_conf,
                                               self.min_plate_width, self.min_blur)
                        if score < 0:
                            continue

                        state = self.tracks.setdefault(track_id, TrackState())
                        state.last_seen_frame = frame_idx
                        if score > state.best_score:
                            state.best_score = score
                            state.best_crop = plate_crop.copy()
                            state.best_frame_idx = frame_idx
                            state.best_plate_bbox = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                            state.best_vehicle_bbox = (x1, y1, x2, y2)
                        plate_boxes_full.append((x1 + px1, y1 + py1, x1 + px2, y1 + py2))

                    if plate_boxes_full:
                        last_plate_boxes = plate_boxes_full
                else:
                    plate_boxes_full = list(last_plate_boxes)

            # finalize stale tracks
            for tid, state in list(self.tracks.items()):
                if not state.finalized and frame_idx - state.last_seen_frame > self.max_missed_frames:
                    self._finalize_track(tid)

            if on_frame is not None:
                for box in boxes:
                    if len(box) == 4:
                        x1, y1, x2, y2 = (int(b) for b in box)
                        cv2.rectangle(infer_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for p in plate_boxes_full:
                    sx1, sy1, sx2, sy2 = (int(v * vehicle_infer_scale) for v in p)
                    cv2.rectangle(infer_frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                on_frame(infer_frame, frame_idx)

            frame_idx += 1
            if on_progress is not None:
                on_progress(frame_idx, total)
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames, {len(self.tracks)} tracks so far")

        cap.release()

        # finalize anything still open at end of video
        for tid in list(self.tracks.keys()):
            self._finalize_track(tid)

        self._write_csv()
        print(f"Done. {len(self.results_rows)} plates saved to {self.output_dir}/plates/")

    def _write_csv(self):
        csv_path = os.path.join(self.output_dir, "results.csv")
        fieldnames = ["track_id", "best_frame_idx", "plate_crop_path",
                      "quality_score", "vehicle_bbox", "plate_bbox",
                      "ocr_text", "ocr_confidence"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results_rows:
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="4K video license plate capture pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--vehicle-model", default="yolov8n.pt",
                         help="Vehicle detection weights (COCO-pretrained works)")
    parser.add_argument("--plate-model", required=True,
                         help="Plate detection weights (must be trained/fine-tuned for plates)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--min-plate-width", type=int, default=80)
    parser.add_argument("--min-blur", type=float, default=50.0)
    parser.add_argument("--max-missed-frames", type=int, default=30)
    parser.add_argument("--vehicle-infer-scale", type=float, default=0.5,
                         help="Downscale factor for vehicle detection pass on 4K input. "
                              "0.5 -> vehicle pass runs at 1080p, plate crops are still taken "
                              "from the full-res 4K frame. Set to 1.0 only if you need to "
                              "detect very distant/small vehicles.")
    parser.add_argument("--ocr", action="store_true", help="Run OCR on final crops")
    parser.add_argument("--device", default=None, help="e.g. 'cuda:0', 'mps', 'cpu'")
    args = parser.parse_args()

    pipeline = PlateCapturePipeline(
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        min_plate_width=args.min_plate_width,
        min_blur=args.min_blur,
        max_missed_frames=args.max_missed_frames,
        run_ocr=args.ocr,
        device=args.device,
    )
    pipeline.process_video(args.video, vehicle_infer_scale=args.vehicle_infer_scale)


if __name__ == "__main__":
    main()
