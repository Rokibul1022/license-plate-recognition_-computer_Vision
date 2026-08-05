"""
Reusable detection engine: YOLO (vehicle detect) + OpenCV + EasyOCR (plate read).
Used by both the CLI (video_demo.py) and the Streamlit demo app (app/demo_app.py).

Draws green vehicle boxes and red plate boxes. Reads the plate immediately
when the plate region is located.
"""

import os
import pathlib

import cv2

# YOLO COCO classes to treat as tracked vehicles (car, motorcycle, bus, truck).
# Filtering by these avoids drawing boxes on people, signs, lights, etc.
VEHICLE_IDS = [2, 3, 5, 7]


def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def load_ocr():
    import easyocr
    return easyocr.Reader(["bn", "en"], gpu=True)


def normalize_plate(text):
    import re
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def is_plate_candidate(text):
    t = normalize_plate(text)
    return (4 <= len(t) <= 18 and any(c.isdigit() for c in t)
            and any(c.isalpha() for c in t))


def detect_plate_region(crop):
    """Classic ANPR: locate a license-plate rectangle in a vehicle crop.

    Returns (x, y, w, h) in crop coords or None.
    """
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, th = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = closed.shape
    best, best_area = None, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 150 or w < 30 or h < 12:
            continue
        ar = w / float(h)
        if not (0.5 < ar < 6.0):
            continue
        if area / float(H * W) < 0.003:
            continue
        # map back to crop coords (we upscaled 3x)
        r = (x // 3, y // 3, w // 3, h // 3)
        if area > best_area:
            best, best_area = r, area
    return best


class PlateTracker:
    """Very lightweight IoU tracker so each passing vehicle accumulates its
    plate read over multiple frames."""

    def __init__(self, iou_thresh=0.2, max_age=25):
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._nid = 0

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter == 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua

    def update(self, boxes, frame_idx, plate_text="", located=False):
        """Match detection boxes to existing tracks. Returns list of active
        track dicts: {id, box, frame, plate_text, located, read}."""
        boxes = list(boxes)
        used = set()

        # Greedy match new boxes to tracks by IoU, largest-first.
        idx = sorted(range(len(boxes)), key=lambda i: -self._area(boxes[i]))
        for bi in idx:
            b = boxes[bi]
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(self.tracks):
                if i in used or t["age"] > self.max_age:
                    continue
                iou = self._iou(b, t["box"])
                if iou > self.iou_thresh and iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0:
                t = self.tracks[best_i]
                t["box"] = b
                t["frame"] = frame_idx
                t["age"] = 0
                if plate_text and not t["read"]:
                    t["plate_text"] = plate_text
                if located:
                    t["located"] = True
                used.add(best_i)
            else:
                self.tracks.append({
                    "id": self._nid, "box": b, "frame": frame_idx, "age": 0,
                    "plate_text": plate_text, "located": located,
                    "plate_id": f"V{self._nid}", "read": bool(plate_text)})
                self._nid += 1

        active = []
        for t in self.tracks:
            t["age"] += 1
            if t["age"] <= self.max_age:
                active.append(t)
        return active

    @staticmethod
    def _area(b):
        return (b[2] - b[0]) * (b[3] - b[1])


def _ocr_plate_crop(reader, crop, scale=4):
    """Upscale a small plate crop and OCR it. Returns plate text or None."""
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    best = None
    for (pts, text, conf) in reader.readtext(up):
        if conf < 0.2 or not is_plate_candidate(text):
            continue
        t = normalize_plate(text)
        if best is None or len(t) > len(best):
            best = t
    return best


def process_video(video_path, output_path, ocr_every=1, yolo_every=1, max_width=1280,
                  max_ocr_vehicles=4, on_progress=None, on_status=None,
                  on_frame=None, show=False, model=None, reader=None):
    """Process a video file. Returns (output_path, plates, summary).

    Frames are downscaled to `max_width` for fast inference; the annotated
    output is written at that resolution.

    Green box = vehicle, red box = plate. Plate text is read immediately
    (every frame, on the small plate crop) as soon as a plate is located.

    on_progress(processed, total): called periodically while decoding.
    on_status(message): called with short status messages.
    on_frame(frame, frame_idx): called with each annotated frame.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if on_status:
        on_status("Loading YOLO detector...")
    model = model if model is not None else load_yolo()

    if on_status:
        on_status("Loading EasyOCR (first run downloads models)...")
    reader = reader if reader is not None else load_ocr()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / src_w) if src_w else 1.0
    width = int(src_w * scale)
    height = int(src_h * scale)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = PlateTracker()
    read_plates = set()
    frame_idx = 0
    while cap.isOpened():
        ret, frame_hi = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame_hi, (width, height), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_hi

        if frame_idx % yolo_every == 0:
            results = model.predict(frame, classes=VEHICLE_IDS, verbose=False, imgsz=640)
            vehicle_boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]),
                              int(b.xyxy[0][3])) for b in results[0].boxes]

        tracks = tracker.update(vehicle_boxes, frame_idx)

        if frame_idx % ocr_every == 0:
            inv = 1.0 / scale if scale > 0 else 1.0
            by_area = sorted(tracks, reverse=True,
                             key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]))
            for t in by_area[:max_ocr_vehicles]:
                if t["read"]:
                    continue
                vx1, vy1, vx2, vy2 = t["box"]
                vw, vh = vx2 - vx1, vy2 - vy1
                if vw < 30 or vh < 30:
                    continue
                # hi-res region
                hx1, hy1 = int(vx1 * inv), int(vy1 * inv)
                hx2, hy2 = int(vx2 * inv), int(vy2 * inv)
                hcy1 = int(hy1 + (hy2 - hy1) * 0.40)
                hcrop = frame_hi[hcy1:hy2, hx1:hx2]
                if hcrop.size == 0:
                    continue
                plate = None
                region = detect_plate_region(hcrop)
                if region:
                    rx, ry, rw, rh = region
                    px1, py1 = hx1 + rx, hcy1 + ry
                    px2, py2 = px1 + rw, py1 + rh
                    t["located"] = True
                    rcrop = frame_hi[py1:py2, px1:px2]
                    plate = _ocr_plate_crop(reader, rcrop) if rcrop.size else None
                    if plate:
                        sx1, sy1 = int(px1 * scale), int(py1 * scale)
                        sx2, sy2 = int(px2 * scale), int(py2 * scale)
                        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                        cv2.putText(frame, plate, (sx1, max(sy1 - 8, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                if plate is None:
                    found = _ocr_plate_crop(reader, hcrop)
                    if found:
                        plate = found
                if plate:
                    t["plate_text"] = plate
                    t["read"] = True
                    read_plates.add(plate)

        # green vehicle boxes
        for t in tracks:
            vx1, vy1, vx2, vy2 = t["box"]
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1
        if show:
            cv2.imshow("Live Detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if on_frame:
            on_frame(frame, frame_idx)
        if on_progress and frame_idx % 10 == 0:
            on_progress(frame_idx, total)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    located = sum(1 for t in tracker.tracks if t["located"])
    read = sum(1 for t in tracker.tracks if t["read"])
    summary = {"vehicles": len(tracker.tracks),
               "plates_located": located, "plates_read": read}
    print(f"SUMMARY: {summary['vehicles']} vehicles, "
          f"{summary['plates_located']} plates located, {summary['plates_read']} read")

    return output_path, sorted(read_plates), summary


def process_camera(camera_index=0, ocr_every=1, yolo_every=1, max_width=1280,
                   max_ocr_vehicles=4, on_status=None, on_frame=None):
    """Real-time webcam detection. Driven by the live camera frame rate.
    Press q or ESC to quit. No output file."""
    if on_status:
        on_status("Loading YOLO detector...")
    model = load_yolo()

    if on_status:
        on_status("Loading EasyOCR (first run downloads models)...")
    reader = load_ocr()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    tracker = PlateTracker()
    read_plates = set()
    frame_idx = 0
    while True:
        ret, frame_hi = cap.read()
        if not ret:
            break
        frame = frame_hi
        h_full, w_full = frame_hi.shape[:2]
        scale = min(1.0, max_width / w_full) if w_full else 1.0
        if scale < 1.0:
            frame = cv2.resize(
                frame_hi, (int(w_full * scale), int(h_full * scale)),
                interpolation=cv2.INTER_AREA)

        if frame_idx % yolo_every == 0:
            res = model.predict(frame, classes=VEHICLE_IDS, verbose=False, imgsz=640)
            vehicle_boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]),
                              int(b.xyxy[0][3])) for b in res[0].boxes]

        tracks = tracker.update(vehicle_boxes, frame_idx)

        if frame_idx % ocr_every == 0:
            inv = 1.0 / scale if scale > 0 else 1.0
            by_area = sorted(tracks, reverse=True,
                             key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]))
            for t in by_area[:max_ocr_vehicles]:
                if t["read"]:
                    continue
                vx1, vy1, vx2, vy2 = t["box"]
                if (vx2 - vx1) < 30 or (vy2 - vy1) < 30:
                    continue
                hx1, hy1 = int(vx1 * inv), int(vy1 * inv)
                hx2, hy2 = int(vx2 * inv), int(vy2 * inv)
                hcy1 = int(hy1 + (hy2 - hy1) * 0.40)
                hcrop = frame_hi[hcy1:hy2, hx1:hx2]
                if hcrop.size == 0:
                    continue
                plate = None
                region = detect_plate_region(hcrop)
                if region:
                    rx, ry, rw, rh = region
                    px1, py1 = hx1 + rx, hcy1 + ry
                    px2, py2 = px1 + rw, py1 + rh
                    t["located"] = True
                    rcrop = frame_hi[py1:py2, px1:px2]
                    plate = _ocr_plate_crop(reader, rcrop) if rcrop.size else None
                    if plate:
                        sx1, sy1 = int(px1 * scale), int(py1 * scale)
                        sx2, sy2 = int(px2 * scale), int(py2 * scale)
                        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                        cv2.putText(frame, plate, (sx1, max(sy1 - 8, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                if plate is None:
                    found = _ocr_plate_crop(reader, hcrop)
                    if found:
                        plate = found
                if plate:
                    t["plate_text"] = plate
                    t["read"] = True
                    read_plates.add(plate)

        for t in tracks:
            vx1, vy1, vx2, vy2 = t["box"]
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)

        if on_frame:
            on_frame(frame, frame_idx)
        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    summary = {"vehicles": len(tracker.tracks),
               "plates_located": sum(1 for t in tracker.tracks if t["located"]),
               "plates_read": sum(1 for t in tracker.tracks if t["read"])}
    print(f"SUMMARY: {summary['vehicles']} vehicles, "
          f"{summary['plates_located']} plates located, {summary['plates_read']} read")
    return summary
