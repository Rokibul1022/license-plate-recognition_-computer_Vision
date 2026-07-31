"""
Verify that ONNX / TensorRT exports preserve detection accuracy.

Runs the ONNX and TensorRT engines on real validation images and scores them
against the PyTorch baseline (treated as pseudo-ground truth):

    mAP@0.5 | Precision@0.5 | Recall@0.5

A large drop between backends means the export silently broke accuracy.

Run from the project root:

    python deploy/verify_accuracy.py [--data-yaml data.yaml] [--limit 100]
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _load_images(data_yaml, images_dir, limit):
    paths = []
    if images_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths += glob.glob(os.path.join(images_dir, "**", ext), recursive=True)
    elif data_yaml:
        import yaml

        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        root = cfg.get("path", ".")
        if not os.path.isabs(root):
            root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
        val = cfg.get("val", "")
        val_dir = val if os.path.isabs(val) else os.path.join(root, val)
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths += glob.glob(os.path.join(val_dir, "**", ext), recursive=True)
    else:
        raise ValueError("Provide --images <dir> or --data-yaml <data.yaml>")

    paths = sorted(p for p in paths if os.path.isfile(p))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError("No images found in the given data source.")
    return paths


def preprocess(img_bgr, imgsz):
    h, w = img_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), r, left, top


def batch_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    y1 = np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    x2 = np.minimum(a[:, 2][:, None], b[:, 2][None, :])
    y2 = np.minimum(a[:, 3][:, None], b[:, 3][None, :])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes, scores, iou_thres):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = batch_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=int)


def postprocess(output, r, left, top, imgsz, conf_thres, iou_thres):
    out = np.asarray(output[0]).T
    boxes = out[:, :4]
    scores = out[:, 4:]
    cls_id = scores.argmax(1)
    conf = scores.max(1)

    mask = conf >= conf_thres
    boxes, cls_id, conf = boxes[mask], cls_id[mask], conf[mask]
    if len(boxes) == 0:
        return []

    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(xyxy, conf, iou_thres)
    xyxy, conf = xyxy[keep], conf[keep]

    scale = 1.0 / r
    xyxy *= scale
    xyxy[:, [0, 2]] -= left * scale
    xyxy[:, [1, 3]] -= top * scale
    return [(box, float(c)) for box, c in zip(xyxy.tolist(), conf.tolist())]


def ap50(gt_boxes, preds):
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    n = len(preds)
    if n == 0:
        return 1.0 if not gt_boxes else 0.0
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)
    matched = set()

    for i, (box, _) in enumerate(preds):
        if not gt_boxes:
            fp[i] = 1
            continue
        ious = batch_iou(np.asarray(box, dtype=np.float32)[None], np.asarray(gt_boxes, dtype=np.float32))[0]
        ious[list(matched)] = -1
        j = int(np.argmax(ious))
        if ious[j] >= 0.5:
            tp[i] = 1
            matched.add(j)
        else:
            fp[i] = 1

    n_gt = max(len(gt_boxes), 1)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    if len(idx) == 0:
        return 1.0 if n == 0 else 0.0
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    p = argparse.ArgumentParser(description="Verify ONNX/TRT accuracy vs PyTorch baseline")
    p.add_argument("--weights", default="outputs/detection/plate_detector/weights/best.pt")
    p.add_argument("--onnx", default="outputs/detection/plate_detector/weights/best.onnx")
    p.add_argument("--engine", default="outputs/deploy/plate_detector_fp16.engine")
    p.add_argument("--data-yaml", default="data.yaml")
    p.add_argument("--images", default=None, help="Image directory (overrides --data-yaml)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    images = _load_images(args.data_yaml, args.images, args.limit)
    print(f"Evaluating {len(images)} images")

    import onnxruntime as ort

    baseline = YOLO(args.weights)
    sess = ort.InferenceSession(
        args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    ort_dtype = np.float16 if "float16" in inp.type else np.float32

    from trt_infer import TRTDetector

    trt = TRTDetector(args.engine, (1, 3, args.imgsz, args.imgsz))
    trt_dtype = trt.inputs[0]["dtype"]

    def run_ort(img):
        x, r, left, top = preprocess(img, args.imgsz)
        out = sess.run(None, {inp.name: x.astype(ort_dtype)})[0]
        return postprocess(out, r, left, top, args.imgsz, args.conf, args.iou)

    def run_trt(img):
        x, r, left, top = preprocess(img, args.imgsz)
        out = trt.infer(x.astype(trt_dtype))[0]
        return postprocess(out, r, left, top, args.imgsz, args.conf, args.iou)

    def run_pt(path):
        res = baseline.predict(path, conf=args.conf, iou=args.iou, verbose=False, device=args.device)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [(box, float(c)) for box, c in zip(boxes, confs)]

    def score(preds, gt_boxes):
        ap = ap50(gt_boxes, preds)
        n_matched = 0
        matched_gt = set()
        for box, _ in preds:
            if not gt_boxes:
                break
            ious = batch_iou(
                np.asarray(box, dtype=np.float32)[None], np.asarray(gt_boxes, dtype=np.float32)
            )[0]
            ious[list(matched_gt)] = -1
            j = int(np.argmax(ious))
            if ious[j] >= 0.5:
                n_matched += 1
                matched_gt.add(j)
        precision = n_matched / max(len(preds), 1)
        recall = n_matched / max(len(gt_boxes), 1)
        return ap, precision, recall

    agg = {"ONNX": [], "TensorRT": []}
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt_boxes = [b for b, _ in run_pt(img_path)]
        agg["ONNX"].append(score(run_ort(img), gt_boxes))
        agg["TensorRT"].append(score(run_trt(img), gt_boxes))

    print(f"\n{'Backend':12s} | {'mAP@0.5':>8s} | {'Prec@0.5':>8s} | {'Recall@0.5':>10s}")
    print("-" * 50)
    for name, scores in agg.items():
        if not scores:
            continue
        mean = np.mean(scores, axis=0)
        print(f"{name:12s} | {mean[0]:8.4f} | {mean[1]:8.4f} | {mean[2]:10.4f}")
    print("\nScored against the PyTorch baseline detections (pseudo-ground truth).")


if __name__ == "__main__":
    main()
