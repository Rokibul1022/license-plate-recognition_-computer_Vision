"""
Export the trained YOLOv8 plate detector (.pt) to ONNX.

    PyTorch -> ONNX

Supports dynamic batch/spatial shapes and optional FP16 export.
Run from the project root:

    python deploy/export_onnx.py [--weights ...] [--imgsz 640] [--half]

Output: outputs/detection/plate_detector/weights/best.onnx
"""

import argparse

from ultralytics import YOLO


def export(weights_path, imgsz=640, dynamic=True, half=False, simplify=True, opset=17):
    model = YOLO(weights_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        half=half,
        simplify=simplify,
        opset=opset,
        device="cuda" if half else "cpu",
    )
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export YOLOv8 (.pt) to ONNX")
    p.add_argument("--weights", default="outputs/detection/plate_detector/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="Export in FP16 (requires a CUDA GPU)")
    p.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        default=True,
        help="Disable dynamic batch/shape (fixed 1x3xHxW)",
    )
    args = p.parse_args()
    export(args.weights, imgsz=args.imgsz, dynamic=args.dynamic, half=args.half)
