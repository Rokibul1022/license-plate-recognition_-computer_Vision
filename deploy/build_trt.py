"""
Build a TensorRT engine from the exported ONNX model using trtexec.

    ONNX -> TensorRT engine

Precision options: fp32 (default), fp16, int8 (requires a calibration cache).

Dynamic-shape handling:
- If the ONNX input is dynamic (default export), min/opt/max shape profiles
  are passed to trtexec so the engine accepts a range of batch sizes.
- If the ONNX input is fixed, shape profiles are skipped automatically.

Run from the project root:

    python deploy/build_trt.py [--onnx ...] [--engine ...] [--precision fp16]

Output: outputs/deploy/plate_detector_fp16.engine
"""

import argparse
import os
import shutil
import subprocess

import onnxruntime as ort


def _input_profile(onnx_path, imgsz, batch):
    """Return (is_dynamic, [min, opt, max] shape strings)."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    is_dynamic = any(dim is None for dim in shape)
    if not is_dynamic:
        return False, []
    name = inp.name
    return True, [
        f"{name}:1x3x{imgsz}x{imgsz}",
        f"{name}:{batch}x3x{imgsz}x{imgsz}",
        f"{name}:{max(batch, 16)}x3x{imgsz}x{imgsz}",
    ]


def build_engine(onnx_path, engine_path, precision="fp16", workspace=4096, imgsz=640, batch=1):
    if shutil.which("trtexec") is None:
        raise RuntimeError(
            "trtexec not found on PATH. Install TensorRT and add its 'bin' "
            "directory (containing trtexec.exe on Windows) to PATH."
        )

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    is_dynamic, profiles = _input_profile(onnx_path, imgsz, batch)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace}",
        "--verbose",
    ]
    if is_dynamic:
        cmd += [f"--minShapes={profiles[0]}", f"--optShapes={profiles[1]}", f"--maxShapes={profiles[2]}"]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine saved to: {engine_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX")
    p.add_argument("--onnx", default="outputs/detection/plate_detector/weights/best.onnx")
    p.add_argument("--engine", default="outputs/deploy/plate_detector_fp16.engine")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--workspace", type=int, default=4096, help="Max workspace size in MB")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1, help="Optimum batch size for dynamic profiles")
    args = p.parse_args()
    build_engine(args.onnx, args.engine, args.precision, args.workspace, args.imgsz, args.batch)
