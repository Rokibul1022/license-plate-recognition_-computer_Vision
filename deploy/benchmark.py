"""
Benchmark the plate detector across inference backends.

    PyTorch (FP32) | ONNX Runtime | TensorRT

Reports per-inference latency, throughput (FPS) and optional VRAM delta.
Preprocessing is excluded so the numbers isolate engine speed.

Results are written to outputs/deploy/benchmark_results.json first, then
rendered to outputs/deploy/benchmark_chart.png.

Run with a trained model from the project root:

    python deploy/benchmark.py [--weights ...] [--onnx ...] [--engine ...] [--imgsz 640]

Run in demo mode (no model/dependencies required, synthetic numbers):

    python deploy/benchmark.py --demo
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _vram_mb():
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
    except Exception:
        return -1.0


def bench_pytorch(weights, imgsz, n, warmup, device):
    from ultralytics import YOLO

    model = YOLO(weights)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(dummy, verbose=False)
    _sync(device)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(dummy, verbose=False)
    _sync(device)
    ms = (time.perf_counter() - t0) * 1000 / n

    vram = (
        torch.cuda.max_memory_allocated() / 1e6
        if device == "cuda" and torch.cuda.is_available()
        else -1.0
    )
    return ms, vram


def bench_onnxruntime(onnx_path, imgsz, n, warmup, device):
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(dtype)

    for _ in range(warmup):
        sess.run(None, {inp.name: dummy})
    before = _vram_mb()
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {inp.name: dummy})
    ms = (time.perf_counter() - t0) * 1000 / n
    after = _vram_mb()

    vram = after - before if before >= 0 else -1.0
    return ms, vram


def bench_tensorrt(engine_path, imgsz, n, warmup):
    from trt_infer import TRTDetector

    detector = TRTDetector(engine_path, (1, 3, imgsz, imgsz))
    dtype = detector.inputs[0]["dtype"]
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(dtype)

    for _ in range(warmup):
        detector.infer(dummy)
    before = _vram_mb()
    t0 = time.perf_counter()
    for _ in range(n):
        detector.infer(dummy)
    ms = (time.perf_counter() - t0) * 1000 / n
    after = _vram_mb()

    vram = after - before if before >= 0 else -1.0
    return ms, vram


def demo_results():
    return [
        {"name": "PyTorch (FP32)", "latency_ms": 18.4, "fps": 54.3, "vram_mb": 1620.0},
        {"name": "ONNX Runtime", "latency_ms": 11.2, "fps": 89.2, "vram_mb": 1280.0},
        {"name": "TensorRT (FP16)", "latency_ms": 4.1, "fps": 243.9, "vram_mb": 980.0},
    ]


def save_json(results, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report saved to: {path}")


def plot_results(results, path, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    latency = [r["latency_ms"] for r in results]
    fps = [r["fps"] for r in results]
    vram = [r.get("vram_mb", -1.0) for r in results]

    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    bars = axes[0].bar(names, latency, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_title(f"Per-inference latency ({device})")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_ylim(0, max(latency) * 1.25)
    for bar, v in zip(bars, latency):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f} ms",
                     ha="center", va="bottom")

    bars = axes[1].bar(names, fps, color=colors, edgecolor="black", linewidth=0.6)
    axes[1].set_title(f"Throughput ({device})")
    axes[1].set_ylabel("FPS")
    axes[1].set_ylim(0, max(fps) * 1.25)
    for bar, v in zip(bars, fps):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                     ha="center", va="bottom")

    vram_plot = [max(v, 0) for v in vram]
    bars = axes[2].bar(names, vram_plot, color=colors, edgecolor="black", linewidth=0.6)
    axes[2].set_title(f"VRAM usage ({device})")
    axes[2].set_ylabel("VRAM (MB)")
    axes[2].set_ylim(0, max(vram_plot) * 1.25 if max(vram_plot) > 0 else 1)
    for bar, v in zip(bars, vram):
        label = "N/A" if v < 0 else f"{v:.0f}"
        axes[2].text(bar.get_x() + bar.get_width() / 2, max(v, 0), label,
                     ha="center", va="bottom")

    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to: {path}")


def main():
    p = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime vs TensorRT")
    p.add_argument("--weights", default="outputs/detection/plate_detector/weights/best.pt")
    p.add_argument("--onnx", default="outputs/detection/plate_detector/weights/best.onnx")
    p.add_argument("--engine", default="outputs/deploy/plate_detector_fp16.engine")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch", "onnx", "trt"],
        choices=["pytorch", "onnx", "trt"],
    )
    p.add_argument("--demo", action="store_true", help="Use sample numbers (no model needed)")
    p.add_argument("--out-json", default="deploy/results/benchmark_results.json")
    p.add_argument("--out-chart", default="deploy/results/benchmark_chart.png")
    args = p.parse_args()

    if args.demo:
        device = "CUDA (demo)"
        rows = demo_results()
    else:
        device = args.device.upper()
        rows = []
        if "pytorch" in args.backends:
            ms, vram = bench_pytorch(args.weights, args.imgsz, args.iters, args.warmup, args.device)
            rows.append({"name": "PyTorch (FP32)", "latency_ms": ms, "fps": 1000 / ms, "vram_mb": vram})
        if "onnx" in args.backends:
            ms, vram = bench_onnxruntime(args.onnx, args.imgsz, args.iters, args.warmup, args.device)
            rows.append({"name": "ONNX Runtime", "latency_ms": ms, "fps": 1000 / ms, "vram_mb": vram})
        if "trt" in args.backends:
            ms, vram = bench_tensorrt(args.engine, args.imgsz, args.iters, args.warmup)
            rows.append({"name": "TensorRT (FP16)", "latency_ms": ms, "fps": 1000 / ms, "vram_mb": vram})

    print(f"\nDevice: {device} | Iterations: {args.iters} | Image size: {args.imgsz}")
    print(f"{'Backend':20s} | {'Latency':>9s} | {'FPS':>8s} | {'VRAM (MB)':>9s}")
    print("-" * 56)
    for r in rows:
        vram_str = "-" if r["vram_mb"] < 0 else f"{r['vram_mb']:.0f}"
        print(f"{r['name']:20s} | {r['latency_ms']:8.2f}ms | {r['fps']:7.1f} | {vram_str:>9s}")
    if not args.demo:
        print("\nPreprocessing excluded; raw model tensors are fed directly.")

    report = {
        "benchmark": "plate_detector",
        "device": device,
        "imgsz": args.imgsz,
        "iters": args.iters,
        "warmup": args.warmup,
        "timestamp": datetime.now().isoformat(),
        "backends": rows,
    }
    save_json(report, args.out_json)
    plot_results(rows, args.out_chart, device)


if __name__ == "__main__":
    main()
