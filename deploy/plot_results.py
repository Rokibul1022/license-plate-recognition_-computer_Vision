"""
Render the benchmark chart from an existing results JSON.

    JSON  ->  chart

Reads outputs/deploy/benchmark_results.json and writes
outputs/deploy/benchmark_chart.png.

Run from the project root:

    python deploy/plot_results.py [--json outputs/deploy/benchmark_results.json]
"""

import argparse
import json

from benchmark import plot_results


def main():
    p = argparse.ArgumentParser(description="Plot benchmark results from JSON")
    p.add_argument("--json", default="deploy/results/benchmark_results.json")
    p.add_argument("--out-chart", default="deploy/results/benchmark_chart.png")
    args = p.parse_args()

    with open(args.json, encoding="utf-8") as f:
        report = json.load(f)

    device = report.get("device", "unknown")
    rows = report["backends"]
    for r in rows:
        expected = 1000.0 / r["latency_ms"]
        if abs(expected - r["fps"]) / expected > 0.05:
            print(
                f"WARNING: '{r['name']}' fps ({r['fps']:.1f}) does not match "
                f"latency_ms ({r['latency_ms']} ms -> {expected:.1f} FPS). "
                f"Fix the JSON or re-run the benchmark."
            )
    print(f"Loaded {len(rows)} backends from {args.json}")
    print(f"{'Backend':20s} | {'Latency':>9s} | {'FPS':>8s}")
    print("-" * 44)
    for r in rows:
        print(f"{r['name']:20s} | {r['latency_ms']:8.2f}ms | {r['fps']:7.1f}")

    plot_results(rows, args.out_chart, device)


if __name__ == "__main__":
    main()
