#!/usr/bin/env python3
"""Render inference, robust latency, speedup, and parity figures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def label(case: dict[str, Any]) -> str:
    if case["runtime"] == "pytorch":
        return f"PyTorch {case['backend']} {case['math_mode']}"
    return f"GGML {case['backend']} {case['model']} {case['math_mode']}"


def case_color(case: dict[str, Any]) -> str:
    return {
        "pytorch": "#e07a1f", "cpu": "#777777",
        "cuda": "#2e8b57", "vulkan": "#3973ac",
    }.get(case["runtime"] if case["runtime"] == "pytorch" else case["backend"], "#555555")


def parity_passed(case: dict[str, Any]) -> bool:
    return case.get("fixture_parity", {}).get("passed", True)


def load_alpha(case: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(resolve(case["output_image"])).getchannel("A"), dtype=np.float32) / 255.0


def plot_inference(cases: list[dict[str, Any]], output: Path) -> None:
    by_id = {case["id"]: case for case in cases if "output_image" in case}
    preferred = [
        "pytorch-cuda-optimized", "ggml-cuda-f16-optimized",
        "ggml-vulkan-f16-optimized", "ggml-cuda-q8-optimized",
    ]
    # Reports generated before the mode rename remain renderable.
    preferred = [
        case_id for case_id in preferred
        if case_id in by_id
    ] or [
        case_id for case_id in (
            "pytorch-cuda-strict", "ggml-cuda-f16-fast",
            "ggml-vulkan-f16-strict", "ggml-cuda-q8-fast",
        ) if case_id in by_id
    ]
    selected = [by_id[case_id] for case_id in preferred if case_id in by_id]
    if len(selected) < 2:
        return
    reference = load_alpha(selected[0])

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(selected), figsize=(4.1 * len(selected), 7.5),
                             constrained_layout=True, squeeze=False)
    for column, case in enumerate(selected):
        alpha = load_alpha(case)
        axes[0, column].imshow(alpha, cmap="gray", vmin=0.0, vmax=1.0)
        status = "PASS" if parity_passed(case) else "FAIL"
        axes[0, column].set_title(f"{label(case)}\n{status}", fontsize=10)
        difference = np.abs(alpha - reference)
        artist = axes[1, column].imshow(difference, cmap="magma", vmin=0.0, vmax=0.05)
        axes[1, column].set_title(
            f"output-PNG abs diff vs PyTorch\n"
            f"max={difference.max():.4f}, mean={difference.mean():.4f}",
            fontsize=10,
        )
        for row in range(2):
            axes[row, column].axis("off")
    fig.colorbar(artist, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02,
                 label="absolute alpha difference")
    fig.suptitle(
        "RMBG-2.0 inference comparison (PASS/FAIL uses the 1024x1024 float fixture; "
        "diff uses resized 8-bit PNGs)"
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_all_outputs(cases: list[dict[str, Any]], output: Path) -> None:
    selected = [case for case in cases if "output_image" in case]
    if not selected:
        return

    import matplotlib.pyplot as plt

    columns = 4
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.3 * rows),
                             constrained_layout=True, squeeze=False)
    for axis, case in zip(axes.flat, selected):
        axis.imshow(load_alpha(case), cmap="gray", vmin=0.0, vmax=1.0)
        status = "PASS" if parity_passed(case) else "FAIL"
        axis.set_title(f"{label(case)} | {status}\n{case['mean_ms']:.1f} ms", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    fig.suptitle("All measured RMBG backend x model outputs")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_metrics(cases: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(
        cases,
        key=lambda case: (
            case["backend"] == "cpu", case["runtime"] != "pytorch",
            case["backend"], case["model"], case["math_mode"],
        ),
    )
    gpu = [case for case in ordered if case["backend"] != "cpu"]
    cpu = [case for case in ordered if case["backend"] == "cpu"]
    ggml = [case for case in ordered if case["runtime"] == "ggml"]
    parity = [case for case in ggml if "fixture_parity" in case]
    labels = [label(case) for case in gpu]
    positions = np.arange(len(gpu))

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    gpu_axis, cpu_axis, speed_axis, parity_axis = axes.flat
    colors = [case_color(case) for case in gpu]
    medians = np.asarray([case["median_ms"] for case in gpu])
    p95 = np.asarray([case["p95_ms"] for case in gpu])
    bars = gpu_axis.barh(positions, medians, xerr=np.maximum(0, p95 - medians),
                         color=colors, edgecolor="#263238", linewidth=0.5)
    for bar, case in zip(bars, gpu):
        if not parity_passed(case):
            bar.set_hatch("////")
        gpu_axis.text(case["median_ms"] + max(medians) * 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      f"{case['median_ms']:.1f}", va="center", fontsize=8)
    gpu_axis.set_yticks(positions, labels, fontsize=8)
    gpu_axis.invert_yaxis()
    gpu_axis.set_xlabel("Latency (ms), median with error bar to p95")
    gpu_axis.set_title("GPU latency (robust statistic)")
    gpu_axis.grid(axis="x", alpha=0.25)

    cpu_pos = np.arange(len(cpu))
    cpu_means = np.asarray([case["median_ms"] for case in cpu])
    cpu_p95 = np.asarray([case["p95_ms"] for case in cpu])
    cpu_bars = cpu_axis.barh(
        cpu_pos, cpu_means, xerr=np.maximum(0, cpu_p95 - cpu_means),
        color=[case_color(case) for case in cpu], edgecolor="#263238", linewidth=0.5,
    )
    for bar, case in zip(cpu_bars, cpu):
        if not parity_passed(case):
            bar.set_hatch("////")
        cpu_axis.text(case["mean_ms"] + max(cpu_means) * 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      f"{case['mean_ms']:.1f}", va="center", fontsize=8)
    cpu_axis.set_yticks(cpu_pos, [label(case) for case in cpu], fontsize=8)
    cpu_axis.invert_yaxis()
    cpu_axis.set_xlabel("Latency (ms), median with error bar to p95")
    cpu_axis.set_title("CPU latency (16 threads)")
    cpu_axis.grid(axis="x", alpha=0.25)

    speed_cases = [
        case for case in ggml
        if "speedup_median_vs_pytorch_cuda_matching_mode" in case
    ]
    speed_values = [
        case["speedup_median_vs_pytorch_cuda_matching_mode"] for case in speed_cases
    ]
    speed_pos = np.arange(len(speed_cases))
    speed_bars = speed_axis.barh(speed_pos, speed_values,
                                 color=[case_color(case) for case in speed_cases])
    for bar, case in zip(speed_bars, speed_cases):
        if not parity_passed(case):
            bar.set_hatch("////")
    speed_axis.axvline(1.0, color="#b03030", linestyle="--", linewidth=1)
    speed_axis.set_yticks(speed_pos, [label(case) for case in speed_cases], fontsize=8)
    speed_axis.invert_yaxis()
    speed_axis.set_xlabel("Median speedup vs PyTorch CUDA with matching mode (x)")
    speed_axis.set_title("Runtime speedup (hatched = parity fail)")
    speed_axis.grid(axis="x", alpha=0.25)

    parity_values = [case["fixture_parity"]["max_abs_diff"] for case in parity]
    parity_pos = np.arange(len(parity))
    parity_axis.barh(
        parity_pos, parity_values,
        color=["#33885a" if parity_passed(case) else "#ba4a48" for case in parity],
    )
    parity_axis.axvline(2e-3, color="#b03030", linestyle="--", linewidth=1,
                        label="2e-3 gate")
    parity_axis.set_xscale("log")
    parity_axis.set_yticks(parity_pos, [label(case) for case in parity], fontsize=8)
    parity_axis.invert_yaxis()
    parity_axis.set_xlabel("Fixture max absolute alpha difference (log scale)")
    parity_axis.set_title("Numerical parity")
    parity_axis.legend()
    parity_axis.grid(axis="x", alpha=0.25)

    fig.suptitle("RMBG-2.0 backend x model benchmark matrix")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("docs/rmbg_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2 or not data.get("cases"):
        raise SystemExit("benchmark data must use schema_version 2 and contain cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_inference(data["cases"], args.output_dir / "rmbg_inference_comparison.png")
    plot_all_outputs(data["cases"], args.output_dir / "rmbg_all_outputs_comparison.png")
    plot_metrics(data["cases"], args.output_dir / "rmbg_latency_comparison.png")


if __name__ == "__main__":
    main()
