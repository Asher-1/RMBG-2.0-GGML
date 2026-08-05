#!/usr/bin/env python3
"""Create the checked-in RMBG inference and latency comparison figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_alpha(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    return np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("docs/rmbg_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    paths = {
        key: (Path(value) if Path(value).is_absolute() else ROOT / value)
        for key, value in data["images"].items()
    }
    rgb = load_rgb(paths["input"])
    alpha = {key: load_alpha(path) for key, path in paths.items() if key != "input"}

    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("Input", rgb),
        ("PyTorch", alpha["pytorch"], "alpha"),
        ("GGML CUDA", alpha["cuda"], "alpha"),
        ("GGML Vulkan", alpha["vulkan"], "alpha"),
        ("abs diff: CUDA - PyTorch", np.abs(alpha["cuda"] - alpha["pytorch"]), "diff"),
        ("abs diff: Vulkan - PyTorch", np.abs(alpha["vulkan"] - alpha["pytorch"]), "diff"),
    ]
    for axis, panel in zip(axes.flat, panels):
        title, image, *kind = panel
        if kind and kind[0] == "diff":
            artist = axis.imshow(image, cmap="magma", vmin=0.0, vmax=0.05)
            axis.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        elif kind and kind[0] == "alpha":
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("RMBG-2.0 output comparison (same input, 1024x1024 inference)")
    fig.savefig(args.output_dir / "rmbg_inference_comparison.png", dpi=160)
    plt.close(fig)

    labels = []
    values = []
    colors = []
    for key, item in data["latency_ms"].items():
        if not item.get("chart", True):
            continue
        labels.append(item["label"])
        values.append(item["mean"])
        colors.append(item.get("color", "#4c78a8"))
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.6)
    axis.set_ylabel("Mean latency (ms, lower is better)")
    axis.set_title("RMBG-2.0 end-to-end inference latency")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}",
                  ha="center", va="bottom", fontsize=9)
    fig.savefig(args.output_dir / "rmbg_latency_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
