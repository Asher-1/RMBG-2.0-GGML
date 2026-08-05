#!/usr/bin/env python3
"""Reproducible end-to-end RMBG GGML vs PyTorch benchmark."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

ROOT = Path(__file__).resolve().parent.parent


def benchmark_env(backend: str, math_mode: str, iterations: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(RMBG_ROOT=str(ROOT), RMBG_DEVICE=backend, RMBG_BENCH_ITERS=str(iterations))
    if backend == "cuda" and math_mode == "strict":
        env["NVIDIA_TF32_OVERRIDE"] = "0"
        env["RMBG_STRICT_MATH"] = "1"
    elif backend == "cuda":
        env.pop("NVIDIA_TF32_OVERRIDE", None)
        env.pop("RMBG_STRICT_MATH", None)
    elif math_mode == "strict":
        env.update(
            GGML_VK_DISABLE_F16="1",
            GGML_VK_DISABLE_COOPMAT="1",
            GGML_VK_DISABLE_COOPMAT2="1",
            GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="1",
        )
        env.pop("RMBG_VULKAN_FAST", None)
    else:
        env["RMBG_VULKAN_FAST"] = "1"
        for key in (
            "GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT",
            "GGML_VK_DISABLE_COOPMAT2", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT",
        ):
            env.pop(key, None)
    return env


def run_ggml(args, output_dir: Path) -> tuple[dict, np.ndarray]:
    build = ROOT / args.build_dir
    test = build / "tests" / "test_rmbg_graph"
    cli = build / "rmbg-cli"
    for executable in (test, cli):
        if not executable.exists():
            raise SystemExit(f"missing {executable}; build the requested backend first")

    env = benchmark_env(args.backend, args.math, args.runs)
    env["RMBG_WEIGHTS"] = str(Path(args.gguf).resolve())
    result = subprocess.run(
        [str(test)], env=env, text=True,
        capture_output=True, check=True, timeout=600,
    )
    log = result.stdout + result.stderr
    parity = re.search(r"max\|d\|=([\deE+.-]+) mean\|d\|=([\deE+.-]+)", log)
    timing = re.search(r"steady-state: iterations=\d+ mean=([\d.]+) ms best=([\d.]+) ms", log)
    if not parity or not timing:
        raise SystemExit(f"unable to parse graph benchmark output:\n{log}")

    output = output_dir / f"ggml-{args.backend}.png"
    begin = time.perf_counter()
    subprocess.run(
        [str(cli), "remove", "--model", args.gguf, "--input", args.input,
         "--output", str(output), "--device", args.backend],
        env=env, check=True, timeout=600,
    )
    cold_ms = (time.perf_counter() - begin) * 1000
    alpha = np.asarray(Image.open(output).getchannel("A"), dtype=np.float32)
    return {
        "mean_ms": float(timing.group(1)), "best_ms": float(timing.group(2)),
        "cold_cli_ms": cold_ms, "max_abs_diff_fixture": float(parity.group(1)),
        "mean_abs_diff_fixture": float(parity.group(2)),
    }, alpha


def run_pytorch(args, output_dir: Path) -> tuple[dict, np.ndarray]:
    device = torch.device(args.pytorch_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForImageSegmentation.from_pretrained(
        args.pytorch_model, trust_remote_code=True, local_files_only=args.local_files_only,
    ).eval().float().to(device)
    image = Image.open(args.input).convert("RGB")
    tensor = transforms.Compose([
        transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(image).unsqueeze(0).to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(2):
            model(tensor)[-1].sigmoid()
            sync()
        for _ in range(args.runs):
            sync()
            begin = time.perf_counter()
            alpha = model(tensor)[-1].sigmoid().cpu()[0, 0]
            sync()
            times.append((time.perf_counter() - begin) * 1000)

    mask = transforms.ToPILImage()(alpha).resize(image.size)
    rgba = image.copy()
    rgba.putalpha(mask)
    rgba.save(output_dir / "pytorch.png")
    return {"mean_ms": float(np.mean(times)), "best_ms": float(np.min(times))}, np.asarray(mask, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gguf", default=str(ROOT / "models" / "rmbg_f16.gguf"))
    parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
    parser.add_argument("--math", choices=("fast", "strict"), default="fast")
    parser.add_argument("--build-dir", default="build-cuda")
    parser.add_argument("--pytorch-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pytorch-model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", default="benchmark_results_current")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ggml, alpha_ggml = run_ggml(args, output_dir)
    pytorch, alpha_pytorch = run_pytorch(args, output_dir)
    diff = np.abs(alpha_ggml - alpha_pytorch)
    report = {
        "configuration": vars(args), "ggml": ggml, "pytorch": pytorch,
        "alpha_u8": {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())},
        "speedup": pytorch["mean_ms"] / ggml["mean_ms"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
