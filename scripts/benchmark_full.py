#!/usr/bin/env python3
"""Run the reproducible PyTorch/GGML backend and model benchmark matrix.

Vulkan modes are deliberately split into ``strict`` (all narrowing disabled),
``optimized`` (the validated direct-conv/CM1 whitelist), and ``unsafe-fast``
(all device fast paths, for diagnostics only). ``fast`` remains an input alias
for ``optimized`` for older automation. The default benchmarks only the production
``optimized`` mode; request ``strict optimized`` for a diagnostic comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

ROOT = Path(__file__).resolve().parent.parent
PARITY_LIMIT = 2e-3


def relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def command_text(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statistics(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "stddev_ms": float(values.std()),
    }


def measurement_context() -> dict[str, Any]:
    return {
        "load_average": list(os.getloadavg()),
        "gpu_utilization": command_text([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,pstate",
            "--format=csv,noheader",
        ]),
    }


def parse_models(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        candidates = {
            "f32": ROOT / "models/rmbg_f32.gguf",
            "f16": ROOT / "models/rmbg_f16.gguf",
            "q8": ROOT / "models/rmbg_q8.gguf",
        }
        models = {name: path for name, path in candidates.items() if path.exists()}
    else:
        models = {}
        for spec in specs:
            if "=" not in spec:
                raise SystemExit(f"--model expects NAME=PATH, got: {spec}")
            name, value = spec.split("=", 1)
            models[name] = Path(value).resolve()
    if not models:
        raise SystemExit("no GGUF models found; pass --model NAME=PATH")
    missing = [str(path) for path in models.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing GGUF model(s): " + ", ".join(missing))
    return models


def benchmark_env(
    backend: str, math_mode: str, iterations: int, warmup: int, cpu_threads: int,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        RMBG_ROOT=str(ROOT), RMBG_DEVICE=backend,
        RMBG_BENCH_ITERS=str(iterations), RMBG_BENCH_WARMUP=str(warmup),
        RMBG_CPU_THREADS=str(cpu_threads),
    )
    for key in (
        "NVIDIA_TF32_OVERRIDE", "RMBG_STRICT_MATH", "RMBG_VULKAN_FAST",
        "RMBG_VULKAN_MODE", "RMBG_VULKAN_STRICT", "RMBG_VK_DIRECT_CONV",
        "RMBG_VK_SCALAR_DIRECT_CONV", "RMBG_VK_COOPMAT_MATMUL",
        "RMBG_VK_QKV_LAYOUT", "RMBG_VK_FLASH_ATTN", "RMBG_VK_DEFORM_PROJECT",
        "RMBG_CUDA_F16_GEMM", "RMBG_CUDA_F16_MIN_STAGE", "RMBG_CUDA_NN_GEMM",
        "GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT",
        "GGML_VK_DISABLE_COOPMAT2", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT",
    ):
        env.pop(key, None)
    mode = "optimized" if math_mode == "fast" else math_mode
    if backend == "cuda" and mode == "strict":
        env.update(NVIDIA_TF32_OVERRIDE="0", RMBG_STRICT_MATH="1")
    elif backend == "vulkan":
        env["RMBG_VULKAN_MODE"] = mode
        if mode == "strict":
            env.update(
                GGML_VK_DISABLE_F16="1", GGML_VK_DISABLE_COOPMAT="1",
                GGML_VK_DISABLE_COOPMAT2="1",
                GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="1",
            )
        elif mode == "optimized":
            env.update(
                GGML_VK_DISABLE_F16="1", GGML_VK_DISABLE_COOPMAT2="1",
                GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="1",
                RMBG_VK_DIRECT_CONV="1", RMBG_VK_SCALAR_DIRECT_CONV="1",
                RMBG_VK_COOPMAT_MATMUL=(
                    "bb_layers_0,bb_layers_1,bb_layers_2,bb_layers_3,"
                    "sq0_,db4_,db3_,db2_,db1_"
                ),
            )
        elif mode == "unsafe-fast":
            env["RMBG_VULKAN_FAST"] = "1"
    return env


def run_ggml_case(
    args: argparse.Namespace, output_dir: Path, model_name: str, model: Path,
    backend: str, math_mode: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    build_dir = {
        "cpu": args.build_cpu, "cuda": args.build_cuda, "vulkan": args.build_vulkan,
    }[backend]
    build = (ROOT / build_dir).resolve()
    test = build / "tests/test_rmbg_graph"
    cli = build / "rmbg-cli"
    for executable in (test, cli):
        if not executable.is_file():
            raise SystemExit(f"missing {executable}; build backend '{backend}' first")

    iterations = args.cpu_runs if backend == "cpu" else args.runs
    warmup = args.cpu_warmup if backend == "cpu" else args.warmup
    env = benchmark_env(backend, math_mode, iterations, warmup, args.cpu_threads)
    env["RMBG_WEIGHTS"] = str(model)
    context = measurement_context()
    result = subprocess.run(
        [str(test)], env=env, text=True, capture_output=True, timeout=args.timeout,
    )
    log = result.stdout + result.stderr
    if result.returncode not in (0, 1):
        raise SystemExit(f"GGML case failed ({backend}/{model_name}/{math_mode}):\n{log}")
    parity = re.search(r"max\|d\|=([\deE+.-]+) mean\|d\|=([\deE+.-]+)", log)
    samples_match = re.search(r"samples-ms:((?:\s+[\d.]+)+)", log)
    actual_backend = re.search(r"^backend:\s*(.+)$", log, re.MULTILINE)
    if not parity or not samples_match or not actual_backend:
        raise SystemExit(f"unable to parse GGML case output:\n{log}")
    samples = [float(value) for value in samples_match.group(1).split()]
    max_diff = float(parity.group(1))
    case_id = f"ggml-{backend}-{model_name}-{math_mode}"
    output = output_dir / f"{case_id}.png"
    alpha = None
    cold_ms = None
    if not args.skip_images:
        begin = time.perf_counter()
        subprocess.run(
            [str(cli), "remove", "--model", str(model), "--input", str(args.input),
             "--output", str(output), "--device", backend],
            env=env, check=True, timeout=args.timeout,
        )
        cold_ms = (time.perf_counter() - begin) * 1000
        alpha = np.asarray(Image.open(output).getchannel("A"), dtype=np.float32)

    case: dict[str, Any] = {
        "id": case_id, "runtime": "ggml", "backend": backend,
        "backend_name": actual_backend.group(1).strip(), "math_mode": math_mode,
        "model": model_name, "build_dir": relative(build),
        "measurement_context": context,
        "samples_ms": samples, **statistics(samples),
        "fixture_parity": {
            "limit": PARITY_LIMIT, "passed": result.returncode == 0 and max_diff <= PARITY_LIMIT,
            "max_abs_diff": max_diff, "mean_abs_diff": float(parity.group(2)),
        },
    }
    if cold_ms is not None:
        case.update(cold_cli_ms=cold_ms, output_image=relative(output))
    return case, alpha


def run_pytorch_case(
    args: argparse.Namespace, output_dir: Path, device_name: str, math_mode: str,
) -> tuple[dict[str, Any], np.ndarray]:
    device = torch.device(device_name)
    if device.type == "cuda":
        strict = math_mode == "strict"
        torch.set_float32_matmul_precision("highest" if strict else "high")
        torch.backends.cuda.matmul.allow_tf32 = not strict
        torch.backends.cudnn.allow_tf32 = not strict
    model = AutoModelForImageSegmentation.from_pretrained(
        args.pytorch_model, trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval().float().to(device)
    image = Image.open(args.input).convert("RGB")
    tensor = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(image).unsqueeze(0).to(device)
    context = measurement_context()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            logits = model(tensor)[-1].sigmoid()
            sync()
        samples = []
        for _ in range(args.runs):
            sync()
            begin = time.perf_counter()
            logits = model(tensor)[-1].sigmoid()
            sync()
            samples.append((time.perf_counter() - begin) * 1000)
        alpha_1024 = logits.detach().cpu()[0, 0]

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    mask = transforms.ToPILImage()(alpha_1024).resize(image.size, resampling)
    case_id = f"pytorch-{device_name}-{math_mode}"
    output = output_dir / f"{case_id}.png"
    rgba = image.copy()
    rgba.putalpha(mask)
    rgba.save(output)
    alpha = np.asarray(mask, dtype=np.float32)
    case = {
        "id": case_id, "runtime": "pytorch", "backend": device_name,
        "backend_name": str(device), "math_mode": math_mode, "model": "fp32",
        "measurement_context": context,
        "samples_ms": samples, **statistics(samples), "output_image": relative(output),
    }
    del model, tensor, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return case, alpha


def metadata(args: argparse.Namespace, models: dict[str, Path]) -> dict[str, Any]:
    commit = command_text(["git", "rev-parse", "HEAD"])
    dirty = bool(command_text(["git", "status", "--porcelain"]))
    model_data = {
        name: {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in models.items()
    }
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "source": {
            "repository_commit": commit, "working_tree_dirty": dirty,
            "ggml_commit": command_text(["git", "-C", "third_party/ggml", "rev-parse", "HEAD"]),
            "ggml_patch_sha256": sha256(ROOT / "third_party/ggml-rmbg.patch"),
        },
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu": command_text(["bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"]),
            "nvidia_smi": command_text([
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]),
        },
        "software": {
            "python": platform.python_version(), "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cmake": command_text(["cmake", "--version"]).splitlines()[0],
        },
        "configuration": {
            "input": relative(args.input), "input_sha256": sha256(args.input),
            "input_size": "1024x1024", "runs": args.runs, "warmup": args.warmup,
            "cpu_runs": args.cpu_runs, "cpu_warmup": args.cpu_warmup,
            "cpu_threads": args.cpu_threads, "parity_limit": PARITY_LIMIT,
            "pytorch_model": args.pytorch_model,
        },
        "models": model_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", action="append", help="GGUF case as NAME=PATH; repeatable")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "cuda", "vulkan"),
                        default=["cpu", "cuda", "vulkan"])
    parser.add_argument(
        "--math-modes", nargs="+",
        choices=("strict", "optimized", "unsafe-fast", "fast"),
        default=["optimized"],
        help="modes to benchmark; use 'strict optimized' for the full diagnostic matrix",
    )
    parser.add_argument("--build-cpu", default="build")
    parser.add_argument("--build-cuda", default="build-cuda")
    parser.add_argument("--build-vulkan", default="build-vulkan")
    parser.add_argument("--pytorch-devices", nargs="+", choices=("cpu", "cuda"), default=["cuda"])
    parser.add_argument("--pytorch-model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cpu-runs", type=int, default=3)
    parser.add_argument("--cpu-warmup", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results_current"))
    parser.add_argument("--report", type=Path, default=Path("docs/rmbg_benchmark.json"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.report = args.report.resolve()
    if (args.runs < 1 or args.cpu_runs < 1 or args.warmup < 0 or
            args.cpu_warmup < 0 or args.cpu_threads < 1):
        raise SystemExit(
            "runs, cpu-runs, and cpu-threads must be positive; warmups must be non-negative"
        )

    models = parse_models(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 2, **metadata(args, models), "cases": []}
    alpha_by_case: dict[str, np.ndarray] = {}

    for device in args.pytorch_devices:
        modes = args.math_modes if device == "cuda" else ["strict"]
        for mode in modes:
            print(f"[benchmark] pytorch {device} {mode}", flush=True)
            case, alpha = run_pytorch_case(args, args.output_dir, device, mode)
            report["cases"].append(case)
            alpha_by_case[case["id"]] = alpha

    for backend in args.backends:
        modes = args.math_modes if backend != "cpu" else ["strict"]
        for model_name, model in models.items():
            for mode in modes:
                print(f"[benchmark] ggml {backend} {model_name} {mode}", flush=True)
                case, alpha = run_ggml_case(
                    args, args.output_dir, model_name, model, backend, mode,
                )
                report["cases"].append(case)
                if alpha is not None:
                    alpha_by_case[case["id"]] = alpha

    # Prefer strict as the cross-backend diagnostic reference when the full matrix was
    # requested; the production-only default has optimized PyTorch as its reference.
    reference_mode = (
        "strict" if "pytorch-cuda-strict" in alpha_by_case else "optimized"
    )
    reference_id = f"pytorch-cuda-{reference_mode}"
    if reference_id in alpha_by_case:
        reference = alpha_by_case[reference_id]
        alpha_u8_key = f"alpha_u8_vs_pytorch_cuda_{reference_mode}"
        alpha_normalized_key = f"alpha_normalized_vs_pytorch_cuda_{reference_mode}"
        for case in report["cases"]:
            alpha = alpha_by_case.get(case["id"])
            if alpha is not None and alpha.shape == reference.shape:
                difference = np.abs(alpha - reference)
                case[alpha_u8_key] = {
                    "max_abs_diff": float(difference.max()),
                    "mean_abs_diff": float(difference.mean()),
                }
                case[alpha_normalized_key] = {
                    "max_abs_diff": float(difference.max() / 255.0),
                    "mean_abs_diff": float(difference.mean() / 255.0),
                }
        pytorch_reference = next(
            case for case in report["cases"] if case["id"] == reference_id
        )
        pytorch_mean = pytorch_reference["mean_ms"]
        pytorch_median = pytorch_reference["median_ms"]
        pytorch_by_mode = {
            case["math_mode"]: case["mean_ms"] for case in report["cases"]
            if case["runtime"] == "pytorch" and case["backend"] == "cuda"
        }
        pytorch_median_by_mode = {
            case["math_mode"]: case["median_ms"] for case in report["cases"]
            if case["runtime"] == "pytorch" and case["backend"] == "cuda"
        }
        for case in report["cases"]:
            if case["runtime"] == "ggml":
                case[f"speedup_vs_pytorch_cuda_{reference_mode}"] = (
                    pytorch_mean / case["mean_ms"]
                )
                case[f"speedup_median_vs_pytorch_cuda_{reference_mode}"] = (
                    pytorch_median / case["median_ms"]
                )
                matching_mean = pytorch_by_mode.get(case["math_mode"])
                if matching_mean is not None:
                    case["speedup_vs_pytorch_cuda_matching_mode"] = (
                        matching_mean / case["mean_ms"]
                    )
                matching_median = pytorch_median_by_mode.get(case["math_mode"])
                if matching_median is not None:
                    case["speedup_median_vs_pytorch_cuda_matching_mode"] = (
                        matching_median / case["median_ms"]
                    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": relative(args.report), "cases": len(report["cases"])}, indent=2))


if __name__ == "__main__":
    main()
