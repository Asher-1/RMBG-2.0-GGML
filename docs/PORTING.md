# RMBG GGML inference audit

Last reviewed on 2026-08-18. This repository implements a BiRefNet-family background-removal
graph on ggml. The checked benchmark fixture and local GGUF files use
`ZhengPeng7/BiRefNet`; they are not a quality evaluation of the gated BRIA weights.

## Acceptance criteria

An optimization is accepted only when all of the following remain true:

1. The workload tuple is unchanged: model hash, input hash, 1024x1024 resolution,
   batch 1, output copy, backend, math mode, warmup, and sample count.
2. The sigmoid alpha output passes the checked fixture gate: maximum absolute
   difference no greater than `2e-3`.
3. Weights and intermediates stay on the selected backend; only input and final alpha
   cross the host/device boundary.
4. Every ggml source modification is reproducible from the pinned submodule commit and
   `third_party/ggml-rmbg.patch` through a normal CMake configure.

This matters because a lower timing that changes precision beyond the gate, uses a
different model, or silently falls back to CPU is not an inference acceleration.

## Implemented graph

The end-to-end graph in `src/rmbg_graph.cpp` contains:

- two-scale Swin-L encoder, 24 blocks per pass;
- context fusion and squeeze module;
- decoder blocks, lateral fusion, GDT attention, and input patches;
- ASPP deformable convolution;
- sigmoid alpha output;
- CPU, CUDA, and Vulkan backend execution;
- original-size RGBA output through the C++ and C APIs.

CUDA uses dedicated shifted-window attention, gather/layout, deformable-im2col, channel
affine, and optional cuDNN convolution paths. The cuDNN implementation caches its
handle, tensor/filter/convolution descriptors, selected algorithm, and workspace size
per device and convolution shape.

Vulkan uses exact F32 gather/layout shaders and now handles both affine and
affine-plus-ReLU in one shader with a push-constant mode. The accepted optimized
profile also uses direct convolution with a scalar F32-accumulation pipeline and a
per-node CM1 whitelist for the four encoder stages plus the validated decoder
projections. Strict attention and deformable convolution retain validated primitive
paths. The experimental fused Vulkan deform-project path remains opt-in because it did
not improve the accepted result.

CPU executes the same graph with an explicit thread count. Merely linking MKL was not
reported as an optimization: this graph is scheduled to the CPU backend, not
automatically partitioned to ggml's separate BLAS backend.

## Numerical modes

`optimized` is the production mode. It is the default for CUDA and Vulkan and keeps
the measured fast paths that pass the `2e-3` alpha gate. Strict mode is not a quality
mode or a separate model: it is a slower arithmetic/reference profile used to detect
rounding regressions and compare backends against an F32 baseline. It is important for
validation, but should not be selected for normal inference.

CUDA optimized permits TF32 GEMMs with FP32 accumulation. Set `RMBG_STRICT_MATH=1`
for strict FP32 math; this also makes an auto-selected Vulkan backend use its strict
profile, so set it only for diagnostics.

Vulkan `RMBG_VULKAN_MODE=optimized` disables unsafe FP16/CM2/integer-dot narrowing,
enables scalar direct convolution, and allows CM1 only for
`bb_layers_0` through `bb_layers_3` and `sq0_`/`db4_`/`db3_`/`db2_`/`db1_`.
`RMBG_VULKAN_MODE=strict` disables all narrowing, CM1, CM2, integer dot product, and
direct convolution. `RMBG_VULKAN_MODE=unsafe-fast` (or legacy
`RMBG_VULKAN_FAST=1`) enables every device fast path and is diagnostic only because it
fails the gate on the checked RTX 3060.

Code that creates a ggml backend directly must call
`rmbg::configure_backend_profile(device)` before `ggml_backend_init*`. The high-level
`load_gguf` API performs this automatically; this ordering is required because Vulkan
device capabilities are fixed during backend initialization.

F16 model weights do not mean that every activation or accumulator is F16. The strict
F16-model cases retain the F32 compute route where required for parity.

| Case | max alpha abs diff | mean alpha abs diff | Gate |
|---|---:|---:|---|
| CUDA F32/F16 strict | 1.131e-4 | 1.383e-7 | pass |
| CUDA F32/F16 optimized | 1.430e-3 | 1.039e-6 | pass |
| Vulkan F32/F16 strict | 1.099e-4 | 1.378e-7 | pass |
| CPU F32/F16 strict | 1.052e-4 | 1.104e-7 | pass |
| Vulkan optimized F32/F16 | 1.533e-3 (measured max) | 9.681e-7 | pass |
| Vulkan unsafe-fast F32/F16 | 5.509e-3 | 1.666e-6 | **fail** |
| Q8, any backend/mode | 3.170e-2 to 3.614e-2 | varies | **fail** |

The reference logits are stored in `tests/fixtures/alpha_ref.gguf`. The test runner
prints the actual initialized backend and returns nonzero when the `2e-3` gate fails.

## Current performance

Hardware and software: NVIDIA RTX 3060 12 GiB, driver 550.144.03, Ryzen 9 5950X,
PyTorch 2.6.0+cu118, CUDA runtime 11.8. GPU results use seven timed iterations after two
warmups. CPU results use 16 threads and two timed iterations. All values below are from
the same schema-v2 report generated at `2026-08-17T21:12:02+08:00`.

### Valid cases

| Runtime / model / mode | Median | p95 | Matching comparison | Result |
|---|---:|---:|---:|---|
| PyTorch CUDA FP32 strict | 722.23 ms | 726.66 ms | baseline | reference |
| PyTorch CUDA FP32 optimized | 559.96 ms | 564.92 ms | baseline | reference |
| GGML CUDA F32 optimized | 573.12 ms | 575.18 ms | 0.977x vs PyTorch optimized | pass |
| GGML CUDA F16 optimized | **570.17 ms** | 573.26 ms | 0.982x vs PyTorch optimized | pass |
| GGML CUDA F32 strict | 745.90 ms | 750.30 ms | 0.968x vs PyTorch strict | pass |
| GGML CUDA F16 strict | 748.71 ms | 754.54 ms | 0.965x vs PyTorch strict | pass |
| GGML Vulkan F32 optimized | 682.32 ms | 684.41 ms | 0.821x vs PyTorch optimized | pass, 1.058x vs strict |
| GGML Vulkan F16 optimized | **675.88 ms** | 692.86 ms | 0.828x vs PyTorch optimized | pass, 1.069x vs strict |
| GGML Vulkan F32 strict | 1259.02 ms | 1267.05 ms | 0.574x vs PyTorch strict | pass |
| GGML Vulkan F16 strict | 1256.48 ms | 1259.14 ms | 0.575x vs PyTorch strict | pass |
| GGML CPU F16 strict | 15201.65 ms | 15281.42 ms | 0.048x vs PyTorch CUDA strict | pass |
| GGML CPU F32 strict | 15545.47 ms | 15577.00 ms | 0.046x vs PyTorch CUDA strict | pass |

The direct answer to whether ggml is faster than local PyTorch CUDA is mode-specific:

- CUDA optimized F32/F16 are 2.3% and 1.8% slower than PyTorch optimized on this machine.
- CUDA strict is 3.2% to 3.7% slower than PyTorch strict.
- Vulkan optimized F32/F16 are 18.0% and 17.2% slower than PyTorch optimized, but 5.8%
  and 6.9% faster than PyTorch strict. Vulkan strict remains 1.74x to 1.75x slower than
  PyTorch strict and is retained for numerical diagnosis.

### Rejected cases

| Runtime / model / mode | Median | max fixture diff | Reason rejected |
|---|---:|---:|---|
| GGML CUDA Q8 optimized | 572.55 ms | 3.334e-2 | parity failure |
| GGML CUDA Q8 strict | 748.68 ms | 3.170e-2 | parity failure |
| GGML Vulkan Q8 optimized | 664.93 ms | 3.638e-2 | parity failure |
| GGML Vulkan Q8 strict | 1271.13 ms | 3.170e-2 | parity failure |
| GGML CPU Q8 strict | 15887.40 ms | 3.173e-2 | parity failure |

Q8 is measured because the artifact exists locally, but it is not a supported deployment
format. A benchmark runner must not rank a failed case as the recommended result.

## GGUF model matrix

| Model | Bytes | MiB | SHA-256 |
|---|---:|---:|---|
| F32 | 882846304 | 841.9 | `73fa93582743128e392b6e5b6be821e5b67361dcd5a5c0deca0ae4077e4c0ddd` |
| F16 | 441451648 | 421.0 | `50aaf0c7570df97b3767909394d9a63c93effe9f01dc863b17fa69d5f76eb8e3` |
| Q8 experimental | 258974848 | 247.0 | `a2f432614d91057614c59745d40a1770b1a84455f5f853174629b05fc7c8e079` |

Newly converted files contain `rmbg.model_id` metadata. The benchmark report records
model size and SHA even for older local files that predate that metadata.

## Benchmark artifacts

`scripts/benchmark_full.py` discovers F32, F16, and Q8 models, validates that the
requested backend is the backend actually initialized, runs all backend/mode cases,
records every raw timing sample, and captures the input/model/source/patch hashes and
measurement load context.

Generated artifacts:

- `docs/rmbg_benchmark.json`: complete schema-v2 data for all 17 cases;
- `docs/rmbg_all_outputs_comparison.png`: every measured output with latency and gate;
- `docs/rmbg_inference_comparison.png`: representative masks and final resized 8-bit PNG
  alpha differences; PASS/FAIL still comes from the raw 1024x1024 float fixture;
- `docs/rmbg_latency_comparison.png`: GPU latency, CPU latency, speedup, and parity.

Reproduce the matrix after building all three backends:

```bash
python scripts/benchmark_full.py --input docs/images/t4.png \
  --pytorch-model ZhengPeng7/BiRefNet --math-modes strict optimized \
  --runs 7 --warmup 2 \
  --cpu-runs 2 --cpu-warmup 0 --cpu-threads 16 --local-files-only
python scripts/plot_benchmarks.py
```

The report records the source commit, working-tree state, pinned ggml commit, and patch
SHA. Regenerate it after source, model, benchmark, or patch changes.

## Reproducible ggml patch integration

The ggml submodule is pinned at:

```text
90951f99af1fbebef3fbdd58ff5b8715b0bb9c43
```

All local changes, including CUDA and Vulkan source, shaders, declarations, CMake
linkage, and ggml version handling, are in `third_party/ggml-rmbg.patch` with SHA-256:

```text
1e707fee4c867a4ab6602f1cd6858249803c96949cf7950496484d8b39cbe67b
```

Top-level CMake performs this sequence:

1. Read the actual submodule `HEAD` and fail if it cannot be resolved.
2. Hash the patch and register it in `CMAKE_CONFIGURE_DEPENDS`.
3. Export the pinned commit with `git archive` into a content-addressed build directory.
4. Run `git apply --check`, apply the patch, then run reverse-check verification.
5. Verify a required RMBG CUDA source exists before writing the stamp.
6. Inject `<ggml-commit>+rmbg.<patch-hash>` into the built ggml version.
7. Call `add_subdirectory` on the patched build-tree copy, never the mutable submodule.

Git repository discovery is explicitly capped at the build directory during apply. This
prevents Git from walking up to the parent repository and incorrectly returning success
while treating every patch path as outside the work tree.

A normal developer build is sufficient:

```bash
git submodule update --init --recursive third_party/ggml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DRMBG_GGML_CUDA=ON
cmake --build build-cuda -j

cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DRMBG_GGML_VULKAN=ON
cmake --build build-vulkan -j
```

No manual `git apply` is part of the developer workflow. A stale or incompatible patch
fails configuration rather than silently dropping a fix.

## Optimization audit

### Accepted this round

- Cached cuDNN handles, shape descriptors, algorithm selection, and workspace sizing.
  This removes repeated host setup while retaining the same convolution math mode.
- Added Vulkan dispatch/support for both channel affine and affine-plus-ReLU, allowing
  BatchNorm affine work to stay fused even where ReLU is absent.
- Preserved the existing exact CUDA deformable sampler and strict Vulkan fallbacks.
- Added raw-sample and p95 reporting, exact backend validation, fixture gates, model and
  patch hashes, output images, and machine-load context.

### Measured and rejected

- A host-dispatched custom CUDA MLP was fixed so it actually activated, then measured at
  1353.55 ms with `5.269e-3` max error. It disrupted whole-graph capture/replay and was
  removed from both the graph and patch.
- F16-input CUDA MLP GEMMs exceeded the `2e-3` gate; they remain experimental and off.
- Vulkan all-cooperative-matrix/FP16 unsafe-fast mode is substantially faster but fails
  parity; only the measured CM1 whitelist is enabled by default.
- Direct Vulkan convolution removes the largest explicit IM2COL allocation and is paired
  with a scalar pipeline so the convolution itself does not accumulate CM1 rounding.
- Vulkan graph submission now flushes on a FLOP budget, matching the upstream scheduler
  fix and avoiding a single oversized command batch.
- The Vulkan deform-project fusion did not improve the accepted route and remains opt-in.
- Q8 reduces file size but fails parity on every backend and does not beat valid CUDA
  optimized F32/F16.

### Remaining acceleration space

The graph still has acceleration space, but the safe opportunities differ by backend:

1. CUDA: fuse bias/residual/GELU epilogues into existing GEMMs and reduce layout copies.
   The accepted CUDA optimized F32 result is 2.3% slower than PyTorch optimized, so
   changes must be measured mode-for-mode rather than against the slower strict baseline.
2. CUDA: retain graph replay by using backend-native nodes; avoid host custom-op
   dispatch inside the captured graph.
3. Vulkan: profile the optimized direct-convolution and CM1 whitelist on multiple
   vendors. The RTX 3060 result is the acceptance baseline, not a guarantee for AMD or
   Intel hardware.
4. Vulkan: replace the scalar direct-convolution output with a tile-fused sampler only
   when boundary interpolation and F32 accumulation remain within the gate.
5. CPU: backend scheduling and data layout are the meaningful levers. Linking a BLAS
   library without moving eligible nodes to that backend is not an optimization.
6. Model level: reaching 100 ms on an RTX 3060 is not realistic for the unchanged
   two-scale Swin-L graph; distillation, structured pruning, or a smaller input is needed
   for that class of target.

Every future optimization should add its case to the same report and remain disabled or
rejected when it cannot pass the fixture gate.

## Build and run

```bash
cmake -S . -B build-cuda -DRMBG_GGML_CUDA=ON -DRMBG_BUILD_TESTS=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda -R rmbg_graph --output-on-failure

./build-cuda/rmbg-cli remove \
  --model models/rmbg_f16.gguf \
  --input input.png --output output.png --device cuda
```

Generate a unified model with explicit source identity:

```bash
python scripts/convert_rmbg_to_gguf.py \
  --model ZhengPeng7/BiRefNet --model-id ZhengPeng7/BiRefNet \
  --out models/rmbg_f16.gguf --ftype 1
```
