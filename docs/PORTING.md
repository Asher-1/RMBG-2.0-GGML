# RMBG-2.0 GGML implementation status

BiRefNet-family background removal with a device-resident GGML graph.

## Status (2026-08-04)

The 1024x1024 end-to-end graph is implemented in `src/rmbg_graph.cpp`:

- two-scale Swin-L encoder (24 blocks per pass)
- context fusion and squeeze module
- decoder blocks, lateral fusion, GDT attention, and input patches
- ASPP deformable convolution with a fused CUDA im2col sampler and a primitive fallback
- Swin-L `head_dim=32` CUDA attention with fused shifted-window pack/unpack,
  QKV bias/split, relative-position bias, shifted mask, softmax, and output projection
- sigmoid alpha output
- CUDA, Vulkan, and CPU backend execution
- encoded image bytes to original-size RGBA PNG through the C++ and C APIs

The runtime keeps weights and intermediates on the selected backend. Only the input
tensor and final alpha tensor cross the host/device boundary.

## Design decision

| Option | Result | Reason |
|---|---|---|
| A: pure GGML primitives | Vulkan fallback | Complete and portable, but generic deform/im2col nodes leave performance on the table |
| B: one GGML graph with backend-fused nodes | Selected | Keeps device ownership and one graph while allowing CUDA deform sampling and cuDNN convolution |
| C: segmented graphs | Reserve only | Useful when a device cannot allocate the full graph, but adds submissions and persistent boundary buffers |

"Hybrid" here means backend-native nodes inside the GGML graph, not CPU/GPU tensor
handoffs. CUDA uses a fused deformable-im2col kernel followed by GGML GEMM. Swin
attention is one GGML custom node per block; dense projections and QK/AV use cuBLAS
inside that node, while layout conversion, shifted masking, relative-position bias,
and softmax use dedicated CUDA kernels. When
cuDNN is found at configure time, ordinary F32 convolutions use cuDNN with a bounded
512 MiB temporary workspace; otherwise they fall back to GGML im2col. Vulkan uses
the same graph builder and primitive implementations, with exact F32 custom gathers
for RMBG input patches and Swin patch merge. Attention and deform sampling remain
primitive Vulkan graphs.

## Numerical modes

CUDA fast mode keeps TF32 enabled and is the production default. Strict regression
mode is enabled with `RMBG_STRICT_MATH=1`, which sets:

- CUDA strict: `NVIDIA_TF32_OVERRIDE=0`
- Vulkan: disable FP16, cooperative-matrix, and integer-dot narrowing

Vulkan applies the listed narrowing-disable flags by default. Set
`RMBG_VULKAN_FAST=1` to enable the device's F16/cooperative-matrix route only when its
separately measured parity is acceptable. Strict settings are required for the strict
parity numbers below.

| Backend | max alpha abs diff | mean alpha abs diff | Result |
|---|---:|---:|---|
| CUDA fast TF32 + fused Swin | 1.315e-3 | 1.029e-6 | pass, no pixel above 2e-3 |
| CUDA FP32 + cuDNN + fused Swin | 1.122e-4 | 1.389e-7 | pass |
| Vulkan FP32 | 1.081e-4 | 1.383e-7 | pass |

The reference is the PyTorch sigmoid output in `tests/fixtures/alpha_ref.gguf`.

## Performance

RTX 3060, 1024x1024, batch 1, output copied to host, warm steady state:

| Runtime | Mean latency | Relative to PyTorch CPU |
|---|---:|---:|
| GGML CUDA fast TF32 + cuDNN | 592.29 ms | 23.07x faster |
| GGML CUDA strict FP32 + cuDNN | 763.56 ms | 17.89x faster |
| GGML Vulkan strict FP32 | 1293.35 ms | 10.56x faster |
| PyTorch CPU FP32 | 13662.23 ms | baseline |
| PyTorch CUDA FP32 | 705.45 ms | GGML CUDA fast is 1.19x faster |

CUDA now uses a dedicated `head_dim=32` Swin attention node. The node packs shifted
windows, runs QKV projection, fuses bias and QKV split into a contiguous head layout,
uses strided-batched QK/AV GEMMs, fuses relative-position bias plus shifted mask plus
softmax, projects the result, and unpacks directly to token order. This removes the
generic `get_rows`, Q/K/V `cont`/`permute`, mask-add, and inverse-window nodes from the
production CUDA graph.

The CUDA fast path exceeds the measured PyTorch CUDA baseline within the accepted
alpha tolerance. The current PyTorch baseline uses PyTorch 2.7.1+cu118 with TF32
disabled and all current CUDA numbers use 12 warm steady-state iterations. The latest
CUDA work shares each deformable-convolution offset and modulation value across its
channel tile, reducing its sampled-im2col kernel from 123.9 ms to 22.8 ms in the CUDA
profile. CUDA Graph replay was tested and did not improve this compute-bound workload.
Vulkan has exact F32 custom gather shaders for input patches and Swin patch merge.
Its attention and deform paths remain primitive graphs; custom attention/deform shaders
are the next backend-specific performance work, not a correctness dependency.

## GGUF model precision

`models/rmbg_f32.gguf` and `models/rmbg_f16.gguf` are the supported complete
end-to-end models. The loader retains their original weight type instead of turning all
tensors into F32 at load time. F32 is the reference; F16 keeps matrix weights in F16 on
Vulkan and is the deployment default.

RTX 3060, batch 1, 1024x1024, five warm steady-state iterations:

| Model | File size | CUDA strict | Vulkan strict | max alpha abs diff |
|---|---:|---:|---:|---:|
| F32 | 841.9 MiB | 644.53 ms | 1293.35 ms | 1.122e-4 |
| F16 | 421.0 MiB | 655.18 ms | 1278.46 ms | 1.122e-4 |

Q8 is not an RMBG deployment format. Its parity-safe hybrid saved only 16.9 MiB relative
to F16 and did not improve CUDA or Vulkan latency; full Q8 quantization failed the alpha
gate, so the artifact and export option were removed. The strict Vulkan path deliberately
disables FP16/cooperative matrix/integer-dot narrowing because the device's
cooperative-matrix path measured 887.10 ms but a 5.048e-3 alpha error. It can be enabled
explicitly with `RMBG_VULKAN_FAST=1` when that relaxed tolerance is valid for the
deployment.

The `rmbg_image_patches_*` and `rmbg_patch_merge_*` Vulkan nodes use no arithmetic
approximation and preserve the strict alpha result. They remove constant-index
`GET_ROWS` work, but do not by themselves make Vulkan competitive with PyTorch CUDA:
the dominant remaining work is deformable `IM2COL`, activation layout copies, and
fragmented Swin attention GEMMs.

## Build and run

```bash
cmake -S . -B build-cuda -DRMBG_GGML_CUDA=ON -DRMBG_BUILD_TESTS=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda -R 'swin_graph_full|rmbg_graph' --output-on-failure

./build-cuda/rmbg-cli remove \
  --model models/rmbg_f16.gguf \
  --input input.png --output output.png --device cuda
```

The split conversion fixtures are deliberately isolated in `models/development/`.
Passing `models/development/encoder_f16.gguf` automatically merges its sibling
`decoder_alpha_f16.gguf`. New deployments should generate one unified file:

```bash
python scripts/convert_rmbg_to_gguf.py \
  --model ZhengPeng7/BiRefNet --out models/rmbg_f16.gguf --ftype 1
```

The converter exports only runtime tensors and shortens decoder names to stay within
ggml's tensor-name limit.
