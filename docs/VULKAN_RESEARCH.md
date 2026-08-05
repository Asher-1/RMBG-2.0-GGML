# Vulkan Acceleration Repository Survey

## Scope and result

On 2026-08-04, the GGML/Vulkan implementations under
`/home/ludahai/develop/code/github/dl` were inspected. The repositories with a Vulkan
GGML backend were BrushNet-GGML, DeepLSD-GGML, EfficientLoFTR-GGML, GFPGAN-GGML,
InstantMesh-GGML, LightGlue-GGML, RMBG-2.0-GGML, T2I-Adapter-GGML, TLC-Calib,
depth-anything-cpp, face-detect-ggml, free-splatter.cpp, locate-anything-ggml,
rf-detr-ggml, sam3-ggml, stable-diffusion-ggml, and trellis-ggml.

Apart from RMBG, only LightGlue-GGML contains model-specific Vulkan compute shaders.
The other projects expose the stock GGML Vulkan backend and stock operators such as
flash attention; none contains an RMBG/Swin/deformable-convolution implementation that
can be copied as a complete solution.

## Reusable pattern, not a turnkey solution

LightGlue's custom ALIKED path is the closest local precedent:

- shaders: `third_party/ggml/src/ggml-vulkan/vulkan-shaders/aliked_*.comp`
- shader registration: `vulkan-shaders-gen.cpp`
- backend dispatch: `ggml-vulkan-aliked.inc.cpp`
- persistent graph/tensor cache: `cpp/src/aliked/gpu_pipeline_cache.cpp`

It implements custom deformable convolution, layout conversion, non-maximum suppression,
and descriptor sampling. That is the right ownership model for RMBG: keep ordinary graph
nodes in GGML and make operations with unusual indexing/layout into one model-specific
GPU dispatch.

It is not a performance result to adopt unchanged. Its own measured RTX 3060 results in
`LightGlue-GGML/cpp/ALIKED_VULKAN.md` report Vulkan at about 3.7--8.4 s versus PyTorch at
888 ms and CUDA at 487 ms; its documentation recommends CUDA. It also records Vulkan
convolution and interpolation parity drift. Therefore copying the ALIKED shaders would
neither match RMBG tensor layouts nor satisfy the RMBG latency/accuracy gate.

## RMBG bottleneck and implementation order

The strict RMBG Vulkan profile is dominated by deformable `IM2COL`, activation layout
copies, and fragmented Swin attention GEMMs. Exact F32 custom shaders now remove the
input-patch, patch-merge, and shifted-window pack/unpack `GET_ROWS` work. A separate
exact F32 deformable-sampling shader replaces the many primitive sampling nodes, but it
still writes the column matrix consumed by the following GEMM. The remaining work should
be ordered by removed memory traffic and launches:

1. Replace QKV split/permute/contiguous copies with a shape-specific `head_dim=32`
   layout kernel, then use it with the existing Vulkan matrix kernels. This removes the
   largest remaining Swin layout traffic without replacing high-throughput GEMM with a
   scalar shader.
2. Add relative bias, shifted-mask, and softmax only after an attention-score fixture
   proves every row against the existing GGML graph. The first direct fusion attempt
   exceeded the alpha gate and was rejected rather than enabled.
3. Fuse RMBG deformable sampling and projection only when the shader can retain the
   sampled values in a tile. It must preserve boundary interpolation and F32
   accumulation; that is what removes the remaining `IM2COL` materialization.
4. Fuse adjacent residual, normalization/layout, and pointwise operations only after
   per-stage profiling proves their traffic dominates.
5. Use F16 for weights and non-sensitive storage only after the F32-accumulator parity
   path passes. Do not use the current cooperative-matrix fast mode as the strict path:
   its measured maximum alpha error is `5.048e-3`, above the `2e-3` gate.

Each step requires the existing PyTorch alpha comparison and a warm steady-state latency
measurement. This keeps speed work falsifiable: a shader is accepted only if it removes
the intended bottleneck and remains under the alpha error gate.
