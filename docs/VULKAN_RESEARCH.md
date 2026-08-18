# Vulkan acceleration audit

This document records the Vulkan optimization work completed on 2026-08-17 and reviewed
on 2026-08-18, plus the remaining falsifiable work. The acceptance invariant is
unchanged: batch 1, 1024x1024,
the checked GGUF/input pair, and maximum alpha error `<= 2e-3` against
`tests/fixtures/alpha_ref.gguf`.

## What the bottleneck is

The graph is a two-scale Swin-L encoder followed by an ASPP/deformable decoder. A
strict run spends most of its time in matrix multiplication, explicit convolution
`IM2COL`, and fragmented graph submission. The Vulkan optimization target is therefore
memory traffic and launch count, not an indiscriminate reduction of every accumulator
to F16.

The local profiler established these facts on an RTX 3060 12 GiB:

- strict Vulkan was about 1.31--1.41 s and passed parity;
- all CM1/FP16 fast paths were about 0.84--0.87 s but reached `5.509e-3` error;
- direct convolution plus scalar accumulation and the encoder/decoder CM1 whitelist
  reached 0.660--0.671 s in two isolated seven-sample runs; the complete 17-case report
  measured F16/F32 medians of 0.676/0.682 s (P95 0.693/0.684 s), with maximum error
  `1.533e-3`;
- the decoder prefixes `sq0_,db4_,db3_,db2_,db1_` are included because the repeated
  runs improved the stable median; broader `attn_*`/`mlp_fc` additions did not;
- a fused deform-project shader was measured at about 1.12 s and remains opt-in.

The timing distribution is bimodal when the GPU is idle between cases. Reports must use
raw samples, median, and P95; a single mean is not an acceptance result.

## Implemented path

### Direct convolution

`src/rmbg_graph.cpp` selects `ggml_conv_2d_direct` for Vulkan when
`RMBG_VK_DIRECT_CONV=1`. The patched backend has two pipeline maps per convolution shape:

- the existing CM1/F16-capable pipelines;
- scalar F32-accumulation pipelines selected by `RMBG_VK_SCALAR_DIRECT_CONV=1`.

The production profile uses the scalar maps. This removes the explicit `IM2COL` tensor,
its reshape/transpose, and the associated memory allocation while preserving the
interpolation and accumulation precision needed by the fixture.

### Fused QKV layout

The Swin projection is flattened to one `[C, N*nW]` matmul and the Vulkan backend
consumes the result through the `rmbg_swin_qkv_layout_*` custom nodes. This keeps Q, K,
and V in their attention layout without materializing the broadcast bias plus three
permute/contiguous chains. The full report includes this path by default; disabling it
with `RMBG_VK_QKV_LAYOUT=0` passed parity but increased the isolated five-sample median
from about 666 ms to 698 ms on the RTX 3060, so it remains enabled.

### Per-node cooperative matrix selection

`RMBG_VK_COOPMAT_MATMUL` is a comma-separated name whitelist. In the default profile it
contains `bb_layers_0,bb_layers_1,bb_layers_2,bb_layers_3,sq0_,db4_,db3_,db2_,db1_`; all
other matmuls use the exact scalar F32 pipeline. `*` enables CM1 everywhere for
diagnostics and `none` disables it everywhere. Node names are assigned in
`src/rmbg_graph.cpp`, so the choice is stable across graph construction and benchmark
runs.

### Submission scheduling

The patched Vulkan scheduler flushes a command batch before adding a node that would
exceed the current FLOP budget, then recomputes the threshold after submission. This is
the upstream ggml scheduling fix backported to the pinned commit, with no wholesale
submodule upgrade. It reduces oversized batches without creating one submit per node.

### Runtime profiles

| Profile | Vulkan flags | Intended use |
|---|---|---|
| `optimized` (default) | direct conv + scalar direct conv + CM1 whitelist; F16/CM2/intdot disabled | production, parity-gated |
| `strict` | F16/CM1/CM2/intdot/direct conv disabled | numerical diagnosis |
| `unsafe-fast` | all device fast paths enabled | diagnostic only; currently parity-fails |

Set `RMBG_VULKAN_MODE` to select a profile. `RMBG_VULKAN_FAST=1` is retained as a
backward-compatible alias for `unsafe-fast`, not as a supported production setting.
Normal Vulkan inference and the benchmark runner both default to `optimized`; strict is
run only when explicitly requested for numerical diagnosis. The runner records the mode
in `docs/rmbg_benchmark.json`.

## Web research applied

The implementation follows primary Vulkan/GGML documentation:

- [ggml build options](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
  and the [GGML CMake options](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt)
  define Vulkan builds, validation, and backend test switches;
- the [GGML operation support table](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)
  is the reference for backend operator coverage;
- Khronos documents that [pipeline barriers have a cost](https://docs.vulkan.org/samples/latest/samples/performance/pipeline_barriers/README.html)
  and that [pipeline caches should be reused](https://docs.vulkan.org/samples/latest/samples/performance/pipeline_cache/README.html);
- NVIDIA's [Vulkan Dos and Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts/)
  recommends minimizing queue submissions, reusing command buffers, using specialization
  constants, and checking GPU gaps with Nsight;
- [VK_KHR_cooperative_matrix](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
  explains why CM1 can accelerate a supported shape but does not guarantee numerical
  equivalence;
- [Nsight Systems Vulkan tracing](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
  and [Nsight Graphics GPU Trace](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html)
  are the recommended next profiling tools when a second vendor/device is available.

These sources justify the optimization direction, but every enabled path is accepted
only through the local output gate and repeated timing samples.

## Remaining TODOs

1. Run the same optimized profile on AMD and Intel Vulkan devices; current numerical and
   latency claims are RTX 3060-specific.
2. Investigate tile-resident deformable sampling/projection. The current experimental
   fusion is slower and remains disabled.
3. Capture a Vulkan trace with Nsight Systems/Graphics when available and record queue
   submit gaps, barriers, and shader occupancy in the benchmark report.

No TODO above is allowed to change the default profile without a parity result and a
median/P95 comparison in the same report schema.
