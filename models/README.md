# RMBG GGUF Assets

Only the following files are supported, deployable RMBG end-to-end models:

| File | Weight storage | Intended use |
|---|---|---|
| `rmbg_f32.gguf` | F32 | Numerical reference and strict parity baseline. |
| `rmbg_f16.gguf` | F16 | Default deployment model. Half the F32 weight footprint. |

There is no Q8 deployment model. The former hybrid Q8 file saved only 16.9 MiB relative
to F16 and did not improve CUDA or Vulkan latency; full Q8 quantization also violated
the `2e-3` alpha gate. It has been removed. GGML's generic Q8 reader remains unchanged,
but this repository no longer exports or distributes an RMBG Q8 artifact.

Development and unit-test assets live in [`development/README.md`](development/README.md),
separate from selectable deployment models.

Generate the two deployable files from the split source weights in `development/` with
`scripts/quantize_rmbg_gguf.py`; the commands and measured parity limits are in the
repository README.
