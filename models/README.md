# RMBG GGUF Assets

Only the following files are supported, deployable RMBG end-to-end models:

| File | Weight storage | Intended use |
|---|---|---|
| `rmbg_f32.gguf` | F32 | Numerical reference and strict parity baseline. |
| `rmbg_f16.gguf` | F16 | Default deployment model. Half the F32 weight footprint. |

There is no supported Q8 deployment model. `rmbg_q8.gguf` may be present in a working
checkout as an experimental benchmark artifact, but it violates the `2e-3` alpha gate
on CPU, CUDA, and Vulkan and is not recommended or released as a deployment file.
GGML's generic Q8 reader remains unchanged.

Development and unit-test assets live in [`development/README.md`](development/README.md),
separate from selectable deployment models.

Generate the two deployable files from the split source weights in `development/` with
`scripts/quantize_rmbg_gguf.py`; the commands and measured parity limits are in the
repository README.
