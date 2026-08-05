# Development and Test GGUF Assets

These files are fixtures for conversion and parity tests. They are not selectable
end-to-end deployment models. Production users should choose only `../rmbg_f16.gguf` or
`../rmbg_f32.gguf`.

| File | Consumer and purpose |
|---|---|
| `encoder_f16.gguf` | Legacy split encoder; full-graph and encoder parity tests. The converter uses it with the sibling decoder to produce a unified model. |
| `decoder_alpha_f16.gguf` | Legacy split decoder/alpha head; paired with `encoder_f16.gguf`. |
| `swin_block0_f16.gguf` | First Swin block and positional-embedding parity tests. |
| `swin_stage0_f16.gguf` | Isolated Swin stage 0 test. |
| `swin_stage01_f16.gguf` | Combined stages 0 and 1 test. |
| `swin_stage2_f16.gguf` | Isolated Swin stage 2 test. |
| `swin_stage3_f16.gguf` | Isolated Swin stage 3 test. |
| `squeeze_f16.gguf` | Decoder squeeze-module tests. |
| `birefnet_f16.gguf` | Early raw-name conversion artifact used only while diagnosing conversion mappings. |

The test defaults point here. Every test also accepts its documented `RMBG_WEIGHTS`,
`RMBG_ENC_WEIGHTS`, or `RMBG_DEC_WEIGHTS` override, so a fixture can be replaced without
copying it back into the deployment directory.
