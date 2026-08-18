#!/usr/bin/env python3
"""Convert briaai/RMBG-2.0 (BiRefNet architecture) safetensors -> GGUF metadata + weights.

Pattern follows depth-anything-cpp/scripts/convert_da2_to_gguf.py and
free-splatter.cpp/scripts/convert.py.

Usage:
  python scripts/convert_rmbg_to_gguf.py --model briaai/RMBG-2.0 --out models/rmbg_f16.gguf
  python scripts/convert_rmbg_to_gguf.py --safetensors /path/model.safetensors --out models/rmbg_f16.gguf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

ARCH = "rmbg"

ENCODER_PREFIXES = ["bb_patch_embed_"]
for _stage, _depth in enumerate((2, 2, 18, 2)):
    if _stage < 3:
        ENCODER_PREFIXES.append(f"bb_layers_{_stage}_downsample_")
    for _block in range(_depth):
        ENCODER_PREFIXES.append(f"bb_layers_{_stage}_blocks_{_block}_")
ENCODER_PREFIXES += [f"bb_norm{i}_" for i in range(4)]

DECODER_REPLACEMENTS = [
    ("decoder_gdt_convs_attn_4_", "gdta4_"),
    ("decoder_gdt_convs_attn_3_", "gdta3_"),
    ("decoder_gdt_convs_attn_2_", "gdta2_"),
    ("decoder_gdt_convs_4_", "gdt4_"),
    ("decoder_gdt_convs_3_", "gdt3_"),
    ("decoder_gdt_convs_2_", "gdt2_"),
    ("decoder_decoder_block4_", "db4_"),
    ("decoder_decoder_block3_", "db3_"),
    ("decoder_decoder_block2_", "db2_"),
    ("decoder_decoder_block1_", "db1_"),
    ("decoder_lateral_block4_", "lat4_"),
    ("decoder_lateral_block3_", "lat3_"),
    ("decoder_lateral_block2_", "lat2_"),
    ("decoder_ipt_blk5_", "ipt5_"),
    ("decoder_ipt_blk4_", "ipt4_"),
    ("decoder_ipt_blk3_", "ipt3_"),
    ("decoder_ipt_blk2_", "ipt2_"),
    ("decoder_ipt_blk1_", "ipt1_"),
    ("decoder_conv_out1_0_", "out1_"),
    ("squeeze_module_0_", "sq0_"),
]


def runtime_name(torch_name: str) -> str | None:
    flat = torch_name.replace(".", "_")
    if flat.endswith("_num_batches_tracked"):
        return None
    if flat.startswith(tuple(ENCODER_PREFIXES)):
        return flat
    for source, target in DECODER_REPLACEMENTS:
        if flat.startswith(source):
            return flat.replace(source, target, 1)
    return None


def load_state(model_id: str | None, safetensors: str | None):
    if safetensors:
        from safetensors.torch import load_file
        return {k: v.float().cpu().numpy() for k, v in load_file(safetensors).items()}
    from transformers import AutoModelForImageSegmentation
    import torch
    m = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    m.eval()
    return {k: v.detach().float().cpu().numpy() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="briaai/RMBG-2.0")
    ap.add_argument("--safetensors", default="")
    ap.add_argument("--model-id", default="",
                    help="source model identity stored in GGUF metadata")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ftype", type=int, default=1, help="0=f32 1=f16")
    args = ap.parse_args()

    sd = load_state(args.model if not args.safetensors else None,
                    args.safetensors or None)
    print(f"loaded {len(sd)} tensors")

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_string("rmbg.backbone", "swin_v1_l")
    writer.add_string(
        "rmbg.model_id", args.model_id or
        (args.safetensors if args.safetensors else args.model),
    )
    writer.add_uint32("rmbg.input_size", 1024)
    writer.add_array("rmbg.img.mean", [0.485, 0.456, 0.406])
    writer.add_array("rmbg.img.std", [0.229, 0.224, 0.225])

    exported = 0
    for name, arr in sorted(sd.items()):
        gguf_name = runtime_name(name)
        if gguf_name is None:
            continue
        if len(gguf_name) >= 64:
            sys.exit(f"runtime tensor name too long: {gguf_name}")
        if args.ftype == 1 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        writer.add_tensor(gguf_name, arr)
        exported += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({exported} tensors, {os.path.getsize(args.out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
