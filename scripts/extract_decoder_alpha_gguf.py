#!/usr/bin/env python3
"""Extract squeeze + decoder weights (short names) for forward_alpha parity."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

REPLACEMENTS = [
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


def shorten(name: str) -> str | None:
    for src, dst in REPLACEMENTS:
        if name.startswith(src):
            return name.replace(src, dst, 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.inp)
    writer = gguf.GGUFWriter(args.out, "rmbg")
    writer.add_string("general.architecture", "rmbg")
    writer.add_string("rmbg.subset", "decoder_alpha")
    n = 0
    for t in reader.tensors:
        out_name = shorten(t.name)
        if not out_name:
            continue
        if len(out_name) >= 64:
            sys.exit(f"name too long: {out_name} ({len(out_name)})")
        arr = np.array(t.data, dtype=np.float32)
        writer.add_tensor(out_name, arr)
        n += 1
    if n == 0:
        sys.exit("no tensors matched")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({n} tensors, {os.path.getsize(args.out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
