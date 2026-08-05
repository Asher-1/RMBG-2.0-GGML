#!/usr/bin/env python3
"""Extract Swin stage-0+1 (patch_embed, 4 blocks, stage0 downsample) into a small GGUF."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

PREFIXES = (
    "bb_patch_embed_",
    "bb_layers_0_blocks_0_",
    "bb_layers_0_blocks_1_",
    "bb_layers_0_downsample_",
    "bb_layers_1_blocks_0_",
    "bb_layers_1_blocks_1_",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.inp)
    writer = gguf.GGUFWriter(args.out, "rmbg")
    writer.add_string("general.architecture", "rmbg")
    writer.add_string("rmbg.subset", "swin_stage01")
    n = 0
    for t in reader.tensors:
        if not t.name.startswith(PREFIXES):
            continue
        if len(t.name) >= 64:
            sys.exit(f"name too long: {t.name}")
        arr = np.array(t.data, dtype=np.float32)
        writer.add_tensor(t.name, arr)
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
