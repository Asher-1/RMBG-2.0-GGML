#!/usr/bin/env python3
"""Extract Swin stage3 weights + bb_norm0-3 for encoder parity."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

STAGE_PREFIXES = []
for s in range(4):
    if s < 3:
        STAGE_PREFIXES.append(f"bb_layers_{s}_downsample_")
    depth = (2, 2, 18, 2)[s]
    for b in range(depth):
        STAGE_PREFIXES.append(f"bb_layers_{s}_blocks_{b}_")
PREFIXES = ("bb_patch_embed_",) + tuple(STAGE_PREFIXES) + tuple(f"bb_norm{i}_" for i in range(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.inp)
    writer = gguf.GGUFWriter(args.out, "rmbg")
    writer.add_string("general.architecture", "rmbg")
    writer.add_string("rmbg.subset", "encoder_4scale")
    n = 0
    for t in reader.tensors:
        if not t.name.startswith(PREFIXES):
            continue
        if len(t.name) >= 64:
            sys.exit(f"name too long: {t.name}")
        writer.add_tensor(t.name, np.array(t.data, dtype=np.float32))
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
