#!/usr/bin/env python3
"""Export supported F32 or F16 RMBG GGUF weights from runtime-named GGUFs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError as exc:
    raise SystemExit("pip install gguf") from exc

def write_metadata(writer: gguf.GGUFWriter, fmt: str) -> None:
    writer.add_string("rmbg.backbone", "swin_v1_l")
    writer.add_string("rmbg.weight_format", fmt)
    writer.add_uint32("rmbg.input_size", 1024)
    writer.add_array("rmbg.img.mean", [0.485, 0.456, 0.406])
    writer.add_array("rmbg.img.std", [0.229, 0.224, 0.225])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, nargs="+",
                        help="runtime-named encoder and optional decoder GGUF files")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--format", choices=("f32", "f16"), required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), "rmbg")
    write_metadata(writer, args.format)
    names: set[str] = set()

    for input_path in args.input:
        for tensor in gguf.GGUFReader(str(input_path)).tensors:
            if tensor.name in names:
                raise SystemExit(f"duplicate tensor name: {tensor.name}")
            names.add(tensor.name)
            data = np.asarray(tensor.data, dtype=np.float32)
            if args.format == "f32":
                writer.add_tensor(tensor.name, data)
            else:
                writer.add_tensor(tensor.name, data.astype(np.float16))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
