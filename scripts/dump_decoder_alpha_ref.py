#!/usr/bin/env python3
"""Golden forward_alpha logits (BiRefNet eval, pre-sigmoid)."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

try:
    import gguf
except ImportError:
    raise SystemExit("pip install gguf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ASSET = os.path.join(os.path.dirname(ROOT), "trellis-ggml", "assets", "example_image", "T.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("RMBG_MODEL", "ZhengPeng7/BiRefNet"))
    ap.add_argument("--image", default=DEFAULT_ASSET)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures", "alpha_ref.gguf"))
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model.eval().float()

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img).unsqueeze(0)

    with torch.no_grad():
        preds = model(x)
        alpha = preds[-1]

    def to_np(t):
        return t.detach().float().cpu().numpy()[0]

    writer = gguf.GGUFWriter(args.out, "rmbg_parity")
    writer.add_string("general.architecture", "rmbg_parity")
    writer.add_string("rmbg.ref_model", args.model)
    writer.add_tensor("input_nchw", to_np(x).astype(np.float32))
    writer.add_tensor("alpha_logits", to_np(alpha).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}")
    print(f"  alpha_logits: {tuple(alpha.shape)}")


if __name__ == "__main__":
    main()
