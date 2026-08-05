#!/usr/bin/env python3
"""Dump Swin stage-3 output (through 2 blocks @ 32x32)."""
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def build_shift_mask(H, W, ws, shift):
    Hp = int(np.ceil(H / ws)) * ws
    Wp = int(np.ceil(W / ws)) * ws
    img_mask = torch.zeros((1, Hp, Wp, 1))
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for h in h_slices:
        for w in h_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mw = window_partition(img_mask, ws).view(-1, ws * ws)
    attn = mw.unsqueeze(1) - mw.unsqueeze(2)
    return attn.masked_fill(attn != 0, -100.0).masked_fill(attn == 0, 0.0)


def run_stage(layer, x, H, W, ws):
    shift = ws // 2
    mask = build_shift_mask(H, W, ws, shift)
    for i, blk in enumerate(layer.blocks):
        blk.H, blk.W = H, W
        x = blk(x, None if i % 2 == 0 else mask)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("RMBG_MODEL", "ZhengPeng7/BiRefNet"))
    ap.add_argument("--image", default=DEFAULT_ASSET)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures", "swin_stage3_ref.gguf"))
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model.eval().float()
    bb = model.bb
    ws = bb.layers[0].window_size

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img).unsqueeze(0)

    with torch.no_grad():
        pe = bb.patch_embed(x)
        Wh, Ww = pe.shape[2], pe.shape[3]
        tokens = pe.flatten(2).transpose(1, 2)
        tokens = run_stage(bb.layers[0], tokens, Wh, Ww, ws)
        tokens = bb.layers[0].downsample(tokens, Wh, Ww)
        H1, W1 = Wh // 2, Ww // 2
        tokens = run_stage(bb.layers[1], tokens, H1, W1, ws)
        tokens = bb.layers[1].downsample(tokens, H1, W1)
        H2, W2 = H1 // 2, W1 // 2
        tokens = run_stage(bb.layers[2], tokens, H2, W2, ws)
        tokens = bb.layers[2].downsample(tokens, H2, W2)
        H3, W3 = H2 // 2, W2 // 2
        s3 = run_stage(bb.layers[3], tokens, H3, W3, ws)

    def to_np(t):
        return t.detach().float().cpu().numpy()

    writer = gguf.GGUFWriter(args.out, "rmbg_parity")
    writer.add_string("general.architecture", "rmbg_parity")
    writer.add_string("rmbg.ref_model", args.model)
    for name, arr in [
        ("input_nchw", to_np(x)),
        ("stage3_out", to_np(s3)),
    ]:
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} shape {s3.shape}")


if __name__ == "__main__":
    main()
