#!/usr/bin/env python3
"""Dump Swin-L stage0 block0 activations for C++ parity.

Output: tests/fixtures/swin_block0_ref.gguf
  input_nchw   [1,3,1024,1024]  ImageNet-normalized
  patch_embed  [1,192,256,256]
  block0_out   [1,65536,192]     after layers.0.blocks.0 (shift=0)

Default model: ZhengPeng7/BiRefNet (open). Use briaai/RMBG-2.0 when HF access granted.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ASSET = os.path.join(os.path.dirname(ROOT), "trellis-ggml", "assets", "example_image", "T.png")


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("RMBG_MODEL", "ZhengPeng7/BiRefNet"))
    ap.add_argument("--image", default=DEFAULT_ASSET)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures", "swin_block0_ref.gguf"))
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model.eval().float().to(device)
    bb = model.bb

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pe = bb.patch_embed(x)
        Wh, Ww = pe.shape[2], pe.shape[3]
        tokens = pe.flatten(2).transpose(1, 2)
        blk = bb.layers[0].blocks[0]
        blk.H, blk.W = Wh, Ww
        B, L, C = tokens.shape
        H, W = Wh, Ww
        shortcut = tokens
        x1 = blk.norm1(tokens).view(B, H, W, C)
        ws = blk.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x1p = torch.nn.functional.pad(x1, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x1p.shape[1], x1p.shape[2]
        xw = window_partition(x1p, ws).view(-1, ws * ws, C)
        aw = blk.attn(xw, None).view(-1, ws * ws, C)
        xr = window_reverse(aw, ws, Hp, Wp)[:, :H, :W, :].contiguous().view(B, L, C)
        after_attn = shortcut + xr
        y = after_attn + blk.mlp(blk.norm2(after_attn))
        norm1_blc = blk.norm1(tokens)

    def to_np(t):
        return t.detach().float().cpu().numpy()

    writer = gguf.GGUFWriter(args.out, "rmbg_parity")
    writer.add_string("general.architecture", "rmbg_parity")
    writer.add_string("rmbg.ref_model", args.model)
    writer.add_uint32("rmbg.embed_dim", 192)
    writer.add_uint32("rmbg.window_size", 12)
    writer.add_uint32("rmbg.num_heads", 6)
    writer.add_uint32("rmbg.input_size", args.size)
    for name, arr in [
        ("input_nchw", to_np(x)),
        ("patch_embed", to_np(pe)),
        ("block0_norm1", to_np(norm1_blc)),
        ("block0_xw0", to_np(xw[0])),
        ("block0_aw0", to_np(aw[0])),
        ("block0_after_attn", to_np(after_attn)),
        ("block0_out", to_np(y)),
    ]:
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} (model={args.model})")
    print(f"  input {tuple(x.shape)} patch {tuple(pe.shape)} block0 {tuple(y.shape)}")


if __name__ == "__main__":
    main()
