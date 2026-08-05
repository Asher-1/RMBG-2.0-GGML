#!/usr/bin/env python3
"""Dump Swin stage-0 (patch_embed + blocks 0+1) activations for C++ parity."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("RMBG_MODEL", "ZhengPeng7/BiRefNet"))
    ap.add_argument("--image", default=DEFAULT_ASSET)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures", "swin_stage0_ref.gguf"))
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model.eval().float()
    bb = model.bb

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img).unsqueeze(0)

    with torch.no_grad():
        pe = bb.patch_embed(x)
        Wh, Ww = pe.shape[2], pe.shape[3]
        tokens = pe.flatten(2).transpose(1, 2)
        layer = bb.layers[0]
        for blk in layer.blocks:
            blk.H, blk.W = Wh, Ww
        # BasicLayer forward (stage 0 only)
        H, W = Wh, Ww
        ws = layer.window_size
        shift = ws // 2
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        Hp = int(np.ceil(H / ws)) * ws
        Wp = int(np.ceil(W / ws)) * ws
        img_mask = torch.zeros((1, Hp, Wp, 1))
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        b0 = layer.blocks[0](tokens, None)
        b1 = layer.blocks[1](b0, attn_mask)

    def to_np(t):
        return t.detach().float().cpu().numpy()

    writer = gguf.GGUFWriter(args.out, "rmbg_parity")
    writer.add_string("general.architecture", "rmbg_parity")
    writer.add_string("rmbg.ref_model", args.model)
    for name, arr in [
        ("input_nchw", to_np(x)),
        ("patch_embed", to_np(pe)),
        ("block0_out", to_np(b0)),
        ("stage0_out", to_np(b1)),
    ]:
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
