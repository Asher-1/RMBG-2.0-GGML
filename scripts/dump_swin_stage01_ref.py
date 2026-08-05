#!/usr/bin/env python3
"""Dump Swin stage-0+1 activations (through patch merge + stage1 blocks)."""
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
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "fixtures", "swin_stage01_ref.gguf"))
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
        layer0 = bb.layers[0]
        layer1 = bb.layers[1]
        for blk in layer0.blocks + layer1.blocks:
            blk.H, blk.W = Wh, Ww

        ws = layer0.window_size
        shift = ws // 2
        pad_r = (ws - Ww % ws) % ws
        pad_b = (ws - Wh % ws) % ws
        Hp = int(np.ceil(Wh / ws)) * ws
        Wp = int(np.ceil(Ww / ws)) * ws
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

        b0 = layer0.blocks[0](tokens, None)
        s0 = layer0.blocks[1](b0, attn_mask)
        merged = layer0.downsample(s0, Wh, Ww)
        H1, W1 = Wh // 2, Ww // 2
        for blk in layer1.blocks:
            blk.H, blk.W = H1, W1

        # Shifted-window mask depends on spatial resolution — rebuild for stage-1.
        Hp1 = int(np.ceil(H1 / ws)) * ws
        Wp1 = int(np.ceil(W1 / ws)) * ws
        img_mask1 = torch.zeros((1, Hp1, Wp1, 1))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask1[:, h, w, :] = cnt
                cnt += 1
        mask1 = window_partition(img_mask1, ws).view(-1, ws * ws)
        attn_mask1 = mask1.unsqueeze(1) - mask1.unsqueeze(2)
        attn_mask1 = attn_mask1.masked_fill(attn_mask1 != 0, -100.0).masked_fill(attn_mask1 == 0, 0.0)

        s1b0 = layer1.blocks[0](merged, None)
        s1 = layer1.blocks[1](s1b0, attn_mask1)

    def to_np(t):
        return t.detach().float().cpu().numpy()

    writer = gguf.GGUFWriter(args.out, "rmbg_parity")
    writer.add_string("general.architecture", "rmbg_parity")
    writer.add_string("rmbg.ref_model", args.model)
    for name, arr in [
        ("input_nchw", to_np(x)),
        ("patch_embed", to_np(pe)),
        ("stage0_out", to_np(s0)),
        ("merge_out", to_np(merged)),
        ("stage1_out", to_np(s1)),
    ]:
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
