#!/usr/bin/env python3
"""PyTorch RMBG-2.0 bridge until GGML inference graph lands.

Produces RGBA PNG for trellis-ggml t2_generate / t2_texture.

  python scripts/rmbg_pytorch_bridge.py --input photo.jpg --output photo_rgba.png
"""
from __future__ import annotations

import argparse
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="briaai/RMBG-2.0")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model.to(args.device).eval()

    image = Image.open(args.input).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(image).unsqueeze(0).to(args.device)
    with torch.no_grad():
        pred = model(x)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(pred[0].squeeze()).resize(image.size)
    image.putalpha(mask)
    image.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
