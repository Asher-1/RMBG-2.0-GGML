#!/usr/bin/env python3
import gguf
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def cpp_window_attention(xin, attn, nh=6, hd=32):
    N, C = xin.shape
    Wq = attn.qkv.weight.detach().numpy()
    bq = attn.qkv.bias.detach().numpy()
    Wp = attn.proj.weight.detach().numpy()
    bp = attn.proj.bias.detach().numpy()
    scale = hd ** -0.5
    qkv = xin @ Wq.T + bq
    rpi = attn.relative_position_index.numpy().astype(int)
    rpb = attn.relative_position_bias_table.detach().numpy()
    out = np.zeros_like(xin)
    attn_m = np.zeros((N, N), dtype=np.float32)
    for h in range(nh):
        a = attn_m
        for i in range(N):
            for j in range(N):
                s = 0.0
                for d in range(hd):
                    s += qkv[i, h * hd + d] * scale * qkv[j, C + h * hd + d]
                s += rpb[rpi[i, j], h]
                a[i, j] = s
        for i in range(N):
            row = a[i]
            e = np.exp(row - row.max())
            a[i] = e / e.sum()
        for i in range(N):
            for d in range(hd):
                s = sum(a[i, j] * qkv[j, 2 * C + h * hd + d] for j in range(N))
                out[i, h * hd + d] = s
    return out @ Wp.T + bp


def main():
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    ).eval().float()
    bb = model.bb
    img = Image.open(
        "/home/ludahai/develop/code/github/dl/trellis-ggml/assets/example_image/T.png"
    ).convert("RGB").resize((1024, 1024))
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = norm(transforms.ToTensor()(img)).unsqueeze(0)
    with torch.no_grad():
        pe = bb.patch_embed(x)
        Wh, Ww = pe.shape[2], pe.shape[3]
        tok = pe.flatten(2).transpose(1, 2)
        blk = bb.layers[0].blocks[0]
        blk.H, blk.W = Wh, Ww
        B, L, C = tok.shape
        H, W = Wh, Ww
        ws = blk.window_size
        shortcut = tok.numpy()
        x1 = blk.norm1(tok).view(B, H, W, C).numpy()
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x1p = np.pad(x1, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        Hp, Wp = x1p.shape[1], x1p.shape[2]
        xw = window_partition(torch.from_numpy(x1p), ws).view(-1, ws * ws, C).numpy()
        aw_pt = blk.attn(torch.from_numpy(xw[:1]).float(), None).numpy()[0]
        aw_cpp = cpp_window_attention(xw[0], blk.attn)
        print("single window attn max", np.abs(aw_pt - aw_cpp).max())

        nW = (Hp // ws) * (Wp // ws)
        win = np.zeros_like(xw)
        for wi in range(nW):
            win[wi] = cpp_window_attention(xw[wi], blk.attn)
        win4 = win.reshape(-1, ws, ws, C)
        xr = window_reverse(torch.from_numpy(win4), ws, Hp, Wp).numpy()[:, :H, :W, :].reshape(B, L, C)
        after = shortcut + xr
        reader = gguf.GGUFReader(
            "/home/ludahai/develop/code/github/dl/RMBG-2.0-GGML/tests/fixtures/swin_block0_ref.gguf"
        )
        ref = {t.name: t.data for t in reader.tensors}["block0_after_attn"]
        print("full after_attn max", np.abs(ref - after).max())


if __name__ == "__main__":
    main()
