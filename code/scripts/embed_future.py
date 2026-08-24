#!/usr/bin/env python
"""CLIP-embed 3D-FUTURE product images for style-aware retrieval (eq. 30).

    a*_i = argmin_j [ lf Df(f^r_i, f^3D_j) + ls Ds(s^req_i, s_j) ]

``Df`` needs an appearance embedding per candidate asset.  3D-FUTURE ships a
product image per model, which is a better appearance proxy than multi-view
renders of the untextured mesh and costs one forward pass per asset.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    a = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image

    ids, paths = [], []
    for root in a.roots:
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d, "image.jpg")
            if os.path.exists(p):
                ids.append(d)
                paths.append(p)
    print(f"{len(paths)} asset images", flush=True)
    if not paths:
        return

    dev = torch.device(a.device)
    model, _, pre = open_clip.create_model_and_transforms(a.model,
                                                          pretrained=a.pretrained)
    model = model.to(dev).eval()
    out = []
    for k in range(0, len(paths), a.batch):
        imgs = []
        for p in paths[k:k + a.batch]:
            try:
                imgs.append(pre(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        with torch.no_grad():
            f = model.encode_image(torch.stack(imgs).to(dev))
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out.append(f.cpu().numpy().astype(np.float32))
        if k % (a.batch * 20) == 0:
            print(f"  {k}/{len(paths)}", flush=True)
    emb = np.concatenate(out, 0)
    np.savez_compressed(a.out, ids=np.array(ids), emb=emb)
    print(f"wrote {emb.shape} -> {a.out}")


if __name__ == "__main__":
    main()
