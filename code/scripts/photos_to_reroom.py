#!/usr/bin/env python
"""Turn MIDI reconstructions of *real photographs* into ReRoom source scenes.

The synthetic path could lean on ground truth for categories and for the global
gauge.  A real photograph has neither, so all three unknowns are resolved the
way a deployed system would have to:

**Category** — CLIP zero-shot over ReRoom's canonical vocabulary, run on the
masked instance crop.  MIDI supplies geometry, not labels; a recogniser
supplies labels.  Both are reported so neither is mistaken for the other.

**Metric scale** — a single image cannot fix it (section 20 says as much and
allows floor calibration).  The scale is anchored on the object whose category
has the tightest real-world size prior, matched to the corpus median for that
category.  One number per room, stated in the output.

**Room outline** — no floor is observed, so the target boundary is the oriented
bounding rectangle of the reconstructed footprints, inflated by a walking
margin, with the room's dominant furniture direction taken as its axis.  This
is the reference room's *inferred* extent, and is only used as the source
geometry that retargeting maps away from.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.core.categories import PRIORS, prior
from reroom.core.scene import ObjectInstance, Room, Scene
from reroom.data.asset_bank import AssetBank

# categories whose real size varies least, best anchors for metric scale
_ANCHOR_PREFERENCE = ["dining_chair", "double_bed", "single_bed", "office_chair",
                      "dining_table", "sofa", "coffee_table", "nightstand",
                      "wardrobe", "desk", "armchair", "tv_stand"]

_PROMPTS = {
    "double_bed": "a double bed", "single_bed": "a single bed",
    "nightstand": "a nightstand", "wardrobe": "a wardrobe",
    "cabinet": "a storage cabinet", "bookcase": "a bookcase",
    "sideboard": "a sideboard", "drawer_chest": "a chest of drawers",
    "tv_stand": "a TV stand", "tv": "a television",
    "sofa": "a sofa", "l_sofa": "an L-shaped sectional sofa",
    "loveseat": "a two-seat sofa", "armchair": "an armchair",
    "lounge_chair": "a lounge chair", "dining_chair": "a dining chair",
    "office_chair": "an office chair", "stool": "a stool",
    "barstool": "a bar stool", "bench": "a bench",
    "dining_table": "a dining table", "coffee_table": "a coffee table",
    "side_table": "a small side table", "desk": "a desk",
    "dressing_table": "a dressing table", "console_table": "a console table",
    "floor_lamp": "a floor lamp", "table_lamp": "a table lamp",
    "pendant_lamp": "a pendant lamp", "ceiling_lamp": "a ceiling lamp",
    "rug": "a rug on the floor", "plant": "a potted plant",
    "decoration": "a small decorative object", "wall_art": "a framed picture",
    "mirror": "a mirror", "fireplace": "a fireplace", "piano": "a piano",
    "shelf": "a shelf", "shoe_cabinet": "a shoe cabinet",
    "wine_cabinet": "a wine cabinet", "kids_bed": "a child's bed",
    "bunk_bed": "a bunk bed", "misc": "a piece of furniture",
}


def classify(crops, device="cpu"):
    """CLIP zero-shot over the canonical vocabulary."""
    import open_clip
    import torch
    cats = [c for c in _PROMPTS if c in PRIORS]
    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok([f"a photo of {_PROMPTS[c]}" for c in cats]
                                  ).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        x = torch.stack([pre(c) for c in crops]).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        sim = (f @ t.T).cpu().numpy()
    best = sim.argmax(1)
    return [cats[b] for b in best], sim.max(1).tolist()


def instance_crops(rgb_path, seg_path, labels):
    from PIL import Image
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    seg = np.array(Image.open(seg_path).convert("L"))
    out = []
    for lb in labels:
        m = seg == lb
        if m.sum() < 32:
            out.append(None)
            continue
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = np.full_like(rgb, 255)
        patch[m] = rgb[m]
        out.append(Image.fromarray(patch[y0:y1, x0:x1]))
    return out


def anchor_scale(cats, sizes, bank_stats) -> tuple[float, str]:
    """Metric scale from the best-anchored object present in the room."""
    for c in _ANCHOR_PREFERENCE:
        idx = [i for i, x in enumerate(cats) if x == c]
        if not idx:
            continue
        pred = np.median([sizes[i] for i in idx], axis=0)
        ref = bank_stats.get(c)
        if ref is None:
            continue
        # match the two largest extents; the smallest is the least reliable
        p = np.sort(pred[:2])[::-1]
        r = np.sort(np.asarray(ref)[:2])[::-1]
        s = float(np.exp(np.mean(np.log(np.maximum(r, 1e-3))
                                 - np.log(np.maximum(p, 1e-3)))))
        if 0.05 < s < 200:
            return s, c
    return 1.0, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="outputs/midi_photos")
    ap.add_argument("--out", default="outputs/photo_scenes")
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--margin", type=float, default=0.7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    stats = ({c: bank.size_stats(c)["mean"] for c in bank.categories()}
             if bank else {})

    rows = []
    for ip in sorted(glob.glob(os.path.join(a.raw, "*", "instances.json"))):
        d = json.load(open(ip))
        key, dd = d["scene_id"], os.path.dirname(ip)
        insts = [i for i in d["instances"] if i.get("footprint_extent")]
        if len(insts) < 3:
            continue
        crops = instance_crops(os.path.join(dd, "rgb.png"),
                               os.path.join(dd, "seg.png"),
                               [i["label"] for i in insts])
        keep = [k for k, c in enumerate(crops) if c is not None]
        if len(keep) < 3:
            continue
        cats, conf = classify([crops[k] for k in keep], a.device)

        raw_xy, raw_size, raw_yaw, raw_z = [], [], [], []
        for k in keep:
            i = insts[k]
            cx, cz = i["footprint_centre_xz"]
            ex, ez = i["footprint_extent"]
            raw_xy.append([cx, -cz])
            raw_size.append([ex, ez, i["y_max"] - i["y_min"]])
            raw_yaw.append(-float(i["footprint_angle"]))
            raw_z.append(i["y_min"])
        raw_xy = np.asarray(raw_xy)
        raw_size = np.asarray(raw_size)
        s, anchor = anchor_scale(cats, raw_size, stats)

        xy = raw_xy * s
        size = raw_size * s
        z = (np.asarray(raw_z) - min(raw_z)) * s

        # dominant furniture direction becomes the room's axis
        ang = np.asarray(raw_yaw) % (math.pi / 2)
        w = size[:, 0] * size[:, 1]
        theta = float(np.average(ang, weights=np.maximum(w, 1e-6)))
        R = np.array([[math.cos(-theta), -math.sin(-theta)],
                      [math.sin(-theta), math.cos(-theta)]])
        xy = xy @ R.T
        yaws = [y - theta for y in raw_yaw]

        lo = xy.min(0) - a.margin
        hi = xy.max(0) + a.margin
        poly = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                         [hi[0], hi[1]], [lo[0], hi[1]]])
        room = Room(polygon=poly, height=2.7, room_type="other")

        objs = []
        for n, k in enumerate(keep):
            c = cats[n]
            yy = yaws[n]
            fwd = np.array([-math.sin(yy), math.cos(yy)])
            centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
            if float(np.dot(fwd, centre - xy[n])) < 0:
                yy += math.pi
            objs.append(ObjectInstance(
                oid=f"p{n}", category=c,
                position=np.array([xy[n, 0], xy[n, 1], max(0.0, float(z[n]))]),
                yaw=float(yy),
                size=np.maximum(size[n], 0.05),
                meta={"source": "midi_photo", "clip_confidence": float(conf[n]),
                      "label": insts[k]["label"],
                      # f^geo (eq. 10) straight off the reconstructed mesh --
                      # a photographed object has no catalogue id, so its shape
                      # descriptor is the only handle retrieval has on it
                      "shape": (np.asarray(insts[k]["shape"], dtype=np.float32)
                                if insts[k].get("shape") else None)}))

        scene = Scene(scene_id=key, room=room, objects=objs, source="photo",
                      meta={"parser": "midi", "dataset": d.get("source_dataset"),
                            "anchor_category": anchor, "metric_scale": s,
                            "room_axis_rad": theta, "rgb": d.get("rgb")})
        scene.save(os.path.join(a.out, f"{key}.json"))
        rows.append({"scene_id": key, "dataset": d.get("source_dataset"),
                     "n_objects": len(objs), "anchor": anchor,
                     "metric_scale": s, "area": room.area,
                     "categories": dict(Counter(cats))})
        print(f"  {key[:44]:46s} {len(objs):2d} obj  anchor={anchor:<13s} "
              f"scale={s:.3f}  room {room.extent[0]:.1f}x{room.extent[1]:.1f} m",
              flush=True)

    with open(os.path.join(a.out, "_manifest.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n{len(rows)} photograph-derived reference scenes -> {a.out}")


if __name__ == "__main__":
    main()
