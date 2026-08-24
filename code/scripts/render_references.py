#!/usr/bin/env python
"""Render 3D-FRONT reference rooms to RGB + exact instance masks.

This is the input a single-image source parser actually needs (section 6), and
the image the appearance metric of section 15.2 should be computed on.  Only
rooms whose every 3D-FUTURE asset is on disk are rendered, so a missing mesh
never silently turns into a missing object in the "reference photograph".

Masks are rendered from the known geometry rather than segmented.  That hands
the parser a *favourable* setting on purpose: whatever error it then makes is
3D reasoning error, not segmentation error, which is the quantity experiment
14.3 is trying to isolate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.render.textured import (available_jids, best_camera,
                                    load_room_assets, scene_asset_coverage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--front", required=True, help="dir of raw 3D-FRONT json")
    ap.add_argument("--future", required=True, nargs="+")
    ap.add_argument("--out", default="outputs/references")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--views", type=int, default=4, help="multi-view renders too")
    ap.add_argument("--min-objects", type=int, default=5)
    ap.add_argument("--min-visible", type=int, default=4)
    ap.add_argument("--scan", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    have = available_jids(a.future)
    print(f"{len(have)} 3D-FUTURE meshes on disk", flush=True)
    scenes = list(iter_scenes(a.corpus, limit=a.scan or None,
                              min_objects=a.min_objects))
    if a.split != "all":
        tr, va, te = split_scenes(scenes)
        scenes = {"train": tr, "val": va, "test": te}[a.split]
    full = [s for s in scenes if scene_asset_coverage(s, have) > 0.999]
    print(f"{len(full)} / {len(scenes)} {a.split} scenes have every asset on disk",
          flush=True)

    manifest = []
    for s in full:
        if len(manifest) >= a.n:
            break
        house = s.meta.get("house")
        inst = s.scene_id.split("__", 1)[1]
        try:
            assets = load_room_assets(os.path.join(a.front, f"{house}.json"),
                                      inst, a.future,
                                      only_oids={o.oid for o in s.objects})
        except Exception as exc:
            print(f"  skip {s.scene_id}: {exc}")
            continue
        if assets is None or len(assets) < a.min_visible:
            continue
        assets.room = s.room
        cam, res = best_camera(assets, s.room, n=a.views,
                               width=a.width, height=a.height)
        vis = {k: v for k, v in res.coverage.items() if v > 0.004}
        if len(vis) < a.min_visible:
            continue

        key = s.scene_id
        d = os.path.join(a.out, key)
        os.makedirs(d, exist_ok=True)
        imageio.imwrite(os.path.join(d, "rgb.png"), res.rgb)
        # label image: 0 = background, 1..N = instances, in the order of ids
        lab = np.zeros(res.instance.shape, np.uint8)
        keep_ids, oid_map = [], {}
        for idx, oid in enumerate(res.ids):
            if res.coverage.get(oid, 0.0) <= 0.004:
                continue
            keep_ids.append(oid)
            lab[res.instance == idx] = len(keep_ids)
            oid_map[str(len(keep_ids))] = oid
        imageio.imwrite(os.path.join(d, "seg.png"), lab)

        extra = []
        for vi, cam2 in enumerate(_other_views(assets, s.room, a.views, cam)):
            r2 = _render(assets, cam2, a.width, a.height)
            p = os.path.join(d, f"view_{vi}.png")
            imageio.imwrite(p, r2.rgb)
            extra.append({"file": os.path.basename(p),
                          "camera": cam2.tolist()})

        rec = {
            "scene_id": key, "house": house, "room": inst,
            "rgb": "rgb.png", "seg": "seg.png",
            "camera": cam.tolist(),
            "yfov_deg": 60.0, "width": a.width, "height": a.height,
            "label_to_oid": oid_map,
            "coverage": {k: float(v) for k, v in res.coverage.items()},
            "room_polygon": s.room.polygon.tolist(),
            "room_height": s.room.height,
            "room_type": s.room.room_type,
            "n_objects": len(s.objects), "n_visible": len(keep_ids),
            "views": extra,
        }
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        s.save(os.path.join(d, "reference_scene.json"))
        manifest.append(rec)
        print(f"  [{len(manifest)}/{a.n}] {key}  visible {len(keep_ids)}"
              f"/{len(s.objects)}", flush=True)

    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{len(manifest)} reference renders -> {a.out}")


def _render(assets, cam, w, h):
    from reroom.render.textured import render_room
    return render_room(assets, cam, w, h)


def _other_views(assets, room, n, chosen):
    from reroom.render.textured import camera_poses
    out = []
    for c in camera_poses(room, n=n):
        if np.allclose(c, chosen):
            continue
        out.append(c)
    return out[:max(n - 1, 0)]


if __name__ == "__main__":
    main()
