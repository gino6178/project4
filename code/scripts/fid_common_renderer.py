#!/usr/bin/env python
"""FID under one renderer, with the ground truth as its own control.

Matching another paper's Blender pipeline exactly turned out to be a rabbit
hole: their script renders this project's scenes as unlit silhouettes, and the
cause is neither the materials (a flat diffuse colour renders equally dark) nor
the light energy (a hundredfold increase changes nothing).  Chasing it further
buys nothing that this does not.

So both sides go through *this* project's textured renderer instead, and the
real 3D-FRONT rooms are rendered by it too.  That last row is the whole point:
it is the floor of the metric.  Whatever FID the real rooms score against
themselves under this pipeline is the part that is renderer, not layout, and
any method's distance has to be read against it.

The consequence, stated plainly: these numbers are internally comparable and
are *not* comparable to InstructScene's published table, which used their
renderer on their splits.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from reroom.core.scene import Scene
from reroom.render.scene3d import CANONICAL_VIEWS, render_scene3d


def render_set(files: list[str], out: str, views, size: float) -> int:
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in files:
        try:
            sc = Scene.load(f)
        except Exception:
            continue
        key = os.path.splitext(os.path.basename(f))[0]
        for k, v in enumerate(views):
            p = os.path.join(out, f"{key}__{k}.png")
            if os.path.exists(p):
                n += 1
                continue
            try:
                render_scene3d(sc, p, view=v, figsize=size, dpi=64)
                n += 1
            except Exception:
                pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, nargs="+",
                    help="<label>=<dir-of-scene-json>")
    ap.add_argument("--work", default="/tmp/fid_common")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--out", default="outputs/fid_common.json")
    a = ap.parse_args()

    from cleanfid import fid

    views = list(CANONICAL_VIEWS)[:a.views]
    dirs = {}
    for spec in a.sets:
        label, d = spec.split("=", 1)
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]
        if a.limit:
            files = files[:a.limit]
        out = os.path.join(a.work, label)
        n = render_set(files, out, views, 4.0)
        dirs[label] = out
        print(f"  {label:14s} {len(files)} scenes -> {n} images", flush=True)

    ref = "real"
    if ref not in dirs:
        raise SystemExit("one of the sets must be labelled 'real'")
    import json
    res = {}
    for label, d in dirs.items():
        if label == ref:
            continue
        f = fid.compute_fid(dirs[ref], d, mode="clean", num_workers=4)
        res[label] = float(f)
        print(f"  FID({label} vs real) = {f:.2f}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
