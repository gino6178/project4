#!/usr/bin/env python
"""FID / KID against InstructScene's own reference images (their Table 1).

Their evaluation renders scenes with Blender at 256px and compares the
synthesised set against the *real* 3D-FRONT set with clean-fid.  They ship
those real renders, which is what makes three of their five columns genuinely
comparable -- so ReRoom's scenes are pushed through the same renderer and the
same metric here rather than being declared incommensurable.

Two settings are reported, because they are not the same difficulty:

``as-is``      the reference room's own floor plan, which is the setting their
               numbers are computed in;
``retargeted`` a deformed target polygon, which is this project's actual task
               and is expected to score worse on a metric that rewards looking
               like the training distribution.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect(src_glob: str, dst: str, limit: int = 0) -> int:
    os.makedirs(dst, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(src_glob)):
        shutil.copy(p, os.path.join(dst, f"{n:06d}.png"))
        n += 1
        if limit and n >= limit:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", required=True,
                    help="InstructScene/threed_front_<type> directory root")
    ap.add_argument("--room-types", default="bedroom,livingroom,diningroom")
    ap.add_argument("--renders", required=True, nargs="+",
                    help="one or more <label>=<dir-of-render-dirs>")
    ap.add_argument("--work", default="/tmp/fid_work")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/fid_instructscene.json")
    a = ap.parse_args()

    from cleanfid import fid

    real_dir = os.path.join(a.work, "real")
    shutil.rmtree(real_dir, ignore_errors=True)
    n_real = 0
    for rt in a.room_types.split(","):
        g = os.path.join(a.real_root, f"threed_front_{rt}", "*",
                         "blender_rendered_scene_256", "*.png")
        n_real += collect(g, real_dir, a.limit)
    print(f"real reference images: {n_real}")

    out = {"n_real": n_real, "settings": {}}
    for spec in a.renders:
        label, d = spec.split("=", 1)
        syn = os.path.join(a.work, f"syn_{label}")
        shutil.rmtree(syn, ignore_errors=True)
        n = collect(os.path.join(d, "*", "*.png"), syn, a.limit)
        if n < 50:
            print(f"  {label}: only {n} images, skipping")
            continue
        f = fid.compute_fid(real_dir, syn, mode="clean", num_workers=4)
        k = fid.compute_kid(real_dir, syn, mode="clean", num_workers=4)
        out["settings"][label] = {"FID": float(f), "KID": float(k), "n": n}
        print(f"  {label:14s} n={n:6d}   FID {f:8.2f}   KID {k*1e3:8.3f} x1e-3",
              flush=True)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n->", a.out)
    print("InstructScene Table 1 (their own generations, for reference):")
    print("  bedroom     FID 114.78  KID 0.32e-3")
    print("  living room FID 110.39  KID 8.16e-3")
    print("  dining room FID 129.76  KID 13.24e-3")
    print("Their FID is per room type against that type's real set; the number "
          "above pools the types, so read the magnitude, not the decimal.")


if __name__ == "__main__":
    main()
