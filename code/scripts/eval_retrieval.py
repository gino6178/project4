#!/usr/bin/env python
"""Evaluate style-aware furniture retrieval on real 3D-FUTURE assets.

    a*_i = argmin_j [ lf Df(f^r_i, f^3D_j) + ls Ds(s^req_i, s_j) ]      (30)

The plan's argument for retrieval over rescaling: when the reference sofa is too
big, the system should fetch a stylistically similar sofa that is *actually*
smaller rather than squash the reference to 70 %.  That claim has two testable
halves, and this script measures both against the two degenerate weightings:

* does retrieval hit the requested size?  (``size error``, log-space)
* does it keep the reference's look?      (``CLIP similarity`` to the original)

``lambda_s = 0`` (appearance only) and ``lambda_f = 0`` (size only) are the
controls: each should win its own metric and lose the other, and the balanced
objective should sit near the frontier of both.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.data.asset_bank import AssetBank, FutureBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.perception.geometry import shape_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--future-root", required=True, nargs="+")
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--shapes", default=None,
                    help="f^geo descriptors from build_future_shapes.py")
    ap.add_argument("--scenes", type=int, default=400)
    ap.add_argument("--out", default="outputs/retrieval.json")
    ap.add_argument("--scales", default="0.7,0.85,1.15,1.4")
    a = ap.parse_args()

    bank = FutureBank.from_dir(a.future_root, bbox_cache=a.bboxes)
    bank.attach_embeddings(a.embeddings)
    if a.shapes and os.path.exists(a.shapes):
        bank.attach_shapes(a.shapes)
        print(f"f^geo on {sum(1 for x in bank.assets if x.shape is not None)} assets")
    have = {x.aid for x in bank.assets if x.embedding is not None}
    print(f"bank: {len(bank)} assets, {len(have)} with embeddings, "
          f"{len(bank.categories())} categories")

    scenes = list(iter_scenes(a.corpus, limit=None, min_objects=4))
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]
    lut = {x.aid: x for x in bank.assets}

    # (lambda_f, lambda_s, lambda_g); the last arm adds the f^geo shape term
    settings = {"balanced": (1.0, 1.0, 0.0), "size_only": (0.0, 1.0, 0.0),
                "appearance_only": (1.0, 0.0, 0.0),
                "balanced+geo": (1.0, 1.0, 0.6)}
    rows = {k: {"size_err": [], "sim": [], "shape_err": [], "n": 0, "miss": 0}
            for k in settings}
    base_err = []
    scales = [float(x) for x in a.scales.split(",")]

    for s in test:
        for o in s.objects:
            src = lut.get(o.jid)
            if src is None or src.embedding is None:
                continue
            if not bank.has(o.category):
                continue
            for sc in scales:
                req = src.size.copy()
                req[:2] = req[:2] * sc
                # what you get by doing nothing: keep the reference asset
                base_err.append(float(np.abs(
                    np.log(np.maximum(src.size[:2], 1e-3))
                    - np.log(np.maximum(req[:2], 1e-3))).mean()))
                for name, (lf, ls, lg) in settings.items():
                    hit = bank.retrieve(o.category, req, src.embedding,
                                        lambda_f=lf, lambda_s=ls, topk=1,
                                        exclude={src.aid},
                                        ref_shape=src.shape, lambda_g=lg)
                    r = rows[name]
                    r["n"] += 1
                    if not hit:
                        r["miss"] += 1
                        continue
                    asset = hit[0][0]
                    r["size_err"].append(float(np.abs(
                        np.log(np.maximum(asset.size[:2], 1e-3))
                        - np.log(np.maximum(req[:2], 1e-3))).mean()))
                    r["shape_err"].append(shape_distance(src.shape, asset.shape))
                    if asset.embedding is not None:
                        e1 = src.embedding
                        e2 = asset.embedding
                        r["sim"].append(float(
                            e1 @ e2 / max(np.linalg.norm(e1) *
                                          np.linalg.norm(e2), 1e-9)))

    out = {"n_queries": rows["balanced"]["n"],
           "no_substitution_size_err": float(np.mean(base_err)) if base_err else None,
           "settings": {}}
    print(f"\n{len(base_err)} queries; doing nothing costs a mean log-size error "
          f"of {np.mean(base_err):.4f}\n")
    print(f"{'setting':18s}{'size err':>12s}{'CLIP sim':>12s}"
          f"{'shape err':>12s}{'n':>8s}")
    for name in settings:
        r = rows[name]
        se = float(np.mean(r["size_err"])) if r["size_err"] else float("nan")
        si = float(np.mean(r["sim"])) if r["sim"] else float("nan")
        sh = float(np.mean(r["shape_err"])) if r["shape_err"] else float("nan")
        out["settings"][name] = {"size_err": se, "clip_sim": si,
                                 "shape_err": sh, "n": r["n"],
                                 "misses": r["miss"]}
        print(f"{name:18s}{se:12.4f}{si:12.4f}{sh:12.4f}{r['n']:8d}")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
