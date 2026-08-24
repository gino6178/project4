#!/usr/bin/env python
"""Experiment 1 -- oracle retargeting (plan section 14.1).

Ground-truth 3D-FRONT scene graphs, no image parsing at all, comparing

  * direct normalized-coordinate scaling
  * affine fitting
  * relation-aware optimisation
  * relation-aware + motif summarization
  * relation-aware + summarization + asset substitution

The plan is explicit about the stakes: if relation-aware retargeting does not
clearly beat direct scaling here, the research hypothesis needs revising and no
amount of generative modelling later will fix it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (METHODS, load_corpus, run_grid, save_rows,
                                table)
from reroom.data.corpus import split_scenes

ORDER = ["source_reference", "reference_rigid", "direct_scaling", "affine_fit",
         "target_only", "relation_only", "relation_summary", "reroom_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/exp1")
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--per-level", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--room-types", default="bedroom,living_room,dining_room,library")
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rts = tuple(x for x in a.room_types.split(",") if x)
    scenes = load_corpus(a.corpus, room_types=rts, min_objects=5, max_objects=20)
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]
    print(f"{len(test)} held-out scenes")

    methods = [m for m in a.methods.split(",") if m in METHODS]
    levels = tuple(int(x) for x in a.levels.split(","))
    rows = run_grid(
        test, methods, out_path=os.path.join(a.out, "rows.json"),
        levels=levels, per_level=a.per_level, seed=a.seed, workers=a.workers,
        base_cfg=dict(restarts=a.restarts, grad_steps=250, proj_steps=120,
                      device="cpu", seed=a.seed),
        elasticity_path=a.elasticity if os.path.exists(a.elasticity) else None,
        bank_path=a.bank, cooc_path=a.cooc)

    txt = table(rows, order=methods,
                title="Experiment 1: oracle retargeting (3D-FRONT GT graphs)")
    print("\n" + txt)
    txt2 = table(rows, by="level_name",
                 title="\nby target floor-geometry difficulty")
    print("\n" + txt2)
    txt3 = table(rows, by="area_bucket",
                 title="\nby target/source area ratio")
    print("\n" + txt3)
    order4 = [f"{m} | {b}" for b in ("a<0.75 (shrink)", "e >1.50 (grow)")
              for m in methods]
    txt4 = table(rows, by="method_x_area", order=order4,
                 title="\nmethod x area ratio (extremes only)")
    print("\n" + txt4)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write("\n\n".join([txt, txt2, txt3, txt4]) + "\n")


if __name__ == "__main__":
    main()
