#!/usr/bin/env python
"""Ablations (plan section 16.2).

Removes one component at a time: relation elasticity, motif-rigid
initialisation, summarization/population, asset substitution and the
constraint-projection stage.  The comparison the plan calls out as the most
important -- normalized coordinate scaling vs relation-aware retargeting (44) --
is included as the two end points.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from experiments.common import METHODS, load_corpus, run_grid, table
from reroom.data.corpus import split_scenes

ORDER = ["direct_scaling", "no_elasticity", "prior_elasticity",
         "no_motif_init", "no_motif_grouping", "size_only_retrieval",
         "no_projection", "relation_only", "relation_summary", "reroom_full",
         "flow_no_projection", "flow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/ablations")
    ap.add_argument("--scenes", type=int, default=150)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--flow", default="outputs/flow/flow.pt")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scenes = load_corpus(a.corpus, room_types=("bedroom", "living_room",
                                               "dining_room", "library"),
                         min_objects=6, max_objects=18)
    _, _, test = split_scenes(scenes)
    methods = [m for m in a.methods.split(",") if m in METHODS]
    rows = run_grid(test[:a.scenes], methods,
                    out_path=os.path.join(a.out, "rows.json"),
                    levels=(1, 2, 3, 4, 5), per_level=1, seed=a.seed,
                    workers=a.workers,
                    base_cfg=dict(restarts=16, grad_steps=200, proj_steps=90,
                                  device="cpu", seed=a.seed),
                    elasticity_path=(a.elasticity
                                     if os.path.exists(a.elasticity) else None),
                    bank_path=a.bank, cooc_path=a.cooc,
                    flow_path=a.flow if os.path.exists(a.flow) else None)
    txt = table(rows, order=methods, title="Ablations (section 16.2)")
    txt2 = table(rows, by="level_name", title="\nby difficulty level")
    print(txt + "\n\n" + txt2)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(txt + "\n\n" + txt2 + "\n")


if __name__ == "__main__":
    main()
