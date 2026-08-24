#!/usr/bin/env python
"""Experiment 2 -- floor-geometry difficulty (plan section 14.2).

Five target-geometry groups, from "same shape, different size" to "arbitrary
concave polygon".  The plan wants this to be one of the paper's main figures,
because a side-by-side across difficulty levels is what actually shows the
difference between reconstruction and retargeting.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from experiments.common import METHODS, load_corpus, run_grid, table
from reroom.data.corpus import split_scenes
from reroom.geom.deform import LEVEL_NAMES
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.topdown import figure_comparison
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig

ORDER = ["direct_scaling", "affine_fit", "target_only", "reroom_full"]


def qualitative_figure(scene, out_path, elasticity_path, bank_path, cooc_path,
                       seed=0, methods=("direct_scaling", "reroom_full")):
    """One reference room retargeted into all five difficulty levels."""
    from experiments.common import _init, make_targets
    _init({"elasticity_path": elasticity_path, "bank_path": bank_path,
           "cooc_path": cooc_path, "flow_path": None, "device": "cpu"})
    from experiments.common import _CTX
    graph = build_motifs(build_scene_graph(scene))
    targets = make_targets(scene, (1, 2, 3, 4, 5), 1, seed)
    cfg = RetargetConfig(restarts=24, device="cpu", seed=seed)
    panels = [("reference", scene)]
    captions = [f"A={scene.room.area:.1f} m$^2$, {len(scene.objects)} objects"]
    for t in targets:
        for m in methods:
            spec = METHODS[m]
            c = RetargetConfig(**{**cfg.__dict__, **{
                k: v for k, v in spec.overrides.items() if k != "no_project"}})
            if spec.kind == "baseline":
                out = run_baseline(spec.baseline, graph, t["room"], cfg=c)
            else:
                from reroom.retarget.optimizer import retarget
                out = retarget(graph, t["room"], elasticity=_CTX["fitted"],
                               bank=_CTX["bank"] if spec.use_bank else None,
                               cooc=_CTX["cooc"], cfg=c).scene
            panels.append((f"{LEVEL_NAMES[t['level']]} / {m}", out))
            captions.append(f"A={t['room'].area:.1f} m$^2$, "
                            f"{len([o for o in out.objects if o.keep])} objects")
    return figure_comparison(panels, out_path, ncols=len(methods) + 1,
                             per_panel=3.0, captions=captions,
                             suptitle="Experiment 2: one reference design, five target geometries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/exp2")
    ap.add_argument("--scenes", type=int, default=150)
    ap.add_argument("--per-level", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--figures", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scenes = load_corpus(a.corpus, room_types=("bedroom", "living_room",
                                               "dining_room", "library"),
                         min_objects=6, max_objects=18)
    _, _, test = split_scenes(scenes)
    methods = [m for m in a.methods.split(",") if m in METHODS]
    el = a.elasticity if os.path.exists(a.elasticity) else None

    rows = run_grid(test[:a.scenes], methods,
                    out_path=os.path.join(a.out, "rows.json"),
                    levels=(1, 2, 3, 4, 5), per_level=a.per_level,
                    seed=a.seed, workers=a.workers,
                    base_cfg=dict(restarts=20, grad_steps=200, proj_steps=90,
                                  device="cpu", seed=a.seed),
                    elasticity_path=el, bank_path=a.bank, cooc_path=a.cooc)

    parts = [table(rows, order=methods, title="Experiment 2: all levels pooled")]
    for lvl in range(1, 6):
        sub = [r for r in rows if r.get("level") == lvl]
        parts.append(table(sub, order=methods,
                           title=f"\nlevel {lvl}: {LEVEL_NAMES[lvl]}"))
    txt = "\n\n".join(parts)
    print(txt)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(txt + "\n")

    for k, s in enumerate(test[:a.figures]):
        p = qualitative_figure(s, os.path.join(a.out, f"figure_{k}.png"),
                               el, a.bank, a.cooc, seed=a.seed + k)
        print("figure ->", p, flush=True)


if __name__ == "__main__":
    main()
