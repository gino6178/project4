#!/usr/bin/env python
"""Qualitative figures: the retargeting story in pictures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from experiments.common import load_corpus, make_targets
from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import split_scenes
from reroom.eval.metrics import evaluate
from reroom.geom.deform import LEVEL_NAMES
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.scene3d import render_scene3d
from reroom.render.topdown import figure_comparison
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.populate import CooccurrenceModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--flow", default="outputs/flow/flow.pt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    el = load_elasticity(a.elasticity) if os.path.exists(a.elasticity) else None
    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    cooc = None
    if os.path.exists(a.cooc):
        d = json.load(open(a.cooc))
        cooc = CooccurrenceModel(
            counts={k: Counter(v) for k, v in d["counts"].items()},
            sizes={k: np.asarray(v) for k, v in d["sizes"].items()},
            n_scenes=d["n_scenes"])
    flow = None
    if os.path.exists(a.flow):
        from reroom.generative.sample import load_flow
        flow = load_flow(a.flow, device="cpu")

    scenes = load_corpus(a.corpus, room_types=("bedroom", "living_room",
                                               "dining_room"),
                         min_objects=7, max_objects=14)
    _, _, test = split_scenes(scenes)
    cfg = RetargetConfig(restarts=24, device="cpu", seed=a.seed)
    made = []

    for ci, s in enumerate(test[:a.n]):
        g = build_motifs(build_scene_graph(s))
        targets = make_targets(s, (1, 2, 3, 4, 5), 1, a.seed + ci)
        # main figure: reference, then each level with baseline vs ours
        panels = [("reference", s)]
        caps = [f"A={s.room.area:.1f} m², {len(s.objects)} objects, "
                f"ρ={s.density():.2f}"]
        for t in targets:
            base = run_baseline("direct_scaling", g, t["room"], cfg=cfg)
            ours = retarget(g, t["room"], elasticity=el, bank=bank, cooc=cooc,
                            cfg=cfg).scene
            mb, mo = evaluate(g, base), evaluate(g, ours)
            panels += [(f"{LEVEL_NAMES[t['level']]}  ·  direct scaling", base),
                       (f"{LEVEL_NAMES[t['level']]}  ·  ReRoom", ours)]
            caps += [f"OOB {mb['R_OOB']:.1%}  col {mb['R_col']:.1%}  "
                     f"S_rel {mb['S_rel']:.2f}",
                     f"OOB {mo['R_OOB']:.1%}  col {mo['R_col']:.1%}  "
                     f"S_rel {mo['S_rel']:.2f}"]
        p = os.path.join(a.out, f"retarget_{ci}.png")
        figure_comparison(panels, p, ncols=3, per_panel=3.1, captions=caps,
                          suptitle="One reference design, five target geometries "
                                   "— direct scaling vs ReRoom")
        made.append(p)
        print("->", p, flush=True)

        if flow is not None and ci == 0:
            from reroom.generative.sample import generative_retarget
            t = targets[3]
            fl = generative_retarget(flow, g, t["room"], elasticity=el,
                                     bank=bank, cooc=cooc, cfg=cfg, k=12).scene
            fl_np = generative_retarget(flow, g, t["room"], elasticity=el,
                                        bank=bank, cooc=cooc, cfg=cfg, k=12,
                                        project=False).scene
            opt = retarget(g, t["room"], elasticity=el, bank=bank, cooc=cooc,
                           cfg=cfg).scene
            ms = [evaluate(g, x) for x in (fl_np, fl, opt)]
            p = os.path.join(a.out, "flow_vs_optimizer.png")
            figure_comparison(
                [("reference", s), ("flow proposal (unprojected)", fl_np),
                 ("flow + constraint projection", fl), ("optimizer only", opt)],
                p, ncols=4, per_panel=3.1,
                captions=[f"A={s.room.area:.1f} m²"] +
                         [f"OOB {m['R_OOB']:.1%}  S_rel {m['S_rel']:.2f}"
                          for m in ms],
                suptitle="Generative proposal → constraint projection (eq. 37)")
            made.append(p)
            print("->", p, flush=True)

        if ci == 0:
            for tag, sc in (("reference", s),
                            ("retargeted", retarget(g, targets[3]["room"],
                                                    elasticity=el, bank=bank,
                                                    cooc=cooc, cfg=cfg).scene)):
                p = os.path.join(a.out, f"scene3d_{tag}.png")
                render_scene3d(sc, p, figsize=4.2, title=tag)
                made.append(p)
    print("\n".join(made))


if __name__ == "__main__":
    main()
