#!/usr/bin/env python
"""Is relation elasticity really about elasticity, or about walls?

Fitted alpha says a bed-to-wardrobe distance follows the room (0.87) while a
chair-to-table distance does not (0.05).  But a bed and a wardrobe are both
backed onto walls, usually opposite ones, so their separation *is* the room's
width minus two depths -- alpha near 1 there is arithmetic, not a design
principle.  A chair and a table are both free-standing and spaced by the human
body.

If that is what alpha is measuring, then alpha should track how wall-mediated a
relation is, and adding it to a solver that already anchors furniture to walls
would be telling it something it has already been told.  This measures the
correlation directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.intent.elasticity import StatElasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--stat", default="outputs/elasticity/stat.json")
    ap.add_argument("--scenes", type=int, default=1500)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--out", default="outputs/alpha_vs_walls.json")
    a = ap.parse_args()

    stat = StatElasticity.load(a.stat)
    scenes = list(iter_scenes(a.corpus, limit=None, min_objects=5))
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]
    print(f"{len(test)} held-out scenes", flush=True)

    both = defaultdict(list)      # bucket -> [1 if both wall-backed else 0]
    cross = defaultdict(list)     # bucket -> [1 if the pair spans two motifs]
    for k, s in enumerate(test):
        g = build_motifs(build_scene_graph(s))
        walled = {w.i for w in g.walls}
        mof = {}
        for m in g.motifs:
            for i in m.members:
                mof[i] = m.mid
        for r in g.relations:
            if r.kind in ("support", "grouped_with"):
                continue
            key = stat._key(s.objects[r.i].category, s.objects[r.j].category,
                            r.kind)
            both[key].append(float(r.i in walled and r.j in walled))
            cross[key].append(float(mof.get(r.i, "a") != mof.get(r.j, "b")))
        if k % 300 == 0:
            print(f"  {k}/{len(test)}", flush=True)

    rows = []
    for key, flags in both.items():
        hit = stat.pair_alpha.get(key)
        if hit is None or len(flags) < a.min_n:
            continue
        alpha, n, r2 = hit
        rows.append({"bucket": key, "alpha": alpha, "n_fit": n,
                     "both_wall_backed": float(np.mean(flags)),
                     "cross_motif": float(np.mean(cross[key])),
                     "n_obs": len(flags)})
    if len(rows) < 4:
        print("not enough buckets")
        return
    al = np.array([r["alpha"] for r in rows])
    wl = np.array([r["both_wall_backed"] for r in rows])
    w = np.array([r["n_obs"] for r in rows], dtype=float)
    xm = np.array([r["cross_motif"] for r in rows])
    c = float(np.corrcoef(al, wl)[0, 1])
    c_cross = float(np.corrcoef(al, xm)[0, 1])
    cov = np.cov(np.vstack([al, wl]), aweights=w)
    cw = float(cov[0, 1] / max(np.sqrt(cov[0, 0] * cov[1, 1]), 1e-12))

    print(f"\n{len(rows)} relation buckets with >= {a.min_n} observations")
    cov2 = np.cov(np.vstack([al, xm]), aweights=w)
    cw2 = float(cov2[0, 1] / max(np.sqrt(cov2[0, 0] * cov2[1, 1]), 1e-12))
    print(f"corr(alpha, both wall-backed)  = {c:.3f}  (weighted {cw:.3f})")
    print(f"corr(alpha, spans two motifs)  = {c_cross:.3f}  (weighted {cw2:.3f})\n")
    rows.sort(key=lambda r: -r["alpha"])
    print(f"{'bucket':50s}{'alpha':>7s}{'wall':>7s}{'cross-motif':>13s}{'n':>7s}")
    for r in rows[:8] + [None] + rows[-8:]:
        if r is None:
            print("   ...")
            continue
        print(f"{r['bucket']:50s}{r['alpha']:7.3f}{r['both_wall_backed']:7.2f}"
              f"{r['cross_motif']:13.2f}{r['n_obs']:7d}")
    with open(a.out, "w") as fh:
        json.dump({"correlation": c, "correlation_weighted": cw,
                   "correlation_cross_motif": c_cross,
                   "correlation_cross_motif_weighted": cw2,
                   "n_buckets": len(rows), "rows": rows}, fh, indent=1)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
