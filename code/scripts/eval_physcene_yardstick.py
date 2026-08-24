#!/usr/bin/env python
"""ReRoom read on PhyScene's yardstick (plan bibliography [11]).

Not a head-to-head -- see `reroom/eval/physcene.py` for why -- but the closest
thing available without their outputs: the same metric definitions, computed on
ReRoom's scenes, reported next to their published table so a reader can see the
order of magnitude rather than guess it.

Two target settings are reported, because they are not equally hard:

``as-is``      the reference room's own floor plan, which is the setting
               PhyScene evaluates in;
``retargeted`` a deformed target polygon, which is this project's actual task
               and strictly harder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.physcene import physcene_metrics
from reroom.geom.deform import deform_room
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget

# PhyScene, CVPR 2024, Table 3 (floor-plan-conditioned synthesis on 3D-FRONT).
# Reproduced here only so the numbers below can be read against something.
PUBLISHED = {
    "bedroom": {
        "ATISS": dict(Col_obj=0.248, Col_scene=0.46, R_out=0.286,
                      R_walkable=0.839, R_reach=0.736),
        "DiffuScene": dict(Col_obj=0.228, Col_scene=0.43, R_out=0.272,
                           R_walkable=0.827, R_reach=0.755),
        "PhyScene": dict(Col_obj=0.187, Col_scene=0.36, R_out=0.245,
                         R_walkable=0.865, R_reach=0.762)},
    "living_room": {
        "ATISS": dict(Col_obj=0.316, Col_scene=0.85, R_out=0.136,
                      R_walkable=0.814, R_reach=0.791),
        "DiffuScene": dict(Col_obj=0.198, Col_scene=0.69, R_out=0.238,
                           R_walkable=0.790, R_reach=0.756),
        "PhyScene": dict(Col_obj=0.191, Col_scene=0.63, R_out=0.219,
                         R_walkable=0.815, R_reach=0.771)},
    "dining_room": {
        "ATISS": dict(Col_obj=0.591, Col_scene=0.96, R_out=0.132,
                      R_walkable=0.874, R_reach=0.848),
        "DiffuScene": dict(Col_obj=0.160, Col_scene=0.55, R_out=0.244,
                           R_walkable=0.787, R_reach=0.847),
        "PhyScene": dict(Col_obj=0.151, Col_scene=0.53, R_out=0.217,
                         R_walkable=0.852, R_reach=0.789)},
}

METHODS = ["reroom_full", "target_only", "direct_scaling"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scenes", type=int, default=150)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--out", default="outputs/physcene_yardstick.json")
    a = ap.parse_args()

    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    scenes = [s for s in iter_scenes(a.corpus, limit=None, min_objects=4)
              if s.room.room_type in PUBLISHED]
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]
    print(f"{len(test)} held-out rooms over {len(PUBLISHED)} room types")

    rows = []
    for k, s in enumerate(test):
        try:
            g = build_motifs(build_scene_graph(s))
        except Exception:
            continue
        targets = {"as-is": s.room.copy(),
                   "retargeted": deform_room(s.room, a.level,
                                             np.random.default_rng(k)).room}
        # the reference design in its own room: the ceiling any method inherits,
        # because 3D-FRONT's own rooms are not collision-free either
        m = physcene_metrics(s)
        m.update({"scene": s.scene_id, "method": "3D-FRONT reference",
                  "target": "as-is", "room_type": s.room.room_type})
        rows.append(m)
        for tname, room in targets.items():
            for name in METHODS:
                try:
                    if name == "reroom_full":
                        out = retarget(g, room, bank=bank,
                                       cfg=RetargetConfig(restarts=16)).scene
                    else:
                        out = run_baseline(name, g, room, cfg=RetargetConfig())
                    m = physcene_metrics(out)
                except Exception as exc:
                    print(f"  {s.scene_id} {name}: {type(exc).__name__}: {exc}")
                    continue
                m.update({"scene": s.scene_id, "method": name,
                          "target": tname, "room_type": s.room.room_type})
                rows.append(m)
        if k % 20 == 0:
            print(f"  {k}/{len(test)}  rows={len(rows)}", flush=True)

    with open(a.out, "w") as fh:
        json.dump(rows, fh)

    keys = ["ps_Col_obj", "ps_Col_scene", "ps_R_out", "ps_R_walkable",
            "ps_R_reach"]
    lab = ["Col_obj↓", "Col_scene↓", "R_out↓", "R_walk↑", "R_reach↑"]

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return None
        return {k: float(np.nanmean([r[k] for r in sub])) for k in keys} | {
            "n": len(sub)}

    for rt in PUBLISHED:
        print(f"\n=== {rt} ===")
        print(f"{'method':34s}" + "".join(f"{x:>12s}" for x in lab) + f"{'n':>7s}")
        for pname, v in PUBLISHED[rt].items():
            print(f"{pname + ' (published)':34s}"
                  + "".join(f"{v[k[3:]]:12.3f}" for k in keys) + f"{'—':>7s}")
        for tname in ("as-is", "retargeted"):
            for name in ["3D-FRONT reference"] + METHODS:
                if name == "3D-FRONT reference" and tname != "as-is":
                    continue
                d = agg(lambda r, n=name, t=tname, x=rt:
                        r["method"] == n and r["target"] == t
                        and r["room_type"] == x)
                if not d:
                    continue
                print(f"{name + ' [' + tname + ']':34s}"
                      + "".join(f"{d[k]:12.3f}" for k in keys)
                      + f"{d['n']:7d}")
    print("\n->", a.out)
    print("These are ReRoom's scenes under PhyScene's metric definitions, not a "
          "head-to-head: PhyScene generates its object set, ReRoom transfers "
          "one; the splits and the floor plans differ.")


if __name__ == "__main__":
    main()
