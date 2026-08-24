#!/usr/bin/env python
"""What a user constraint costs (plan section 1: ``C_t``).

The plan's problem statement includes user constraints alongside the reference
and the target polygon, but says nothing about what honouring them is worth.
Two are implemented -- pinned objects and no-go floor -- and both are hard
constraints, so the interesting number is the price: how much legality and
design preservation does the solver give up to obey them, and does it obey
them at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from shapely.geometry import Polygon

from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import aggregate, evaluate
from reroom.geom.deform import deform_room
from reroom.geom.polygon import object_polygon
from reroom.intent.importance import object_importance
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.optimizer import RetargetConfig, retarget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--out", default="outputs/constraints.json")
    a = ap.parse_args()

    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    scenes = list(iter_scenes(a.corpus, limit=None, min_objects=5))
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]

    rows = {"free": [], "pinned": [], "keepout": []}
    obeyed = {"pin": 0, "pin_total": 0, "zone_free": [], "zone_kept": []}
    for k, s in enumerate(test):
        try:
            g = build_motifs(build_scene_graph(s))
            room = deform_room(s.room, a.level, np.random.default_rng(k)).room
        except Exception:
            continue
        cfg = lambda: RetargetConfig(restarts=16, seed=0)

        base = retarget(g, room, bank=bank, cfg=cfg()).scene
        rows["free"].append(evaluate(g, base))

        # (a) pin the most important object at its reference pose
        zeta = object_importance(g)
        pin_i = int(np.argmax(zeta))
        # `build_scene_graph` keeps a *reference* to the scene it was given,
        # so pinning through a second graph built from the same scene would
        # also pin it for the keep-out arm below.  Copy first.
        gp = build_motifs(build_scene_graph(s.copy()))
        tgt = gp.scene.objects[pin_i]
        tgt.locked = True
        pose = (tgt.xy.copy(), float(tgt.yaw))
        out = retarget(gp, room, bank=bank, cfg=cfg()).scene
        got = out.by_id(tgt.oid)
        obeyed["pin_total"] += 1
        if got is not None and got.keep and \
                float(np.linalg.norm(got.xy - pose[0])) < 1e-5 and \
                abs(got.yaw - pose[1]) < 1e-5:
            obeyed["pin"] += 1
        rows["pinned"].append(evaluate(g, out))

        # (b) forbid a quarter of the room
        b = room.bbox
        lo, mid = b[0], (b[0] + b[1]) / 2
        zone = np.array([[lo[0], lo[1]], [mid[0], lo[1]],
                         [mid[0], mid[1]], [lo[0], mid[1]]])
        rk = room.copy()
        rk.keepout = [zone]
        outk = retarget(g, rk, bank=bank, cfg=cfg()).scene
        rows["keepout"].append(evaluate(g, outk))
        z = Polygon(zone)
        inside = lambda sc: sum(object_polygon(o).intersection(z).area
                                for o in sc.objects if o.keep and o.z < 1.9)
        obeyed["zone_free"].append(inside(base))
        obeyed["zone_kept"].append(inside(outk))
        if k % 10 == 0:
            print(f"  {k}/{len(test)}", flush=True)

    out = {k: aggregate(v) for k, v in rows.items() if v}
    out["pin_respected"] = obeyed["pin"] / max(obeyed["pin_total"], 1)
    out["zone_area_free"] = float(np.mean(obeyed["zone_free"]))
    out["zone_area_constrained"] = float(np.mean(obeyed["zone_kept"]))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\n{'setting':12s}{'legality':>11s}{'S_rel':>9s}{'S_motif':>10s}"
          f"{'score':>9s}{'n':>6s}")
    for k in ("free", "pinned", "keepout"):
        if k in out:
            d = out[k]
            print(f"{k:12s}{d['legality']:11.4f}{d['S_rel']:9.4f}"
                  f"{d['S_motif']:10.4f}{d['score']:9.4f}{d['n']:6d}")
    print(f"\npinned object kept its exact pose in "
          f"{out['pin_respected']:.1%} of rooms")
    print(f"furniture inside the forbidden quarter: "
          f"{out['zone_area_free']:.2f} m^2 unconstrained -> "
          f"{out['zone_area_constrained']:.2f} m^2 with the zone")
    print("\n->", a.out)


if __name__ == "__main__":
    main()
