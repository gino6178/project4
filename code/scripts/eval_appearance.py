#!/usr/bin/env python
"""Is the global appearance score of section 15.2 worth anything?

The plan asserts that a whole-image CLIP similarity "cannot substitute for
relation and motif evaluation" -- it is dominated by colour and by the largest
object.  That is a testable claim, not a matter of taste, so this measures it:
for the same retargetings it computes

* ``S_rel`` / ``S_motif``      the design-preservation metrics (42), (43);
* ``appearance_object``        the object-matched CLIP score;
* ``appearance_global``        the whole-render CLIP score, section 15.2;

and reports how each appearance number *ranks* methods compared with how the
preservation metrics rank them.  If the global score is the blunt instrument
the plan says it is, it will fail to separate methods that S_rel separates
clearly, and will correlate mostly with how many objects survived.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.appearance import ClipEncoder
from reroom.eval.metrics import aggregate, evaluate
from reroom.geom.deform import deform_room
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget

METHODS = ["reference_rigid", "direct_scaling", "reroom_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/appearance.json")
    a = ap.parse_args()

    enc = ClipEncoder(device=a.device)
    if not enc.ok:
        raise SystemExit("no CLIP encoder available; section 15.2 needs one")
    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None

    scenes = list(iter_scenes(a.corpus, limit=None, min_objects=5))
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]
    rows = []
    for k, s in enumerate(test):
        try:
            g = build_motifs(build_scene_graph(s))
            room = deform_room(s.room, a.level, np.random.default_rng(k)).room
        except Exception:
            continue
        for name in METHODS:
            try:
                if name == "reroom_full":
                    out = retarget(g, room, bank=bank,
                                   cfg=RetargetConfig(restarts=16)).scene
                else:
                    out = run_baseline(name, g, room, cfg=RetargetConfig())
                m = evaluate(g, out, bank=bank, encoder=enc,
                             global_appearance=True)
            except Exception as exc:
                print(f"  {s.scene_id} {name}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            m.update({"scene": s.scene_id, "method": name,
                      "kept": sum(1 for o in out.objects if o.keep),
                      "n_source": len(s.objects)})
            rows.append(m)
        if k % 5 == 0:
            print(f"  {k}/{len(test)}  rows={len(rows)}", flush=True)

    with open(a.out, "w") as fh:
        json.dump(rows, fh)

    cols = ["S_rel", "S_motif", "legality", "appearance_object",
            "appearance_global", "appearance_pooled"]
    print(f"\n{'method':18s}" + "".join(f"{c:>19s}" for c in cols)
          + f"{'kept':>7s}")
    per = {}
    for name in METHODS:
        sub = [r for r in rows if r["method"] == name]
        ag = aggregate(sub)
        per[name] = ag
        print(f"{name:18s}" + "".join(f"{ag.get(c, float('nan')):19.4f}"
                                      for c in cols)
              + f"{ag.get('kept', float('nan')):7.1f}")

    # does the global score rank methods the way the preservation metrics do?
    print("\nSpread between the best and worst method, per metric "
          "(a metric that cannot separate them is not measuring the design):")
    for c in cols:
        vals = [per[m].get(c) for m in METHODS if per[m].get(c) is not None]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2:
            continue
        print(f"  {c:20s} {max(vals) - min(vals):.4f}")

    ok = [r for r in rows if np.isfinite(r.get("appearance_global", np.nan))]
    if len(ok) > 3:
        print("\nCorrelation with the whole-render score, over "
              f"{len(ok)} retargetings:")
        gv = np.array([r["appearance_global"] for r in ok])
        for c in ["S_rel", "S_motif", "legality", "appearance_object", "kept"]:
            v = np.array([float(r.get(c, np.nan)) for r in ok])
            good = np.isfinite(v) & np.isfinite(gv)
            if good.sum() > 3 and v[good].std() > 1e-9:
                print(f"  corr(appearance_global, {c:18s}) "
                      f"= {np.corrcoef(gv[good], v[good])[0, 1]:+.3f}")
    print("\n->", a.out)


if __name__ == "__main__":
    main()
