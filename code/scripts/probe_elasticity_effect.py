#!/usr/bin/env python
"""Does relation elasticity actually change the output, and where?

The ablation grid finds almost no aggregate difference between alpha = 0 and a
fitted alpha.  That is a real result, but it is measured over deformations that
mostly keep gamma inside [0.7, 1.4], where eq. (9) can move a distance by at
most 40 % even at alpha = 1.  This probes the regime where the term can
actually bite: strong uniform rescalings, and only the relations whose fitted
alpha is far from the mean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from experiments.common import _ZeroElasticity, load_corpus
from reroom.core.scene import Room
from reroom.data.corpus import split_scenes
from reroom.eval.metrics import aggregate, evaluate
from reroom.geom.deform import uniform_scale
from reroom.intent.elasticity import PriorElasticity, load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph, relation_features
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.target import build_design_intent


def rigid_vs_elastic_error(graph, target_scene, intent):
    """Split relation error by how elastic the relation is."""
    tmap = {o.oid: o for o in target_scene.objects if o.keep}
    src = graph.scene.objects
    from reroom.intent.relations import relation_distance
    out = {"rigid": [], "elastic": []}
    for r in intent.relations:
        ta, tb = tmap.get(src[r.i].oid), tmap.get(src[r.j].oid)
        if ta is None or tb is None:
            continue
        d = relation_distance(r.phi_des, relation_features(ta, tb))
        out["rigid" if r.alpha < 0.25 else "elastic"].append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--scales", default="0.6,0.75,1.4,1.8")
    ap.add_argument("--out", default="outputs/elasticity_effect.json")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    scenes = load_corpus(a.corpus, room_types=("bedroom", "living_room",
                                               "dining_room"),
                         min_objects=6, max_objects=14)
    _, _, test = split_scenes(scenes)
    test = test[:a.scenes]
    models = {"alpha=0 (rigid)": _ZeroElasticity(),
              "prior alpha": PriorElasticity(),
              "fitted f_psi": load_elasticity(a.elasticity)}
    scales = [float(x) for x in a.scales.split(",")]
    rows = defaultdict(list)
    split = defaultdict(lambda: {"rigid": [], "elastic": []})

    for si, s in enumerate(test):
        g = build_motifs(build_scene_graph(s))
        for sc in scales:
            room = Room(polygon=uniform_scale(s.room.polygon, sc),
                        height=s.room.height,
                        openings=[o.copy() for o in s.room.openings],
                        room_type=s.room.room_type)
            for name, el in models.items():
                cfg = RetargetConfig(restarts=16, device="cpu", seed=si)
                r = retarget(g, room, elasticity=el, cfg=cfg)
                m = evaluate(g, r.scene)
                m["scale"] = sc
                rows[name].append(m)
                # judge every method against the *same* yardstick: the prior
                # elasticity model's desired relations
                ref_intent = build_design_intent(g, room, PriorElasticity())
                d = rigid_vs_elastic_error(g, r.scene, ref_intent)
                split[name]["rigid"] += d["rigid"]
                split[name]["elastic"] += d["elastic"]
        if si % 10 == 0:
            print(f"  {si}/{len(test)}", flush=True)

    out = {}
    print(f"\n{'model':18s}{'S_rel':>9s}{'S_rel_el':>10s}{'S_motif':>9s}"
          f"{'legality':>10s}{'score':>8s}{'rigid err':>11s}{'elastic err':>12s}")
    for name in models:
        agg = aggregate(rows[name])
        re_ = float(np.mean(split[name]["rigid"])) if split[name]["rigid"] else float("nan")
        ee = float(np.mean(split[name]["elastic"])) if split[name]["elastic"] else float("nan")
        out[name] = {"agg": {k: v for k, v in agg.items()},
                     "rigid_relation_error": re_, "elastic_relation_error": ee}
        print(f"{name:18s}{agg['S_rel']:9.4f}{agg['S_rel_elastic']:10.4f}"
              f"{agg['S_motif']:9.4f}{agg['legality']:10.4f}{agg['score']:8.4f}"
              f"{re_:11.4f}{ee:12.4f}")
    for sc in scales:
        print(f"\n  scale {sc}:")
        for name in models:
            sub = aggregate([r for r in rows[name] if r["scale"] == sc])
            print(f"    {name:18s} S_rel {sub['S_rel']:.4f}  "
                  f"S_rel_elastic {sub['S_rel_elastic']:.4f}  "
                  f"S_motif {sub['S_motif']:.4f}  score {sub['score']:.4f}")
            out.setdefault("per_scale", {}).setdefault(str(sc), {})[name] = {
                "S_rel": sub["S_rel"], "S_rel_elastic": sub["S_rel_elastic"],
                "S_motif": sub["S_motif"], "score": sub["score"]}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
