#!/usr/bin/env python
"""Why does coherence collapse when the room shrinks?  Probe the 0.75x case."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from reroom.core.scene import Room
from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import evaluate, preservation_metrics
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.geom.polygon import as_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.optimizer import RetargetConfig
from reroom.retarget.target import build_design_intent
from reroom.retarget.summarize import plan_summarization, area_budget
from reroom.generative.sample import generative_retarget, load_flow
from reroom.intent.elasticity import load_elasticity


def scaled_room(room, s):
    poly = uniform_scale(room.polygon, s)
    anchored = _anchor_openings(room)
    op = _replace_openings(poly, anchored, len(room.polygon))
    return Room(polygon=poly, height=room.height, openings=op, room_type=room.room_type)


bank = AssetBank.load("outputs/priors/assets_future.pkl")
flow = load_flow("outputs/flow_wall/flow.pt", device="cpu")
el = load_elasticity("outputs/elasticity/neural.pt")
cfg = RetargetConfig(restarts=24)

scenes = [s for s in iter_scenes("/home/gino/data/reroom/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
_, _, test = split_scenes(scenes)
src = test[6]
g = build_motifs(build_scene_graph(src))
ref_area = as_polygon(src.room).area
print(f"reference {ref_area:.1f} m^2, {sum(o.keep for o in src.objects)} objects")

for s in (0.75, 1.0, 1.35):
    room = scaled_room(src.room, s)
    area = as_polygon(room).area
    intent = build_design_intent(g, room, elasticity=el)
    demanded = float(sum(o.footprint_area for o in g.scene.objects if o.keep))
    budget = area_budget(intent, room, slack=0.0)
    sm = plan_summarization(intent, room, allow_drop=cfg.allow_removal)
    n_keep = int(np.sum(sm.keep))
    out = generative_retarget(flow, g, room, elasticity=el, bank=bank, cfg=cfg, k=16).scene
    pm = preservation_metrics(g, out)
    m = evaluate(g, out)
    n_final = sum(o.keep for o in out.objects)
    print(f"\n=== {s:.2f}x  area={area:.1f} m^2  demanded_fp={demanded:.1f}  budget={budget:.1f} "
          f"(demand/budget={demanded/budget:.2f}) ===")
    print(f"  summarize kept {n_keep}/{len(g.scene.objects)}   final objects {n_final}")
    print(f"  S_rel(all)={pm['S_rel']:.2f}  S_rel_kept={pm.get('S_rel_kept', float('nan')):.2f}  "
          f"S_motif={pm.get('S_motif', float('nan')):.2f}  col={100*m['R_col']:.0f}%")
    if sm.log:
        print(f"  summarize log: {sm.log}")
