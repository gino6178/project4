#!/usr/bin/env python
"""D1 eval: learned joint-flow pruning vs greedy Summarise, on SMALL rooms.

Same reference scenes shrunk hard (0.5/0.6/0.75x).  Compare:
  * shipped  = flow_bfresh + greedy plan_summarization (hand priority table)
  * D1       = flow_d1mask, existence head decides drops jointly with pose
On drop-count, R_col, R_OOB, S_rel, and PhyScene walkability."""
import os, sys, argparse
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
import numpy as np
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.eval.physcene import physcene_metrics
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.data.asset_bank import AssetBank

def scaled_room(room, s):
    poly = uniform_scale(room.polygon, s); a=_anchor_openings(room)
    return Room(polygon=poly, height=room.height,
                openings=_replace_openings(poly,a,len(room.polygon)), room_type=room.room_type)

ap=argparse.ArgumentParser()
ap.add_argument("--d1", default="outputs/flow_d1mask/flow_best.pt")
ap.add_argument("--base", default="outputs/flow_bfresh/flow_best.pt")
ap.add_argument("--seeds", default="6,8,25,2,10,14")
ap.add_argument("--sizes", default="0.5,0.6,0.75")
a=ap.parse_args()

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6)
        if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[int(x) for x in a.seeds.split(",")]; sizes=[float(x) for x in a.sizes.split(",")]

flow_base=load_flow(a.base, device="cuda:0")
flow_d1=load_flow(a.d1, device="cuda:0")
print("d1 mask_flow:", getattr(flow_d1,"_mask_flow",False))

def run(flow, g, room, learned):
    cfg=RetargetConfig(restarts=16, regularity_snap=True, device="cuda:0")
    res=generative_retarget(flow,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False)
    sc=res.scene; m=evaluate(g,sc); ps=physcene_metrics(sc)
    ndrop=sum(1 for o in g.scene.objects if o.keep)-sum(1 for o in sc.objects if o.keep)
    return dict(ndrop=ndrop, R_col=100*m["R_col"], R_OOB=100*m["R_OOB"],
                S_rel=m["S_rel"], walk=ps["ps_R_walkable"],
                info=res.info.get("learned_drop"))

agg={"shipped":[], "d1":[]}
for sd in seeds:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
        room=scaled_room(src.room,s)
        agg["shipped"].append(run(flow_base,g,room,False))
        agg["d1"].append(run(flow_d1,g,room,True))

def mean(rows,k): return np.nanmean([r[k] for r in rows])
print(f"\n{'method':<10}{'drop':>6}{'R_col%':>8}{'R_OOB%':>8}{'S_rel':>8}{'walk':>7}")
for k in ("shipped","d1"):
    r=agg[k]
    print(f"{k:<10}{mean(r,'ndrop'):>6.2f}{mean(r,'R_col'):>8.2f}{mean(r,'R_OOB'):>8.2f}{mean(r,'S_rel'):>8.3f}{mean(r,'walk'):>7.2f}")
print("\nDONE_EVAL_D1")
