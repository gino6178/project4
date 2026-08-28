#!/usr/bin/env python
"""Probe 2: does the principled partial-GW selection (gwselect.gw_select) match
or beat the greedy-relational proxy and the flow's learned mask, at the same
affine placement, under forced pruning?"""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.generative.train import warp_scene
from reroom.retarget.gwselect import gw_select
from reroom.data.asset_bank import AssetBank
from scipy import stats

def small_room(r,s):
    p=uniform_scale(r.polygon,s); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)
def keep_only(scene,oids):
    sc=scene.copy()
    for o in sc.objects: o.keep=(o.oid in oids)
    return sc
def incident(g,present):
    ref=g.scene; w={o.oid:0.0 for o in ref.objects}
    for r in g.relations:
        a=ref.objects[r.i].oid; b=ref.objects[r.j].oid
        if a in present and b in present: w[a]+=r.weight; w[b]+=r.weight
    return w
def greedy(g,allo,K):
    present=set(allo)
    while len(present)>K:
        w=incident(g,present); present.discard(min(present,key=lambda o:w[o]))
    return present

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
res={k:[] for k in ("flow_sel","greedy_sel","gw_sel","full")}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    room=small_room(src.room,0.65); allo=[o.oid for o in src.objects]
    fs=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
    kept=set(o.oid for o in fs.objects if o.keep); K=max(len(kept),2)
    aff=warp_scene(src,room)
    posr=np.array([np.array(o.xy) for o in src.objects],float)
    tgt=np.array([np.array(o.xy) for o in aff.objects],float)
    incw=incident(g,set(allo)); relw=np.array([incw[o.oid] for o in src.objects],float)
    gw_idx=gw_select(posr,tgt,relw,K)
    gw_oids=set(src.objects[i].oid for i in gw_idx)
    res["full"].append(evaluate(g,keep_only(aff,set(allo)))["S_rel"])
    res["flow_sel"].append(evaluate(g,keep_only(aff,kept))["S_rel"])
    res["greedy_sel"].append(evaluate(g,keep_only(aff,greedy(g,allo,K)))["S_rel"])
    res["gw_sel"].append(evaluate(g,keep_only(aff,gw_oids))["S_rel"])
  except Exception as e: print("skip",sd,repr(e),flush=True)
def m(k): a=np.array(res[k]); return a.mean(),a.std()
print(f"\nN={len(res['gw_sel'])}  (affine placement, forced prune @0.65x)")
for k,l in [("full","no-prune ceiling"),("flow_sel","FLOW selection"),("greedy_sel","greedy-relational"),("gw_sel","PARTIAL-GW selection")]:
    mm,ss=m(k); print(f"  {l:<24}{mm:.3f}±{ss:.3f}")
for a,b,l in [("gw_sel","flow_sel","GW − flow"),("gw_sel","greedy_sel","GW − greedy")]:
    d=np.array(res[a])-np.array(res[b])
    try:_,p=stats.wilcoxon(d)
    except:p=float('nan')
    print(f"  {l}: Δ={d.mean():+.3f} win {100*(d>0).mean():.0f}% p={p:.3g}")
print("DONE_PROBE2")
