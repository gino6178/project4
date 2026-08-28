#!/usr/bin/env python
"""OFFLINE PROBE for the GW direction: under capacity-forced pruning, does a
relational-optimal (Gromov-Wasserstein-style) SELECTION of which objects to keep
preserve more S_rel than (a) the flow's learned selection and (b) size/importance
heuristics -- all judged at the SAME affine placement to isolate SELECTION?

If greedy-relational >> flow selection, the unbalanced-GW-for-pruning mechanism
has real headroom and is worth building/training.  Fast, no training.
"""
import os, sys, copy, numpy as np
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
from reroom.data.asset_bank import AssetBank

def small_room(r, s):
    p=uniform_scale(r.polygon,s); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)

def keep_only(scene, oids):
    sc=scene.copy()
    for o in sc.objects: o.keep = (o.oid in oids)
    return sc

def incident_weight(g, present):
    """weight of each object = sum of reference-relation weights incident to it
    with the other endpoint still present (the relational 'mass' it carries)."""
    ref=g.scene; w={o.oid:0.0 for o in ref.objects}
    for r in g.relations:
        a=ref.objects[r.i].oid; b=ref.objects[r.j].oid
        if a in present and b in present:
            w[a]+=r.weight; w[b]+=r.weight
    return w

def greedy_relational(g, all_oids, K):
    """greedy removal: repeatedly drop the object carrying the LEAST incident
    relational weight until K remain (keeps the relationally-central objects) --
    a submodular proxy for where unbalanced GW concentrates transported mass."""
    present=set(all_oids)
    while len(present)>K:
        w=incident_weight(g, present)
        drop=min(present, key=lambda o: w[o])
        present.discard(drop)
    return present

def topk_size(scene, oids, K):
    objs=[o for o in scene.objects if o.oid in oids]
    objs.sort(key=lambda o: -float(o.size[0])*float(o.size[1]))
    return set(o.oid for o in objs[:K])

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
res={k:[] for k in ("flow_full","flow_sel","size_sel","greedy_sel","full_noprune")}
Ks=[]
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    room=small_room(src.room, 0.65)          # force ~pruning
    allo=[o.oid for o in src.objects]
    # flow: its own selection + placement
    fs=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
    kept=set(o.oid for o in fs.objects if o.keep); K=max(len(kept),2); Ks.append(K)
    res["flow_full"].append(evaluate(g,fs)["S_rel"])
    aff=warp_scene(src, room)                # fixed affine placement for all
    res["full_noprune"].append(evaluate(g, keep_only(aff, set(allo)))["S_rel"])
    res["flow_sel"].append(evaluate(g, keep_only(aff, kept))["S_rel"])
    res["size_sel"].append(evaluate(g, keep_only(aff, topk_size(aff, allo, K)))["S_rel"])
    res["greedy_sel"].append(evaluate(g, keep_only(aff, greedy_relational(g, allo, K)))["S_rel"])
  except Exception as e:
    print("skip",sd,repr(e),flush=True)
import numpy as np
from scipy import stats
print(f"\nN={len(res['flow_sel'])} scenes, mean kept K={np.mean(Ks):.1f} of ~{np.mean([len(test[s].objects) for s in seeds]):.1f} objects (forced prune @0.65x)")
def m(k): a=np.array(res[k]); return a.mean(),a.std()
print(f"{'selection (affine placement)':<34}{'S_rel':>12}")
for k,l in [("full_noprune","(no pruning, ceiling)"),("size_sel","size top-K"),("flow_sel","FLOW selection"),("greedy_sel","GREEDY-RELATIONAL (GW-style)")]:
    mm,ss=m(k); print(f"{l:<34}{mm:6.3f}±{ss:<5.3f}")
print(f"\n{'(context) flow full pipeline S_rel':<34}{m('flow_full')[0]:6.3f}")
d=np.array(res["greedy_sel"])-np.array(res["flow_sel"])
try:_,p=stats.wilcoxon(d)
except:p=float('nan')
print(f"\nGREEDY-REL − FLOW selection (same placement): Δ={d.mean():+.3f} (win {100*(d>0).mean():.0f}%), Wilcoxon p={p:.3g}")
print("DONE_PROBE")
