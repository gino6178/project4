#!/usr/bin/env python
"""A / Prop 3 decisive eval: does TRAIN-THROUGH the differentiable projection
give Pi_theta a measurable, non-confounded benefit (reviewer #1)?

Compares the shipped flow (flow_bfresh) vs the train-through model (flow_proj,
final epoch), BOTH decoded to world and BOTH passed through the SAME test-time
Pi_theta (project_scene, 40 steps).  Per ref x size we measure:
  * raw R_col / S_rel / wall-snap of each model's endpoint;
  * projection DISPLACEMENT = mean object move (metres) the deployed Pi_theta
    induces  -- the headline: a feasible-by-construction flow should need the
    projection to move it far LESS;
  * R_col / S_rel AFTER Pi_theta.
Reports means + paired (per-cell and reference-clustered) Wilcoxon.
"""
import os, sys, math, argparse, copy
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
import numpy as np
from scipy import stats
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, aspect_deform, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.regularity import WALL_CATS
from reroom.retarget.diffproj import project_scene
from reroom.geom.polygon import as_polygon
from reroom.data.asset_bank import AssetBank

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="outputs/flow_bfresh/flow_best.pt")
ap.add_argument("--proj", default="outputs/flow_proj/flow.pt")   # FINAL, full train-through
ap.add_argument("--seeds", default="6,8,25,2,10,14,1,3,5,7,9,11")
ap.add_argument("--aniso", action="store_true")
a = ap.parse_args()

def scaled_room(r, s):
    p = aspect_deform(r.polygon, float(s[0]), float(s[1])) if isinstance(s,(tuple,list)) else uniform_scale(r.polygon, s)
    an=_anchor_openings(r); return Room(polygon=p,height=r.height,openings=_replace_openings(p,an,len(r.polygon)),room_type=r.room_type)

def snap_pct(scene):
    poly=as_polygon(scene.room); ring=np.asarray(poly.exterior.coords)[:-1]
    edges=[(ring[i],ring[(i+1)%len(ring)]) for i in range(len(ring))]; ok=tot=0
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        tot+=1; cs=o.corners(); best=None
        for u,v in edges:
            d=v-u; L=np.linalg.norm(d)+1e-9; t=d/L; nrm=np.array([-t[1],t[0]])
            rel=cs-u; perp=np.abs(rel@nrm); pj=rel@t; ins=(pj>-0.05)&(pj<L+0.05)
            if not ins.any(): continue
            gap=perp[ins].min()
            if best is None or gap<best: best=gap
        if best is not None and best<=0.12: ok+=1
    return 100*ok/max(tot,1)

def disp(a_sc,b_sc):
    d=[np.linalg.norm(np.array(x.xy)-np.array(y.xy)) for x,y in zip(a_sc.objects,b_sc.objects) if x.keep]
    return float(np.mean(d)) if d else 0.0

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[int(x) for x in a.seeds.split(",")]
sizes=[(1.5,0.75),(0.75,1.5),(1.7,0.85),(0.85,1.7)] if a.aniso else [0.75,1.0,1.35]
print(("ANISO" if a.aniso else "UNIFORM"),"sizes",sizes,flush=True)
fb=load_flow(a.base,device="cuda:0"); fp=load_flow(a.proj,device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
KEYS=("R_col_raw","S_rel_raw","snap_raw","proj_disp","R_col_proj","S_rel_proj")
rows={n:{k:[] for k in KEYS} for n in ("base","proj")}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=scaled_room(src.room,s)
      for name,fm in (("base",fb),("proj",fp)):
        raw=generative_retarget(fm,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
        pj=project_scene(raw,room,iters=40,lr=0.03,device="cuda:0")
        mr=evaluate(g,raw); mp=evaluate(g,pj)
        rows[name]["R_col_raw"].append(100*mr["R_col"]); rows[name]["S_rel_raw"].append(mr["S_rel"])
        rows[name]["snap_raw"].append(snap_pct(raw)); rows[name]["proj_disp"].append(disp(raw,pj))
        rows[name]["R_col_proj"].append(100*mp["R_col"]); rows[name]["S_rel_proj"].append(mp["S_rel"])
  except Exception as e: print("skip",sd,e,flush=True)

def ms(x): x=np.array(x); x=x[~np.isnan(x)]; return x.mean(),x.std()
n=len(rows["base"]["proj_disp"]); nsz=len(sizes)
print(f"\nN cells = {n}")
print(f"{'model':<26}{'Rcol_raw':>10}{'Srel_raw':>10}{'snap_raw':>10}{'projDISP(m)':>13}{'Rcol_proj':>11}{'Srel_proj':>11}")
for nm,lab in (("base","flow_bfresh (shipped)"),("proj","flow_proj (train-through)")):
    r=rows[nm]; f=lambda k:ms(r[k])
    print(f"{lab:<26}{f('R_col_raw')[0]:9.2f}{f('S_rel_raw')[0]:10.3f}{f('snap_raw')[0]:10.0f}{f('proj_disp')[0]:13.3f}{f('R_col_proj')[0]:11.2f}{f('S_rel_proj')[0]:11.3f}")
def paired(kk,lbl,invert=False):
    aA=np.array(rows["proj"][kk]); bB=np.array(rows["base"][kk]); d=aA-bB
    dr=d.reshape(-1,nsz).mean(1); dr=dr[~np.isnan(dr)]
    try: _,p=stats.wilcoxon(d)
    except: p=float('nan')
    try: _,pc=stats.wilcoxon(dr)
    except: pc=float('nan')
    print(f"  {lbl}: proj−base Δ={d.mean():+.3f}  per-cell p={p:.3g} | clustered(n={len(dr)}) Δ={dr.mean():+.3f} p={pc:.3g}")
print("\nPaired (train-through − shipped):")
paired("proj_disp","projection DISPLACEMENT (m, want −, lower=Pi_theta barely moves it)")
paired("R_col_raw","raw R_col% (want −)")
paired("S_rel_raw","raw S_rel (want ≥0)")
paired("R_col_proj","post-Pi_theta R_col% (want −)")
paired("S_rel_proj","post-Pi_theta S_rel (want ≥0)")
import json
json.dump({n_:{k:[float(x) for x in v] for k,v in d_.items()} for n_,d_ in rows.items()},
          open(("outputs/eval_proj_through_aniso.json" if a.aniso else "outputs/eval_proj_through.json"),"w"))
print("DONE_PROJ_THROUGH")
