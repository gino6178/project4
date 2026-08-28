#!/usr/bin/env python
"""Evaluate flow_gw (GW-relational-loss fine-tune) vs flow_bfresh on INDEPENDENT
metrics.  The GW loss trains pairwise relational structure; the honest,
non-circular test is S_motif (motif-group preservation, a different metric) and
the real cross-pairing OOD set.  We also report S_rel for context."""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, aspect_deform, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.data.asset_bank import AssetBank
from scipy import stats
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--aniso",action="store_true"); a=ap.parse_args()

def sr(r,s):
    p=aspect_deform(r.polygon,float(s[0]),float(s[1])) if isinstance(s,(tuple,list)) else uniform_scale(r.polygon,s)
    an=_anchor_openings(r); return Room(polygon=p,height=r.height,openings=_replace_openings(p,an,len(r.polygon)),room_type=r.room_type)
el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
sizes=[(1.5,0.75),(0.75,1.5),(1.7,0.85),(0.85,1.7)] if a.aniso else [0.75,1.0,1.35]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
fg=load_flow("outputs/flow_gw/flow.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
KEYS=("S_rel","S_motif","S_rel_kept","R_col")
rows={n:{k:[] for k in KEYS} for n in ("base","gw")}
nsz=len(sizes)
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=sr(src.room,s)
      for nm,fm in (("base",fb),("gw",fg)):
        sc=generative_retarget(fm,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
        m=evaluate(g,sc)
        for k in KEYS: rows[nm][k].append(m.get(k, m["S_rel"] if k=="S_rel_kept" else np.nan))
        rows[nm]["R_col"][-1]=100*rows[nm]["R_col"][-1]
  except Exception as e: print("skip",sd,repr(e),flush=True)
def ms(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean(),x.std()
n=len(rows["base"]["S_rel"]); print(f"\n{'ANISO' if a.aniso else 'UNIFORM'}  N={n} ({n//nsz} refs x {nsz})")
print(f"{'model':<24}{'S_rel':>11}{'S_motif*':>11}{'S_rel_kept':>12}{'R_col%':>9}")
for nm,l in (("base","flow_bfresh"),("gw","flow_gw (GW loss)")):
    r=rows[nm]; print(f"{l:<24}{ms(r['S_rel'])[0]:8.3f}{ms(r['S_motif'])[0]:11.3f}{ms(r['S_rel_kept'])[0]:12.3f}{ms(r['R_col'])[0]:9.2f}")
print("\n* S_motif = the INDEPENDENT metric (GW loss trains pairwise S_rel-like structure)")
def paired(k):
    aa=np.array(rows["gw"][k],float); bb=np.array(rows["base"][k],float)
    msk=~(np.isnan(aa)|np.isnan(bb)); aa,bb=aa[msk],bb[msk]; d=aa-bb
    dr=d[:len(d)//nsz*nsz].reshape(-1,nsz).mean(1)
    try:_,p=stats.wilcoxon(d)
    except:p=float('nan')
    try:_,pc=stats.wilcoxon(dr)
    except:pc=float('nan')
    print(f"  {k:12s}: Δ={d.mean():+.3f} percell p={p:.3g} | clustered(n={len(dr)}) Δ={dr.mean():+.3f} p={pc:.3g}")
print("Paired (flow_gw − bfresh):")
for k in KEYS: paired(k)
print("DONE_GW_EVAL")
