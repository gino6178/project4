#!/usr/bin/env python
"""Sweep wall_pull weight: for each room size report mean/max wall float,
OOB, collision and S_rel, so we can pick a weight that seats floaters without
overshooting objects out of the room."""
from __future__ import annotations
import os, sys, math
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
from reroom.generative.sample import generative_retarget, load_flow
from reroom.generative.guidance import GuidanceConfig
from reroom.intent.elasticity import load_elasticity

WALL_CATS = ("bookcase","wardrobe","tv_stand","tv","sofa","sofa_bed","bed",
             "double_bed","single_bed","desk","dressing_table","cabinet",
             "shelf","sideboard","console")

def sr(room,s):
    p=uniform_scale(room.polygon,s);a=_anchor_openings(room)
    return Room(polygon=p,height=room.height,openings=_replace_openings(p,a,len(room.polygon)),room_type=room.room_type)

def floats(scene):
    poly=as_polygon(scene.room); ring=np.asarray(poly.exterior.coords)[:-1]
    edges=[(ring[i],ring[(i+1)%len(ring)]) for i in range(len(ring))]
    gaps=[]
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        cor=o.corners(); best=None
        for a,b in edges:
            d=b-a; L=np.linalg.norm(d)+1e-9; t=d/L; n=np.array([-t[1],t[0]])
            rel=cor-a; perp=np.abs(rel@n); proj=rel@t
            ins=(proj>-0.05)&(proj<L+0.05)
            if not ins.any(): continue
            g=perp[ins].min()
            if best is None or g<best: best=g
        if best is not None: gaps.append(best)
    return np.array(gaps) if gaps else np.array([0.0])

bank=AssetBank.load("outputs/priors/assets_future.pkl");flow=load_flow("outputs/flow_wall/flow.pt")
el=load_elasticity("outputs/elasticity/neural.pt");cfg=RetargetConfig(restarts=24)
scenes=[s for s in iter_scenes("/home/gino/data/reroom/processed",limit=None,min_objects=6) if s.room.room_type in("bedroom","living_room")]
_,_,test=split_scenes(scenes);src=test[6];g=build_motifs(build_scene_graph(src))

print(f"{'w':>4} {'size':>5} {'meanflt':>7} {'maxflt':>6} {'OOB%':>5} {'col%':>5} {'Srel':>5}")
for w in (0.0, 1.5, 2.5, 3.5):
    for s in (0.75,1.35):
        gc=GuidanceConfig(wall_pull=w)
        out=generative_retarget(flow,g,sr(src.room,s),elasticity=el,bank=bank,cfg=cfg,k=16,guidance=gc).scene
        fl=floats(out); m=evaluate(g,out); pm=preservation_metrics(g,out)
        print(f"{w:>4.1f} {s:>5.2f} {100*fl.mean():>7.1f} {100*fl.max():>6.1f} "
              f"{100*m['R_OOB']:>5.1f} {100*m['R_col']:>5.1f} {pm['S_rel']:>5.2f}")
