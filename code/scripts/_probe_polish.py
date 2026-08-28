import os, sys, numpy as np
import os,sys as _s
_s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from reroom.intent.elasticity import load_elasticity

WALL_CATS = ("bookcase","wardrobe","tv_stand","tv","sofa","sofa_bed","bed",
             "double_bed","single_bed","desk","dressing_table","cabinet",
             "shelf","sideboard","console")
def sr(room,s):
    p=uniform_scale(room.polygon,s);a=_anchor_openings(room)
    return Room(polygon=p,height=room.height,openings=_replace_openings(p,a,len(room.polygon)),room_type=room.room_type)
def gaps(sc):
    poly=as_polygon(sc.room);ring=np.asarray(poly.exterior.coords)[:-1]
    edges=[(ring[i],ring[(i+1)%len(ring)]) for i in range(len(ring))]
    out=[]
    for o in sc.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        cor=o.corners();best=None
        for a,b in edges:
            d=b-a;L=np.linalg.norm(d)+1e-9;t=d/L;n=np.array([-t[1],t[0]])
            rel=cor-a;perp=np.abs(rel@n);proj=rel@t;ins=(proj>-0.05)&(proj<L+0.05)
            if not ins.any(): continue
            g=perp[ins].min()
            if best is None or g<best: best=g
        if best is not None: out.append(best)
    return np.array(out) if out else np.array([0.0])

bank=AssetBank.load("/home/gino/project/project4/outputs/priors/assets_future.pkl")
flow=load_flow("/home/gino/project/project4/outputs/flow_wall/flow.pt")
el=load_elasticity("/home/gino/project/project4/outputs/elasticity/neural.pt")
scenes=[s for s in iter_scenes("/home/gino/data/reroom/processed",limit=None,min_objects=6) if s.room.room_type in("bedroom","living_room")]
_,_,test=split_scenes(scenes)
rng=np.random.default_rng(0);picks=rng.choice(len(test),size=6,replace=False)
subset=[test[int(i)] for i in picks]

print(f"{'mode':>16} {'sz':>4} {'mean':>6} {'max':>6} {'OOB':>5} {'col':>5} {'Srel':>5}")
for polish_on, anchor in [(False, 0.0), (True, 6.0), (True, 30.0), (True, 100.0)]:
    cfg=RetargetConfig(restarts=24, polish_anchor=anchor)
    for sz in (0.75, 1.0, 1.35):
        allg=[]; oob=[]; cl=[]; sr_=[]
        for src in subset:
            g=build_motifs(build_scene_graph(src))
            room=sr(src.room, sz)
            try:
                out=generative_retarget(flow,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=polish_on).scene
                gs=gaps(out); m=evaluate(g,out); pm=preservation_metrics(g,out)
                allg.extend(gs.tolist()); oob.append(m['R_OOB']); cl.append(m['R_col']); sr_.append(pm['S_rel'])
            except Exception as e:
                print(f"  err: {e}")
        gs=np.array(allg)
        tag = f"nopolish" if not polish_on else f"polish{anchor:.0f}"
        print(f"{tag:>16} {sz:>4.2f} {100*gs.mean():>5.1f} {100*gs.max():>5.1f} {100*np.mean(oob):>4.1f} {100*np.mean(cl):>4.1f} {np.mean(sr_):>5.2f}")
