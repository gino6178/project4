import os, json
os.environ.setdefault("PYOPENGL_PLATFORM","egl"); os.environ.pop("DISPLAY",None)
import numpy as np, imageio.v2 as imageio
from reroom.core.scene import Scene, scene_from_dict
from reroom.render.textured import load_room_assets, repose_assets, render_scene_textured
FRONT="/home/gino/data/reroom/3D-FRONT_raw/3D-FRONT"
FUTURE=[f"/home/gino/data/reroom/3D-FUTURE/3D-FUTURE-model-part{i}" for i in (1,2,3,4)]
DUMP="/home/gino/project/project4/outputs/scenes_dump"
ref=Scene.load(f"{DUMP}/ref.json")
house=ref.meta.get("house"); room_id=ref.scene_id.split("__",1)[1]
assets=load_room_assets(f"{FRONT}/{house}.json",room_id,FUTURE)
print("assets",len(assets.objects),flush=True)
data=json.load(open(f"{DUMP}/trajectory.json"))["frames"]
PHASE_COL={0:(168,65,42),1:(168,65,42),2:(42,122,58)}
PHASE_TXT={0:"① prior",1:"① rectified flow",2:"② projection Πθ"}
frames=[]
for k,fd in enumerate(data):
    ph=fd.get("_phase",1)
    sc=scene_from_dict({kk:vv for kk,vv in fd.items() if not kk.startswith("_")})
    a=repose_assets(assets, ref, sc)
    cam,rr=render_scene_textured(a, sc.room, width=720, height=560, elevated=True)
    frames.append(rr.rgb)
    if k%5==0: print("frame",k,fd.get("_label"),flush=True)
imageio.mimsave(f"{DUMP}/flow_traj_tex.gif", frames, duration=0.13, loop=0)
print("wrote flow_traj_tex.gif frames",len(frames))
print("DONE_TRAJ_TEX")
