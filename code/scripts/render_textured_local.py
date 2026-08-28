#!/usr/bin/env python
"""LOCAL textured render (PhyScene-style) of the flow-retargeted scenes, using
the local 3D-FUTURE meshes + pyrender/EGL. Run in the `reroom` conda env.

  ~/miniconda3/envs/reroom/bin/python scripts/render_textured_local.py
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from reroom.core.scene import Scene
from reroom.render.textured import load_room_assets, repose_assets, render_scene_textured

FRONT = "/home/gino/data/reroom/3D-FRONT_raw/3D-FRONT"
FUTURE = [f"/home/gino/data/reroom/3D-FUTURE/3D-FUTURE-model-part{i}" for i in (1, 2, 3, 4)]
DUMP = "/home/gino/project/project4/outputs/scenes_dump"
OUT = "/home/gino/project/project4/outputs/fig_tex"
os.makedirs(OUT, exist_ok=True)

ref = Scene.load(f"{DUMP}/ref.json")
house = ref.meta.get("house"); room_id = ref.scene_id.split("__", 1)[1]
print("house", house, "room", room_id, flush=True)
assets = load_room_assets(f"{FRONT}/{house}.json", room_id, FUTURE)
if assets is None:
    raise SystemExit("load_room_assets returned None — check room_instanceid")
print("assets meshes:", len(assets.objects), flush=True)

panels = [("reference", ref, ref)]
for s in (0.75, 1.0, 1.35):
    t = Scene.load(f"{DUMP}/t_{s}.json")
    panels.append((f"target {s}×", ref, t))

imgs = []
for title, srcsc, tgtsc in panels:
    a = repose_assets(assets, srcsc, tgtsc)
    cam, rr = render_scene_textured(a, tgtsc.room, width=900, height=680, elevated=True)
    p = f"{OUT}/tex_{title.split()[0]}_{title.split()[-1]}.png".replace("×", "x").replace(" ", "")
    imageio.imwrite(p, rr.rgb)
    imgs.append((title, p))
    print("rendered", title, rr.rgb.shape, "meshes", len(a.objects), flush=True)

fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
for ax, (title, p) in zip(axes, imgs):
    ax.imshow(imageio.imread(p)); ax.set_axis_off()
    ax.set_title(title, fontsize=12)
fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.02)
fig.savefig(f"{OUT}/textured_sheet.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote textured_sheet.png")
print("DONE_TEX")
