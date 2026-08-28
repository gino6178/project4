#!/usr/bin/env python
"""Produce paper figures from the REAL model:
  (A) 3d_sheet.png     — matplotlib 3D box-render (GL-free) of the flow result
                          at 0.75x / 1.0x / 1.35x (no 3D-FUTURE meshes needed).
  (B) flow_traj.gif    — the ACTUAL 50-step ODE trajectory (informative prior ->
                          rectified flow) then the differentiable projection snap.
"""
import os, sys, math
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
os.makedirs("outputs/fig", exist_ok=True)
import numpy as np, torch, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.core.scene import Room, Scene
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.target import build_design_intent
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.generative.tokens import build_tokens, collate, from_frame
from reroom.generative.model import to_relative, to_world
from reroom.render.topdown import draw_scene
from reroom.render.scene3d import render_scene3d
from reroom.retarget.diffproj import project_scene
from reroom.data.asset_bank import AssetBank

DEV = "cuda:0"
def scaled_room(r, s):
    p = uniform_scale(r.polygon, s); a = _anchor_openings(r)
    return Room(polygon=p, height=r.height, openings=_replace_openings(p, a, len(r.polygon)), room_type=r.room_type)

flow = load_flow("outputs/flow_bfresh/flow_best.pt", device=DEV)
el = load_elasticity("outputs/elasticity/neural.pt")
bank = AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
_, _, test = split_scenes(scenes)
src = test[6]; g = build_motifs(build_scene_graph(src))
print("ref:", src.scene_id, flush=True)

# ---------- (A) 3D box-render multi-scale sheet ----------
def retarget(s):
    room = scaled_room(src.room, s)
    return generative_retarget(flow, g, room, elasticity=el, bank=bank,
        cfg=RetargetConfig(restarts=16, regularity_snap=True, device=DEV), k=16, polish=False).scene
panels = [("reference", src)]
for s, tag in [(0.75, "target 0.75×"), (1.0, "target 1.0×"), (1.35, "target 1.35×")]:
    panels.append((tag, retarget(s)))
tmp = []
for i, (title, sc) in enumerate(panels):
    p = f"outputs/fig/_p{i}.png"
    render_scene3d(sc, path=p, view=(30.0, -58.0), figsize=4.2, title=title, dpi=150)
    tmp.append(p)
fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0))
for ax, p in zip(axes, tmp):
    ax.imshow(imageio.imread(p)); ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
fig.savefig("outputs/fig/3d_sheet.png", dpi=140, bbox_inches="tight", facecolor="white")
plt.close(fig); print("wrote 3d_sheet.png", flush=True)

# ---------- (B) real ODE trajectory GIF ----------
room = scaled_room(src.room, 1.35)
intent = build_design_intent(g, room, elasticity=el)
item = build_tokens(intent, room, None)
batch = collate([item], device=DEV)
fr = item.meta["frame_tgt"]; n = len(item.cat)
par = batch["parent"]; fh = batch["frame_h"]
gseed = torch.Generator(device="cpu").manual_seed(6)
noise = torch.randn(batch["state"].shape, generator=gseed).to(DEV)
prior = batch["cond"][..., 10:14]
x_world = prior + getattr(flow, "_prior_noise", 0.3) * noise
prel = getattr(flow, "parent_relative", False)
x = to_relative(x_world, par, fh) if prel else x_world
base = Scene(scene_id="traj", room=room.copy(),
             objects=[o.copy() for o in intent.source.objects], source="flow")

def scene_from_x(xc):
    xw = to_world(xc, par, fh) if prel else xc
    xn = xw[0].detach().cpu().numpy()
    sc = Scene(scene_id="f", room=room.copy(),
               objects=[o.copy() for o in base.objects], source="flow")
    for i in range(n):
        p, a = from_frame(xn[i], fr)
        sc.objects[i].xy = p; sc.objects[i].yaw = a
    return sc

def frame_img(sc, label, col):
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    draw_scene(ax, sc, labels=True, fontsize=7.2, min_label_size=1.0, show_front=False)
    b = __import__("reroom.geom.polygon", fromlist=["as_polygon"]).as_polygon(room).bounds
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; r = max(b[2]-b[0], b[3]-b[1])/2 + 0.6
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_axis_off()
    ax.set_title(label, fontsize=11, color=col, pad=6)
    fig.tight_layout(pad=0.2); fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig); return arr

frames = []
steps = 50; dt = 1.0 / steps
frames += [frame_img(scene_from_x(x), "① 仿射先驗 prior", "#a8412a")] * 6
with torch.no_grad():
    for s in range(steps):
        t = s * dt; tau = torch.full((1,), t, device=DEV)
        v = flow(x, tau, batch)
        if getattr(flow, "_mask_flow", False):
            v = v[..., :4]
        x = x + dt * v
        if s % 2 == 0:
            frames.append(frame_img(scene_from_x(x), f"① 整流流 rectified flow  {s+1}/50", "#a8412a"))
flow_scene = scene_from_x(x)
frames += [frame_img(flow_scene, "① flow 完成", "#a8412a")] * 3
for K in [6, 14, 26, 40]:
    proj = project_scene(flow_scene, room, iters=K, lr=0.03, device=DEV)
    frames.append(frame_img(proj, f"② 可微投影 Πθ  ({K} 步)", "#2a7a2a"))
frames += [frame_img(proj, "✓ 輸出 result", "#2a7a2a")] * 8
imageio.mimsave("outputs/fig/flow_traj.gif", frames, duration=0.11, loop=0)
print("wrote flow_traj.gif frames=", len(frames), flush=True)
print("DONE_FIGS")
