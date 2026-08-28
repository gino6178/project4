#!/usr/bin/env python
"""Diagnostic: one reference design retargeted to three ROOM SIZES.

Follows the hero figure (one design -> several targets), but instead of varying
the *shape* it varies the *size* -- small / same / large -- so we can see, in
one sheet, how the shipped pipeline behaves across the whole size range and
audit every wall-affinity object for gap-to-wall and skew.
"""
from __future__ import annotations
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reroom.core.scene import Room
from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import evaluate
from reroom.geom.deform import (uniform_scale, _anchor_openings, _replace_openings)
from reroom.geom.polygon import as_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.topdown import draw_scene
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import generative_retarget, load_flow
from reroom.intent.elasticity import load_elasticity

WALL_CATS = ("bookcase", "wardrobe", "tv_stand", "tv", "sofa", "sofa_bed",
             "bed", "double_bed", "single_bed", "desk", "dressing_table",
             "cabinet", "shelf", "sideboard", "console")


def scaled_room(room: Room, s: float) -> Room:
    poly = uniform_scale(room.polygon, s)
    anchored = _anchor_openings(room)
    openings = _replace_openings(poly, anchored, len(room.polygon))
    return Room(polygon=poly, height=room.height, openings=openings,
                room_type=room.room_type)


def wall_report(scene: Room, tag: str):
    """For every wall-affinity object: nearest wall, gap, skew (deg)."""
    poly = as_polygon(scene.room)
    ring = np.asarray(poly.exterior.coords)[:-1]
    edges = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    lines = []
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS:
            continue
        corners = o.corners()
        # nearest wall = the edge minimising mean corner distance
        best = None
        for wi, (a, b) in enumerate(edges):
            d = b - a
            L = np.linalg.norm(d) + 1e-9
            t = d / L
            n = np.array([-t[1], t[0]])
            # signed perpendicular distance of each corner to the infinite line
            rel = corners - a
            perp = np.abs(rel @ n)
            proj = rel @ t
            inside = (proj > -0.05) & (proj < L + 0.05)
            if not inside.any():
                continue
            gap = perp[inside].min()
            if best is None or gap < best[1]:
                # skew: angle of object's nearest edge vs the wall direction
                # object edges: use the side whose normal is closest to wall n
                oc = np.vstack([corners, corners[0]])
                ang_wall = math.atan2(t[1], t[0])
                # object's two axis directions
                ax0 = np.array([math.cos(o.yaw), math.sin(o.yaw)])
                ax1 = np.array([-math.sin(o.yaw), math.cos(o.yaw)])
                sk = min(
                    abs(((math.atan2(ax0[1], ax0[0]) - ang_wall + math.pi/2) % math.pi) - math.pi/2),
                    abs(((math.atan2(ax1[1], ax1[0]) - ang_wall + math.pi/2) % math.pi) - math.pi/2),
                )
                best = (wi, gap, math.degrees(sk))
        if best is None:
            lines.append(f"    {o.category:14s} gap=  --   (no wall)")
        else:
            wi, gap, sk = best
            flag = "  <-- FLOAT" if gap > 0.12 else ("  <-- SKEW" if sk > 4 else "")
            lines.append(f"    {o.category:14s} wall#{wi} gap={gap*100:5.1f}cm "
                         f"skew={sk:4.1f}deg{flag}")
    print(f"  [{tag}]")
    for l in lines:
        print(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--flow", default="outputs/flow_wall/flow.pt")
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--sizes", default="0.75,1.0,1.35")
    ap.add_argument("--regularity", action="store_true",
                    help="ReRoom 2.0 Step 1: snap layout to orthogonal/flush/slot")
    ap.add_argument("--walkable", action="store_true",
                    help="PhyScene-style walkability: capacity prune + door box + nav rank")
    ap.add_argument("--no-polish", dest="no_polish", action="store_true",
                    help="drop the 25-step Adam polish; Flow + Regularity only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    flow = load_flow(a.flow, device="cpu")
    el = load_elasticity(a.elasticity) if os.path.exists(a.elasticity) else None
    cfg = RetargetConfig(restarts=24, regularity_snap=a.regularity, walkable=a.walkable)

    scenes = [s for s in iter_scenes(a.corpus, limit=None, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    _, _, test = split_scenes(scenes)
    src = test[a.seed]
    g = build_motifs(build_scene_graph(src))

    sizes = [float(x) for x in a.sizes.split(",")]
    names = {0.75: "small (0.75x)", 1.0: "same (1.0x)", 1.35: "large (1.35x)"}

    print(f"reference: {src.scene_id}  {as_polygon(src.room).area:.1f} m^2  "
          f"{sum(1 for o in src.objects if o.keep)} objects  flow={a.flow}")
    wall_report(src, "reference")

    panels = [("reference", src,
               f"{as_polygon(src.room).area:.1f} m^2   "
               f"{sum(1 for o in src.objects if o.keep)} obj")]
    for s in sizes:
        room = scaled_room(src.room, s)
        out = generative_retarget(flow, g, room, elasticity=el, bank=bank,
                                  cfg=cfg, k=16, polish=not a.no_polish).scene
        m = evaluate(g, out)
        cap = (f"OOB {100*m['R_OOB']:.1f}%  col {100*m['R_col']:.1f}%  "
               f"S_rel {m['S_rel']:.2f}")
        nm = names.get(s, f"{s:g}x")
        panels.append((nm, out, cap))
        wall_report(out, nm)

    # render sheet
    box_w = box_h = 0.0
    for _, sc, _ in panels:
        b = as_polygon(sc.room).bounds
        box_w = max(box_w, b[2] - b[0]); box_h = max(box_h, b[3] - b[1])
    box = (box_w + 0.8, box_h + 0.8)
    ar = box[1] / box[0]
    fig, axes = plt.subplots(2, 2, figsize=(3.9 * 2, 3.9 * 2 * ar * 1.16))
    for ax in axes.ravel():
        ax.axis("off")
    for k, (title, sc, cap) in enumerate(panels):
        ax = axes.ravel()[k]; ax.axis("on")
        draw_scene(ax, sc, labels=True, fontsize=7.4, min_label_size=1.0,
                   show_front=False)
        b = as_polygon(sc.room).bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        ax.set_xlim(cx - box[0]/2, cx + box[0]/2)
        ax.set_ylim(cy - box[1]/2, cy + box[1]/2)
        ax.set_title(title, fontsize=10, pad=5)
        ax.text(0.5, -0.035, cap, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.6, color="#5c5c5c")
    fig.subplots_adjust(hspace=0.22, wspace=0.05)
    path = os.path.join(a.out, "three_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
