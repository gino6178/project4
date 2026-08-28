#!/usr/bin/env python
"""Three versions x three sizes (+reference) side-by-side comparison.

Row A: flow_wall + guidance only (no polish)   -- earlier §8.2 state
Row B: flow_wall + polish (shipped default)    -- current §8.5 shipped
Row C: flow_metric20 + polish                  -- §8.6 new training

Column 0: reference room
Column 1: 0.75x
Column 2: 1.0x
Column 3: 1.35x

Shared view box so panels line up; small labels so caption text stays readable.
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
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.geom.polygon import as_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.topdown import draw_scene
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import generative_retarget, load_flow
from reroom.intent.elasticity import load_elasticity


def scaled_room(room, s):
    poly = uniform_scale(room.polygon, s)
    anchored = _anchor_openings(room)
    op = _replace_openings(poly, anchored, len(room.polygon))
    return Room(polygon=poly, height=room.height, openings=op, room_type=room.room_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--flow-wall", default="outputs/flow_wall/flow.pt")
    ap.add_argument("--flow-metric", default="outputs/flow_metric20/flow_best.pt")
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--sizes", default="0.75,1.0,1.35")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    el = load_elasticity(a.elasticity) if os.path.exists(a.elasticity) else None
    flow_wall = load_flow(a.flow_wall, device="cpu")
    flow_metric = load_flow(a.flow_metric, device="cpu")
    cfg = RetargetConfig(restarts=24)

    scenes = [s for s in iter_scenes(a.corpus, limit=None, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    _, _, test = split_scenes(scenes)
    src = test[a.seed]
    g = build_motifs(build_scene_graph(src))
    sizes = [float(x) for x in a.sizes.split(",")]

    rows = [
        ("(A) guidance only\n(no polish)", flow_wall, {"polish": False}),
        ("(B) SHIPPED\nguidance + polish", flow_wall, {"polish": True}),
        ("(C) flow_metric20\n+ polish (new train)", flow_metric, {"polish": True}),
    ]

    def cap(m):
        return f"OOB {100*m['R_OOB']:.1f}%  col {100*m['R_col']:.1f}%  S_rel {m['S_rel']:.2f}"

    # generate everything first so view box is shared
    grid = []
    box_w = box_h = 0.0
    for label, flow, kw in rows:
        row_scenes = []
        for s in sizes:
            room = scaled_room(src.room, s)
            out = generative_retarget(flow, g, room, elasticity=el, bank=bank,
                                      cfg=cfg, k=16, **kw).scene
            row_scenes.append((s, out, cap(evaluate(g, out))))
        grid.append((label, row_scenes))

    # include reference in view box
    all_scenes = [src] + [sc for _, rs in grid for _, sc, _ in rs]
    for sc in all_scenes:
        b = as_polygon(sc.room).bounds
        box_w = max(box_w, b[2] - b[0])
        box_h = max(box_h, b[3] - b[1])
    box = (box_w + 0.6, box_h + 0.6)
    ar = box[1] / box[0]

    cols = len(sizes) + 1
    fig, axes = plt.subplots(3, cols, figsize=(3.2 * cols, 3.2 * 3 * ar * 1.10))
    for ax in axes.ravel():
        ax.axis("off")

    # column headers
    col_titles = ["reference"] + [f"{s:g}x" for s in sizes]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11, pad=6)

    for i, (label, row_scenes) in enumerate(grid):
        # column 0: reference (once per row for visual continuity)
        ax = axes[i, 0]; ax.axis("on")
        draw_scene(ax, src, labels=True, fontsize=6.2, min_label_size=1.0,
                   show_front=False)
        b = as_polygon(src.room).bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        ax.set_xlim(cx - box[0]/2, cx + box[0]/2)
        ax.set_ylim(cy - box[1]/2, cy + box[1]/2)
        # row label as ylabel-like text on left
        ax.text(-0.10, 0.5, label, transform=ax.transAxes, ha="right",
                va="center", fontsize=9.5, fontweight="bold", color="#a8412a",
                rotation=0)

        for j, (s, sc, capt) in enumerate(row_scenes, start=1):
            ax = axes[i, j]; ax.axis("on")
            draw_scene(ax, sc, labels=True, fontsize=6.2, min_label_size=1.0,
                       show_front=False)
            b = as_polygon(sc.room).bounds
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            ax.set_xlim(cx - box[0]/2, cx + box[0]/2)
            ax.set_ylim(cy - box[1]/2, cy + box[1]/2)
            ax.text(0.5, -0.02, capt, transform=ax.transAxes, ha="center",
                    va="top", fontsize=7.4, color="#5c5c5c")

    fig.subplots_adjust(hspace=0.22, wspace=0.06, left=0.09)
    out_path = os.path.join(a.out, "three_versions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
