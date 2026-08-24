#!/usr/bin/env python
"""Figures sized for the paper page, not for a full-resolution report.

The report's sheets pack eleven panels into one image.  Dropped into an 820-px
column that leaves each panel about 250 px wide, the labels become unreadable
and the grid goes ragged, because every panel is scaled by its own room's
aspect ratio.  These are drawn instead at a fixed panel count, on a shared view
box so the grid is regular, with labels suppressed on objects too small to hold
them.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import evaluate
from reroom.geom.deform import deform_room
from reroom.geom.polygon import as_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.topdown import draw_scene
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget


def _common_box(scenes, pad=0.4):
    """One view box for every panel, so the grid does not go ragged."""
    w = h = 0.0
    for s in scenes:
        b = as_polygon(s.room).bounds
        w = max(w, b[2] - b[0])
        h = max(h, b[3] - b[1])
    return w + 2 * pad, h + 2 * pad


def _panel(ax, scene, box, title, caption, fs):
    draw_scene(ax, scene, labels=True, fontsize=fs, min_label_size=1.0,
               show_front=False)
    b = as_polygon(scene.room).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    ax.set_xlim(cx - box[0] / 2, cx + box[0] / 2)
    ax.set_ylim(cy - box[1] / 2, cy + box[1] / 2)
    ax.set_title(title, fontsize=fs * 1.35, pad=5)
    if caption:
        ax.text(0.5, -0.035, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=fs * 1.05, color="#5c5c5c")


def sheet(panels, path, cols, per=3.4, fs=7.2):
    rows = int(math.ceil(len(panels) / cols))
    box = _common_box([s for _, s, _ in panels])
    ar = box[1] / box[0]
    fig, axes = plt.subplots(rows, cols,
                             figsize=(per * cols, per * rows * ar * 1.16))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for k, (title, sc, cap) in enumerate(panels):
        ax = np.atleast_1d(axes).ravel()[k]
        ax.axis("on")
        _panel(ax, sc, box, title, cap, fs)
    fig.subplots_adjust(hspace=0.20, wspace=0.04)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None

    scenes = [s for s in iter_scenes(a.corpus, limit=None, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    _, _, test = split_scenes(scenes)

    src = test[a.seed]
    g = build_motifs(build_scene_graph(src))
    cfg = RetargetConfig(restarts=24)
    cap = lambda m: (f"OOB {100*m['R_OOB']:.1f}%   collision {100*m['R_col']:.1f}%"
                     f"   S_rel {m['S_rel']:.2f}")

    # ---- hero: one design, three target shapes, four equal panels ----
    hero = [("the reference room", src,
             f"{as_polygon(src.room).area:.1f} m²   "
             f"{sum(1 for o in src.objects if o.keep)} objects")]
    for lvl, name in ((2, "narrower"), (4, "a corner removed"), (5, "a slanted wall")):
        room = deform_room(src.room, lvl, np.random.default_rng(a.seed + lvl)).room
        out = retarget(g, room, bank=bank, cfg=cfg).scene
        hero.append((name, out, cap(evaluate(g, out))))
    sheet(hero, os.path.join(a.out, "hero.png"), cols=2, per=3.7, fs=7.6)

    # ---- the comparison, two large panels on the hardest target ----
    room = deform_room(src.room, 4, np.random.default_rng(a.seed + 4)).room
    ds = run_baseline("direct_scaling", g, room, cfg=cfg)
    rr = retarget(g, room, bank=bank, cfg=cfg).scene
    sheet([("normalised-coordinate scaling", ds, cap(evaluate(g, ds))),
           ("ReRoom", rr, cap(evaluate(g, rr)))],
          os.path.join(a.out, "compare.png"), cols=2, per=3.9, fs=8.0)

    # ---- the curriculum: five outlines in one row, no labels needed ----
    lv = [("reference", src, "")]
    for lvl, name in ((1, "uniform"), (2, "aspect"), (3, "slanted"), (4, "corner cut")):
        room = deform_room(src.room, lvl, np.random.default_rng(a.seed + lvl)).room
        out = retarget(g, room, bank=bank, cfg=cfg).scene
        lv.append((name, out, ""))
    sheet(lv, os.path.join(a.out, "levels.png"), cols=5, per=2.5, fs=5.4)


if __name__ == "__main__":
    main()
