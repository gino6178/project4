#!/usr/bin/env python
"""Visualise the *warp-based pseudo-GT* training pair (currently NOT used).

Reversed direction from the current RetargetPairs:
    reference = real 3D-FRONT scene (INPUT, kept as-is)
    target_room = deform(scene.room) (a made-up new room)
    target_layout GT = affine warp of the real layout into the new room

The GT is *not* perfect -- it is a naive projection -- but it teaches the
model the *direction* of "target room is different from reference; here is a
starting point".  Combined with the current backward pair, it doubles the
number of retargeting samples and gives explicit exposure to reference->
new-target scaling, which is exactly what large-target inference needs.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import deform_room
from reroom.geom.polygon import as_polygon
from reroom.generative.train import warp_scene
from reroom.render.topdown import draw_scene


def main():
    corpus = "/home/gino/data/reroom/processed"
    scenes = [s for s in iter_scenes(corpus, limit=None, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    _, _, test = split_scenes(scenes)
    scene = test[6]
    print(f"scene: {scene.scene_id}  area={as_polygon(scene.room).area:.1f} m²")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    levels_shown = [1, 2, 4]
    names = ["Level 1: uniform_scale", "Level 2: aspect_ratio", "Level 4: corner_cut"]

    box_w = box_h = 0.0
    for L in levels_shown:
        rr = np.random.default_rng(42)
        tr = deform_room(scene.room, L, rr).room
        for r in (scene.room, tr):
            b = as_polygon(r).bounds
            box_w = max(box_w, b[2]-b[0])
            box_h = max(box_h, b[3]-b[1])

    for i, (L, name) in enumerate(zip(levels_shown, names)):
        rr = np.random.default_rng(42 + i)
        target_room = deform_room(scene.room, L, rr).room
        # target GT = affine warp of real scene into the deformed room
        target_warp = warp_scene(scene, target_room)

        ax = axes[i]
        ax.axis("off")
        # left: reference = real scene (unchanged)
        ax_l = ax.inset_axes([0.0, 0.0, 0.42, 1.0])
        draw_scene(ax_l, scene, labels=False, fontsize=7, min_label_size=1.0,
                   show_front=False)
        b = as_polygon(scene.room).bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        ax_l.set_xlim(cx - box_w/2*1.05, cx + box_w/2*1.05)
        ax_l.set_ylim(cy - box_h/2*1.05, cy + box_h/2*1.05)
        ax_l.set_title("reference = real scene\n(INPUT)", fontsize=10,
                       color="#2a7a2a", pad=6)
        # right: warp-derived target (pseudo-GT)
        ax_r = ax.inset_axes([0.58, 0.0, 0.42, 1.0])
        draw_scene(ax_r, target_warp, labels=False, fontsize=7, min_label_size=1.0,
                   show_front=False)
        b = as_polygon(target_warp.room).bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        ax_r.set_xlim(cx - box_w/2*1.05, cx + box_w/2*1.05)
        ax_r.set_ylim(cy - box_h/2*1.05, cy + box_h/2*1.05)
        ax_r.set_title("target (deformed room)\npseudo-GT = affine warp",
                       fontsize=10, color="#a8412a", pad=6)
        ax.annotate("", xy=(0.58, 0.5), xycoords="axes fraction",
                    xytext=(0.42, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#333"))
        ax.text(0.5, 0.53, "reference is real,\ntarget GT is a bootstrap",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, style="italic", color="#555")
        ax.text(0.5, 1.05, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, weight="bold")
        a_r = as_polygon(scene.room).area
        a_t = as_polygon(target_room).area
        ratio = a_t / a_r
        ax.text(0.21, -0.02, f"{a_r:.1f} m²",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
        ax.text(0.79, -0.02, f"{a_t:.1f} m²",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
        ax.text(0.5, -0.07, f"target / reference = {ratio:.2f}×",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="#a8412a" if ratio > 1.05 else ("#2a7a2a" if ratio < 0.95 else "#555"),
                weight="bold")

    fig.suptitle("Warp-based pseudo-GT training pair (currently NOT active)",
                 fontsize=13, y=1.03)
    fig.text(0.5, -0.07,
             "Reference = real 3D-FRONT scene (untouched) · target room made by deforming · GT layout = affine warp (a bootstrap, not perfect)\n"
             "Teaches the model reference → different-sized target directly. Complements the current backward pair, especially for target > reference cases.",
             ha="center", fontsize=10.5, color="#333")
    plt.tight_layout()
    out = "outputs/diag/warp_pair.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
