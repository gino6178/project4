#!/usr/bin/env python
"""Visualise how a training pair is manufactured for RetargetPairs.

Left: pseudo-reference (INPUT to the model) -- the original scene's objects
      after being warped into a deformed version of the room.  Looks
      unrealistic because it's an affine warp, not a real design.

Right: target (GT for training) -- the original real scene, professionally
       designed in the original room.

The model learns:  pseudo-reference -> target.
No 'target GT for a new room' is invented -- the real room is the target and
the reference is manufactured backwards.
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
    scene = test[6]  # same scene as three-sizes bench
    print(f"scene: {scene.scene_id}  area={as_polygon(scene.room).area:.1f} m²")

    # deform + warp — this is exactly what RetargetPairs.__getitem__ does
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    levels_shown = [1, 2, 4]         # uniform_scale, aspect, corner_cut
    names = ["Level 1: uniform_scale", "Level 2: aspect_ratio", "Level 4: corner_cut"]

    box_w = box_h = 0.0
    for L in levels_shown:
        rr = np.random.default_rng(42)
        ref_room = deform_room(scene.room, L, rr).room
        for r in (scene.room, ref_room):
            b = as_polygon(r).bounds
            box_w = max(box_w, b[2]-b[0])
            box_h = max(box_h, b[3]-b[1])

    for i, (L, name) in enumerate(zip(levels_shown, names)):
        rr = np.random.default_rng(42 + i)
        ref_room = deform_room(scene.room, L, rr).room
        pseudo = warp_scene(scene, ref_room)

        ax = axes[i]
        # draw two rooms side by side within one subplot using inset axes:
        # left half = pseudo-reference (deformed room)
        # right half = target (original scene)
        ax.axis("off")
        # left: pseudo-ref
        ax_l = ax.inset_axes([0.0, 0.0, 0.42, 1.0])
        draw_scene(ax_l, pseudo, labels=False, fontsize=7, min_label_size=1.0,
                   show_front=False)
        b = as_polygon(pseudo.room).bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        ax_l.set_xlim(cx - box_w/2*1.05, cx + box_w/2*1.05)
        ax_l.set_ylim(cy - box_h/2*1.05, cy + box_h/2*1.05)
        ax_l.set_title("pseudo-reference\n(INPUT)", fontsize=10, color="#a8412a", pad=6)
        # right: target (original scene)
        ax_r = ax.inset_axes([0.58, 0.0, 0.42, 1.0])
        draw_scene(ax_r, scene, labels=False, fontsize=7, min_label_size=1.0,
                   show_front=False)
        b = as_polygon(scene.room).bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        ax_r.set_xlim(cx - box_w/2*1.05, cx + box_w/2*1.05)
        ax_r.set_ylim(cy - box_h/2*1.05, cy + box_h/2*1.05)
        ax_r.set_title("target = real scene\n(GT)", fontsize=10, color="#2a7a2a", pad=6)
        # arrow between
        ax.annotate("", xy=(0.58, 0.5), xycoords="axes fraction",
                    xytext=(0.42, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#333"))
        ax.text(0.5, 0.53, "model learns\nto recover", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5, style="italic", color="#555")
        # curriculum level name at top
        ax.text(0.5, 1.05, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, weight="bold")
        # area annotations
        a_p = as_polygon(pseudo.room).area
        a_t = as_polygon(scene.room).area
        ratio = a_p / a_t
        ax.text(0.21, -0.02, f"{a_p:.1f} m²",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
        ax.text(0.79, -0.02, f"{a_t:.1f} m²",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")
        ax.text(0.5, -0.07, f"ref / target = {ratio:.2f}×",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="#a8412a" if ratio > 1.05 else ("#2a7a2a" if ratio < 0.95 else "#555"),
                weight="bold")

    fig.suptitle("How each training pair is manufactured (RetargetPairs.__getitem__)",
                 fontsize=13, y=1.03)
    fig.text(0.5, -0.06,
             "Real 3D-FRONT scene stays as TARGET (GT) · reference is manufactured backwards by deforming the room and warping objects in\n"
             "Both « ref smaller → target bigger » (expand) and « ref bigger → target smaller » (shrink) directions appear naturally in training",
             ha="center", fontsize=10.5, color="#333")
    plt.tight_layout()
    out = "outputs/diag/training_pair.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
