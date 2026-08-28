#!/usr/bin/env python
"""Render every training-pair variant we have or could have, side by side.

Writes four PNGs:
  1. training_pair_backward.png   -- current default in RetargetPairs
  2. training_pair_scramble.png   -- backward + 50 % positions scrambled
  3. training_pair_warp.png       -- reversed direction (candidate, not active)
  4. training_pair_subst.png      -- backward + substitution (candidate, not active)

Each row of each PNG shows the INPUT (reference) side by side with the TARGET
(GT) side, for three deform levels.  Purpose: sanity-check that every pair
variant produces a real, reasonable-looking sample the model could learn from.
"""
from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point as ShPoint
from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import deform_room
from reroom.geom.polygon import as_polygon
from reroom.generative.train import warp_scene
from reroom.render.topdown import draw_scene


LEVELS = [1, 2, 4]
LEVEL_NAMES = {1: "L1 uniform_scale", 2: "L2 aspect_ratio", 4: "L4 corner_cut"}


def _bbox(scenes):
    w = h = 0.0
    for s in scenes:
        b = as_polygon(s.room).bounds
        w = max(w, b[2]-b[0])
        h = max(h, b[3]-b[1])
    return w, h


def _draw_pair(fig, ax, left_scene, right_scene, box_w, box_h,
               left_title, right_title, left_color, right_color,
               arrow_label, level_name, ratio_label, ratio_value):
    ax.axis("off")
    ax_l = ax.inset_axes([0.0, 0.0, 0.42, 1.0])
    draw_scene(ax_l, left_scene, labels=False, fontsize=7, min_label_size=1.0,
               show_front=False)
    b = as_polygon(left_scene.room).bounds
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    ax_l.set_xlim(cx - box_w/2*1.05, cx + box_w/2*1.05)
    ax_l.set_ylim(cy - box_h/2*1.05, cy + box_h/2*1.05)
    ax_l.set_title(left_title, fontsize=9.5, color=left_color, pad=6)
    ax_r = ax.inset_axes([0.58, 0.0, 0.42, 1.0])
    draw_scene(ax_r, right_scene, labels=False, fontsize=7, min_label_size=1.0,
               show_front=False)
    b = as_polygon(right_scene.room).bounds
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    ax_r.set_xlim(cx - box_w/2*1.05, cx + box_w/2*1.05)
    ax_r.set_ylim(cy - box_h/2*1.05, cy + box_h/2*1.05)
    ax_r.set_title(right_title, fontsize=9.5, color=right_color, pad=6)
    ax.annotate("", xy=(0.58, 0.5), xycoords="axes fraction",
                xytext=(0.42, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#333"))
    ax.text(0.5, 0.53, arrow_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, style="italic", color="#555")
    ax.text(0.5, 1.05, level_name, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10.5, weight="bold")
    ax.text(0.5, -0.06, f"{ratio_label} = {ratio_value:.2f}×",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#a8412a" if ratio_value > 1.05 else ("#2a7a2a" if ratio_value < 0.95 else "#555"),
            weight="bold")


# ---------------------------------------------------------------------------
# 1. backward (current)
# ---------------------------------------------------------------------------
def make_backward(scene, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    # pre-compute box
    scs = [scene]
    for L in LEVELS:
        rr = np.random.default_rng(42)
        scs.append(warp_scene(scene, deform_room(scene.room, L, rr).room))
    w, h = _bbox(scs)
    for i, L in enumerate(LEVELS):
        rr = np.random.default_rng(42 + i)
        ref_room = deform_room(scene.room, L, rr).room
        pseudo = warp_scene(scene, ref_room)
        ratio = as_polygon(pseudo.room).area / as_polygon(scene.room).area
        _draw_pair(fig, axes[i], pseudo, scene, w, h,
                   "pseudo-reference\n(INPUT)", "target = real scene\n(GT)",
                   "#a8412a", "#2a7a2a",
                   "model learns to recover", LEVEL_NAMES[L],
                   "ref / target", ratio)
    fig.suptitle("(1) Backward pair · CURRENT DEFAULT",
                 fontsize=13, y=1.03, color="#111")
    fig.text(0.5, -0.05,
             "Real 3D-FRONT scene stays as target · reference manufactured by deforming the room + affine warp of objects.  "
             "Both « ref smaller → target bigger » and the reverse appear naturally.",
             ha="center", fontsize=10, color="#333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 2. backward + scramble (positions uniform-random inside pseudo room)
# ---------------------------------------------------------------------------
def make_scramble(scene, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    scs = [scene]
    for L in LEVELS:
        rr = np.random.default_rng(42)
        scs.append(warp_scene(scene, deform_room(scene.room, L, rr).room))
    w, h = _bbox(scs)
    for i, L in enumerate(LEVELS):
        rr = np.random.default_rng(42 + i)
        ref_room = deform_room(scene.room, L, rr).room
        pseudo = warp_scene(scene, ref_room)
        # scramble: uniform sample inside polygon
        poly = as_polygon(pseudo.room)
        minx, miny, maxx, maxy = poly.bounds
        rng = np.random.default_rng(500 + i)
        for o in pseudo.objects:
            for _ in range(20):
                px = rng.uniform(minx, maxx); py = rng.uniform(miny, maxy)
                if poly.contains(ShPoint(px, py)):
                    o.xy = np.array([px, py]); break
            o.yaw = float(rng.uniform(-math.pi, math.pi))
        ratio = as_polygon(pseudo.room).area / as_polygon(scene.room).area
        _draw_pair(fig, axes[i], pseudo, scene, w, h,
                   "scrambled reference\n(INPUT)", "target = real scene\n(GT)",
                   "#a8412a", "#2a7a2a",
                   "model must recover\nfrom noise", LEVEL_NAMES[L],
                   "ref / target", ratio)
    fig.suptitle("(2) Backward + SCRAMBLE (positions randomised) · CURRENTLY ACTIVE (50 % of samples)",
                 fontsize=12.5, y=1.03, color="#111")
    fig.text(0.5, -0.05,
             "50 % of training samples have their pseudo-reference positions completely randomised inside the deformed room "
             "(yaw uniform, category / motif / size kept). Forces the model to rebuild layout from semantic tokens alone.",
             ha="center", fontsize=10, color="#333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 3. warp-based pseudo-GT (reversed direction, not active)
# ---------------------------------------------------------------------------
def make_warp(scene, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    scs = [scene]
    for L in LEVELS:
        rr = np.random.default_rng(42)
        tr = deform_room(scene.room, L, rr).room
        scs.append(warp_scene(scene, tr))
    w, h = _bbox(scs)
    for i, L in enumerate(LEVELS):
        rr = np.random.default_rng(42 + i)
        target_room = deform_room(scene.room, L, rr).room
        target_warp = warp_scene(scene, target_room)
        ratio = as_polygon(target_room).area / as_polygon(scene.room).area
        _draw_pair(fig, axes[i], scene, target_warp, w, h,
                   "reference = real scene\n(INPUT)",
                   "target (deformed room)\npseudo-GT = affine warp",
                   "#2a7a2a", "#a8412a",
                   "reference is real,\ntarget GT is a bootstrap",
                   LEVEL_NAMES[L], "target / reference", ratio)
    fig.suptitle("(3) Warp-based pseudo-GT pair · CANDIDATE (not active)",
                 fontsize=13, y=1.03, color="#111")
    fig.text(0.5, -0.05,
             "Reversed direction: reference = untouched real scene, target = deformed room, GT = affine warp (imperfect but well-defined). "
             "Teaches reference → different-sized target directly.",
             ha="center", fontsize=10, color="#333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 4. backward + substitution (swap objects from bank, not active)
# ---------------------------------------------------------------------------
def make_subst(scene, bank, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    scs = [scene]
    for L in LEVELS:
        rr = np.random.default_rng(42)
        scs.append(warp_scene(scene, deform_room(scene.room, L, rr).room))
    w, h = _bbox(scs)
    for i, L in enumerate(LEVELS):
        rr = np.random.default_rng(42 + i)
        ref_room = deform_room(scene.room, L, rr).room
        pseudo = warp_scene(scene, ref_room)
        subst_target = scene.copy() if hasattr(scene, "copy") else scene
        # swap in random same-category, DIFFERENT-size assets from the bank on
        # BOTH input and target so category matches but size varies
        rng = np.random.default_rng(700 + i)
        swapped = 0
        for o_input, o_target in zip(pseudo.objects, subst_target.objects):
            if rng.random() > 0.6:                     # ~40 % swapped
                continue
            if not bank.has(o_input.category):
                continue
            idx = list(bank.by_category[o_input.category])
            aid_idx = int(rng.choice(idx))
            new_size = bank.assets[aid_idx].size.copy()
            # apply same new_size to both sides so relation types are held
            o_input.size = new_size
            o_target.size = new_size.copy()
            swapped += 1
        ratio = as_polygon(pseudo.room).area / as_polygon(scene.room).area
        title_left = f"pseudo-ref + subst.\n({swapped} objs resized)"
        _draw_pair(fig, axes[i], pseudo, subst_target, w, h,
                   title_left, "target (same swaps)\n(GT)",
                   "#a8412a", "#2a7a2a",
                   "learn α: which distances\nshould stretch vs stay rigid",
                   LEVEL_NAMES[L], "ref / target", ratio)
    fig.suptitle("(4) Backward + SUBSTITUTION (swap object identities) · CANDIDATE (not active)",
                 fontsize=12.5, y=1.03, color="#111")
    fig.text(0.5, -0.05,
             "During training, some objects get swapped with a same-category but DIFFERENT-size asset from the bank on both sides.  "
             "This is the missing signal for α_ij: same relation type at various object sizes → the model learns which distances are rigid vs elastic.",
             ha="center", fontsize=10, color="#333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


def main():
    corpus = "/home/gino/data/reroom/processed"
    scenes = [s for s in iter_scenes(corpus, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    _, _, test = split_scenes(scenes)
    scene = test[6]
    bank = AssetBank.load("outputs/priors/assets_future.pkl")
    print(f"scene: {scene.scene_id}   bank size: {len(bank)}")
    os.makedirs("outputs/diag", exist_ok=True)
    make_backward(scene, "outputs/diag/pair_1_backward.png")
    make_scramble(scene, "outputs/diag/pair_2_scramble.png")
    make_warp(scene,     "outputs/diag/pair_3_warp.png")
    make_subst(scene, bank, "outputs/diag/pair_4_subst.png")


if __name__ == "__main__":
    main()
