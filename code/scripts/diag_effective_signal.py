#!/usr/bin/env python
"""How many training samples actually match the *shape-preserving* test?

The three-sizes bench only scales the room uniformly (same shape, different
size). But the training data also includes aspect deformations, slanted
walls, corner cuts, concave polygons — those change *shape*, not just size.

This script:
  1. samples many training pairs
  2. classifies each by (a) deform level (b) shape-preserving vs shape-changing
     and (c) whether its ref/target size ratio matches an inference test point
  3. plots the fraction of pairs that are *directly relevant* to the test.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import deform_room
from reroom.geom.polygon import as_polygon


def shape_signature(poly_np):
    """A cheap rotation/scale-invariant shape descriptor.

    Uses (perimeter^2 / area) plus vertex count and normalised extents to
    tell 'rectangle' from 'L-shape' from 'trapezoid'.  Two rooms with the
    same signature -> same *shape*.
    """
    poly = Polygon(poly_np)
    ext = poly.exterior
    per = ext.length
    area = abs(poly.area)
    isoperim = per ** 2 / max(area, 1e-6)
    # aspect of axis-aligned bbox as a shape hint
    minx, miny, maxx, maxy = poly.bounds
    dx, dy = maxx - minx, maxy - miny
    aspect = max(dx, dy) / max(min(dx, dy), 1e-6)
    return isoperim, aspect, len(poly_np)


def is_shape_preserving(ref_room, target_room, tol=0.03):
    """True iff ref_room is (nearly) a uniform scale of target_room."""
    r_sig = shape_signature(ref_room.polygon)
    t_sig = shape_signature(target_room.polygon)
    # isoperim scale-invariant, aspect scale-invariant, vcount identical
    d_iso = abs(r_sig[0] - t_sig[0]) / max(t_sig[0], 1e-6)
    d_asp = abs(r_sig[1] - t_sig[1]) / max(t_sig[1], 1e-6)
    return d_iso < tol and d_asp < tol and r_sig[2] == t_sig[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/reroom/processed")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="outputs/diag/effective_signal.png")
    a = ap.parse_args()

    scenes = [s for s in iter_scenes(a.corpus, min_objects=5)
              if len(s.objects) <= 24]
    train, _, _ = split_scenes(scenes)
    rng = np.random.default_rng(0)
    LEVELS = (1, 2, 3, 4, 5)

    rows = []                       # (level, ratio, shape_preserving)
    for _ in range(a.n):
        s = train[rng.integers(0, len(train))]
        L = int(rng.choice(LEVELS))
        for _try in range(4):
            ref_room = deform_room(s.room, L, rng).room
            if as_polygon(ref_room).area > 3.0:
                break
        ratio = as_polygon(ref_room).area / as_polygon(s.room).area
        sp = is_shape_preserving(ref_room, s.room)
        rows.append((L, ratio, sp))

    ratios = np.array([r[1] for r in rows])
    sp     = np.array([r[2] for r in rows])
    levels = np.array([r[0] for r in rows])

    tests = [(0.75, "test 0.75x", 1.0/0.75),
             (1.35, "test 1.35x", 1.0/1.35)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))

    # top: separate the pooled histogram into shape-preserving vs shape-changing
    ax = axes[0]
    bins = np.linspace(0.2, 3.0, 60)
    ax.hist(ratios[sp], bins=bins, color="#2a7a2a", alpha=0.7,
            label=f"shape-preserving  n={sp.sum()} ({100*sp.mean():.0f}%)",
            histtype="stepfilled", edgecolor="white")
    ax.hist(ratios[~sp], bins=bins, color="#a8412a", alpha=0.5,
            label=f"shape-changing  n={(~sp).sum()} ({100*(~sp).mean():.0f}%)",
            histtype="stepfilled", edgecolor="white")
    for _, lbl, eq in tests:
        ax.axvline(eq, color="#111", ls="--", lw=1.2)
        ax.text(eq, ax.get_ylim()[1]*0.9,
                f"  {lbl}\n  (ratio={eq:.2f})",
                fontsize=8.5, ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax.set_xlabel("ref / target area ratio")
    ax.set_ylabel("count")
    ax.set_title(f"Training pairs pooled ({a.n} samples): "
                 "shape-preserving vs shape-changing", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0.2, 3.0)
    ax.grid(alpha=0.25)

    # bottom: how many pairs are *both* shape-preserving AND ratio-relevant
    ax2 = axes[1]
    # for each test point, count pairs within a window of ±0.05
    window = 0.05
    labels = ["all L1..L5\n(any ratio)",
              "shape-preserving\n(any ratio)",
              "0.75× ratio-relevant\n(any shape)",
              "0.75× + shape-preserving",
              "1.35× ratio-relevant\n(any shape)",
              "1.35× + shape-preserving"]
    fracs = [
        1.0,
        sp.mean(),
        ((ratios > tests[0][2]-window) & (ratios < tests[0][2]+window)).mean(),
        ((ratios > tests[0][2]-window) & (ratios < tests[0][2]+window) & sp).mean(),
        ((ratios > tests[1][2]-window) & (ratios < tests[1][2]+window)).mean(),
        ((ratios > tests[1][2]-window) & (ratios < tests[1][2]+window) & sp).mean(),
    ]
    colors = ["#888", "#2a7a2a", "#a8412a", "#7a2a2a", "#a8412a", "#7a2a2a"]
    bars = ax2.barh(range(len(labels)), [f*100 for f in fracs], color=colors,
                     alpha=0.85)
    for i, (b, f) in enumerate(zip(bars, fracs)):
        ax2.text(f*100 + 0.5, i, f" {f*100:.1f}%",
                 va="center", fontsize=10, weight="bold")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9.5)
    ax2.set_xlabel("fraction of training samples (%)")
    ax2.set_title("The effective signal: pairs that BOTH match test size AND preserve shape",
                  fontsize=11)
    ax2.grid(alpha=0.25, axis="x")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")
    print("\n===== summary =====")
    for i, l in enumerate(labels):
        print(f"  {l.replace(chr(10),' ')}:  {fracs[i]*100:.2f}%")


if __name__ == "__main__":
    main()
