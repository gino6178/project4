#!/usr/bin/env python
"""Sample many training pairs and plot ref/target area ratio distribution.

The claim is `uniform_scale s in [0.7, 1.4]`, but aspect_deform / corner_cut
can push area ratios wider or narrower.  This script measures what the model
actually sees across a large sample and marks where inference tests
(0.75x / 1.35x) fall on that distribution.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import deform_room
from reroom.geom.polygon import as_polygon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/reroom/processed")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="outputs/diag/train_ratios.png")
    a = ap.parse_args()

    scenes = [s for s in iter_scenes(a.corpus, min_objects=5)
              if len(s.objects) <= 24]
    train, _, _ = split_scenes(scenes)
    rng = np.random.default_rng(0)
    LEVELS = (1, 2, 3, 4, 5)

    ratios_per_level = {L: [] for L in LEVELS}
    ratios_all = []
    for _ in range(a.n):
        s = train[rng.integers(0, len(train))]
        L = int(rng.choice(LEVELS))
        for _try in range(4):
            ref_room = deform_room(s.room, L, rng).room
            if as_polygon(ref_room).area > 3.0:
                break
        r_area = as_polygon(ref_room).area
        t_area = as_polygon(s.room).area
        ratio = r_area / t_area
        ratios_per_level[L].append(ratio)
        ratios_all.append(ratio)

    # In backward pair: ref = deformed, target = original. Model sees this as
    # its "reference" (input). If we invert to think "what the test asks", the
    # test's target/reference (target vs original reference) corresponds to
    # 1/ratio here. So both perspectives on the same axis.
    inv_all = [1.0 / r for r in ratios_all]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    ax = axes[0]
    colors = {1: "#c1442e", 2: "#1f78b4", 3: "#33a02c", 4: "#ff7f00", 5: "#6a3d9a"}
    bins = np.linspace(0.2, 3.0, 60)
    for L in LEVELS:
        ax.hist(ratios_per_level[L], bins=bins, alpha=0.6, color=colors[L],
                label=f"Level {L} (n={len(ratios_per_level[L])})",
                histtype="stepfilled", linewidth=1.2, edgecolor="white")
    # test points where our three-sizes bench sits (target/reference)
    # translated to ref/target axis by taking reciprocal
    for tr, lbl in [(0.75, "target=0.75×"), (1.0, "target=1.0×"), (1.35, "target=1.35×")]:
        eq_ratio = 1.0 / tr
        ax.axvline(eq_ratio, color="#111", ls="--", lw=1.2, alpha=0.7)
        ax.text(eq_ratio, ax.get_ylim()[1] * 0.9, f"  {lbl}  \n  (ref/target={eq_ratio:.2f})",
                fontsize=8.5, ha="left", va="top", color="#111",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax.set_xlabel("ref / target area ratio  (as seen at training)")
    ax.set_ylabel("count")
    ax.set_title(f"Training-pair area ratio distribution across {a.n} sampled pairs",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlim(0.2, 3.0)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    all_arr = np.array(ratios_all)
    ax2.hist(all_arr, bins=bins, color="#555", alpha=0.85,
             histtype="stepfilled", edgecolor="white")
    # p10/p50/p90 markers
    p10, p50, p90 = np.percentile(all_arr, [10, 50, 90])
    for p, name in [(p10, "p10"), (p50, "p50"), (p90, "p90")]:
        ax2.axvline(p, color="#a8412a", ls=":", lw=1.5)
        ax2.text(p, ax2.get_ylim()[1] * 0.95, f"{name}={p:.2f}",
                 fontsize=8.5, ha="left", va="top", color="#a8412a",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    for tr, lbl in [(0.75, "test 0.75×"), (1.35, "test 1.35×")]:
        eq_ratio = 1.0 / tr
        ax2.axvline(eq_ratio, color="#111", ls="--", lw=1.2)
        ax2.text(eq_ratio, ax2.get_ylim()[1] * 0.7, f"  {lbl}  \n  (ratio={eq_ratio:.2f})",
                 fontsize=8.5, ha="left", va="top", color="#111",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax2.set_xlabel("ref / target area ratio  (all levels pooled)")
    ax2.set_ylabel("count")
    ax2.set_title("Pooled distribution + inference-test points", fontsize=11)
    ax2.set_xlim(0.2, 3.0)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")
    # summary print
    print(f"\npooled n={len(ratios_all)}: p10={p10:.2f}  p50={p50:.2f}  p90={p90:.2f}")
    print(f"fraction of pairs with ref/target > 1.33 (like target=0.75×): "
          f"{(all_arr > 1.33).mean():.1%}")
    print(f"fraction of pairs with ref/target < 0.74 (like target=1.35×): "
          f"{(all_arr < 0.74).mean():.1%}")


if __name__ == "__main__":
    main()
