#!/usr/bin/env python
"""Render the curriculum-eval summary table + three-sizes bench as a chart."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CURRICULUM = {
    # cm
    "bnd200 (baseline)":       {"scramble": 41.73, "scale075": 4.64, "scale135": 5.02, "subst": 8.91},
    "p1 (+scramble 30 ep)":    {"scramble": 42.82, "scale075": 6.59, "scale135": 6.63, "subst": 10.31},
    "p2 (+wider L1, 30 ep)":   {"scramble": 42.90, "scale075": 6.59, "scale135": 6.67, "subst": 9.98},
    "p2b (lr=1e-5, 10 ep)":    {"scramble": 41.75, "scale075": 5.38, "scale135": 5.73, "subst": 9.35},
}

BENCH = {
    # cm (mean_float @ each size, S_rel)
    "flow_wall (old shipped)": {"0.75": 8.7,  "1.00": 14.4, "1.35": 20.0, "1.35_srel": 0.86},
    "bnd200 (new candidate)":  {"0.75": 7.9,  "1.00": 10.8, "1.35": 23.9, "1.35_srel": 0.87},
    "p2":                      {"0.75": 7.6,  "1.00": 11.6, "1.35": 22.9, "1.35_srel": 0.87},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: curriculum eval — 4 metric bars, 4 checkpoints
ax = axes[0]
metrics = ["scramble", "scale075", "scale135", "subst"]
metric_names = ["scramble\n(Phase 1 goal)", "scale 0.75×\n(Phase 2 goal)",
                "scale 1.35×\n(Phase 2 goal)", "subst\n(Phase 3 goal)"]
names = list(CURRICULUM.keys())
colors = ["#2a7a2a", "#c98b2c", "#1f78b4", "#a8412a"]
x = np.arange(len(metrics))
w = 0.2
for i, name in enumerate(names):
    v = [CURRICULUM[name][m] for m in metrics]
    b = ax.bar(x + i*w - 1.5*w, v, w, color=colors[i], alpha=0.85, label=name)
    for xi, vi in zip(x + i*w - 1.5*w, v):
        ax.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=9)
ax.set_ylabel("metric-space wall error (cm) · lower = better")
ax.set_title("Curriculum eval on 40 fixed val scenes (τ=0.9 one-step prediction)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 55)

# Right: 3-sizes bench on 6 real scenes
ax2 = axes[1]
sizes = ["0.75", "1.00", "1.35"]
bench_names = list(BENCH.keys())
bench_colors = ["#c1442e", "#1f78b4", "#2a7a2a"]
x2 = np.arange(len(sizes))
w2 = 0.28
for i, name in enumerate(bench_names):
    v = [BENCH[name][s] for s in sizes]
    ax2.bar(x2 + i*w2 - w2, v, w2, color=bench_colors[i], alpha=0.85, label=name)
    for xi, vi in zip(x2 + i*w2 - w2, v):
        ax2.text(xi, vi + 0.4, f"{vi:.1f}", ha="center", fontsize=9)
ax2.set_xticks(x2); ax2.set_xticklabels([f"{s}×" for s in sizes], fontsize=10)
ax2.set_ylabel("mean wall float (cm) · lower = better")
ax2.set_title("Three-sizes bench on 6 real scenes (shipped polish enabled)")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(alpha=0.3, axis="y")
ax2.set_ylim(0, 30)

fig.suptitle("Curriculum result summary: no fine-tune beat bnd200 across the board", fontsize=12)
plt.tight_layout()
out = "outputs/diag/curriculum_summary.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
