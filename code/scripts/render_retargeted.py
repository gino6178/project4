#!/usr/bin/env python
"""Render reference and retargeted rooms with their real materials.

Section 14.4 asks whether a person still recognises the target as carrying the
reference room's design.  That is a question about *pictures*, and top-down
boxes cannot answer it — so this renders the same 3D-FUTURE assets, moved to
their retargeted poses, inside a shell built for the new floor polygon.

One sheet per reference: the reference room, then each prescribed target floor,
with the same objects visibly rearranged rather than redrawn.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp4_realworld import six_targets
from reroom.core.scene import Scene
from reroom.data.asset_bank import AssetBank
from reroom.eval.metrics import evaluate
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.textured import (load_room_assets, render_room,
                                    render_scene_textured, repose_assets)
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.populate import CooccurrenceModel


def sheet(panels, path, cols=4, per=4.1, suptitle=None):
    n = len(panels)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(per * cols, per * rows * 0.86))
    axes = np.atleast_1d(axes).ravel()
    for k, (title, img, cap) in enumerate(panels):
        ax = axes[k]
        ax.imshow(img)
        ax.set_title(title, fontsize=11.5, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        if cap:
            ax.text(0.5, -0.045, cap, transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.6, color="#555555")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13.5, y=0.995)
    fig.subplots_adjust(hspace=0.22, wspace=0.05, top=0.90)
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="outputs/references")
    ap.add_argument("--front", required=True)
    ap.add_argument("--future", required=True, nargs="+")
    ap.add_argument("--out", default="outputs/render_sheets")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--width", type=int, default=860)
    ap.add_argument("--height", type=int, default=620)
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--restarts", type=int, default=28)
    ap.add_argument("--with-baseline", action="store_true",
                    help="also render direct scaling for every target")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    el = load_elasticity(a.elasticity) if os.path.exists(a.elasticity) else None
    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    cooc = None
    if os.path.exists(a.cooc):
        d = json.load(open(a.cooc))
        cooc = CooccurrenceModel(
            counts={k: Counter(v) for k, v in d["counts"].items()},
            sizes={k: np.asarray(v) for k, v in d["sizes"].items()},
            n_scenes=d["n_scenes"])

    made = []
    metas = sorted(glob.glob(os.path.join(a.refs, "*", "meta.json")))
    for mp in metas:
        if len(made) >= a.n:
            break
        d = os.path.dirname(mp)
        meta = json.load(open(mp))
        ref = Scene.load(os.path.join(d, "reference_scene.json"))
        assets = load_room_assets(os.path.join(a.front, f"{meta['house']}.json"),
                                  meta["room"], a.future,
                                  only_oids={o.oid for o in ref.objects})
        if assets is None or len(assets) < 4:
            continue
        assets.room = ref.room
        graph = build_motifs(build_scene_graph(ref))

        _, res = render_scene_textured(assets, ref.room, a.width, a.height)
        panels = [("reference", res.rgb,
                   f"{ref.room.area:.1f} m$^2$ · {len(ref.objects)} objects")]

        cfg = RetargetConfig(restarts=a.restarts, device="cpu", seed=a.seed)
        for name, room in six_targets(ref.room):
            out = retarget(graph, room, elasticity=el, bank=bank, cooc=cooc,
                           cfg=cfg).scene
            m = evaluate(graph, out)
            ra = repose_assets(assets, ref, out)
            _, r2 = render_scene_textured(ra, room, a.width, a.height)
            panels.append((f"{name} · ReRoom", r2.rgb,
                           f"{room.area:.1f} m$^2$ · "
                           f"{len([o for o in out.objects if o.keep])} objects · "
                           f"outside {m['R_OOB']:.1%}, overlap {m['R_col']:.1%}"))
            if a.with_baseline:
                bs = run_baseline("direct_scaling", graph, room, cfg=cfg)
                mb = evaluate(graph, bs)
                rb = repose_assets(assets, ref, bs)
                _, r3 = render_scene_textured(rb, room, a.width, a.height)
                panels.append((f"{name} · direct scaling", r3.rgb,
                               f"outside {mb['R_OOB']:.1%}, "
                               f"overlap {mb['R_col']:.1%}"))

        key = meta["scene_id"]
        p = os.path.join(a.out, f"{key}.png")
        sheet(panels, p, cols=(4 if not a.with_baseline else 4), per=4.1,
              suptitle=f"{ref.room.room_type.replace('_', ' ')} — reference and "
                       f"six target floors, same furniture")
        made.append(p)
        print("->", p, flush=True)
    print("\n".join(made))


if __name__ == "__main__":
    main()
