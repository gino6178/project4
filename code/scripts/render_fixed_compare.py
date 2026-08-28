#!/usr/bin/env python
"""Render fixed-testset cross cases: ref | real GT | bnd200 | xscene, side by side."""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reroom.data.corpus import iter_scenes
from reroom.geom.polygon import as_polygon
from reroom.render.topdown import draw_scene
from reroom.data.asset_bank import AssetBank
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import generative_retarget, load_flow
from reroom.generative.xscene import make_cross_pair_filtered


def run(flow, ref, troom, cfg, el, bank):
    g = build_motifs(build_scene_graph(ref))
    return generative_retarget(flow, g, troom, elasticity=el, bank=bank, cfg=cfg, k=16).scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed")
    ap.add_argument("--testset", default="outputs/fixed_testset.json")
    ap.add_argument("--bnd", default="outputs/flow_bnd200/flow_best.pt")
    ap.add_argument("--xscene", default="outputs/flow_xscene/flow_best.pt")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="outputs/fixed_compare.png")
    a = ap.parse_args()
    payload = json.load(open(a.testset))
    by = {s.scene_id: s for s in iter_scenes(a.corpus, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")}
    cases = [c for c in payload["cases"] if c["type"] == "cross"][:a.n]
    bank = AssetBank.load("outputs/priors/assets_future.pkl")
    el = load_elasticity("outputs/elasticity/neural.pt")
    cfg = RetargetConfig(restarts=24)
    fb = load_flow(a.bnd, device="cpu"); fx = load_flow(a.xscene, device="cpu")

    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.6 * n))
    cols = ["ref (INPUT)", "real human GT", "bnd200", "xscene (new)"]
    ccol = ["#a8412a", "#2a7a2a", "#555", "#3060c0"]
    for r, c in enumerate(cases):
        ref = by[c["ref_id"]]; tgt = by[c["tgt_id"]]
        trip = make_cross_pair_filtered(ref, tgt)
        if trip is None:
            continue
        _, troom, gt = trip
        ob = run(fb, ref, troom, cfg, el, bank)
        ox = run(fx, ref, troom, cfg, el, bank)
        scenes = [ref, gt, ob, ox]
        rooms = [ref.room, troom, troom, troom]
        # unified extent per row
        W = H = 0
        for s, rm in zip(scenes, rooms):
            b = as_polygon(rm).bounds
            W = max(W, b[2]-b[0]); H = max(H, b[3]-b[1])
        for k in range(4):
            ax = axes[r][k] if n > 1 else axes[k]
            ax.axis("off")
            draw_scene(ax, scenes[k], labels=False, show_front=False)
            bb = as_polygon(rooms[k]).bounds
            cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            ax.set_xlim(cx-W/2*1.1, cx+W/2*1.1); ax.set_ylim(cy-H/2*1.1, cy+H/2*1.1)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(cols[k], fontsize=11, color=ccol[k], pad=6)
        aл = axes[r][0] if n > 1 else axes[0]
        aл.text(-0.08, 0.5, f"{c['room_type']}\n{c['area_ratio']:.2f}x",
                transform=aл.transAxes, fontsize=9, rotation=90, va="center", ha="center")
    fig.suptitle("Fixed test set — same reference retargeted to a real target boundary\n"
                 "real human GT vs bnd200 vs xscene (new hybrid-trained)", fontsize=13, y=1.005)
    plt.tight_layout()
    fig.savefig(a.out, dpi=115, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
