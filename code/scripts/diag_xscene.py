#!/usr/bin/env python
"""Cross-Scene Pairing validation + visualization.

1. Builds the pair index over the training split, reports coverage & match
   quality, and compares intra-motif rigidity of the GT under three pairing
   strategies (cross-scene / motif-rigid warp / plain affine warp).
2. Renders example triplets: Reference (real) -> Target boundary -> GT (real).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.polygon import as_polygon
from reroom.geom.deform import deform_room
from reroom.render.topdown import draw_scene
from reroom.intent.relations import build_scene_graph
from reroom.intent.motifs import build_motifs
from reroom.generative.train import warp_scene
from reroom.generative.xscene import (build_pair_index, match_objects,
                                      make_cross_pair, motif_rigid_warp, jaccard,
                                      catset)

CORPUS = "/home/gino/data/reroom/processed"
OUT = "outputs/diag"


def motif_stretch(src_scene, gt_scene):
    """Mean |‖offset_gt‖ − ‖offset_src‖| over intra-motif child->head pairs,
    in cm.  Measures how much a pairing strategy stretches motif geometry."""
    g = build_motifs(build_scene_graph(src_scene))
    lut_gt = {o.oid: o for o in gt_scene.objects}
    lut_s = {o.oid: o for o in src_scene.objects}
    errs = []
    for m in g.motifs:
        if len(m.members) < 2: continue
        h = src_scene.objects[m.members[0]].oid
        if h not in lut_gt or h not in lut_s: continue
        for ci in m.members[1:]:
            c = src_scene.objects[ci].oid
            if c not in lut_gt or c not in lut_s: continue
            d_s = np.linalg.norm(lut_s[c].xy - lut_s[h].xy)
            d_g = np.linalg.norm(lut_gt[c].xy - lut_gt[h].xy)
            errs.append(abs(d_g - d_s))
    return float(np.mean(errs)) if errs else None


def main():
    os.makedirs(OUT, exist_ok=True)
    scenes = [s for s in iter_scenes(CORPUS, limit=None, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    tr, va, te = split_scenes(scenes)
    print(f"train={len(tr)}")

    idx = build_pair_index(tr, thresh=0.6, max_partners=16, seed=0)
    cov = len(idx) / len(tr)
    tot_partners = sum(len(v) for v in idx.values())
    print(f"pair-index: {len(idx)}/{len(tr)} scenes have >=1 partner "
          f"({100*cov:.1f}%)  mean_partners={tot_partners/max(len(idx),1):.1f}")

    # match quality + motif stretch comparison on a sample
    rng = np.random.default_rng(1)
    keys = list(idx.keys())
    picks = rng.choice(keys, size=min(300, len(keys)), replace=False)
    match_frac = []; xs_stretch = []; rigid_stretch = []; affine_stretch = []
    for a in picks:
        ref = tr[a]; b = int(rng.choice(idx[a])); tgt = tr[b]
        pairs = match_objects(ref, tgt)
        nkeep = sum(1 for o in ref.objects if o.keep)
        match_frac.append(len(pairs) / max(nkeep, 1))
        trip = make_cross_pair(ref, tgt)
        if trip is not None:
            _, _, gt = trip
            s = motif_stretch(ref, gt)
            if s is not None: xs_stretch.append(s)
        # motif-rigid warp vs plain affine warp on a deformed target
        rr = np.random.default_rng(int(a))
        dtgt = deform_room(ref.room, int(rr.integers(1, 6)), rr).room
        gt_rig = motif_rigid_warp(ref, dtgt)
        gt_aff = warp_scene(ref, dtgt)
        s1 = motif_stretch(ref, gt_rig); s2 = motif_stretch(ref, gt_aff)
        if s1 is not None: rigid_stretch.append(s1)
        if s2 is not None: affine_stretch.append(s2)
    print(f"\nmatch fraction (ref objs with GT counterpart): "
          f"mean={100*np.mean(match_frac):.0f}%  p10={100*np.quantile(match_frac,.1):.0f}%")
    print(f"\nintra-motif stretch (cm, lower=more rigid GT):")
    print(f"  cross-scene GT      : {100*np.mean(xs_stretch):5.1f}   (real target, not our concern to be 0)")
    print(f"  motif-rigid warp GT : {100*np.mean(rigid_stretch):5.1f}")
    print(f"  plain affine warp GT: {100*np.mean(affine_stretch):5.1f}   <- current pipeline")

    # ---- render example triplets ----
    show = rng.choice(keys, size=6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for k, a in enumerate(show):
        ref = tr[a]
        # pick the partner with highest jaccard for a clean example
        b = max(idx[a], key=lambda j: jaccard(catset(ref), catset(tr[j])))
        tgt = tr[b]
        trip = make_cross_pair(ref, tgt)
        if trip is None: continue
        _, troom, gt = trip
        jac = jaccard(catset(ref), catset(tgt))
        ax = axes[k // 3][k % 3]; ax.axis("off")
        # three panes: ref | empty target boundary | gt
        axl = ax.inset_axes([0.0, 0.0, 0.30, 1.0])
        axm = ax.inset_axes([0.35, 0.0, 0.30, 1.0])
        axr = ax.inset_axes([0.70, 0.0, 0.30, 1.0])
        draw_scene(axl, ref, labels=False, show_front=False)
        # empty target: draw gt scene's room only by drawing gt but hiding objs
        from reroom.core.scene import Scene as _S
        draw_scene(axm, _S(scene_id="pt", room=troom, objects=[]), labels=False, show_openings=True)
        draw_scene(axr, gt, labels=False, show_front=False)
        for axx, tt, col in ((axl, "S_ref (real INPUT)", "#a8412a"),
                             (axm, "target boundary P_t", "#555"),
                             (axr, "S_tgt real layout (GT)", "#2a7a2a")):
            b2 = as_polygon(gt.room if axx is not axl else ref.room).bounds
            axx.set_title(tt, fontsize=9, color=col, pad=4)
        ax.text(0.5, 1.06, f"{ref.room.room_type}  ·  Jaccard={jac:.2f}",
                transform=ax.transAxes, ha="center", fontsize=11, weight="bold")
    fig.suptitle("Cross-Scene Pairing — two real rooms, same type, Jaccard>0.6\n"
                 "reference design (left) retargeted to a different real room's boundary (mid); its real layout is the GT (right)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT, "xscene_pairs.png")
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
