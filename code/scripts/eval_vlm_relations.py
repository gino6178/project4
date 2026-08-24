#!/usr/bin/env python
"""Measure the VLM relation extractor of section 20 against the geometric rules.

The plan lists "LLM/VLM relation extraction unstable" as a risk and mitigates
it by keeping the VLM in a supplementary role.  This turns that judgement into
a number: over the rendered reference rooms, how often does a CLIP-scored
semantic relation agree with the deterministic one, how many does it invent,
and how many real ones does it miss.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.core.scene import Scene
from reroom.eval.appearance import ClipEncoder
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.vlm import (augment_with_vlm, extract_semantic_relations,
                               vlm_agreement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="outputs/references")
    ap.add_argument("--out", default="outputs/vlm_relations.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-margin", type=float, default=0.04)
    a = ap.parse_args()

    enc = ClipEncoder(device=a.device)
    if not enc.ok:
        raise SystemExit("section 20 needs a CLIP backend")

    rows = []
    for mp in sorted(glob.glob(os.path.join(a.refs, "*", "meta.json"))):
        d = os.path.dirname(mp)
        meta = json.load(open(mp))
        sp = os.path.join(d, "reference_scene.json")
        if not os.path.exists(sp):
            continue
        scene = Scene.load(sp)
        try:
            graph = build_motifs(build_scene_graph(scene))
        except Exception:
            continue
        vl = extract_semantic_relations(
            scene, os.path.join(d, meta["rgb"]), os.path.join(d, meta["seg"]),
            dict(meta["label_to_oid"]), encoder=enc, min_margin=a.min_margin)
        r = vlm_agreement(graph, vl)
        r["scene"] = meta["scene_id"]
        r["n_visible"] = int(meta.get("n_visible", 0))
        rows.append(r)
        by_kind = {}
        for v in vl:
            by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        r["by_kind"] = by_kind
        print(f"  {meta['scene_id'][:44]:44s} geo={r['n_geometric']:3d} "
              f"vlm={r['n_vlm']:3d} overlap={r['n_overlap']:3d} {by_kind}",
              flush=True)

    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)

    tot_g = sum(r["n_geometric"] for r in rows)
    tot_v = sum(r["n_vlm"] for r in rows)
    tot_o = sum(r["n_overlap"] for r in rows)
    print(f"\n{len(rows)} rooms")
    print(f"  geometric semantic relations : {tot_g}")
    print(f"  VLM-proposed relations       : {tot_v}")
    print(f"  proposed *and* geometric     : {tot_o}")
    if tot_v:
        print(f"  precision vs geometry        : {tot_o / tot_v:.3f}")
    if tot_g:
        print(f"  recall vs geometry           : {tot_o / tot_g:.3f}")
    novel = tot_v - tot_o
    print(f"  relations geometry did not have: {novel}"
          "   <- what the VLM is actually for")
    print("\n->", a.out)


if __name__ == "__main__":
    main()
