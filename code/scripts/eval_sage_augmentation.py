#!/usr/bin/env python
"""What SAGE-10k actually buys the asset bank (plan section 17, month 5).

The plan assigns SAGE a narrow role -- object diversity and open-vocabulary
augmentation, explicitly *not* room geometry.  The measurable form of that role
is retrieval coverage: how many reference objects have any substitution
candidate at all.  A category the bank has never seen cannot be swapped for a
better-sized asset no matter how good the optimiser is, so those objects fall
back to being rescaled or dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes


def coverage(bank, scenes):
    have = miss = 0
    missing = Counter()
    for s in scenes:
        for o in s.objects:
            if bank.has(o.category):
                have += 1
            else:
                miss += 1
                missing[o.category] += 1
    return have, miss, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--augmented", default="outputs/priors/assets_future_sage.pkl")
    ap.add_argument("--scenes", type=int, default=400)
    ap.add_argument("--out", default="outputs/sage_augmentation.json")
    a = ap.parse_args()

    scenes = list(iter_scenes(a.corpus, limit=a.scenes, min_objects=4))
    b1, b2 = AssetBank.load(a.base), AssetBank.load(a.augmented)
    h1, m1, miss1 = coverage(b1, scenes)
    h2, m2, miss2 = coverage(b2, scenes)
    gained = sorted(set(b2.categories()) - set(b1.categories()))

    out = {"n_scenes": len(scenes), "n_objects": h1 + m1,
           "base_assets": len(b1), "augmented_assets": len(b2),
           "base_coverage": h1 / max(h1 + m1, 1),
           "augmented_coverage": h2 / max(h2 + m2, 1),
           "new_categories": gained,
           "still_missing": dict(miss2.most_common(10))}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{out['n_objects']} reference objects over {len(scenes)} rooms")
    print(f"  bank alone      {len(b1):6d} assets -> {out['base_coverage']:.1%} covered")
    print(f"  + SAGE          {len(b2):6d} assets -> {out['augmented_coverage']:.1%} covered")
    print(f"  categories gained: {gained}")
    if miss2:
        print(f"  still uncovered: {dict(miss2.most_common(6))}")
    print("\n->", a.out)


if __name__ == "__main__":
    main()
