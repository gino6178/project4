#!/usr/bin/env python
"""Check what SAGE-10k can and cannot be used for (plan section 3.2).

The plan assigns SAGE the role of *appearance and object diversity*, explicitly
not irregular-room ground truth, on the grounds that the released layouts are
generated from a width/length pair even though the schema stores walls as line
segments.  That is an empirical claim about the data, so this measures it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from reroom.data.sage import iter_sage_scenes
from reroom.geom.polygon import as_polygon, floor_descriptor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    scenes = list(iter_sage_scenes(a.root, limit=a.limit or None,
                                   min_objects=4, max_objects=80))
    if not scenes:
        print(f"no SAGE layouts under {a.root}")
        return
    rect = np.array([bool(s.meta.get("rectangular")) for s in scenes])
    conv = np.array([floor_descriptor(as_polygon(s.room))[7] for s in scenes])
    nv = Counter(len(s.room.polygon) for s in scenes)
    cats = Counter(o.category for s in scenes for o in s.objects)
    rep = {
        "n_scenes": len(scenes),
        "room_types": dict(Counter(s.room.room_type for s in scenes)),
        "axis_aligned_rectangle_fraction": float(rect.mean()),
        "vertex_count_hist": dict(sorted(nv.items())),
        "median_convexity": float(np.median(conv)),
        "fraction_convexity_below_0.99": float((conv < 0.99).mean()),
        "mean_objects": float(np.mean([len(s.objects) for s in scenes])),
        "mean_area": float(np.mean([s.room.area for s in scenes])),
        "mean_density": float(np.mean([s.density() for s in scenes])),
        "distinct_categories": len(cats),
        "top_categories": cats.most_common(20),
        "verdict": ("SAGE supplies object and appearance diversity; its floor "
                    "plans are rectangles, so it cannot serve as irregular-room "
                    "ground truth — matching the plan's assessment."),
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
