#!/usr/bin/env python
"""Corpus statistics: how irregular are 3D-FRONT floor plans, really?

The plan (section 3.1, following DeBaRA) warns that the *benchmark subsets*
usually taken from 3D-FRONT are dominated by simple rectangles.  That is true
of the preprocessed subsets -- but only because the preprocessing replaces the
floor with its bounding box.  Parsing the raw ``Floor`` meshes keeps the real
polygon, and this script quantifies what that recovers.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from reroom.data.corpus import iter_scenes
from reroom.geom.polygon import as_polygon, floor_descriptor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    cats = Counter()
    rts = Counter()
    for s in iter_scenes(a.corpus, limit=a.limit or None):
        poly = as_polygon(s.room)
        g = floor_descriptor(poly)
        rows.append({
            "room_type": s.room.room_type, "area": float(poly.area),
            "n_vertices": len(s.room.polygon),
            "aspect": float(g[6]), "convexity": float(g[7]),
            "rect_fill": float(g[8]), "reflex_frac": float(g[10]),
            "n_objects": len(s.objects), "density": s.density(),
            "n_openings": len(s.room.openings),
        })
        rts[s.room.room_type] += 1
        for o in s.objects:
            cats[o.category] += 1

    arr = {k: np.array([r[k] for r in rows], dtype=float)
           for k in rows[0] if k != "room_type"}
    n = len(rows)

    def frac(mask):
        return float(np.mean(mask))

    rect = (arr["n_vertices"] <= 4) & (arr["rect_fill"] > 0.97)
    near_rect = arr["rect_fill"] > 0.95
    concave = arr["reflex_frac"] > 1e-6
    strong_concave = arr["convexity"] < 0.92

    report = {
        "n_scenes": n,
        "room_types": dict(rts),
        "floor_shape": {
            "exact_rectangle": frac(rect),
            "near_rectangular_rect_fill>0.95": frac(near_rect),
            "has_reflex_vertex": frac(concave),
            "convexity<0.92": frac(strong_concave),
            "n_vertices_hist": dict(sorted(Counter(
                arr["n_vertices"].astype(int).tolist()).items())),
            "median_convexity": float(np.median(arr["convexity"])),
            "median_rect_fill": float(np.median(arr["rect_fill"])),
        },
        "scale": {
            "area_mean": float(arr["area"].mean()),
            "area_p10": float(np.percentile(arr["area"], 10)),
            "area_p90": float(np.percentile(arr["area"], 90)),
            "aspect_mean": float(arr["aspect"].mean()),
            "objects_mean": float(arr["n_objects"].mean()),
            "density_mean": float(arr["density"].mean()),
            "density_p90": float(np.percentile(arr["density"], 90)),
            "openings_mean": float(arr["n_openings"].mean()),
        },
        "top_categories": cats.most_common(30),
    }
    print(json.dumps(report, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"report": report, "rows": rows}, fh)


if __name__ == "__main__":
    main()
