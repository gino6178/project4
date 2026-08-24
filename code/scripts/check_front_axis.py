#!/usr/bin/env python
"""Determine 3D-FUTURE's canonical front axis empirically.

Objects with a strong wall affinity (wardrobes, TV stands, beds, sideboards)
are placed with their *back* to the wall.  So the correct yaw offset is the one
under which those objects' forward vectors point away from the nearest wall.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from reroom.core.categories import prior
from reroom.data.corpus import iter_scenes
from reroom.geom.polygon import as_polygon

WALL_CATS = ("wardrobe", "tv_stand", "bookcase", "sideboard", "cabinet",
             "drawer_chest", "double_bed", "single_bed", "desk", "shelf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=1500)
    a = ap.parse_args()

    votes = Counter()
    n = 0
    for s in iter_scenes(a.corpus, limit=a.limit, min_objects=4):
        walls = s.room.walls()
        for o in s.objects:
            if o.category not in WALL_CATS or prior(o.category).wall < 0.7:
                continue
            # nearest wall and its inward normal
            best, bd, bn = None, 1e9, None
            for p, q in walls:
                d = q - p
                L = float(np.linalg.norm(d))
                if L < 1e-6:
                    continue
                t = d / L
                nn = np.array([-t[1], t[0]])
                s_ = float(np.clip(np.dot(o.xy - p, t) / L, 0.0, 1.0))
                proj = p + t * (s_ * L)
                dist = float(np.linalg.norm(o.xy - proj))
                if dist < bd:
                    best, bd, bn = (p, q), dist, nn
            if best is None or bd > 1.2:
                continue
            n += 1
            f = o.forward
            votes["as_is"] += 1 if float(np.dot(f, bn)) > 0 else 0
            votes["plus_pi"] += 1 if float(np.dot(-f, bn)) > 0 else 0
    print(f"{n} wall-adjacent objects")
    for k, v in votes.items():
        print(f"  yaw {k:8s}: forward points into the room {v / max(n, 1):.3f}")
    print("\n-> use", "yaw + pi" if votes["plus_pi"] > votes["as_is"] else "yaw as-is")


if __name__ == "__main__":
    main()
