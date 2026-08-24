#!/usr/bin/env python
"""PhyScene's generated layouts -> ReRoom scenes, for one common evaluator.

The point of this file is that a comparison is only a comparison when both
sides are measured by the same code.  PhyScene reports its own numbers; ReRoom
reports its own; putting the two published tables side by side compares two
different quantities on two different samples.  So PhyScene's generated
layouts are converted into ReRoom ``Scene`` objects here and scored by
``reroom.eval.physcene``, exactly like ReRoom's own outputs.

The conversion is mechanical.  PhyScene works in a y-up frame centred on the
room's floor-plan centroid, stores half-extents, and encodes yaw as a cosine
and sine pair; ReRoom is z-up, metres, full extents, yaw in radians.  Objects
are marked by an ``objectness`` score, negative meaning "this slot holds a real
object".
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.core.categories import canonical_category
from reroom.core.scene import ObjectInstance, Room, Scene
from reroom.geom.polygon import normalize_polygon


def floor_polygon(boxes_npz: str) -> tuple[np.ndarray, np.ndarray]:
    """Room outline in the same centred frame the layout uses."""
    d = np.load(boxes_npz, allow_pickle=True)
    v = np.asarray(d["floor_plan_vertices"], dtype=float)
    f = np.asarray(d["floor_plan_faces"], dtype=int)
    c = np.asarray(d["floor_plan_centroid"], dtype=float)
    v = v - c
    # y-up -> z-up: the floor lives in (x, z)
    pts = v[:, [0, 2]]
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    tris = [Polygon(pts[t]) for t in f if len(set(t)) == 3]
    tris = [t for t in tris if t.is_valid and t.area > 1e-9]
    if not tris:
        return None, c
    hull = unary_union(tris)
    if isinstance(hull, MultiPolygon):
        hull = max(hull.geoms, key=lambda g: g.area)
    hull = hull.simplify(0.02)
    return np.asarray(hull.exterior.coords)[:-1], c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="PhyScene livingroom.json")
    ap.add_argument("--cache", required=True,
                    help="PhyScene preprocessed_data/<RoomType> directory")
    ap.add_argument("--out", default="outputs/physcene_scenes")
    ap.add_argument("--room-type", default="living_room")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = json.load(open(a.json))
    labels = np.asarray(d["class_labels"])
    trans = np.asarray(d["translations"])
    sizes = np.asarray(d["sizes"])
    angles = np.asarray(d["angles"])
    objness = np.asarray(d["objectness"])
    ids = list(d["scene_ids"])

    stats = json.load(open(os.path.join(a.cache, "dataset_stats.txt")))
    classes = stats["class_labels"]

    # the cached rooms are keyed by full scene uid; the JSON keeps only the
    # room part, so index the cache by its suffix
    by_suffix = {}
    for p in glob.glob(os.path.join(a.cache, "*", "boxes.npz")):
        by_suffix.setdefault(os.path.basename(os.path.dirname(p)).split("_")[-1], p)

    n_written = 0
    for k, sid in enumerate(ids):
        box_p = by_suffix.get(sid)
        if box_p is None:
            continue
        poly, _ = floor_polygon(box_p)
        if poly is None or len(poly) < 3:
            continue
        room = Room(polygon=normalize_polygon(np.asarray(poly, dtype=float)),
                    height=2.6, room_type=a.room_type)

        objs = []
        for j in range(labels.shape[1]):
            if float(objness[k, j, 0]) >= 0:      # empty slot
                continue
            ci = int(np.argmax(labels[k, j]))
            name = classes[ci] if ci < len(classes) else "misc"
            if name in ("empty", "start", "end"):
                continue
            t = trans[k, j]
            s = np.abs(sizes[k, j]) * 2.0         # half-extents -> full
            ang = angles[k, j]
            yaw = float(math.atan2(ang[1], ang[0])) if ang.shape[0] > 1 \
                else float(ang[0])
            objs.append(ObjectInstance(
                oid=f"ps_{k}_{j}", category=canonical_category(name, ""),
                raw_category=name,
                # y-up -> z-up, and the stored translation is the box centre
                position=np.array([t[0], t[2], max(0.0, t[1] - s[1] / 2)]),
                yaw=yaw,
                size=np.array([max(s[0], 0.05), max(s[2], 0.05),
                               max(s[1], 0.05)]),
                meta={"source": "physcene"}))
        if not objs:
            continue
        sc = Scene(scene_id=f"physcene__{sid}__{k}", room=room, objects=objs,
                   source="physcene",
                   meta={"method": "PhyScene", "room": sid})
        sc.save(os.path.join(a.out, f"{sc.scene_id}.json"))
        n_written += 1

    print(f"{n_written}/{len(ids)} PhyScene scenes -> {a.out}")


if __name__ == "__main__":
    main()
