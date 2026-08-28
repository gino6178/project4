#!/usr/bin/env python
"""Measure OOB rate in training-pair *inputs*.

At inference the reference is a real scene: every object sits inside the
room polygon.  At training we manufacture the reference by warping / deforming
/ scrambling, so an object can end up outside its (pseudo-)ref room.  This is
a train/test distribution gap that wastes capacity teaching the model to
"push OOB things back in" -- a skill it never needs at inference.
"""
from __future__ import annotations
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shapely.geometry import Point as ShPoint, Polygon as ShPoly
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import deform_room
from reroom.geom.polygon import as_polygon
from reroom.generative.train import warp_scene


def obj_corners(o):
    hx, hy = o.size[0]/2, o.size[1]/2
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    rot = np.array([[c, -s], [s, c]])
    local = np.array([[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy]])
    return local @ rot.T + o.xy


def oob_stats(scene):
    poly = as_polygon(scene.room)
    counts = {"any_oob": 0, "center_oob": 0, "n": 0}
    for o in scene.objects:
        if not o.keep: continue
        counts["n"] += 1
        cs = obj_corners(o)
        oob_corners = sum(1 for c in cs if not poly.contains(ShPoint(*c)))
        if oob_corners > 0: counts["any_oob"] += 1
        cx, cy = o.xy
        if not poly.contains(ShPoint(cx, cy)): counts["center_oob"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/reroom/processed")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    scenes = [s for s in iter_scenes(a.corpus, min_objects=5)
              if len(s.objects) <= 24]
    train, _, _ = split_scenes(scenes)
    rng = np.random.default_rng(0)
    LEVELS = (1, 2, 3, 4, 5)

    # collect stats for (a) real scenes (inference-time input distribution),
    # (b) backward pair pseudo-ref (current default),
    # (c) backward + scramble,
    # (d) warp-based pair target GT (candidate)
    agg = {name: {"any_oob": 0, "center_oob": 0, "n": 0, "scenes": 0}
           for name in ("real", "backward", "scramble", "warp_target")}

    for _ in range(a.n):
        s = train[rng.integers(0, len(train))]
        # real scene input
        r = oob_stats(s)
        for k in ("any_oob", "center_oob", "n"): agg["real"][k] += r[k]
        agg["real"]["scenes"] += 1

        L = int(rng.choice(LEVELS))
        for _try in range(4):
            ref_room = deform_room(s.room, L, rng).room
            if as_polygon(ref_room).area > 3.0: break

        # (b) backward pseudo-ref = warp(scene, ref_room)
        pseudo = warp_scene(s, ref_room)
        r = oob_stats(pseudo)
        for k in ("any_oob", "center_oob", "n"): agg["backward"][k] += r[k]
        agg["backward"]["scenes"] += 1

        # (c) backward + scramble (rejection sample inside ref polygon)
        pseudo_scr = warp_scene(s, ref_room)
        poly = as_polygon(pseudo_scr.room)
        minx, miny, maxx, maxy = poly.bounds
        for o in pseudo_scr.objects:
            ok = False
            for _ in range(20):
                px = rng.uniform(minx, maxx); py = rng.uniform(miny, maxy)
                if poly.contains(ShPoint(px, py)):
                    o.xy = np.array([px, py]); ok = True; break
            o.yaw = float(rng.uniform(-math.pi, math.pi))
        r = oob_stats(pseudo_scr)
        for k in ("any_oob", "center_oob", "n"): agg["scramble"][k] += r[k]
        agg["scramble"]["scenes"] += 1

        # (d) warp-based pair TARGET GT = warp(scene, deformed_target)
        # (In this pair the reference is untouched real, so no OOB there;
        #  the *target* is what has potential OOB.)
        target_warp = warp_scene(s, ref_room)
        r = oob_stats(target_warp)
        for k in ("any_oob", "center_oob", "n"): agg["warp_target"][k] += r[k]
        agg["warp_target"]["scenes"] += 1

    print(f"{'source':>18}  {'objs':>6}  {'any-OOB obj':>11}  {'center-OOB obj':>14}  scenes")
    print("-" * 70)
    for name in ("real", "backward", "scramble", "warp_target"):
        st = agg[name]
        p_any = 100 * st["any_oob"] / max(st["n"], 1)
        p_c = 100 * st["center_oob"] / max(st["n"], 1)
        print(f"{name:>18}  {st['n']:>6}  {p_any:>9.1f}%  {p_c:>12.1f}%  {st['scenes']:>6}")

    print("\nreal-input baseline (inference distribution):",
          f"{100 * agg['real']['any_oob'] / max(agg['real']['n'], 1):.2f}% any-OOB")


if __name__ == "__main__":
    main()
