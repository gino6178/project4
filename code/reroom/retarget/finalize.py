"""Rule-based finalisation: snap satellites toward their anchors and wall-lovers
toward their walls, subject to hard collision and boundary constraints.

The optimiser produces a physically clean layout that stays close to the
reference.  The reference itself, though, is not always the answer the user
wants -- dining chairs 1.6 m from the table because summarisation removed one
of them, a sofa 0.5 m from the wall because the target room is a hair wider
than the source.  A small deterministic pass at the end nudges the obvious
cases toward what a person would actually do, while refusing any move that
would violate the constraints the solver just satisfied.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point

from ..core.categories import prior
from ..core.scene import Scene
from ..eval.functional import COMPANION_RULES
from ..geom.polygon import as_polygon, object_polygon

__all__ = ["snap_functional"]

# a move is accepted only if the swept object does not intersect anything below
# 2 m in height, and the new footprint stays inside the room polygon
STEP = 0.04                   # metres per attempt
MAX_STEPS = 40


def _blockers(scene: Scene, ignore) -> list:
    return [object_polygon(o) for o in scene.objects
            if o.keep and o is not ignore and o.z < 1.6]


def _try_move(o, target_xy, room_poly, blockers, step=STEP,
              max_steps=MAX_STEPS):
    """March the object toward ``target_xy`` while it stays legal."""
    start = o.xy.copy()
    d = np.asarray(target_xy) - start
    dist = float(np.linalg.norm(d))
    if dist < 1e-3:
        return False
    direction = d / dist
    best = start.copy()
    for k in range(max_steps):
        step_xy = start + direction * min(step * (k + 1), dist)
        o.position[:2] = step_xy
        fp = object_polygon(o)
        if not room_poly.contains(fp):
            break
        hit = any(fp.intersects(b) and fp.intersection(b).area > 1e-4
                  for b in blockers)
        if hit:
            break
        best = step_xy
        if np.linalg.norm(step_xy - target_xy) < 1e-3:
            break
    o.position[:2] = best
    return not np.allclose(best, start)


def snap_functional(scene: Scene) -> Scene:
    """Companion-snap and wall-flush, both under hard constraints.

    Skipped per-rule when the current score is already high: the pass exists
    to raise a low value, not to make small improvements to a good one --
    every move is a chance to move the wrong thing, and the measurements
    showed that once past 0.95 the pass hurts as often as it helps.
    """
    from ..eval.functional import functional_score as _fs
    before = _fs(scene)
    poly = as_polygon(scene.room)
    kept = [o for o in scene.objects if o.keep]
    by_cat: dict[str, list] = {}
    for o in kept:
        by_cat.setdefault(o.category, []).append(o)

    # ---- companion snap: pull each satellite toward its nearest partner ----
    if not (before.get("companion") is not None and before["companion"] >= 0.995):
     for sat, anchors, dmax in COMPANION_RULES:
        for o in by_cat.get(sat, []):
            partners = [a for c in anchors for a in by_cat.get(c, [])]
            if not partners:
                continue
            anchor = min(partners, key=lambda a: np.linalg.norm(o.xy - a.xy))
            d = float(np.linalg.norm(o.xy - anchor.xy))
            if d <= dmax * 0.55:
                continue
            # aim just outside the anchor's footprint, on the line from the
            # anchor to the object -- this keeps chairs pointing at the table
            # rather than lining up on one side
            outward = (o.xy - anchor.xy) / max(d, 1e-6)
            want = np.asarray(anchor.xy) + outward * (
                float(anchor.half.max()) + float(o.half.max()) + 0.10)
            _try_move(o, want, poly, _blockers(scene, o))

    # ---- wall flush: pull wall-lovers to the nearest wall segment ----
    if before.get("wall") is not None and before["wall"] >= 0.995:
        return scene
    walls = scene.room.walls()
    for o in kept:
        pw = prior(o.category).wall
        if pw < 0.6 or o.z >= 1.4:
            continue
        fp = object_polygon(o)
        # nearest wall segment to the current footprint
        best_k, best_gap = -1, math.inf
        for k, (a, b) in enumerate(walls):
            g = fp.distance(LineString([a, b]))
            if g < best_gap:
                best_gap, best_k = g, k
        if best_k < 0 or best_gap < 0.03:
            continue
        a, b = walls[best_k]
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-6:
            continue
        t = seg / L
        n_in = np.array([-t[1], t[0]])
        # move the object along the inward normal by the gap
        want = np.asarray(o.xy) - n_in * best_gap
        _try_move(o, want, poly, _blockers(scene, o))
    return scene
