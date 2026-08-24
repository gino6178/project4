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


def _slots_around(anchor, n_slots, sat_half=0.30):
    """Symmetric slots around the anchor.

    For ``n_slots`` satellites the layout is fixed by symmetry rather than
    greedy choice: chairs at a dining table go on the two long sides in equal
    numbers when the count is even, and one on a short side when it is odd.
    An earlier version handed out slots on all four sides in a fixed ratio,
    which put a lone chair at the head even when the two long sides could hold
    every chair symmetrically -- and, because the assignment was greedy, would
    leave one side of the table empty while piling three chairs on the other.
    """
    fwd = anchor.forward
    side = np.array([-fwd[1], fwd[0]])
    hx, hy = float(anchor.half[0]), float(anchor.half[1])
    # local axes: `fwd` is the anchor's own +y, so the *long* sides are the
    # ones normal to fwd if hx > hy, and normal to side otherwise
    if hx >= hy:
        long_axis, short_axis, hl, hs = side, fwd, hx, hy
    else:
        long_axis, short_axis, hl, hs = fwd, side, hy, hx
    r_long = hs + sat_half + 0.02
    r_short = hl + sat_half + 0.02

    slots = []
    if n_slots <= 0:
        return slots
    # split: as many pairs on the long sides as fit, then odd one at a short end
    n_pairs_long = n_slots // 2
    n_odd = n_slots % 2
    per_side = n_pairs_long                    # equal on both long sides
    # spacing along the long edge, symmetric around the anchor centre
    if per_side >= 1:
        span = 2.0 * hl - 0.1                  # small margin from the corners
        # evenly spaced positions from -span/2 to +span/2
        if per_side == 1:
            offsets = [0.0]
        else:
            offsets = list(np.linspace(-span / 2.0, span / 2.0, per_side))
        for sign in (+1, -1):
            for u in offsets:
                slots.append(anchor.xy + sign * long_axis * r_long
                             + u * short_axis)
    if n_odd == 1:
        # one satellite at the head of the table
        slots.append(anchor.xy + long_axis * 0.0 + short_axis * r_short)
    return slots


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

    # ---- companion snap: satellites take slots around their anchor ----
    if not (before.get("companion") is not None and before["companion"] >= 0.995):
     for sat, anchors, dmax in COMPANION_RULES:
        pool = by_cat.get(sat, [])
        if not pool:
            continue
        groups = {}
        for o in pool:
            partners = [a for c in anchors for a in by_cat.get(c, [])]
            if not partners:
                continue
            anchor = min(partners, key=lambda a: np.linalg.norm(o.xy - a.xy))
            groups.setdefault(id(anchor), [anchor, []])[1].append(o)
        for anchor, sats in groups.values():
            # symmetry means every satellite of this group participates in the
            # allocation, not just the ones outside the threshold: leaving one
            # chair in place and moving the others only inherits the reference's
            # asymmetry.  Skipping remains cheap when every chair is already at
            # its slot, because _try_move exits immediately when the target is
            # under 1 mm away.
            if all(float(np.linalg.norm(s.xy - anchor.xy)) <= dmax * 0.55
                   for s in sats) and len(sats) < 2:
                continue
            slots = _slots_around(anchor, len(sats))
            if not slots:
                continue
            # global-optimal assignment (Hungarian on centre-to-slot distance)
            from scipy.optimize import linear_sum_assignment
            cost = np.array([[float(np.linalg.norm(s.xy - slots[j]))
                              for j in range(len(slots))] for s in sats])
            rows, cols = linear_sum_assignment(cost)
            # Chairs get in each other's way: a march that stops when it hits
            # another chair is not the assignment we asked for.  Two passes
            # over the assignment fix it -- the first snaps chairs to the
            # nearest legal position on the way to their slot (usually far
            # enough away that another chair no longer blocks), the second
            # completes the reach from that position.  If a chair still
            # cannot land, its slot is empty rather than filled by the wrong
            # chair.
            for _ in range(2):
                for i in range(len(rows)):
                    _try_move(sats[rows[i]], slots[cols[i]], poly,
                              _blockers(scene, sats[rows[i]]))

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
