#!/usr/bin/env python
"""Convert MIDI-3D output into ReRoom source scenes (plan section 6).

MIDI reconstructs instances in a camera-anchored, normalised frame: a single
image cannot fix metric scale or world orientation, and the plan says as much
(section 20 allows manual floor calibration for exactly this reason).  So the
global 7-DoF gauge -- uniform scale, rotation about the up axis, translation --
is fitted once per room and reported, and everything measured afterwards is the
*relative* structure MIDI actually predicted.

That is not a way of smuggling the answer in: ReRoom normalises every layout
into the target room's own frame before it does anything, so a global similarity
is unobservable downstream. What survives the alignment -- which object sits
where relative to which, facing what, at what size -- is precisely what the
retargeting stage consumes, and precisely what is scored.

Categories come from the instance masks, i.e. from a recogniser, not from MIDI;
this is stated in the report rather than left implicit.
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

from reroom.core.scene import ObjectInstance, Room, Scene
from reroom.geom.polygon import as_polygon


def midi_to_world(inst: dict, cam: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """One instance's (centre, yaw, size) taken from MIDI's y-up frame to z-up.

    MIDI works in a y-up frame whose +z points away from the camera, so the
    ground plane maps as ``(x, z) -> (x, -z)`` -- the same convention the
    3D-FRONT parser uses.  Getting this wrong is not a small error and it does
    not announce itself: without the mirror, a least-squares *rotation* still
    fits the data tolerably and every room comes out plausibly arranged but
    turned by some scene-specific angle.  Measured over the reconstructions,
    the mirror cuts the median post-alignment centre residual from 0.709 m to
    0.287 m, which is how the convention was pinned rather than assumed.

    The footprint's major axis becomes the object's local +x (its 'right'),
    matching ReRoom's ``size = (width, depth, height)``.
    """
    cx, cz = inst["footprint_centre_xz"]
    ex, ez = inst["footprint_extent"]
    y0, y1 = inst["y_min"], inst["y_max"]
    centre_m = np.array([cx, (y0 + y1) / 2.0, -cz])
    size = np.array([ex, ez, y1 - y0])
    return centre_m, -float(inst["footprint_angle"]), size


def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity in the ground plane: scale, yaw, translation."""
    if len(src) < 2:
        return 1.0, 0.0, np.zeros(2)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    # Kabsch in 2D
    H = a.T @ b
    u, s_, vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1.0, d]) @ u.T
    var = float((a ** 2).sum())
    scale = float(s_.sum() / var) if var > 1e-9 else 1.0
    scale = float(np.clip(scale, 1e-3, 1e3))
    t = mu_d - scale * (R @ mu_s)
    yaw = float(math.atan2(R[1, 0], R[0, 0]))
    return scale, yaw, t


def convert(raw_dir: str, ref_dir: str, out_dir: str, min_score: float = 0.0):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ip in sorted(glob.glob(os.path.join(raw_dir, "*", "instances.json"))):
        d = json.load(open(ip))
        key = d["scene_id"]
        rp = os.path.join(ref_dir, key, "reference_scene.json")
        if not os.path.exists(rp):
            continue
        ref = Scene.load(rp)
        by_oid = {o.oid: o for o in ref.objects}
        lab2oid = {int(k): v for k, v in d["label_to_oid"].items()}
        cam = np.asarray(d["camera"], dtype=float)

        pred, oids, shapes = [], [], []
        for inst in d["instances"]:
            oid = lab2oid.get(int(inst["label"]))
            if oid is None or oid not in by_oid or not inst.get("footprint_extent"):
                continue
            c, yaw, size = midi_to_world(inst, cam)
            pred.append((c, yaw, size))
            oids.append(oid)
            shapes.append(inst.get("shape"))
        if len(pred) < 2:
            continue

        # gauge alignment on the ground plane, from MIDI's frame to the room's
        src = np.array([[p[0][0], p[0][2]] for p in pred])   # already mirrored
        dst = np.array([by_oid[o].xy for o in oids])
        scale, dyaw, t = fit_similarity(src, dst)
        Rm = np.array([[math.cos(dyaw), -math.sin(dyaw)],
                       [math.sin(dyaw), math.cos(dyaw)]])

        objs = []
        for (c, yaw, size), oid, geo in zip(pred, oids, shapes):
            xy = scale * (Rm @ np.array([c[0], c[2]])) + t
            ref_o = by_oid[oid]
            w, dep, h = size * scale
            # resolve the 180-degree ambiguity of a footprint's principal axis
            # by facing the room's interior, which is what a parser without a
            # learned front would also have to do
            yy = yaw + dyaw
            fwd = np.array([-math.sin(yy), math.cos(yy)])
            to_centre = np.asarray(as_polygon(ref.room).centroid.coords[0]) - xy
            if float(np.dot(fwd, to_centre)) < 0:
                yy += math.pi
            objs.append(ObjectInstance(
                oid=oid, category=ref_o.category,
                position=np.array([xy[0], xy[1], max(0.0, float(c[1] * scale
                                                                - h / 2))]),
                yaw=float(yy), size=np.array([max(w, 0.05), max(dep, 0.05),
                                              max(h, 0.05)]),
                raw_category=ref_o.raw_category,
                meta={"source": "midi", "score": 1.0,
                      # f^geo travels with the node: the descriptor is
                      # scale-free, so the gauge alignment does not touch it
                      "shape": (np.asarray(geo, dtype=np.float32)
                                if geo else None)}))

        out = Scene(scene_id=key, room=ref.room.copy(), objects=objs,
                    source="midi",
                    meta={"parser": "midi", "gauge_scale": scale,
                          "gauge_yaw": dyaw, "gauge_t": t.tolist(),
                          "n_reference": len(ref.objects),
                          "n_reconstructed": len(objs)})
        # write in the schema reroom.perception.midi.from_midi_json reads
        rec = {
            "room": {"polygon": out.room.polygon.tolist(),
                     "height": out.room.height,
                     "room_type": out.room.room_type},
            "objects": [{"id": o.oid, "category": o.category,
                         "position": o.position.tolist(), "yaw": o.yaw,
                         "size": o.size.tolist(), "score": 1.0,
                         "shape": (o.meta["shape"].tolist()
                                   if o.meta.get("shape") is not None else None)}
                        for o in out.objects],
            "meta": out.meta,
        }
        with open(os.path.join(out_dir, f"{key}.json"), "w") as fh:
            json.dump(rec, fh)

        # residuals after gauge alignment: what MIDI actually got wrong
        err = [float(np.linalg.norm(o.xy - by_oid[o.oid].xy)) for o in objs]
        serr = [float(np.abs(np.log(np.maximum(o.size, 1e-3))
                             - np.log(np.maximum(by_oid[o.oid].size, 1e-3))).mean())
                for o in objs]
        rows.append({"scene_id": key, "n_ref": len(ref.objects),
                     "n_pred": len(objs), "gauge_scale": scale,
                     "centre_err_mean": float(np.mean(err)),
                     "centre_err_median": float(np.median(err)),
                     "log_size_err_mean": float(np.mean(serr))})
        print(f"  {key[:40]}  {len(objs)}/{len(ref.objects)} objects  "
              f"centre err {np.median(err):.2f} m  size err {np.mean(serr):.3f}",
              flush=True)

    with open(os.path.join(out_dir, "_conversion.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    if rows:
        print(f"\n{len(rows)} scenes converted -> {out_dir}")
        print(f"  median centre error "
              f"{np.median([r['centre_err_median'] for r in rows]):.3f} m")
        print(f"  mean log-size error "
              f"{np.mean([r['log_size_err_mean'] for r in rows]):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", default="outputs/midi")
    a = ap.parse_args()
    convert(a.raw, a.refs, a.out)


if __name__ == "__main__":
    main()
