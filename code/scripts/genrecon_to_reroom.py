#!/usr/bin/env python
"""GenRecon reconstructions -> ReRoom design-intent graphs (plan section 3.3).

GenRecon is the plan's *multi-view* source parser: given several photographs of
a reference room it returns complete scene geometry, where MIDI returns
per-instance meshes from one image.  That difference is the whole problem here.
The reconstruction arrives as one fused mesh with no notion of "sofa" in it, and
a design-intent graph needs instances.

Labels are lifted from the multi-view instance masks that were rendered with
the input views: every reconstructed vertex is projected into each camera, a
cheap point-splat z-buffer decides whether it is the surface seen along that
ray, and the mask under that pixel casts one vote.  The majority label wins.
The 3D is therefore entirely GenRecon's; what is supplied is the segmentation,
exactly the concession made for MIDI in experiment three, and for the same
reason -- so the measurement isolates 3D reasoning from segmentation.

Each instance then gets an oriented footprint box (principal axis of its
points), a height from its own z-range, and an ``f^geo`` descriptor (eq. 10)
straight off its vertices.
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
from reroom.perception.geometry import shape_descriptor


def load_mesh_vertices(path: str, max_points: int, seed: int = 0):
    import trimesh

    m = trimesh.load(path, process=False)
    v = np.asarray(m.vertices, dtype=np.float64)
    if len(v) > max_points:
        rng = np.random.default_rng(seed)
        v = v[rng.choice(len(v), max_points, replace=False)]
    return v


def label_vertices(v: np.ndarray, frames: list, fov_x: float, root: str,
                   splat: int = 2, min_votes: int = 3,
                   majority: float = 0.6) -> np.ndarray:
    """Multi-view majority vote of instance labels onto the vertices."""
    from PIL import Image

    votes = None
    for fr in frames:
        seg_path = os.path.join(root, os.path.splitext(
            os.path.basename(fr["file_path"]))[0] + "_seg.png")
        if not os.path.exists(seg_path):
            continue
        seg = np.asarray(Image.open(seg_path))
        if seg.ndim == 3:
            seg = seg[..., 0]
        H, W = seg.shape
        if votes is None:
            votes = np.zeros((len(v), int(seg.max()) + 2), dtype=np.int32)

        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        p = (w2c[:3, :3] @ v.T).T + w2c[:3, 3]
        # the render convention is OpenGL: camera looks down -z, +y up
        z = -p[:, 2]
        ok = z > 1e-3
        f = 0.5 * W / math.tan(0.5 * fov_x)
        u = (p[:, 0] / np.maximum(z, 1e-6)) * f + 0.5 * W
        w = (-p[:, 1] / np.maximum(z, 1e-6)) * f + 0.5 * H
        ui, wi = np.round(u).astype(np.int64), np.round(w).astype(np.int64)
        ok &= (ui >= 0) & (ui < W) & (wi >= 0) & (wi < H)
        if not ok.any():
            continue

        # point-splat z-buffer: a vertex votes only if nothing of the same
        # reconstruction sits measurably in front of it along that ray
        idx = np.flatnonzero(ok)
        flat = wi[idx] * W + ui[idx]
        near = np.full(H * W, np.inf)
        np.minimum.at(near, flat, z[idx])
        if splat > 1:                       # dilate, so thin geometry occludes
            nb = near.reshape(H, W).copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = np.minimum(nb, np.roll(np.roll(near.reshape(H, W),
                                                        dy, 0), dx, 1))
            near = nb.ravel()
        visible = idx[z[idx] <= near[flat] * 1.02 + 0.02]
        lab = seg.ravel()[wi[visible] * W + ui[visible]].astype(np.int64)
        votes[visible, lab] += 1

    if votes is None:
        return np.zeros(len(v), dtype=np.int64)
    top = votes.max(1)
    best = votes.argmax(1)
    tot = votes.sum(1)
    # A single stray vote is not evidence.  Grazing rays let floor and wall
    # vertices slip through the point-splat z-buffer and land inside an
    # object's mask, and a thin sheet of those is enough to triple a footprint
    # while leaving the height untouched -- which is exactly what the first
    # run produced.  Demanding several views *and* a clear majority removes
    # them, because a mislabelled vertex rarely wins twice from different
    # angles.
    best[(top < min_votes) | (top < majority * np.maximum(tot, 1))] = 0
    return best


def _trim(pts: np.ndarray, voxel: float = 0.06,
          max_span: float = 3.0) -> np.ndarray:
    """Keep the largest spatially connected piece of a labelled point set.

    Stray votes do not arrive as a diffuse halo -- they arrive as a *sheet*:
    floor and wall vertices that a grazing ray slipped past the z-buffer.  A
    statistical trim is the wrong tool for that, because it treats a thin
    chair back as an outlier too; the first attempt shrank a 0.61 m chair to
    0.16 m while leaving a sideboard spanning the whole room.  What separates
    the object from the sheet is not distance from the median but *contact*,
    so the points are voxelised and only the largest connected component is
    kept.
    """
    from scipy import ndimage

    if len(pts) < 32:
        return pts
    lo = pts.min(0)
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    shape = idx.max(0) + 3
    if int(np.prod(shape.astype(np.float64))) > 40_000_000:
        return pts                       # too spread out to voxelise cheaply
    grid = np.zeros(shape, dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return pts
    comp = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    counts = np.bincount(comp, minlength=n + 1)
    counts[0] = 0
    # Largest is not always right.  When a few votes leak onto the room shell,
    # the shell is one enormous connected component and swallows the object --
    # which is how a sideboard came out 7 m wide.  So the largest component
    # that is still a plausible piece of furniture wins, and only if none is
    # does the overall largest get used.
    order = np.argsort(counts)[::-1]
    for c in order:
        if counts[c] == 0:
            break
        q = pts[comp == c]
        span = float(np.percentile(q[:, :2], 99.5, axis=0).max()
                     - np.percentile(q[:, :2], 0.5, axis=0).min())
        if span <= max_span and len(q) >= 32:
            return q
    keep = int(counts.argmax())
    mine = comp == keep
    return pts[mine] if mine.sum() >= 32 else pts


def oriented_box(pts: np.ndarray, pct: float = 0.5):
    """Footprint centre, extent and yaw from the principal axis, plus height."""
    pts = _trim(pts)
    xy = pts[:, :2]
    c = np.median(xy, axis=0)
    d = xy - c
    try:
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        ax = vt[0]
    except np.linalg.LinAlgError:
        ax = np.array([1.0, 0.0])
    ang = float(np.arctan2(ax[1], ax[0]))
    R = np.array([[math.cos(-ang), -math.sin(-ang)],
                  [math.sin(-ang), math.cos(-ang)]])
    loc = d @ R.T
    lo = np.percentile(loc, pct, axis=0)
    hi = np.percentile(loc, 100.0 - pct, axis=0)
    ctr = c + (np.linalg.inv(R.T) @ ((lo + hi) / 2.0))
    ext = hi - lo
    z0 = float(np.percentile(pts[:, 2], pct))
    z1 = float(np.percentile(pts[:, 2], 100.0 - pct))
    return ctr, ext, ang, z0, z1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, help="GenRecon output directory")
    ap.add_argument("--input", required=True, help="genrecon_input directory")
    ap.add_argument("--refs", required=True, help="outputs/references")
    ap.add_argument("--out", default="outputs/genrecon")
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument("--min-points", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    manifest = {m["sage_id"]: m for m in
                json.load(open(os.path.join(a.input, "manifest.json")))}
    rows = []
    for mesh_path in sorted(glob.glob(os.path.join(a.recon, "*", "mesh.ply"))):
        sid = os.path.basename(os.path.dirname(mesh_path))
        man = manifest.get(sid)
        if man is None:
            continue
        views = os.path.join(a.input, "renders_room", sid)
        tf = os.path.join(views, "transforms.json")
        if not os.path.exists(tf):
            continue
        meta = json.load(open(tf))
        ref_path = os.path.join(a.refs, man["scene_id"], "reference_scene.json")
        if not os.path.exists(ref_path):
            continue
        ref = Scene.load(ref_path)

        v = load_mesh_vertices(mesh_path, a.max_points)
        lab = label_vertices(v, meta["frames"], float(meta["camera_angle_x"]),
                             views)
        shift = np.asarray(man["shift"], dtype=float)

        # the render labels are 1..K in the order prepare_genrecon_input wrote
        oids = man["instance_oids"]
        by_oid = {o.oid: o for o in ref.objects}
        objs, errs, serrs = [], [], []
        for k, oid in enumerate(oids, start=1):
            pts = v[lab == k]
            if len(pts) < a.min_points or oid not in by_oid:
                continue
            ctr, ext, ang, z0, z1 = oriented_box(pts)
            pts = _trim(pts)
            xy = ctr - shift
            src = by_oid[oid]
            # resolve the 180-degree ambiguity of a principal axis the same way
            # the MIDI adapter does: face the interior
            yy = ang
            fwd = np.array([-math.sin(yy), math.cos(yy)])
            to_c = np.asarray(ref.room.polygon[:, :2].mean(0)) - xy
            if float(np.dot(fwd, to_c)) < 0:
                yy += math.pi
            size = np.array([max(ext[0], 0.05), max(ext[1], 0.05),
                             max(z1 - z0, 0.05)])
            geo = shape_descriptor(pts, yaw=ang,
                                   centre=np.array([ctr[0], ctr[1],
                                                    (z0 + z1) / 2.0]))
            objs.append(ObjectInstance(
                oid=oid, category=src.category, raw_category=src.raw_category,
                position=np.array([xy[0], xy[1], max(0.0, z0)]),
                yaw=float(yy), size=size,
                meta={"source": "genrecon", "score": 1.0,
                      "n_points": int(len(pts)), "shape": geo}))
            errs.append(float(np.linalg.norm(xy - src.xy)))
            serrs.append(float(np.abs(np.log(np.maximum(size, 1e-3))
                                      - np.log(np.maximum(src.size, 1e-3))).mean()))

        if not objs:
            print(f"  {sid}: no instances recovered", flush=True)
            continue
        out = Scene(scene_id=man["scene_id"], room=ref.room.copy(), objects=objs,
                    source="genrecon",
                    meta={"parser": "genrecon", "sage_id": sid,
                          "n_reference": len(ref.objects),
                          "n_reconstructed": len(objs),
                          "labelled_points": int((lab > 0).sum()),
                          "total_points": int(len(v))})
        rec = {"room": {"polygon": out.room.polygon.tolist(),
                        "height": out.room.height,
                        "room_type": out.room.room_type},
               "objects": [{"id": o.oid, "category": o.category,
                            "position": o.position.tolist(), "yaw": o.yaw,
                            "size": o.size.tolist(), "score": 1.0,
                            "shape": (np.asarray(o.meta["shape"]).tolist()
                                      if o.meta.get("shape") is not None
                                      else None)}
                           for o in out.objects],
               "meta": out.meta}
        with open(os.path.join(a.out, f"{man['scene_id']}.json"), "w") as fh:
            json.dump(rec, fh)
        rows.append({"scene_id": man["scene_id"], "sage_id": sid,
                     "n_ref": len(ref.objects), "n_pred": len(objs),
                     "centre_err_median": float(np.median(errs)),
                     "centre_err_mean": float(np.mean(errs)),
                     "log_size_err_mean": float(np.mean(serrs)),
                     "labelled_frac": float((lab > 0).mean())})
        print(f"  {sid:26s} {len(objs)}/{len(ref.objects)} objects  "
              f"centre err {np.median(errs):.2f} m  size err {np.mean(serrs):.3f}"
              f"  labelled {100 * (lab > 0).mean():.0f}%", flush=True)

    with open(os.path.join(a.out, "_conversion.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    if rows:
        print(f"\n{len(rows)} rooms -> {a.out}")
        print(f"  median centre error {np.median([r['centre_err_median'] for r in rows]):.3f} m")
        print(f"  mean log-size error {np.mean([r['log_size_err_mean'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
