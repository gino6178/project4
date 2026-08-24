#!/usr/bin/env python
"""Run MIDI-3D over the rendered reference rooms (plan section 6, 14.3).

Runs inside MIDI's own conda environment and depends on nothing from ReRoom, so
the two dependency stacks stay separate.  For each reference it writes the
per-instance geometry MIDI produced, reduced to what a source parser has to
hand downstream: an oriented footprint and a height per instance.  The
conversion into a ReRoom scene happens afterwards, in the ReRoom environment.

Usage (from the MIDI-3D checkout):
    python run_midi.py --refs outputs/references --out outputs/midi_raw
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def instance_summary(mesh) -> dict:
    """Reduce one generated instance to a pose-and-size description."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(v) < 4:
        return {}
    lo, hi = v.min(0), v.max(0)
    # MIDI works in a y-up, camera-anchored frame
    xz = v[:, [0, 2]]
    c = xz.mean(0)
    d = xz - c
    # principal axis of the footprint gives the oriented rectangle
    try:
        u, s_, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
        ax = vt[0]
    except np.linalg.LinAlgError:
        ax = np.array([1.0, 0.0])
    ang = float(np.arctan2(ax[1], ax[0]))
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    loc = d @ R.T
    ext = loc.max(0) - loc.min(0)
    ctr = c + (loc.max(0) + loc.min(0)) / 2.0 @ np.linalg.inv(R).T
    # f^geo of eq. (10): a real parser hands back a mesh, so the node can carry
    # a shape descriptor and not just a box.  MIDI is y-up, ReRoom z-up, and the
    # footprint angle is the object's own yaw -- undo both so the descriptor is
    # in the same canonical frame as the ones built from 3D-FUTURE meshes.
    try:
        from reroom.perception.geometry import descriptor_from_mesh
        vz = v[:, [0, 2, 1]]
        f = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, "faces") else None
        geo = descriptor_from_mesh(
            vz, f, centre=np.array([ctr[0], ctr[1], (lo[1] + hi[1]) / 2.0]),
            yaw=-ang).tolist()
    except Exception:
        geo = None
    return {
        "shape": geo,
        "aabb_min": lo.tolist(), "aabb_max": hi.tolist(),
        "footprint_centre_xz": [float(ctr[0]), float(ctr[1])],
        "footprint_extent": [float(ext[0]), float(ext[1])],
        "footprint_angle": ang,
        "height": float(hi[1] - lo[1]),
        "y_min": float(lo[1]), "y_max": float(hi[1]),
        "n_vertices": int(len(v)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--midi-root", default=".")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-meshes", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.midi_root))
    os.chdir(a.midi_root)
    import torch
    from scripts.inference_midi import prepare_pipeline, run_midi

    os.makedirs(a.out, exist_ok=True)
    refs = sorted(glob.glob(os.path.join(a.refs, "*", "meta.json")))
    if a.limit:
        refs = refs[:a.limit]
    print(f"{len(refs)} reference rooms", flush=True)

    pipe = prepare_pipeline("cuda", torch.bfloat16)
    done = 0
    for k, mp in enumerate(refs):
        d = os.path.dirname(mp)
        meta = json.load(open(mp))
        key = meta["scene_id"]
        dst = os.path.join(a.out, key)
        if os.path.exists(os.path.join(dst, "instances.json")):
            done += 1
            continue
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        try:
            scene = run_midi(pipe, os.path.join(d, meta["rgb"]),
                             os.path.join(d, meta["seg"]), a.seed,
                             num_inference_steps=a.steps,
                             guidance_scale=a.guidance)
        except Exception as exc:
            print(f"  [{k}] {key}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        geoms = list(scene.geometry.values())
        insts = []
        for i, m in enumerate(geoms):
            s = instance_summary(m)
            s["index"] = i
            s["label"] = i + 1          # split_rgb_mask sorts labels ascending
            insts.append(s)
            if a.save_meshes:
                m.export(os.path.join(dst, f"inst_{i:02d}.ply"))
        with open(os.path.join(dst, "instances.json"), "w") as fh:
            json.dump({"scene_id": key, "n_instances": len(insts),
                       "label_to_oid": meta["label_to_oid"],
                       "camera": meta["camera"], "yfov_deg": meta["yfov_deg"],
                       "instances": insts,
                       "steps": a.steps, "guidance": a.guidance,
                       "seconds": time.time() - t0}, fh, indent=1)
        done += 1
        print(f"  [{k+1}/{len(refs)}] {key}: {len(insts)} instances "
              f"in {time.time()-t0:.1f}s", flush=True)
    print(f"\n{done} scenes -> {a.out}")


if __name__ == "__main__":
    main()
