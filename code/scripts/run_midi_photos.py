#!/usr/bin/env python
"""Run MIDI-3D on arbitrary reference photographs (plan section 14.4).

The experiment the plan actually wants is a real one: someone hands the system
a photograph of a room they like.  This runs MIDI over paired
``<name>_rgb.png`` / ``<name>_seg.png`` files, which is the format MIDI's own
released example data uses, so real captures (ScanNet, Matterport3D) can be fed
through the same source-parser path as the synthetic references.

Runs inside MIDI's environment; the conversion into a ReRoom scene happens
afterwards in the ReRoom environment, where there is no ground truth to lean on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, nargs="+",
                    help="directories holding <name>_rgb.png / <name>_seg.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--midi-root", default=".")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-meshes", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.midi_root))
    os.chdir(a.midi_root)
    import torch
    from PIL import Image
    from scripts.inference_midi import prepare_pipeline, run_midi
    # one definition, in this repo.  A copy of it living next to the MIDI
    # checkout silently kept the photo path on an older version that emitted
    # no f^geo, which is exactly the kind of drift a second copy invites.
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
    sys.path.insert(0, _HERE)
    from run_midi import instance_summary

    jobs = []
    for d in a.pairs:
        for rgb in sorted(glob.glob(os.path.join(d, "*_rgb.png"))):
            seg = rgb.replace("_rgb.png", "_seg.png")
            if os.path.exists(seg):
                jobs.append((os.path.basename(d), rgb, seg))
    print(f"{len(jobs)} reference photographs", flush=True)
    os.makedirs(a.out, exist_ok=True)
    pipe = prepare_pipeline("cuda", torch.bfloat16)

    for k, (src, rgb, seg) in enumerate(jobs):
        key = f"{src}__{os.path.basename(rgb)[:-8]}"
        dst = os.path.join(a.out, key)
        if os.path.exists(os.path.join(dst, "instances.json")):
            continue
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        try:
            scene = run_midi(pipe, rgb, seg, a.seed,
                             num_inference_steps=a.steps,
                             guidance_scale=a.guidance)
        except Exception as exc:
            print(f"  [{k}] {key}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        geoms = list(scene.geometry.values())
        labels = sorted(int(v) for v in
                        np.unique(np.array(Image.open(seg).convert("L"))) if v > 0)
        insts = []
        for i, m in enumerate(geoms):
            s = instance_summary(m)
            s["index"] = i
            s["label"] = labels[i] if i < len(labels) else i + 1
            insts.append(s)
            if a.save_meshes:
                m.export(os.path.join(dst, f"inst_{i:02d}.ply"))
        Image.open(rgb).convert("RGB").save(os.path.join(dst, "rgb.png"))
        Image.open(seg).convert("L").save(os.path.join(dst, "seg.png"))
        with open(os.path.join(dst, "instances.json"), "w") as fh:
            json.dump({"scene_id": key, "source_dataset": src,
                       "n_instances": len(insts), "instances": insts,
                       "rgb": os.path.abspath(rgb), "seg": os.path.abspath(seg),
                       "steps": a.steps, "guidance": a.guidance,
                       "seconds": time.time() - t0}, fh, indent=1)
        print(f"  [{k+1}/{len(jobs)}] {key}: {len(insts)} instances "
              f"in {time.time()-t0:.1f}s", flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
