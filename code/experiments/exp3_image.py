#!/usr/bin/env python
"""Experiment 3 -- image-conditioned retargeting (plan section 14.3).

Measures the gap between

    I_r -> G^_r -> S_t          (38)   parsed source graph
    G^GT_r -> S_t               (39)   ground-truth source graph

so perception error and retargeting error stay separable.  The source graph is
degraded by a *calibrated* noise budget (detection, category, pose, size and
metric-scale error), which traces the whole degradation curve rather than a
single operating point; a real parser (MIDI) is then one measured point on that
curve, and is used automatically when its outputs are present on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from reroom.core.scene import scene_from_dict
from reroom.data.corpus import split_scenes
from reroom.eval.metrics import aggregate, evaluate
from reroom.geom.deform import deform_room
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.perception.base import (NoisyOracleParser, OracleParser,
                                    PerceptionNoise)
from reroom.perception.midi import GenReconAdapter, MIDIAdapter
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget
from experiments.common import load_corpus, table

# noise operating points, from "perfect" to "a bad day for a single-image parser"
LEVELS = {
    "oracle": None,
    "noise_light": dict(miss_rate=0.05, hallucination_rate=0.02,
                        category_error=0.05, translation_std=0.04,
                        yaw_std_deg=3.0, size_log_std=0.04,
                        yaw_flip_rate=0.01, room_scale_log_std=0.02),
    "noise_medium": dict(miss_rate=0.10, hallucination_rate=0.04,
                         category_error=0.10, translation_std=0.08,
                         yaw_std_deg=6.0, size_log_std=0.08,
                         yaw_flip_rate=0.04, room_scale_log_std=0.04),
    "noise_heavy": dict(miss_rate=0.20, hallucination_rate=0.08,
                        category_error=0.20, translation_std=0.16,
                        yaw_std_deg=12.0, size_log_std=0.15,
                        yaw_flip_rate=0.10, room_scale_log_std=0.08),
    "noise_severe": dict(miss_rate=0.32, hallucination_rate=0.14,
                         category_error=0.32, translation_std=0.28,
                         yaw_std_deg=20.0, size_log_std=0.22,
                         yaw_flip_rate=0.18, room_scale_log_std=0.12),
}

_CTX: dict = {}


def _init(ctx):
    global _CTX
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    _CTX = dict(ctx)
    p = ctx.get("elasticity_path")
    _CTX["el"] = load_elasticity(p) if p and os.path.exists(p) else None
    midi = ctx.get("midi_dir")
    _CTX["midi"] = MIDIAdapter(midi) if midi and os.path.isdir(midi) else None
    gr = ctx.get("genrecon_dir")
    _CTX["genrecon"] = (GenReconAdapter(gr) if gr and os.path.isdir(gr)
                        else None)


def _one(job):
    sd, seed, levels, cfg_kw = job
    truth = scene_from_dict(sd)
    rng = np.random.default_rng(seed)
    # the evaluation reference is always the *ground truth* design: the point is
    # how much of the true design intent survives a noisy reading of it
    gt_graph = build_motifs(build_scene_graph(truth))
    targets = [deform_room(truth.room, lvl, rng).room for lvl in (1, 2, 4)]
    rows = []
    for name in levels:
        cfgd = LEVELS.get(name)
        if name == "oracle":
            parsed = OracleParser().parse(truth)
        elif name in ("midi", "genrecon"):
            if _CTX.get(name) is None:
                continue
            try:
                parsed = _CTX[name].parse(truth.scene_id)
            except FileNotFoundError:
                continue
        else:
            nz = PerceptionNoise(seed=int(seed), **cfgd)
            parsed = NoisyOracleParser(nz).parse(truth)
        try:
            graph = build_motifs(build_scene_graph(parsed.scene))
        except Exception as exc:
            rows.append({"error": str(exc), "perception": name})
            continue
        for ti, room in enumerate(targets):
            cfg = RetargetConfig(**cfg_kw)
            # section 16.1 asks for the perception stage paired with the naive
            # coordinate map as well, so the table separates "the parse is bad"
            # from "the layout stage did nothing": a better parser cannot save
            # direct scaling, and direct scaling cannot spoil a good parse.
            for how in ("reroom", "direct"):
                try:
                    if how == "reroom":
                        out = retarget(graph, room, elasticity=_CTX["el"],
                                       cfg=cfg).scene
                    else:
                        out = run_baseline("direct_scaling", graph, room,
                                           cfg=cfg)
                    m = evaluate(gt_graph, out)
                except Exception as exc:
                    rows.append({"error": f"{type(exc).__name__}: {exc}",
                                 "perception": name, "solver": how})
                    continue
                m.update({"scene": truth.scene_id,
                          "method": name if how == "reroom" else f"{name}+direct",
                          "perception": name, "solver": how, "target": ti,
                          "n_parsed": len(parsed.scene.objects),
                          "n_true": len(truth.objects)})
                rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/exp3")
    ap.add_argument("--scenes", type=int, default=120)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--midi-dir", default="outputs/midi")
    ap.add_argument("--genrecon-dir", default="outputs/genrecon")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scene-ids", default=None,
                    help="restrict to these scene ids (one per line, or 'midi' "
                         "to use exactly the rooms a real parser was run on)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scenes = load_corpus(a.corpus, room_types=("bedroom", "living_room",
                                               "dining_room", "library"),
                         min_objects=6, max_objects=18)
    if a.scene_ids in ("midi", "genrecon", "parsers"):
        # a real parser only ran on the rooms it was given, so its row has to
        # be measured on exactly those rooms -- comparing it against an oracle
        # computed over a different sample would not be a comparison at all.
        # "parsers" is the intersection, the only sample on which MIDI and
        # GenRecon can be compared with each other.
        import glob

        def _ids(d):
            return {os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(d, "*.json"))
                    if not p.endswith("_conversion.json")}

        if a.scene_ids == "midi":
            ids = _ids(a.midi_dir)
        elif a.scene_ids == "genrecon":
            ids = _ids(a.genrecon_dir)
        else:
            ids = _ids(a.midi_dir) & _ids(a.genrecon_dir)
        test = [s for s in scenes if s.scene_id in ids]
        print(f"restricted to {len(test)} rooms with parser output", flush=True)
    elif a.scene_ids:
        ids = {l.strip() for l in open(a.scene_ids) if l.strip()}
        test = [s for s in scenes if s.scene_id in ids]
    else:
        _, _, test = split_scenes(scenes)
        test = test[:a.scenes]
    levels = (list(LEVELS)
              + (["midi"] if os.path.isdir(a.midi_dir) else [])
              + (["genrecon"] if os.path.isdir(a.genrecon_dir) else []))
    print(f"{len(test)} scenes, perception levels: {levels}", flush=True)

    ctx = {"elasticity_path": a.elasticity, "midi_dir": a.midi_dir,
           "genrecon_dir": a.genrecon_dir}
    cfg_kw = dict(restarts=16, grad_steps=200, proj_steps=90, device="cpu",
                  seed=a.seed)
    jobs = [(s.to_dict(), a.seed + k, levels, cfg_kw) for k, s in enumerate(test)]
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(ctx,)) as ex:
        for k, part in enumerate(ex.map(_one, jobs, chunksize=1)):
            rows.extend(part)
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}  rows={len(rows)}", flush=True)

    with open(os.path.join(a.out, "rows.json"), "w") as fh:
        json.dump(rows, fh)
    ok = [r for r in rows if "error" not in r]
    txt = table([r for r in ok if r.get("solver") == "reroom"],
                by="perception", order=levels,
                title="Experiment 3: retargeting quality vs source-perception quality")
    txt += "\n\n" + table([r for r in ok if r.get("solver") == "direct"],
                           by="perception", order=levels,
                           title="the same parses, retargeted by direct scaling (16.1)")
    base = aggregate([r for r in ok if r["perception"] == "oracle"
                      and r.get("solver") == "reroom"])
    lines = [txt, "", "gap to oracle (39) - (38):"]
    for name in levels:
        sub = aggregate([r for r in ok if r["perception"] == name
                         and r.get("solver") == "reroom"])
        if not sub or not base:
            continue
        lines.append(f"  {name:14s} dS_rel={sub['S_rel'] - base['S_rel']:+.4f}"
                     f"  dS_motif={sub['S_motif'] - base['S_motif']:+.4f}"
                     f"  dR_OOB={sub['R_OOB'] - base['R_OOB']:+.4f}"
                     f"  dlegality={sub['legality'] - base['legality']:+.4f}")
    lines += ["", "perception vs solver (16.1): what each stage is worth"]
    for name in levels:
        a_ = aggregate([r for r in ok if r["perception"] == name
                        and r.get("solver") == "reroom"])
        b_ = aggregate([r for r in ok if r["perception"] == name
                        and r.get("solver") == "direct"])
        if not a_ or not b_:
            continue
        lines.append(f"  {name:14s} reroom legality {a_['legality']:.3f} "
                     f"S_rel {a_['S_rel']:.3f}   |   direct legality "
                     f"{b_['legality']:.3f} S_rel {b_['S_rel']:.3f}")
    if "midi" not in levels:
        lines += ["", "MIDI not evaluated: no parser outputs found in "
                  f"{a.midi_dir}. Run MIDI on the reference images and write "
                  "one JSON per scene (see reroom/perception/midi.py) to place "
                  "a real parser on this curve."]
    out = "\n".join(lines)
    print("\n" + out)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    main()
