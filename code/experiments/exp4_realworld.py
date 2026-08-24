#!/usr/bin/env python
"""Experiment 4 -- real reference rooms and the human study (section 14.4).

Each reference room gets the six prescribed target floors: original-like, 70 %
smaller, narrow, wide, L-shaped and slanted-wall.  The measured question is not
reconstruction accuracy -- it is whether a person still recognises the target as
carrying the reference room's design, which is what the two-question A/B study
of section 15.3 asks.

Reference cases come from ``--cases`` (a directory of ReRoom scene JSONs, e.g.
written by the MIDI adapter from photographed rooms).  With no such directory,
held-out 3D-FRONT rooms are used and the study still runs end to end; the report
says plainly which source was used, because "we photographed 30 showrooms" and
"we held out 30 designed rooms" are not the same claim.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from experiments.common import METHODS, load_corpus, table
from reroom.core.scene import Room, Scene
from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import split_scenes
from reroom.eval.metrics import evaluate
from reroom.eval.userstudy import Trial, build_study
from reroom.geom.deform import (aspect_deform, corner_cut, slant_wall,
                                uniform_scale, validate_polygon)
from reroom.geom.polygon import normalize_polygon
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.render.topdown import figure_comparison
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.populate import CooccurrenceModel


def six_targets(room: Room) -> list[tuple[str, Room]]:
    """The six target floors the plan prescribes, deterministically."""
    p = room.polygon
    out: list[tuple[str, np.ndarray]] = [
        ("original_like", uniform_scale(p, 1.0)),
        ("70pct_smaller", uniform_scale(p, float(np.sqrt(0.7)))),
        ("narrow", aspect_deform(p, 0.72, 1.25)),
        ("wide", aspect_deform(p, 1.30, 0.95)),
        ("l_shaped", corner_cut(p, 0, 0.42, 0.42, 0.0)),
        ("slanted_wall", slant_wall(p, 1, 0.30 * float(
            np.linalg.norm(p[2 % len(p)] - p[1 % len(p)])), "normal")),
    ]
    rooms = []
    for name, poly in out:
        poly = normalize_polygon(poly)
        if not validate_polygon(poly):
            poly = normalize_polygon(uniform_scale(p, 1.0))
        rooms.append((name, Room(polygon=poly, height=room.height,
                                 openings=[o.copy() for o in room.openings],
                                 room_type=room.room_type)))
    return rooms


def load_cases(path: str) -> list[Scene]:
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        try:
            out.append(Scene.load(f))
        except Exception as exc:
            print(f"  skipping {f}: {exc}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--cases", default=None,
                    help="directory of reference scene JSONs (real rooms)")
    ap.add_argument("--out", default="outputs/exp4")
    ap.add_argument("--n-cases", type=int, default=30)
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--bank", default="outputs/priors/assets.pkl")
    ap.add_argument("--cooc", default="outputs/priors/cooc.json")
    ap.add_argument("--compare", default="direct_scaling")
    ap.add_argument("--study-trials", type=int, default=24)
    ap.add_argument("--three-d", action="store_true")
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.cases and os.path.isdir(a.cases):
        cases = load_cases(a.cases)[:a.n_cases]
        provenance = f"user-supplied reference rooms from {a.cases}"
    else:
        scenes = load_corpus(a.corpus, room_types=("bedroom", "living_room",
                                                   "dining_room"),
                             min_objects=7, max_objects=16)
        _, _, test = split_scenes(scenes)
        cases = test[:a.n_cases]
        provenance = ("held-out 3D-FRONT rooms (no photographed reference rooms "
                      "were supplied; pass --cases to use real ones)")
    print(f"{len(cases)} reference cases: {provenance}", flush=True)

    el = load_elasticity(a.elasticity) if os.path.exists(a.elasticity) else None
    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None
    cooc = None
    if os.path.exists(a.cooc):
        from collections import Counter
        d = json.load(open(a.cooc))
        cooc = CooccurrenceModel(counts={k: Counter(v) for k, v in d["counts"].items()},
                                 sizes={k: np.asarray(v) for k, v in d["sizes"].items()},
                                 n_scenes=d["n_scenes"])

    cfg = RetargetConfig(restarts=a.restarts, device="cpu", seed=a.seed)
    rows, trials = [], []
    for ci, scene in enumerate(cases):
        graph = build_motifs(build_scene_graph(scene))
        panels = [("reference", scene)]
        for name, room in six_targets(scene.room):
            ours = retarget(graph, room, elasticity=el, bank=bank, cooc=cooc,
                            cfg=cfg).scene
            other = run_baseline(a.compare, graph, room, cfg=cfg)
            for tag, sc in (("reroom_full", ours), (a.compare, other)):
                m = evaluate(graph, sc, bank=bank)
                m.update({"scene": scene.scene_id, "method": tag,
                          "target": name, "case": ci})
                rows.append(m)
            panels.append((f"{name} / reroom", ours))
            empty = Scene(scene_id=f"{scene.scene_id}__{name}__empty",
                          room=room.copy(), objects=[], source="empty")
            trials.append(Trial(case_id=f"c{ci:03d}_{name}", reference=scene,
                                target_empty=empty, a=ours, b=other,
                                method_a="reroom_full", method_b=a.compare,
                                note=name))
        if ci < 4:
            figure_comparison(panels, os.path.join(a.out, f"case_{ci}.png"),
                              ncols=4, per_panel=3.0,
                              suptitle=f"case {ci}: six prescribed target floors")
        if ci % 5 == 0:
            print(f"  case {ci}/{len(cases)}", flush=True)

    with open(os.path.join(a.out, "rows.json"), "w") as fh:
        json.dump(rows, fh)
    from experiments.common import APPEARANCE_COLUMNS
    parts = [f"Experiment 4: reference cases -- {provenance}",
             table(rows, order=["reroom_full", a.compare],
                   columns=APPEARANCE_COLUMNS)]
    for name in [t[0] for t in six_targets(cases[0].room)]:
        sub = [r for r in rows if r.get("target") == name]
        parts.append(table(sub, order=["reroom_full", a.compare],
                           title=f"\ntarget floor: {name}"))
    txt = "\n\n".join(parts)
    print("\n" + txt)
    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(txt + "\n")

    rng = np.random.default_rng(a.seed)
    sel = list(rng.permutation(len(trials))[:a.study_trials])
    man = build_study([trials[i] for i in sel],
                      os.path.join(a.out, "study"),
                      title="ReRoom: reference-guided 3D scene retargeting",
                      three_d=a.three_d, seed=a.seed)
    print("\nuser study ->", man)


if __name__ == "__main__":
    main()
