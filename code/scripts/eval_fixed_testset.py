#!/usr/bin/env python
"""Evaluate a flow checkpoint on the frozen fixed test set.

Headline number: mean position error (cm) of the model's placement against the
REAL human target layout on the held-out cross pairs.  Plus per-family
plausibility metrics.  Run the same command on any checkpoint to compare.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.core.scene import Room
from reroom.geom.polygon import as_polygon
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.data.asset_bank import AssetBank
from reroom.intent.elasticity import load_elasticity
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.optimizer import RetargetConfig
from reroom.eval.metrics import evaluate, preservation_metrics
from reroom.generative.sample import generative_retarget, load_flow
from reroom.generative.xscene import make_cross_pair_filtered, real_collisions
from bench_wall_float import wall_gaps, motif_integrity

TESTSET = "outputs/fixed_testset.json"


def procrustes_mae(P, Q):
    """Min mean ||R·P + t − Q|| over rotation R and translation t (rigid, no
    scale, no reflection).  Factors out only the global-frame ambiguity of
    cross-scene retargeting so the number reflects *relative arrangement*
    fidelity, not which way the whole layout happens to face.  Scale is NOT
    freed because both layouts live in the same metric target room.  P, Q are
    (n,2) in metres.  Returns metres."""
    if len(P) < 2:
        return float(np.mean(np.linalg.norm(P - Q, axis=1)))
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    aligned = (R @ Pc.T).T
    return float(np.mean(np.linalg.norm(aligned - Qc, axis=1)))


def sr(room, s):
    p = uniform_scale(room.polygon, s); a = _anchor_openings(room)
    return Room(polygon=p, height=room.height,
                openings=_replace_openings(p, a, len(room.polygon)),
                room_type=room.room_type)


def eval_cross(flow, ref, tgt, cfg, el, bank):
    trip = make_cross_pair_filtered(ref, tgt)
    if trip is None:
        return None
    _, troom, gt = trip
    g = build_motifs(build_scene_graph(ref))
    res = generative_retarget(flow, g, troom, elasticity=el, bank=bank, cfg=cfg, k=16)
    out = res.scene
    lut = {o.oid: o for o in out.objects}
    pos_err = []; yaw_err = []; P = []; Q = []
    for go in gt.objects:
        if not go.keep or go.oid not in lut:
            continue
        oo = lut[go.oid]
        pos_err.append(float(np.linalg.norm(oo.xy - go.xy)))
        dy = float(oo.yaw - go.yaw)
        yaw_err.append(abs(np.degrees(np.arctan2(np.sin(dy), np.cos(dy)))))
        P.append(oo.xy); Q.append(go.xy)
    if not pos_err:
        return None
    # ---- PDF metrics (the ones §15 actually asks for): intrinsic feasibility
    # of the output + preservation of the REFERENCE design intent.  None of
    # these need the human GT's absolute coordinates (§4: design intent is not
    # absolute coordinates).
    m = evaluate(g, out)
    pm = preservation_metrics(g, out)
    # ---- reference-only (§4 says NOT to optimise these): distance to one
    # human's specific layout.
    aligned = procrustes_mae(np.array(P), np.array(Q))
    coll_added = max(0, real_collisions(out) - real_collisions(gt))
    return {
        # PDF headline
        "S_rel": pm["S_rel"], "S_rel_elastic": pm["S_rel_elastic"],
        "S_motif": pm["S_motif"], "R_OOB": m["R_OOB"], "R_col": m["R_col"],
        # reference only
        "pos": np.array(pos_err), "yaw": np.array(yaw_err),
        "aligned": aligned, "coll_added": coll_added}


def eval_three_sizes(flow, ref, scales, cfg, el, bank):
    g = build_motifs(build_scene_graph(ref))
    rows = {}
    for s in scales:
        room = sr(ref.room, s)
        res = generative_retarget(flow, g, room, elasticity=el, bank=bank, cfg=cfg, k=16)
        out = res.scene
        gaps, skew = wall_gaps(out)
        m = evaluate(g, out)
        mi_err, mi_pass = motif_integrity(out, ref, res.intent)
        snap = float(((gaps <= 0.08) & (skew <= 5.0)).mean()) if len(gaps) else 0.0
        rows[s] = {"float": 100 * float(gaps.mean()), "snap": 100 * snap,
                   "OOB": 100 * m["R_OOB"], "col": 100 * m["R_col"],
                   "motif_pass": 100 * mi_pass}
    return rows


def bench(flow_path, tag, cases, by_id, cfg, el, bank):
    flow = load_flow(flow_path, device="cpu")
    all_pos = []; all_yaw = []; coll = []; aligned = []
    pdf = {k: [] for k in ("S_rel", "S_rel_elastic", "S_motif", "R_OOB", "R_col")}
    per_cross = []
    fwd_rows = {s: {k: [] for k in ("float", "snap", "OOB", "col", "motif_pass")}
                for s in (0.75, 1.0, 1.35)}
    for c in cases:
        if c["type"] == "cross":
            r = eval_cross(flow, by_id[c["ref_id"]], by_id[c["tgt_id"]], cfg, el, bank)
            if r is None:
                continue
            all_pos.append(r["pos"]); all_yaw.append(r["yaw"]); coll.append(r["coll_added"])
            aligned.append(r["aligned"])
            for k in pdf:
                pdf[k].append(r[k])
            per_cross.append((c["room_type"], c["area_ratio"],
                              100 * float(r["pos"].mean())))
        else:
            rows = eval_three_sizes(flow, by_id[c["ref_id"]], c["scales"], cfg, el, bank)
            for s, d in rows.items():
                for k in fwd_rows[s]:
                    fwd_rows[s][k].append(d[k])
    pos = np.concatenate(all_pos) if all_pos else np.array([0.0])
    yaw = np.concatenate(all_yaw) if all_yaw else np.array([0.0])
    def mean(k): return float(np.mean(pdf[k])) if pdf[k] else 0.0
    print(f"\n=== {tag} ===")
    print(f"CROSS ({len(per_cross)} pairs) -- PDF metrics (§15), higher S_* better, lower R_* better:")
    print(f"  S_rel        (relation preservation, eq42) : {mean('S_rel'):.3f}")
    print(f"  S_rel_elastic(elasticity-adjusted, §19)    : {mean('S_rel_elastic'):.3f}")
    print(f"  S_motif      (motif preservation, eq43)    : {mean('S_motif'):.3f}")
    print(f"  R_OOB        (out-of-bound ratio, eq40)    : {100*mean('R_OOB'):.1f}%")
    print(f"  R_col        (collision ratio, eq41)       : {100*mean('R_col'):.1f}%")
    print(f"  -- reference only (§4: NOT the objective) --")
    print(f"  position MAE vs one human layout : raw={100*pos.mean():.0f}cm  aligned={100*np.mean(aligned):.0f}cm  yaw={yaw.mean():.0f}deg")
    print(f"THREE-SIZES (physical plausibility):")
    for s in (0.75, 1.0, 1.35):
        d = fwd_rows[s]
        print(f"  {s:.2f}x: float={np.mean(d['float']):5.1f}cm  snap={np.mean(d['snap']):3.0f}%  "
              f"OOB={np.mean(d['OOB']):.1f}%  col={np.mean(d['col']):.1f}%  motif_pass={np.mean(d['motif_pass']):3.0f}%")
    return 100 * pos.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", nargs="+", required=True, help="tag=path or path")
    ap.add_argument("--testset", default=TESTSET)
    ap.add_argument("--corpus", default="",
                    help="override the corpus path stored in the testset JSON "
                         "(the JSON records the machine it was built on)")
    a = ap.parse_args()
    payload = json.load(open(a.testset))
    corpus = a.corpus or payload["corpus"]
    if not os.path.isdir(corpus):
        raise SystemExit(f"corpus not found: {corpus} (pass --corpus)")
    scenes = [s for s in iter_scenes(corpus, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    # Resolve frozen scene_ids against ALL scenes, not just the test split: the
    # set was selected from the held-out split at creation time, and building
    # the lookup from every scene makes evaluation robust to any split-boundary
    # shift between machines / corpus orderings.
    by_id = {s.scene_id: s for s in scenes}
    bank = AssetBank.load("outputs/priors/assets_future.pkl")
    el = load_elasticity("outputs/elasticity/neural.pt")
    cfg = RetargetConfig(restarts=24)
    print(f"fixed test set: {payload['n_cross']} cross + {payload['n_three_sizes_scenes']} three-sizes scenes")
    for spec in a.flows:
        tag, path = spec.split("=", 1) if "=" in spec else (os.path.basename(os.path.dirname(spec)), spec)
        bench(path, tag, payload["cases"], by_id, cfg, el, bank)


if __name__ == "__main__":
    main()
