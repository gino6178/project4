#!/usr/bin/env python
"""Benchmark: for a set of test scenes, retarget to 0.75x/1.0x/1.35x and
measure mean/max wall-object float against the nearest wall.  Compare two
checkpoints side by side.
"""
from __future__ import annotations
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from reroom.core.scene import Room
from reroom.data.asset_bank import AssetBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import evaluate, preservation_metrics
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.geom.polygon import as_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import generative_retarget, load_flow
from reroom.intent.elasticity import load_elasticity

WALL_CATS = ("bookcase","wardrobe","tv_stand","tv","sofa","sofa_bed","bed",
             "double_bed","single_bed","desk","dressing_table","cabinet",
             "shelf","sideboard","console")

def sr(room, s):
    p = uniform_scale(room.polygon, s); a = _anchor_openings(room)
    return Room(polygon=p, height=room.height,
                openings=_replace_openings(p, a, len(room.polygon)),
                room_type=room.room_type)

def wall_gaps(scene):
    """Returns per-wall-affinity object: (gap_cm, skew_deg)."""
    poly = as_polygon(scene.room); ring = np.asarray(poly.exterior.coords)[:-1]
    edges = [(ring[i], ring[(i+1)%len(ring)]) for i in range(len(ring))]
    out_gap = []; out_skew = []
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        cor = o.corners(); best_gap = None; best_edge = None
        for a,b in edges:
            d = b-a; L = np.linalg.norm(d)+1e-9; t = d/L; n = np.array([-t[1],t[0]])
            rel = cor-a; perp = np.abs(rel@n); proj = rel@t
            ins = (proj>-0.05)&(proj<L+0.05)
            if not ins.any(): continue
            g = perp[ins].min()
            if best_gap is None or g<best_gap:
                best_gap = g; best_edge = (t, n)
        if best_gap is not None:
            # skew: angle between object's forward and wall normal
            fx = -np.sin(o.yaw); fy = np.cos(o.yaw)
            fwd = np.array([fx, fy])
            _, n = best_edge
            cos_a = min(1.0, abs(float(fwd @ n)))
            skew_deg = min(np.degrees(np.arccos(cos_a)),
                            np.degrees(np.arccos(cos_a)) - 90) if cos_a < 0.5 else np.degrees(np.arccos(cos_a))
            # simpler: skew = angle between fwd and n, taking min of that and (90-)
            skew_deg = np.degrees(np.arccos(cos_a))
            skew_deg = min(skew_deg, 90 - skew_deg)  # normalise to [0, 45]
            out_gap.append(best_gap); out_skew.append(skew_deg)
    if not out_gap:
        return np.array([0.0]), np.array([0.0])
    return np.array(out_gap), np.array(out_skew)


def motif_integrity(scene, ref_scene, ref_intent):
    """Motif Integrity Score: mean geometric distortion of Child objects
    relative to their Parent (head) across all motifs.  Returns (mean_err_cm,
    pass_rate).  Pass = child within 25cm of ref relative offset AND yaw
    error < 15°.  This is the strict test that catches "chairs escaping the
    dining table" while snap/S_rel look fine.
    """
    lut_out = {o.oid: o for o in scene.objects if o.keep}
    lut_ref = {o.oid: o for o in ref_scene.objects if o.keep}
    errs = []; passes = []
    for m in ref_intent.motifs:
        if len(m.members) < 2: continue
        head_oid = ref_scene.objects[m.members[0]].oid
        if head_oid not in lut_out or head_oid not in lut_ref: continue
        h_out = lut_out[head_oid]; h_ref = lut_ref[head_oid]
        for child_i in m.members[1:]:
            c_oid = ref_scene.objects[child_i].oid
            if c_oid not in lut_out or c_oid not in lut_ref: continue
            c_out = lut_out[c_oid]; c_ref = lut_ref[c_oid]
            off_out = c_out.xy - h_out.xy
            off_ref = c_ref.xy - h_ref.xy
            drift = float(np.linalg.norm(off_out - off_ref))
            # yaw error relative to parent
            dy_out = float(c_out.yaw - h_out.yaw)
            dy_ref = float(c_ref.yaw - h_ref.yaw)
            yaw_err = float(np.degrees(abs(np.arctan2(np.sin(dy_out - dy_ref),
                                                     np.cos(dy_out - dy_ref)))))
            errs.append(drift)
            passes.append(drift <= 0.25 and yaw_err <= 15.0)
    mean_err = float(np.mean(errs)) if errs else 0.0
    pass_rate = float(np.mean(passes)) if passes else 1.0
    return mean_err, pass_rate


ZONE_CATS = {
    "living": ("sofa", "sofa_bed", "coffee_table", "tv_stand", "tv",
               "armchair", "chaise", "l_shaped_sofa"),
    "dining": ("dining_table", "dining_chair", "chinese_dining_table",
               "sideboard"),
    "sleep":  ("bed", "double_bed", "single_bed", "nightstand",
               "wardrobe", "dressing_table", "kids_bed", "baby_bed"),
    "work":   ("desk", "office_desk", "office_chair", "bookcase"),
}


def zone_disentangle(scene):
    """Zone Disentanglement Score S_zone: 1 - (union-normalised overlap of
    living-zone convex hull with dining-zone convex hull, or living/sleep
    etc).  If any pair of zone hulls overlaps significantly, the score drops.
    Reports the mean over the (up to C(k,2)) present zone pairs.
    """
    from shapely.geometry import MultiPoint
    hulls = {}
    for zone, cats in ZONE_CATS.items():
        pts = []
        for o in scene.objects:
            if not o.keep or o.category not in cats: continue
            # use footprint corners for a fair hull
            for c in o.corners():
                pts.append((float(c[0]), float(c[1])))
        if len(pts) >= 3:
            hulls[zone] = MultiPoint(pts).convex_hull
    if len(hulls) < 2:
        return 1.0
    scores = []
    keys = list(hulls.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A = hulls[keys[i]]; B = hulls[keys[j]]
            if A.area < 1e-6 or B.area < 1e-6: continue
            inter = A.intersection(B).area
            union = A.union(B).area
            scores.append(1.0 - inter / max(union, 1e-6))
    return float(np.mean(scores)) if scores else 1.0


CORE_ANCHORS = {
    "living_room": ("sofa", "sofa_bed"),
    "bedroom":     ("bed", "double_bed", "single_bed"),
    "dining_room": ("dining_table",),
    "office":      ("desk", "office_desk"),
}


def core_recall(scene):
    """Core Object Recall R_core: fraction of expected anchor categories that
    are (a) present in scene AND (b) don't clip >5% overlap with another
    object.  For living_room this checks sofa; for bedroom, bed; etc.
    """
    rt = getattr(scene.room, "room_type", "") or ""
    expected = CORE_ANCHORS.get(rt, ())
    if not expected: return 1.0
    total = 0; ok = 0
    for cat in expected:
        anchors = [o for o in scene.objects if o.keep and o.category == cat]
        if not anchors:
            total += 1
            continue
        for a in anchors:
            total += 1
            a_area = float(a.footprint_area) + 1e-6
            # check overlap with any other keep object
            from shapely.geometry import Polygon as _Pg
            a_poly = _Pg(a.corners())
            max_overlap = 0.0
            for other in scene.objects:
                if not other.keep or other is a: continue
                try:
                    o_poly = _Pg(other.corners())
                    if not a_poly.is_valid or not o_poly.is_valid: continue
                    over = a_poly.intersection(o_poly).area / a_area
                    max_overlap = max(max_overlap, over)
                except Exception:
                    pass
            if max_overlap < 0.05:
                ok += 1
    return ok / max(total, 1)

def bench(flow_path, tag, scenes, cfg, el, bank):
    flow = load_flow(flow_path, device="cpu")
    all_gaps = {0.75: [], 1.0: [], 1.35: []}
    all_skew = {0.75: [], 1.0: [], 1.35: []}
    metrics = {s: {'OOB': [], 'col': [], 'Srel': [], 'motif_err': [],
                    'motif_pass': [], 'zone': [], 'core': []}
               for s in all_gaps}
    for idx, src in enumerate(scenes):
        g = build_motifs(build_scene_graph(src))
        for s in (0.75, 1.0, 1.35):
            try:
                room = sr(src.room, s)
                res = generative_retarget(flow, g, room, elasticity=el,
                                          bank=bank, cfg=cfg, k=16)
                out = res.scene
                gaps, skew = wall_gaps(out)
                m = evaluate(g, out); pm = preservation_metrics(g, out)
                mi_err, mi_pass = motif_integrity(out, src, res.intent)
                zs = zone_disentangle(out)
                cr = core_recall(out)
                all_gaps[s].extend(gaps.tolist())
                all_skew[s].extend(skew.tolist())
                metrics[s]['OOB'].append(m['R_OOB'])
                metrics[s]['col'].append(m['R_col'])
                metrics[s]['Srel'].append(pm['S_rel'])
                metrics[s]['motif_err'].append(mi_err)
                metrics[s]['motif_pass'].append(mi_pass)
                metrics[s]['zone'].append(zs)
                metrics[s]['core'].append(cr)
            except Exception as e:
                print(f"  scene {idx} @ {s}x: {e}")
    print(f"\n=== {tag} ===")
    for s in (0.75, 1.0, 1.35):
        gaps = np.array(all_gaps[s])
        skew_a = np.array(all_skew[s])
        M = metrics[s]
        snap = float(((gaps <= 0.08) & (skew_a <= 5.0)).mean()) if len(gaps) else 0.0
        motif_err_cm = 100 * float(np.mean(M['motif_err'])) if M['motif_err'] else 0.0
        motif_pass = float(np.mean(M['motif_pass'])) if M['motif_pass'] else 1.0
        zone = float(np.mean(M['zone'])) if M['zone'] else 1.0
        core = float(np.mean(M['core'])) if M['core'] else 1.0
        print(f"  {s:.2f}x: mean_float={100*gaps.mean():5.1f}  "
              f"OOB={100*np.mean(M['OOB']):.1f}%  "
              f"col={100*np.mean(M['col']):.1f}%  "
              f"Srel={np.mean(M['Srel']):.2f}  "
              f"snap={100*snap:2.0f}%  "
              f"motif_pass={100*motif_pass:2.0f}%  "
              f"zone={zone:.2f}  "
              f"core={100*core:.0f}%")
    return all_gaps, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--flows", nargs="+", required=True,
                    help="flow.pt paths, tagged as name=path")
    ap.add_argument("--n-scenes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    scenes = [s for s in iter_scenes(a.corpus, limit=None, min_objects=6)
              if s.room.room_type in ("bedroom","living_room")]
    _,_,test = split_scenes(scenes)
    rng = np.random.default_rng(a.seed)
    picks = rng.choice(len(test), size=min(a.n_scenes, len(test)), replace=False)
    subset = [test[int(i)] for i in picks]
    print(f"benching {len(subset)} scenes: {[i for i in picks]}")
    bank = AssetBank.load("outputs/priors/assets_future.pkl")
    el = load_elasticity("outputs/elasticity/neural.pt")
    cfg = RetargetConfig(restarts=24)
    for spec in a.flows:
        if "=" in spec:
            tag, path = spec.split("=", 1)
        else:
            tag, path = os.path.basename(os.path.dirname(spec)), spec
        bench(path, tag, subset, cfg, el, bank)

if __name__ == "__main__":
    main()
