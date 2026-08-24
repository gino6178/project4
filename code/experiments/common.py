"""Shared machinery for the four experiments of plan section 14."""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reroom.core.scene import Scene, scene_from_dict
from reroom.data.asset_bank import AssetBank, StatisticalBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import aggregate, evaluate
from reroom.geom.deform import LEVEL_NAMES, deform_room
from reroom.geom.polygon import as_polygon, floor_descriptor
from reroom.intent.elasticity import (ElasticityModel, PriorElasticity,
                                      StatElasticity, load_elasticity)
from reroom.intent.motifs import build_motifs, strip_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.baselines import run_baseline
from reroom.retarget.energy import EnergyWeights
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.populate import CooccurrenceModel

__all__ = ["MethodSpec", "METHODS", "make_targets", "run_grid", "table",
           "save_rows", "load_corpus"]


@dataclass
class MethodSpec:
    """One row of an experiment table."""

    name: str
    kind: str = "reroom"              # 'reroom' | 'baseline' | 'flow'
    baseline: str = ""
    overrides: dict = field(default_factory=dict)
    elasticity: str = "fitted"        # 'fitted' | 'prior' | 'none'
    use_bank: bool = False
    no_motifs: bool = False           # run on the motif-free graph (16.2)
    retrieval: dict = field(default_factory=dict)   # lambda_f / lambda_s / lambda_g
    note: str = ""


# The five variants of experiment one (section 14.1), plus the two pure
# coordinate maps they are meant to beat.
METHODS: dict[str, MethodSpec] = {
    "source_reference": MethodSpec(
        "source_reference", kind="baseline", baseline="source_reference",
        note="reference design in its own room (reference point, not a method)"),
    "direct_scaling": MethodSpec(
        "direct_scaling", kind="baseline", baseline="direct_scaling",
        note="normalised-coordinate scaling"),
    "affine_fit": MethodSpec(
        "affine_fit", kind="baseline", baseline="affine_fit",
        note="best-fit similarity between room frames"),
    "reference_rigid": MethodSpec(
        "reference_rigid", kind="baseline", baseline="reference_rigid",
        note="copy the reference layout unchanged"),
    "target_only": MethodSpec(
        "target_only", kind="baseline", baseline="target_only",
        note="floor-plan-conditioned synthesis, reference ignored"),
    "relation_only": MethodSpec(
        "relation_only", overrides=dict(allow_removal=False, allow_addition=False,
                                        allow_substitution=False),
        note="relation-aware optimisation"),
    "relation_summary": MethodSpec(
        "relation_summary", overrides=dict(allow_removal=True, allow_addition=True,
                                           allow_substitution=False),
        note="+ motif summarisation and population"),
    "reroom_full": MethodSpec(
        "reroom_full", overrides=dict(allow_removal=True, allow_addition=True,
                                      allow_substitution=True),
        use_bank=True, note="+ style-aware asset substitution"),
    # ablations (section 16.2)
    "no_elasticity": MethodSpec(
        "no_elasticity", overrides=dict(allow_removal=True, allow_addition=True,
                                        allow_substitution=True),
        elasticity="none", use_bank=True, note="alpha := 0 (rigid relations)"),
    "prior_elasticity": MethodSpec(
        "prior_elasticity", overrides=dict(allow_removal=True, allow_addition=True,
                                           allow_substitution=True),
        elasticity="prior", use_bank=True, note="hand-specified alpha"),
    "no_motif_grouping": MethodSpec(
        "no_motif_grouping", overrides=dict(use_motif_init=False,
                                            allow_removal=True, allow_addition=True,
                                            allow_substitution=True),
        use_bank=True, no_motifs=True,
        note="no motif layer at all: flat objects, flat selection"),
    "size_only_retrieval": MethodSpec(
        "size_only_retrieval", overrides=dict(allow_removal=True, allow_addition=True,
                                              allow_substitution=True),
        use_bank=True, retrieval=dict(lambda_f=0.0, lambda_g=0.0),
        note="substitution by size alone (lambda_f = 0)"),
    "no_motif_init": MethodSpec(
        "no_motif_init", overrides=dict(use_motif_init=False, allow_removal=True,
                                        allow_addition=True, allow_substitution=True),
        use_bank=True, note="no motif-rigid initialisation"),
    "no_projection": MethodSpec(
        "no_projection", overrides=dict(proj_steps=0, projection_scale=1.0,
                                        allow_removal=True, allow_addition=True,
                                        allow_substitution=True),
        use_bank=True, note="no feasibility projection"),
    "flow": MethodSpec("flow", kind="flow", use_bank=True,
                       note="flow-matching proposal + constraint projection"),
    "flow_no_projection": MethodSpec(
        "flow_no_projection", kind="flow", use_bank=True,
        overrides=dict(no_project=True),
        note="flow-matching proposal, unprojected"),
}


class _ZeroElasticity(ElasticityModel):
    """alpha = 0 everywhere: relations are copied rigidly."""

    def alpha(self, ctx) -> float:
        return 0.0


def load_corpus(root: str, room_types=None, limit: int | None = None,
                min_objects: int = 5, max_objects: int = 24):
    scenes = [s for s in iter_scenes(root, room_types=room_types,
                                     limit=None, min_objects=min_objects)
              if len(s.objects) <= max_objects]
    return scenes


def make_targets(scene: Scene, levels=(1, 2, 3, 4, 5), per_level: int = 1,
                 seed: int = 0):
    """The five floor-geometry difficulty groups of section 14.2."""
    rng = np.random.default_rng(seed)
    out = []
    for lvl in levels:
        for r in range(per_level):
            res = deform_room(scene.room, lvl, rng)
            out.append({"level": lvl, "name": LEVEL_NAMES[lvl], "rep": r,
                        "room": res.room, "spec": res.spec.describe(),
                        "area_ratio": res.room.area / max(scene.room.area, 1e-6)})
    return out


# ---------------------------------------------------------------- worker ---
_CTX: dict = {}


def _init(ctx: dict):
    global _CTX
    # each worker solves small problems; letting torch grab every core inside
    # every worker oversubscribes the machine by an order of magnitude
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _CTX = dict(ctx)
    el = ctx.get("elasticity_path")
    _CTX["fitted"] = load_elasticity(el) if el else PriorElasticity()
    _CTX["prior"] = PriorElasticity()
    _CTX["none"] = _ZeroElasticity()
    bank_path = ctx.get("bank_path")
    _CTX["bank"] = AssetBank.load(bank_path) if bank_path and os.path.exists(bank_path) else None
    cp = ctx.get("cooc_path")
    if cp and os.path.exists(cp):
        with open(cp) as fh:
            d = json.load(fh)
        from collections import Counter
        _CTX["cooc"] = CooccurrenceModel(
            counts={k: Counter(v) for k, v in d["counts"].items()},
            sizes={k: np.asarray(v) for k, v in d["sizes"].items()},
            n_scenes=d["n_scenes"])
    else:
        _CTX["cooc"] = None
    fp = ctx.get("flow_path")
    if fp and os.path.exists(fp):
        from reroom.generative.sample import load_flow
        _CTX["flow"] = load_flow(fp, device=ctx.get("device", "cpu"))
    else:
        _CTX["flow"] = None


def _run_one(job):
    scene_dict, methods, targets_seed, levels, per_level, base_cfg = job
    scene = scene_from_dict(scene_dict)
    try:
        graph = build_motifs(build_scene_graph(scene))
    except Exception as exc:
        return [{"error": f"graph: {exc}", "scene": scene.scene_id}]
    targets = make_targets(scene, levels, per_level, targets_seed)
    rows = []
    for t in targets:
        for name in methods:
            spec = METHODS[name]
            cfg = RetargetConfig(**{**base_cfg, **{
                k: v for k, v in spec.overrides.items() if k != "no_project"}})
            el = _CTX.get(spec.elasticity, _CTX["prior"])
            bank = _CTX["bank"] if spec.use_bank else None
            # the motif-free arm optimises on a stripped graph but is *scored*
            # against the intact one, so S_motif still asks the same question
            g_in = strip_motifs(graph) if spec.no_motifs else graph
            if spec.retrieval:
                cfg.retrieval = dict(spec.retrieval)
            try:
                if spec.kind == "baseline":
                    out = run_baseline(spec.baseline, g_in, t["room"], cfg=cfg)
                elif spec.kind == "flow":
                    if _CTX["flow"] is None:
                        continue
                    from reroom.generative.sample import generative_retarget
                    out = generative_retarget(
                        _CTX["flow"], g_in, t["room"], elasticity=el, bank=bank,
                        cooc=_CTX["cooc"], cfg=cfg,
                        project=not spec.overrides.get("no_project", False)).scene
                else:
                    out = retarget(g_in, t["room"], elasticity=el, bank=bank,
                                   cooc=_CTX["cooc"], cfg=cfg).scene
                m = evaluate(graph, out, bank=(bank if _CTX.get("appearance")
                                               else None))
            except Exception as exc:
                rows.append({"scene": scene.scene_id, "method": name,
                             "level": t["level"], "error": f"{type(exc).__name__}: {exc}"})
                continue
            m.update({"scene": scene.scene_id, "method": name,
                      "level": t["level"], "level_name": t["name"],
                      "rep": t["rep"], "area_ratio": t["area_ratio"],
                      "room_type": scene.room.room_type,
                      "n_source": len(scene.objects),
                      "src_convexity": float(scene.meta.get("convexity", 1.0))})
            rows.append(m)
    return rows


def run_grid(scenes, methods, out_path: str | None = None,
             levels=(1, 2, 3, 4, 5), per_level: int = 1, seed: int = 0,
             workers: int = 8, base_cfg: dict | None = None,
             elasticity_path: str | None = None, bank_path: str | None = None,
             cooc_path: str | None = None, flow_path: str | None = None,
             device: str = "cpu", progress_every: int = 20,
             appearance: bool = False) -> list[dict]:
    base_cfg = base_cfg or dict(restarts=16, grad_steps=250, proj_steps=120,
                                device=device, seed=seed)
    ctx = {"elasticity_path": elasticity_path, "bank_path": bank_path,
           "cooc_path": cooc_path, "flow_path": flow_path, "device": device,
           "appearance": appearance}
    jobs = [(s.to_dict(), methods, seed + k, levels, per_level, base_cfg)
            for k, s in enumerate(scenes)]
    rows: list[dict] = []
    if workers <= 1:
        _init(ctx)
        for k, j in enumerate(jobs):
            rows.extend(_run_one(j))
            if k % progress_every == 0:
                print(f"  {k}/{len(jobs)} scenes", flush=True)
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
            for k, part in enumerate(ex.map(_run_one, jobs, chunksize=1)):
                rows.extend(part)
                if k % progress_every == 0:
                    print(f"  {k}/{len(jobs)} scenes  rows={len(rows)}", flush=True)
    if out_path:
        save_rows(rows, out_path)
    return rows


def save_rows(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rows, fh)


DEFAULT_COLUMNS = ("R_OOB", "R_col", "clearance_violation_ratio",
                   "door_blockage", "reachable_ratio", "S_rel", "S_rel_scaled",
                   "S_rel_elastic", "S_rel_kept", "S_motif", "object_retention",
                   "legality", "score")
APPEARANCE_COLUMNS = DEFAULT_COLUMNS + ("appearance_object", "n_substituted")


def bucket_area_ratio(r: float) -> str:
    if r < 0.75:
        return "a<0.75 (shrink)"
    if r < 0.95:
        return "b 0.75-0.95"
    if r < 1.15:
        return "c 0.95-1.15"
    if r < 1.5:
        return "d 1.15-1.50"
    return "e >1.50 (grow)"


def table(rows: list[dict], by: str = "method",
          columns=DEFAULT_COLUMNS, order: list[str] | None = None,
          title: str | None = None) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    n_err = 0
    for r in rows:
        if "error" in r:
            n_err += 1
            continue
        if by == "area_bucket":
            groups[bucket_area_ratio(float(r.get("area_ratio", 1.0)))].append(r)
        elif by == "method_x_area":
            groups[f"{r.get('method')} | "
                   f"{bucket_area_ratio(float(r.get('area_ratio', 1.0)))}"].append(r)
        else:
            groups[str(r.get(by))].append(r)
    keys = order or sorted(groups)
    keys = [k for k in keys if k in groups]
    w = max([len(k) for k in keys] + [len(by)]) + 2
    head = f"{by:<{w}}" + "".join(f"{c[:11]:>12}" for c in columns) + f"{'n':>7}"
    lines = []
    if title:
        lines += [title, "=" * len(head)]
    lines += [head, "-" * len(head)]
    for k in keys:
        a = aggregate(groups[k])
        lines.append(f"{k:<{w}}" + "".join(f"{a.get(c, float('nan')):>12.4f}"
                                           for c in columns) + f"{a['n']:>7d}")
    if n_err:
        lines.append(f"({n_err} failed runs omitted)")
    return "\n".join(lines)
