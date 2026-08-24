"""Sampling from the flow proposal, then projecting onto the constraints.

    Generative Proposal -> Constraint Projection -> Final Scene           (37)

The generative model supplies diversity and a global layout prior; the
optimizer supplies collision-freedom, containment, clearance and functional
validity.  Neither is asked to do the other's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..core.scene import Room, Scene
from ..retarget.energy import EnergyWeights, TorchProblem, exact_energy
from ..retarget.optimizer import (RetargetConfig, RetargetResult, _apply_supports,
                                  _write_back, refine_continuous)
from ..retarget.populate import CooccurrenceModel, plan_population
from ..retarget.summarize import plan_summarization
from ..retarget.target import DesignIntent, build_design_intent
from ..intent.relations import SceneGraph
from .model import FlowModel
from .tokens import build_tokens, collate, from_frame

__all__ = ["sample_layouts", "generative_retarget", "load_flow"]


def load_flow(path: str, device: str = "cpu", use_ema: bool = True) -> FlowModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    m = FlowModel(cfg.get("d_model", 256), cfg.get("depth", 6),
                  cfg.get("heads", 8)).to(device)
    m.load_state_dict(ck["ema" if use_ema and "ema" in ck else "model"])
    m.eval()
    return m


@torch.no_grad()
def sample_layouts(model: FlowModel, intent: DesignIntent, target_room: Room,
                   k: int = 8, steps: int = 50, device: str = "cpu",
                   temperature: float = 1.0, seed: int = 0,
                   guidance=None) -> np.ndarray:
    """Integrate the probability-flow ODE to draw ``k`` candidate layouts.

    Returns ``(k, N, 2)`` positions and ``(k, N)`` yaws, already mapped back
    from the room frame into metres.

    ``guidance`` (a :class:`GuidanceConfig`) turns on PhyScene-style in-sampling
    physical guidance: at every ODE step the predicted clean endpoint is scored
    for feasibility (out-of-floor, non-nestable collision) and its gradient
    nudges the trajectory.  This replaces the post-hoc constraint projection --
    feasibility stays inside the generative loop, and legitimate overlaps
    (a chair under its table) are left intact.
    """
    from .guidance import feasibility_grad
    item = build_tokens(intent, target_room, None)
    batch = collate([item] * k, device=device)
    frames = [item.meta["frame_tgt"]] * k
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch["state"].shape, generator=g).to(device) * temperature
    dt = 1.0 / steps
    for s in range(steps):
        t = s * dt
        tau = torch.full((x.shape[0],), t, device=device)
        with torch.no_grad():
            v = model(x, tau, batch)
        x = x + dt * v
        if guidance is not None and t >= guidance.start:
            # ramp the step in over the tail of the trajectory
            ramp = (t - guidance.start) / max(1.0 - guidance.start, 1e-6)
            x1_hat = x + (1.0 - t) * v
            # this function runs under @torch.no_grad(); the guidance needs the
            # graph, so re-enable it just for the feasibility gradient
            with torch.enable_grad():
                grad = feasibility_grad(x1_hat, batch, frames, guidance)
            x = x - dt * ramp * grad
    x = x.cpu().numpy()
    fr = item.meta["frame_tgt"]
    n = len(item.cat)
    xy = np.zeros((k, n, 2))
    yaw = np.zeros((k, n))
    for c in range(k):
        for i in range(n):
            p, a = from_frame(x[c, i], fr)
            xy[c, i] = p
            yaw[c, i] = a
    return xy, yaw


def generative_retarget(model: FlowModel, graph: SceneGraph, target_room: Room,
                        elasticity=None, bank=None,
                        cooc: CooccurrenceModel | None = None,
                        cfg: RetargetConfig | None = None,
                        k: int = 12, steps: int = 50,
                        temperature: float = 1.0,
                        project: bool = False,
                        guidance="default") -> RetargetResult:
    """Stage-two pipeline: propose with ``p_theta``, guided in-sampling (37).

    The default path is PhyScene-style: no post-hoc projection, feasibility is a
    guidance gradient inside the sampler (``guidance``).  Pass ``project=True``
    for the old constraint-projection behaviour, or ``guidance=None`` to sample
    the raw proposal with no feasibility correction at all.
    """
    from .guidance import GuidanceConfig
    if guidance == "default":
        guidance = GuidanceConfig()
    cfg = cfg or RetargetConfig()
    rng = np.random.default_rng(cfg.seed)
    src = graph.scene
    intent = build_design_intent(graph, target_room, elasticity=elasticity)

    sm = plan_summarization(intent, target_room, allow_drop=cfg.allow_removal)
    out = Scene(scene_id=f"{src.scene_id}__flow", room=target_room.copy(),
                objects=[o.copy() for o in src.objects], source="reroom_flow",
                meta={"source_scene": src.scene_id, "method": "flow"})
    for i, o in enumerate(out.objects):
        o.keep = bool(sm.keep[i])

    xy, yaw = sample_layouts(model, intent, target_room, k=k, steps=steps,
                             device=cfg.device, temperature=temperature,
                             seed=cfg.seed, guidance=guidance)
    info: dict = {"k": k, "steps": steps, "summarization": sm.log,
                  "projected": bool(project),
                  "guided": guidance is not None}

    if project:
        problem = TorchProblem(out, intent, cfg.weights, device=cfg.device)
        # freeze_yaw: keep the proposal's orientation and correct only
        # positions.  The flow prior orients furniture the way real rooms do
        # (measured: 89 % of wall objects within 0.5 deg of parallel, median
        # skew 0.58 deg); letting the surrogate re-optimise yaw during
        # projection collapses that back to the optimiser's few-degree local
        # minimum.  This is the eq. (37) division of labour made literal --
        # the transformer owns orientation, the projection owns feasibility.
        xy, yaw, e, _ = refine_continuous(problem, xy, yaw, cfg.grad_steps,
                                          cfg.lr, freeze_yaw=True)
        # escalating projection: the proposal is a good layout prior but knows
        # nothing about hard constraints, so feasibility weights are raised
        # until the best candidate is actually legal (37)
        for scale in (cfg.projection_scale, cfg.projection_scale * 4.0):
            xy, yaw, e, _ = refine_continuous(problem, xy, yaw, cfg.proj_steps,
                                              cfg.lr * 0.35, scale,
                                              freeze_yaw=True)
            best_c = int(np.argmin(e))
            _write_back(out, xy[best_c], yaw[best_c])
            _apply_supports(out, intent)
            ex = exact_energy(out, intent, cfg.weights)
            info["last_projection_scale"] = scale
            if ex["E_bound"] <= cfg.bound_tol and ex["E_col"] <= cfg.col_tol:
                break
        order = np.argsort(e)[:cfg.exact_topk]
    else:
        order = range(len(xy))

    best, best_ex = None, None
    for c in order:
        _write_back(out, xy[c], yaw[c])
        _apply_supports(out, intent)
        ex = exact_energy(out, intent, cfg.weights)
        if best_ex is None or ex["E"] < best_ex["E"]:
            best, best_ex = int(c), ex
    _write_back(out, xy[best], yaw[best])
    _apply_supports(out, intent)

    # ---- growth (section 10) and substitution (section 11) --------------
    # The transformer can only place the objects it has tokens for -- the
    # reference set.  The remaining two-thirds of eq. (18), "selection +
    # substitution", are handled around it, exactly as in the optimiser path:
    #   * population proposes furniture to fill a room that grew, placed by the
    #     planner relative to the surviving motifs (the flow cannot invent
    #     tokens for objects with no reference relations);
    #   * substitution swaps each asset for the closest-fitting real one in the
    #     bank, so a reference sofa too large for the target becomes a smaller
    #     sofa of similar style rather than a non-physically shrunk one;
    #   * a feasibility gate then admits each addition only while it costs the
    #     layout nothing -- filling a bigger room is the right instinct, doing
    #     it for free is not.
    from ..retarget.optimizer import _add_wall_targets, _vet_additions
    from ..retarget.retrieval import substitute_assets
    if cfg.allow_addition:
        pop = plan_population(intent, target_room,
                              np.array([o.keep for o in out.objects], dtype=bool),
                              cooc, bank, rng)
        if pop is not None and pop.additions:
            n_ref = len(out.objects)
            for o in pop.additions:
                a = o.copy()
                a.meta = dict(a.meta or {}); a.meta["added"] = True
                out.objects.append(a)
            _add_wall_targets(intent, out, n_ref, target_room)
    if cfg.allow_substitution and bank is not None and len(bank):
        substitute_assets(out, intent, bank, rng=rng, **cfg.retrieval)
    if cfg.allow_addition and any(o.meta.get("added") for o in out.objects):
        _vet_additions(out, intent, cfg)

    out.objects = [o for o in out.objects if o.keep]
    return RetargetResult(scene=out, intent=intent,
                          energy=exact_energy(out, intent, cfg.weights),
                          info=info)
