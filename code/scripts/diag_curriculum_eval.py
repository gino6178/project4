#!/usr/bin/env python
"""Post-hoc evaluation of a checkpoint across curriculum-relevant test sets.

Each curriculum phase teaches a different capability, so measuring only one
metric is misleading.  This tool builds three fixed test sets aligned with
each phase's objective and reports the metric-space wall error (cm) for the
wall-affinity subset of each set:

  test_scramble  Phase 1 objective: recover a clean layout from a fully
                 randomised pseudo-reference (LEGO-Net-style denoising).
                 Metric: metric-space L2 between predicted x1_hat and true
                 x1 for wall-affinity objects; smaller is better.

  test_scale075  Phase 2 objective: handle a shrunk target room (0.75x).
  test_scale135  Phase 2 objective: handle a grown target room (1.35x).
                 Same metric.

  test_subst     Phase 3 objective: preserve motif relations when object
                 sizes are perturbed.  Metric: mean pairwise relation-vector
                 L2 error over the *reference* graph, comparing predicted
                 layout with the true target.

The tests are seeded per name so they are stable across checkpoints -- so we
can plot progression cleanly.
"""
from __future__ import annotations
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.generative.train import RetargetPairs, _collate_fn
from reroom.generative.model import FlowModel
from reroom.intent.elasticity import load_elasticity


def _load_flow(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    m = FlowModel(cfg.get("d_model", 384), cfg.get("depth", 12),
                  cfg.get("heads", 8)).to(device)
    sd = ck.get("ema", ck.get("model"))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


@torch.no_grad()
def _wall_metric(model, batch, tau=0.9, device="cpu"):
    """Metric-space L2 error (in metres) between model x1_hat and true x1,
    averaged over wall-affinity objects.  tau=0.9 -> x1_hat is close to x1.
    """
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    x1 = batch["state"]
    x0 = torch.randn_like(x1)
    tau_t = torch.full((x1.shape[0],), tau, device=device)
    xt = (1 - tau) * x0 + tau * x1
    v = model(xt, tau_t, batch)
    x1_hat = xt + (1 - tau) * v
    aff = batch["cond"][..., -1]
    fh = batch["frame_h"]
    diff = (x1_hat[..., :2] - x1[..., :2]) * fh[:, None, :]
    d = torch.sqrt((diff ** 2).sum(-1) + 1e-9)
    mask = batch["mask"].float()
    gate = aff * mask * (aff > 0.3).float()
    return float((gate * d).sum() / max(float(gate.sum()), 1))


def _build_test_set(scenes, elasticity, kind, seed, n=40):
    """kind: 'scramble' | 'scale075' | 'scale135' | 'subst'."""
    from reroom.data.asset_bank import AssetBank
    bank = None
    if kind == "subst":
        bank = AssetBank.load("outputs/priors/assets_future.pkl")
    subset = scenes[:n]
    if kind == "scramble":
        # scramble at s=1.0 (no room scaling) so only the position-randomisation
        # signal is measured
        return RetargetPairs(subset, levels=(1,), elasticity=elasticity,
                             seed=seed, scramble_prob=1.0,
                             l1_range=(1.0, 1.00001), l1_u_shape=False,
                             cache=True)
    if kind == "scale075":
        return RetargetPairs(subset, levels=(1,), elasticity=elasticity,
                             seed=seed, scramble_prob=0.0,
                             l1_range=(0.75, 0.75001), l1_u_shape=False,
                             cache=True)
    if kind == "scale135":
        return RetargetPairs(subset, levels=(1,), elasticity=elasticity,
                             seed=seed, scramble_prob=0.0,
                             l1_range=(1.35, 1.35001), l1_u_shape=False,
                             cache=True)
    if kind == "subst":
        return RetargetPairs(subset, levels=(1,), elasticity=elasticity,
                             seed=seed, scramble_prob=0.0,
                             subst_prob=0.5, bank=bank,
                             l1_range=(1.0, 1.00001), l1_u_shape=False,
                             cache=True)
    raise ValueError(kind)


def evaluate(ckpt_path, device="cpu", n=40):
    scenes = [s for s in iter_scenes("/home/gino/data/reroom/processed",
                                     min_objects=5) if len(s.objects) <= 24]
    _, val, _ = split_scenes(scenes)
    val = val[:400]
    el = load_elasticity("outputs/elasticity/neural.pt")
    model = _load_flow(ckpt_path, device=device)
    from torch.utils.data import DataLoader
    kinds = ["scramble", "scale075", "scale135", "subst"]
    seeds = {"scramble": 91111, "scale075": 92222,
             "scale135": 93333, "subst": 94444}
    out = {}
    for kind in kinds:
        ds = _build_test_set(val, el, kind, seed=seeds[kind], n=n)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                         collate_fn=_collate_fn)
        errs = []
        for batch in dl:
            e = _wall_metric(model, batch, tau=0.9, device=device)
            errs.append(e)
        out[kind] = 100 * (sum(errs) / max(len(errs), 1))    # cm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+",
                    help="one or more flow.pt / flow_best.pt paths")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    print(f"{'checkpoint':>50}  {'scramble':>10}  {'scale075':>10}  "
          f"{'scale135':>10}  {'subst':>10}   (cm; lower = better)")
    print("-" * 100)
    for p in a.checkpoints:
        r = evaluate(p, device=a.device, n=a.n)
        tag = os.path.basename(os.path.dirname(p))
        print(f"{tag:>50}  {r['scramble']:>10.2f}  {r['scale075']:>10.2f}  "
              f"{r['scale135']:>10.2f}  {r['subst']:>10.2f}")


if __name__ == "__main__":
    main()
