"""Training the graph-conditioned flow-matching proposal (section 13).

There is no corpus of *paired* (reference room -> retargeted room) layouts, so
the supervision is manufactured from real designs by running the retargeting
problem backwards:

1. take a professionally designed scene ``(L, P)`` from 3D-FRONT;
2. sample a curriculum deformation ``P_r = T_delta(P)`` (section 12) and warp
   the layout into it, giving a *pseudo-reference* design ``(L_r, P_r)``;
3. train the model to recover ``L`` in ``P`` given the design intent extracted
   from ``(L_r, P_r)``.

The target is therefore always a real, human-designed layout, and the input is
always the same design seen in a differently shaped room -- exactly the
retargeting task, with free supervision.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.scene import Scene, scene_from_dict
from ..geom.deform import deform_room
from ..geom.polygon import as_polygon
from ..intent.elasticity import ElasticityModel, PriorElasticity
from ..intent.motifs import build_motifs
from ..intent.relations import build_scene_graph
from ..retarget.optimizer import _map_point, _mrr_frame
from ..retarget.target import build_design_intent
from .model import FlowModel
from .tokens import build_tokens, collate

__all__ = ["RetargetPairs", "TrainConfig", "train_flow", "warp_scene"]


def warp_scene(scene: Scene, new_room) -> Scene:
    """Move a layout into a differently shaped room by the MRR-frame map.

    Used only to *manufacture* a pseudo-reference: it is the affine baseline,
    which is precisely what a reference design looks like when it has been
    naively transplanted, so the model learns to undo that transplant.
    """
    out = scene.copy()
    src = _mrr_frame(as_polygon(scene.room))
    tgt = _mrr_frame(as_polygon(new_room))
    dang = tgt[5] - src[5]
    out.room = new_room.copy()
    for o in out.objects:
        o.xy = _map_point(o.xy, src, tgt)
        o.yaw = o.yaw + dang
    return out


@dataclass
class TrainConfig:
    epochs: int = 30
    batch: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 8
    device: str = "cuda:0"
    d_model: int = 256
    depth: int = 6
    heads: int = 8
    levels: tuple = (1, 2, 3, 4, 5)
    ema: float = 0.999
    grad_clip: float = 1.0
    out: str = "outputs/flow"
    log_every: int = 50
    seed: int = 0


class RetargetPairs(Dataset):
    """On-the-fly (pseudo-reference, true layout) pairs."""

    def __init__(self, scenes: list[Scene], levels=(1, 2, 3, 4, 5),
                 elasticity: ElasticityModel | None = None,
                 samples_per_scene: int = 1, seed: int = 0):
        self.dicts = [s.to_dict() for s in scenes]
        self.levels = tuple(levels)
        self.spp = samples_per_scene
        self.seed = seed
        self._elast = elasticity or PriorElasticity()

    def __len__(self) -> int:
        return len(self.dicts) * self.spp

    def __getitem__(self, idx: int):
        base = idx // self.spp
        scene = scene_from_dict(self.dicts[base])
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2 ** 32))
        level = int(rng.choice(self.levels))
        for _ in range(4):
            ref_room = deform_room(scene.room, level, rng).room
            if ref_room.area > 3.0:
                break
        pseudo = warp_scene(scene, ref_room)
        graph = build_motifs(build_scene_graph(pseudo))
        intent = build_design_intent(graph, scene.room, elasticity=self._elast)
        return build_tokens(intent, scene.room, scene)


def _collate_fn(items):
    return collate([b for b in items if b is not None])


def train_flow(scenes: list[Scene], val_scenes: list[Scene] | None = None,
               cfg: TrainConfig | None = None,
               elasticity: ElasticityModel | None = None) -> FlowModel:
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    dev = torch.device(cfg.device)

    train_ds = RetargetPairs(scenes, cfg.levels, elasticity, seed=cfg.seed)
    dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                    num_workers=cfg.workers, collate_fn=_collate_fn,
                    drop_last=True, persistent_workers=cfg.workers > 0)
    val_dl = None
    if val_scenes:
        val_ds = RetargetPairs(val_scenes, cfg.levels, elasticity, seed=cfg.seed + 7)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                            num_workers=max(cfg.workers // 2, 1),
                            collate_fn=_collate_fn)

    model = FlowModel(cfg.d_model, cfg.depth, cfg.heads).to(dev)
    ema = FlowModel(cfg.d_model, cfg.depth, cfg.heads).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(len(dl), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(steps, 1), pct_start=0.06)

    step = 0
    history = []
    for ep in range(cfg.epochs):
        model.train()
        tot, cnt = 0.0, 0
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            x1 = batch["state"]
            x0 = torch.randn_like(x1)
            tau = torch.rand(x1.shape[0], device=dev)
            xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * x1
            v_target = x1 - x0
            v = model(xt, tau, batch)
            m = batch["mask"][..., None].float()
            loss = ((v - v_target) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(cfg.ema).add_(pm, alpha=1 - cfg.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += float(loss) * int(m.sum())
            cnt += int(m.sum())
            step += 1
            if step % cfg.log_every == 0:
                print(f"  ep {ep} step {step}/{steps} loss {tot / max(cnt, 1):.4f}",
                      flush=True)
        row = {"epoch": ep, "train_loss": tot / max(cnt, 1)}
        if val_dl is not None:
            row["val_loss"] = _validate(ema, val_dl, dev)
        history.append(row)
        print(f"[flow] epoch {ep}: " +
              "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"),
              flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "cfg": cfg.__dict__, "history": history},
                   os.path.join(cfg.out, "flow.pt"))
    return ema


@torch.no_grad()
def _validate(model: FlowModel, dl, dev) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for batch in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        x1 = batch["state"]
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=dev)
        xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * x1
        v = model(xt, tau, batch)
        m = batch["mask"][..., None].float()
        tot += float((((v - (x1 - x0)) ** 2) * m).sum())
        cnt += int(m.sum())
    model.train()
    return tot / max(cnt, 1)
