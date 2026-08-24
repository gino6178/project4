"""PhyScene-style physical guidance for the flow sampler (learned, in-sampling).

The plan (section 13) hands feasibility to a projection stage *after* the
generative proposal.  Measured, that projection destroys exactly the coherence
the proposal learned -- it scatters a dining set to drive the collision count to
zero.  PhyScene does not do this: it keeps feasibility inside the generative
loop as a guidance gradient on the predicted clean sample, and it *tolerates*
the legitimate overlaps of a real layout (chairs tucked under a table) rather
than eliminating them.

This module is the flow-matching analogue of PhyScene's eq. (6)-(8), and it
also carries the section-8 energy's boundary/collision/clearance terms as
active priors during sampling: at each ODE step the predicted endpoint
``x1_hat = x + (1-tau) v`` is scored by a differentiable feasibility objective,
and its gradient nudges the state.  Three terms, all in world metres:

* ``phi_bound`` (E_bound) -- footprint outside the floor polygon, via the
  boundary samples the batch already carries (concave rooms included);
* ``phi_coll`` (E_col) -- pairwise overlap, but *only* between category pairs
  that should not overlap.  Nestable pairs (a chair and its table, a nightstand
  and its bed) are exempt, which lets the proposal keep its groups intact;
* ``phi_clear`` (E_clear) -- a walkable corridor between objects in *different*
  functional motifs, without spreading a single group apart.

No hand-written marching, no post-hoc snapping: the whole correction is a
gradient inside the sampler.
"""
from __future__ import annotations

import numpy as np
import torch

from .tokens import CATS

__all__ = ["NESTABLE_PAIRS", "GuidanceConfig", "feasibility_grad"]

# Category pairs whose footprints may legitimately overlap in a real layout, so
# the collision guidance must not push them apart.  Symmetric; stored as frozen
# sets of the two category names.
NESTABLE_PAIRS = frozenset({
    frozenset({"dining_chair", "dining_table"}),
    frozenset({"office_chair", "desk"}),
    frozenset({"dining_chair", "desk"}),
    frozenset({"stool", "dining_table"}),
    frozenset({"stool", "desk"}),
    frozenset({"stool", "console_table"}),
    frozenset({"barstool", "console_table"}),
    frozenset({"barstool", "dining_table"}),
    frozenset({"armchair", "coffee_table"}),
    frozenset({"lounge_chair", "coffee_table"}),
    frozenset({"stool", "dressing_table"}),
    frozenset({"nightstand", "double_bed"}),
    frozenset({"nightstand", "single_bed"}),
    frozenset({"nightstand", "kids_bed"}),
    frozenset({"bench", "dining_table"}),
})


class GuidanceConfig:
    """Strength and schedule of the in-sampling guidance."""

    def __init__(self, bound: float = 3.0, coll: float = 1.0,
                 clear: float = 0.2, corridor: float = 0.25,
                 start: float = 0.5, nestable_slack: float = 0.6,
                 bound_margin: float = 0.12):
        # E_clear (section 8): objects from *different* functional groups need a
        # gap a person can pass through.  ``corridor`` is that gap in metres;
        # ``clear`` weights it.  Only cross-motif pairs are pushed -- objects
        # within one motif (a sofa and its coffee table) are meant to be close.
        self.clear = clear
        self.corridor = corridor
        # a corner may hang up to ``bound_margin`` metres past the wall before
        # the bound term acts, so a wall-hugging object (edge on the wall) is
        # left alone and only a gross overhang is pulled in
        self.bound_margin = bound_margin
        # weights of each feasibility term
        self.bound = bound
        self.coll = coll
        # guidance is off for the first ``start`` fraction of the ODE (the
        # trajectory is pure noise there and a gradient is meaningless); it
        # ramps in over the remainder
        self.start = start
        # nestable pairs are not fully exempt -- a chair may slide *under* a
        # table but must not pass clean through it -- so their allowed overlap
        # is this fraction of the smaller footprint's inscribed radius
        self.nestable_slack = nestable_slack


def _cat_names(batch) -> list[int]:
    return batch["cat"]


def feasibility_grad(x1_hat: torch.Tensor, batch: dict, frames: list,
                     cfg: GuidanceConfig) -> torch.Tensor:
    """Gradient of the feasibility objective w.r.t. the predicted endpoint.

    ``x1_hat`` is (B, N, 4) in each room's normalised MRR frame.  Returns a
    tensor of the same shape; only the (u, v) position channels are non-zero.
    ``frames`` is the list of per-batch ``RoomFrame`` (constants) used to turn
    normalised coordinates into world metres so distances are isotropic.
    """
    B, N, _ = x1_hat.shape
    dev = x1_hat.device
    x = x1_hat.detach().clone().requires_grad_(True)
    uv = x[..., :2]

    # per-batch frame constants -> world map: world = c + u*h1*a1 + v*h2*a2
    h1 = torch.tensor([f.half1 for f in frames], device=dev, dtype=torch.float32)
    h2 = torch.tensor([f.half2 for f in frames], device=dev, dtype=torch.float32)
    a1 = torch.tensor(np.stack([f.axis1 for f in frames]), device=dev,
                      dtype=torch.float32)                          # (B, 2)
    a2 = torch.tensor(np.stack([f.axis2 for f in frames]), device=dev,
                      dtype=torch.float32)
    # world offset from room centre, (B, N, 2)
    world = (uv[..., 0:1] * h1[:, None, None] * a1[:, None, :]
             + uv[..., 1:2] * h2[:, None, None] * a2[:, None, :])

    mask = batch["mask"].float()                                   # (B, N)
    sx = torch.exp(batch["cond"][..., 0]).clamp(0.05, 6.0)
    sy = torch.exp(batch["cond"][..., 1]).clamp(0.05, 6.0)
    rad = 0.5 * torch.minimum(sx, sy)                              # inscribed

    # world footprint corners, from the predicted orientation (cos_r, sin_r are
    # the yaw relative to the frame; add the frame angle to get world yaw).  The
    # bound term uses corners rather than the centre so a wall-hugging object
    # (centre half a depth off the wall, edge on it) is *not* flagged, while an
    # object whose corner actually pokes outside is.
    ang = torch.tensor([f.angle for f in frames], device=dev, dtype=torch.float32)
    cr, sr = x[..., 2], x[..., 3]
    ca, sa = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    cw = cr * ca - sr * sa                                          # cos(world yaw)
    sw = sr * ca + cr * sa                                          # sin(world yaw)
    fwd = torch.stack([-sw, cw], -1)                               # local +y (B,N,2)
    right = torch.stack([cw, sw], -1)                              # local +x
    hx, hy = (0.5 * sx)[..., None], (0.5 * sy)[..., None]
    corners = torch.stack([
        world + right * hx + fwd * hy,
        world + right * hx - fwd * hy,
        world - right * hx + fwd * hy,
        world - right * hx - fwd * hy,
    ], dim=2)                                                      # (B, N, 4, 2)

    phi = x.new_zeros(())

    # ---- phi_bound: footprint centre pushed outside the floor -------------
    # boundary samples: (B, Nb, 4) = point(u,v) + inward normal(world).  Map the
    # sample points to world too, and penalise a centre that lies on the
    # outward side of its nearest boundary segment by more than its radius.
    bnd = batch["boundary"]                                        # (B, Nb, 4)
    bp_uv = bnd[..., :2]
    # the stored normal is in frame-basis components (n.axis1, n.axis2); rebuild
    # it in world coordinates so it matches the world-space displacement below
    bn_fb = bnd[..., 2:]                                          # (B, Nb, 2)
    bn = (bn_fb[..., 0:1] * a1[:, None, :]
          + bn_fb[..., 1:2] * a2[:, None, :])                     # world normal
    bp_world = (bp_uv[..., 0:1] * h1[:, None, None] * a1[:, None, :]
                + bp_uv[..., 1:2] * h2[:, None, None] * a2[:, None, :])
    # for each of the four corners, signed distance to its nearest boundary
    # sample; penalise only corners that actually lie outside (signed < 0).
    cflat = corners.reshape(B, N * 4, 2)                          # (B, N*4, 2)
    d = torch.cdist(cflat, bp_world)                             # (B, N*4, Nb)
    nn = d.argmin(-1)                                            # (B, N*4)
    bp_sel = torch.gather(bp_world, 1, nn[..., None].expand(-1, -1, 2))
    bn_sel = torch.gather(bn, 1, nn[..., None].expand(-1, -1, 2))
    signed = ((cflat - bp_sel) * bn_sel).sum(-1)                 # (B, N*4)
    # tolerate up to bound_margin of overhang (a wall-hugging edge), penalise
    # only what pokes out beyond it
    outside = torch.relu(-signed - cfg.bound_margin).reshape(B, N, 4)
    phi = phi + cfg.bound * (outside ** 2 * mask[..., None]).sum()

    # ---- phi_coll: overlap between non-nestable pairs ---------------------
    cats = batch["cat"]                                           # (B, N) int
    # pairwise centre distances in world metres
    dw = torch.cdist(world, world)                               # (B, N, N)
    rr = rad[:, :, None] + rad[:, None, :]                        # (B, N, N) sum of radii
    pen = torch.relu(rr - dw)                                     # overlap depth proxy
    # build a (B, N, N) multiplier: 1 for non-nestable valid pairs, else 0
    eye = torch.eye(N, device=dev).bool()[None]
    valid = (mask[:, :, None] * mask[:, None, :]).bool() & ~eye
    # nestable exemption + slack: allow overlap up to slack*min_radius
    cat_np = cats.cpu().numpy()
    slack = torch.zeros(B, N, N, device=dev)
    exempt = torch.zeros(B, N, N, device=dev, dtype=torch.bool)
    for b in range(B):
        names = [CATS[int(c)] if 0 <= int(c) < len(CATS) else "misc"
                 for c in cat_np[b]]
        for i in range(N):
            for j in range(i + 1, N):
                if frozenset({names[i], names[j]}) in NESTABLE_PAIRS:
                    exempt[b, i, j] = exempt[b, j, i] = True
    # for exempt pairs, subtract the allowed slide-under depth
    min_rad = torch.minimum(rad[:, :, None], rad[:, None, :])
    pen = torch.where(exempt,
                      torch.relu(pen - cfg.nestable_slack * min_rad),
                      pen)
    pen = pen * valid.float()
    phi = phi + cfg.coll * (pen ** 2).sum() * 0.5

    # ---- phi_clear (E_clear, section 8): a walkable corridor between groups --
    # Require a passable gap between objects in *different* functional motifs.
    # Same-motif pairs (a sofa and its coffee table) and objects with no motif
    # are left alone, so this buys walkability without spreading a group apart
    # or scattering loose clutter.
    if cfg.clear > 0.0:
        motif = batch["motif"]                                   # (B, N) ids
        mi = motif[:, :, None]; mj = motif[:, None, :]
        cross = (mi != mj) & (mi > 0) & (mj > 0)                 # different real motifs
        gap = dw - rr                                            # footprint gap proxy
        clear_pen = (torch.relu(cfg.corridor - gap)
                     * cross.float() * valid.float() * (~exempt).float())
        phi = phi + cfg.clear * (clear_pen ** 2).sum() * 0.5

    # degenerate case: a fully feasible predicted layout makes phi structurally
    # zero, with no gradient to take -- the correction is legitimately nothing.
    if not phi.requires_grad or float(phi) == 0.0:
        return torch.zeros_like(x1_hat)
    g, = torch.autograd.grad(phi, x)
    out = torch.zeros_like(x1_hat)
    out[..., :2] = torch.nan_to_num(g[..., :2])
    return out
