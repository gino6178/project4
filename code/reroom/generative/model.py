"""Graph-conditioned conditional flow matching for layout proposal (section 13).

    p_theta(L_t | G_r, P_t, z_style)                                   (34)
    X_tau = (1 - tau) X_0 + tau X_1,   tau in [0, 1]                   (35)
    v_theta(X_tau, tau, G_r, P_t, z_style)                             (36)

The network is a permutation-equivariant transformer over object tokens.  All
scene structure enters as *bias*, never as order:

* the design-intent graph becomes an additive attention bias, one scalar per
  head per edge, computed from the relation type, its weight, its fitted
  elasticity ``alpha``, the room-scale ratio ``gamma`` and the elasticity-
  adjusted target relation ``phi~``;
* the target floor polygon enters as a set of boundary points with inward
  normals, mean-pooled and broadcast, so concave and slanted rooms are
  representable rather than being flattened to (W, D);
* the flow time ``tau`` modulates every block through adaptive layer norm.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens import (EDGE_DIM, GLOBAL_DIM, N_BOUNDARY, N_CAT, N_MOTIF,
                     STATE_DIM, TOKEN_COND_DIM)
from ..core.categories import ROOM_TYPES

__all__ = ["FlowModel", "timestep_embedding"]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device, dtype=torch.float32)
                      / half)
    a = t.float()[:, None] * freqs[None] * max_period ** 0.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class BiasedAttention(nn.Module):
    """Multi-head self-attention with an additive per-edge bias."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, bias, key_mask):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dk).transpose(1, 2)
        k = k.view(B, N, self.h, self.dk).transpose(1, 2)
        v = v.view(B, N, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if bias is not None:
            att = att + bias
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer block with adaptive (time-conditioned) LayerNorm."""

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.att = BiasedAttention(d, heads)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(d * mlp_ratio), d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, bias, key_mask):
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + g1[:, None] * self.att(h, bias, key_mask)
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


class FlowModel(nn.Module):
    def __init__(self, d: int = 256, depth: int = 6, heads: int = 8,
                 cat_emb: int = 64, motif_emb: int = 32):
        super().__init__()
        self.d = d
        self.heads = heads
        self.e_cat = nn.Embedding(N_CAT, cat_emb)
        self.e_motif = nn.Embedding(N_MOTIF, motif_emb)
        self.e_room = nn.Embedding(len(ROOM_TYPES), 32)
        self.tok = nn.Linear(STATE_DIM + TOKEN_COND_DIM + cat_emb + motif_emb, d)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.bound = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64))
        self.glob = nn.Sequential(
            nn.Linear(GLOBAL_DIM + 64 + 64 + 32, d), nn.SiLU(), nn.Linear(d, d))
        self.edge = nn.Sequential(nn.Linear(EDGE_DIM, 128), nn.GELU(),
                                  nn.Linear(128, heads))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, STATE_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.out_ada[1].weight)
        nn.init.zeros_(self.out_ada[1].bias)

    def _edge_bias(self, batch, N: int):
        ef = batch["edge_feat"]                       # (B, E, EDGE_DIM)
        ei = batch["edge_index"]                      # (B, 2, E)
        em = batch["edge_mask"]                       # (B, E)
        B, E, _ = ef.shape
        w = self.edge(ef) * em[..., None]             # (B, E, heads)
        bias = ef.new_zeros(B, self.heads, N, N)
        if E == 0:
            return bias
        src = ei[:, 0].clamp(0, N - 1)
        dst = ei[:, 1].clamp(0, N - 1)
        flat = (src * N + dst)                        # (B, E)
        w = w.transpose(1, 2)                         # (B, heads, E)
        bias = bias.view(B, self.heads, N * N)
        bias.scatter_add_(2, flat[:, None].expand(-1, self.heads, -1), w)
        rev = (dst * N + src)
        bias.scatter_add_(2, rev[:, None].expand(-1, self.heads, -1), w)
        return bias.view(B, self.heads, N, N)

    def forward(self, x, tau, batch):
        """``v_theta(X_tau, tau, G_r, P_t, z_style)``; ``x`` is (B, N, 4)."""
        mask = batch["mask"]
        B, N, _ = x.shape
        tok = torch.cat([x, batch["cond"], self.e_cat(batch["cat"]),
                         self.e_motif(batch["motif"])], dim=-1)
        h = self.tok(tok)
        bfeat = (self.bound(batch["boundary"])).mean(1)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        g = self.glob(torch.cat([batch["glob"], bfeat, pooled[:, :64],
                                 self.e_room(batch["room_type"])], dim=-1))
        c = g + self.time(timestep_embedding(tau, self.d))
        bias = self._edge_bias(batch, N)
        for blk in self.blocks:
            h = blk(h, c, bias, mask)
        s, b = self.out_ada(c).chunk(2, dim=-1)
        h = self.out_norm(h) * (1 + s[:, None]) + b[:, None]
        return self.head(h) * mask[..., None]
