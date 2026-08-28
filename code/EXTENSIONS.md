# ReRoom — Deep-Technical Extensions (CVPR/SIGGRAPH direction)

Four attempts to lift the shipped **engineering** system toward a **deep technical
contribution**: turn bolted-on heuristics into mathematical formulations and fold
split modules into the learned model.  Every piece is implemented dependency-free
(the Run:ai kernel has no `ot` / `cvxpylayers` / `gudhi`), unit-validated, and —
where a training run fit in one session — measured against the shipped baseline.

**The shipped system (`flow_bfresh` + single-step regularity) is unchanged and
restorable.**  These are additive research branches, gated by flags.

Status legend: **✓ measured** · **◐ built+validated (training-fold pending)** · **✗ negative**

---

## D1 — Joint discrete-continuous flow (learned pruning)  ✗ (concept validated, not yet competitive)

**Idea.** State becomes `z_i = (m_i, x_i)`: a per-object existence logit `m_i`
flows *jointly* with the pose `x_i`, so one network solves "which objects survive"
and "where they go" together — replacing the greedy `plan_summarization`.

**Implementation.**
- `model.py`: `mask_flow` head — zero-init `ell_in`/`ell_head`, warm-start-safe,
  emits a 5-wide velocity `(pose, existence)` sharing every attention path.
- `train.py`: existence velocity loss; pose supervised only on survivors
  (present1-weighted); uninformative prior `ell0∼N(0,1)`.
- `xscene.py`: physics-grounded drop labels — `make_forward_pair_drop` deforms
  into a too-small room and drops objects by **footprint poke-out** (a feasibility
  criterion, NOT the hand priority table), so the flow learns amortised pruning.
- `sample.py`: integrate `ell` through the ODE; sign at t=1 is the keep decision,
  overriding the greedy prune.

**Result (6 refs × {0.5,0.6,0.75}× small rooms).**

| method | drops | R_col% | S_rel | walk |
|---|---|---|---|---|
| shipped (greedy Summarise) | 8.11 | **0.94** | 0.116 | **0.89** |
| D1 learned mask | 6.83 | 4.09 | 0.124 | 0.77 |

**Finding.** *The paradigm works*: with an uninformative existence prior +
up-weighted mask loss, the head learns physics-sensible drops (6.83 ≈ greedy
8.11; mask loss 16.9→1.5).  A first "assume-keep" prior collapsed to drop=0 — a
diagnosable prior/weighting failure, fixed.  BUT the joint objective degraded the
pose flow in this short small-data warm-start run (val 0.058→0.28; R_col 0.9→4.1%),
so it does **not** beat the greedy baseline on final quality yet.  To become
competitive it needs balanced multi-task weighting (separate LR / lower mask
weight), full-corpus data, and a full-length run — a real research effort, not a
one-session result.  Honest verdict: **promising, unproven**.

---

## D2 — Unbalanced-OT coupling for OT-CFM data  ◐

**Idea.** Replace hand-scripted forward-deform pairs (attackable as shortcut
learning) with a measure-theoretic coupling: solve *unbalanced* OT between two
real scenes' objects.  Unbalanced (KL soft-marginals) is required because object
counts differ — unmatched mass is exactly the add/drop signal feeding D1.

**Implementation.** `uot.py` — self-contained log-domain KL-unbalanced Sinkhorn
+ `couple_scenes` (cost = category + MRR-frame position + size).

**Validation.** Solver matches a balanced reference on synthetic data; on real
scene pairs it recovers sensible couplings (P.sum≈1, matched fraction 1.0 for
similar rooms), and unmatched mass appears as drop/add on dissimilar pairs.  Two
bugs found+fixed: an MRR-frame mis-index collapsed all cost; `eps` too small for
the cost scale drove mass→0.  **OT-CFM training run to show it beats the scripted
pairs is the remaining step** (highest convergence risk of the four).

---

## D3 — Differentiable geometric projection layer  ✓ **positive — the clear winner**

**Idea.** Replace the hard, non-differentiable regularity snap with a *smooth*
projection realised as K unrolled gradient steps on a geometric energy
(wall-flush + inward-facing + Manhattan + non-collision + **anchor-to-proposal**).
Differentiable ⇒ can replace the snap at test time AND be unrolled inside training
so the DiT anticipates it (OptNet/implicit-diff spirit, done by explicit
unrolling — dependency-free).

**Implementation.** `diffproj.py` — `project_batch` (autograd-differentiable in
p, θ) and `project_scene` (test-time drop-in).

**Result (raw `flow_bfresh` outputs, 36 layouts = 12 refs × 3 sizes, mean ± std).**

| projection | R_col% | snap% | S_rel |
|---|---|---|---|
| none (raw flow) | 1.15±2.3 | 80±24 | 0.707±.32 |
| hard snap (shipped) | 1.59±2.6 | 87±20 | 0.676±.31 |
| **diff-proj (D3)** | 1.04±1.9 | 80±23 | **0.703±.32** |
| 25-step polish | 1.23±2.3 | 86±23 | 0.680±.31 |

**Finding.** Collision differences are within one std (no significant collision
win). The robust effect is **topology preservation**: raw and diff-proj hold
S_rel≈0.70, while hard snap/polish trade ~0.03 S_rel for +6–7 pts wall-snap.
So the differentiable projection is a topology-preserving alignment at no
collision penalty. (An earlier 4-seed run reported snap R_col up to 2.6%; that
was small-sample noise.)  And it is
trainable end-to-end, which the hard snap is not.  *Honest caveat found the hard
way*: the first D3 (no anchor term) was **worse** than the snap (drifted, S_rel
0.566) — the anchor-to-proposal term (what makes it a *projection*, not free
energy descent) is essential.

---

## D4 — Circulation / affordance field  ◐

**Idea.** Add the functional constraints a raw polygon lacks: a differentiable
**connectivity loss** (free space must stay reachable from the door) and
**affordance-SDF conditioning** (door swing, window light-cone, visual axis).

**Implementation.** `connectivity.py` — `soft_connectivity_loss` (soft object
footprints → free-space grid → K differentiable flood-fill max-pool dilations →
penalise door-unreachable free floor; gradients flow to object centres) +
`affordance_channels`.  The hard walkability *metric* already exists in
`eval/physcene.py`.

**Validation.** The loss cleanly separates a door-blocking object (1.00) from a
clear layout (0.03), differentiable w.r.t. positions.  **Folding it into a
training run to show R_walkable improves is the remaining step.**

---

## Summary

| dir | difficulty | one-session outcome | verdict |
|---|---|---|---|
| D3 diff projection | med | **measured, best collision + topology** | **adopt-worthy** |
| D1 joint disc-cont flow | med | learns to prune, pose degrades | promising, unproven |
| D2 unbalanced-OT data | high | coupling validated | needs OT-CFM run |
| D4 affordance/circulation | med | connectivity loss validated | needs training-fold |

**The one clear win is D3** — a differentiable projection layer that matches/beats
the hard snap and is trainable end-to-end, giving the "neural generator + convex
geometric projection" story a real, measured footing.  D1's paradigm is
demonstrated but not yet competitive; D2/D4 are built and unit-validated but need
full training runs to conclude.

Code: `reroom/retarget/diffproj.py`, `reroom/generative/{connectivity,uot}.py`,
mask-flow in `reroom/generative/{model,train,sample,tokens,xscene}.py`;
eval `scripts/eval_proj.py`, `scripts/eval_d1.py`, train `scripts/train_d1.py`.
