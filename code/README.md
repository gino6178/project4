# ReRoom — Reference-Guided 3D Interior Scene Retargeting

An implementation of the research plan *"ReRoom: Reference-Guided 3D Interior
Scene Retargeting under Spatial and Geometric Constraints"* (2026-08-21).

**The problem.** Given one or more reference room images (or a reference 3D
scene) and a *different* target room — different size, aspect ratio, even shape —
produce an editable 3D furniture layout that is physically placeable in the
target while preserving the reference's furniture composition, spatial
relations, design motifs and visual style.

```
X = (I_r, P_t, C_t)                                                  (1)
S_t = {(c_i, a_i, p_i, R_i, s_i)}                                    (3)
Reference Image(s) + New Floor Geometry -> Retargeted Editable 3D Scene  (4)
```

The target room is an arbitrary **simple polygon** `P_t = {v_1..v_M}` (2), so
L-shaped, trapezoidal, slanted-wall and concave rooms are first-class, not
special cases.

---

## The central hypothesis, and how it is tested

> Design intent lives in *hierarchical relations*, not in absolute coordinates.
> `Design Intent = Object Selection + Scene Motifs + Spatial Relations + Appearance Style` (7)

Different relations tolerate different amounts of stretching. That is made
concrete as **relation elasticity** `alpha_r in [0,1]` (8) with

```
d~_ij = (1 - alpha) d^r_ij + alpha * gamma_ij * d^r_ij                (9)
```

`alpha ~ 0` means human-scale and rigid (a dining chair's distance to its
table); `alpha ~ 1` means it should follow the room (a sofa's distance to the
TV). The claim is falsifiable, and `scripts/fit_elasticity.py` tests it on
3D-FRONT by regressing `log d` on `log gamma` per (category-pair, relation)
bucket. Measured values are in `outputs/elasticity/report.json`.

---

## Pipeline

```
reference image(s)                                    target floor polygon P_t
       |                                                        |
       v                                                        v
 Stage I  scene understanding  ->  Stage II  design-intent graph + motifs
       (reroom/perception)              (reroom/intent)
                                                |
                                                v
                          Stage III  geometry-aware retargeting
                                (reroom/retarget)
                    placement + selection + substitution        (18)
                                                |
                                                v
                       Stage IV  style-aware asset realization
                            -> editable 3D scene S_t
```

| stage | plan section | module |
|---|---|---|
| scene representation, polygons, curriculum | 8.1, 12 | `reroom/core`, `reroom/geom` |
| datasets (3D-FRONT, SAGE-10k, procedural) | 3 | `reroom/data` |
| relations, motifs, elasticity, importance | 4, 7, 9, 19 | `reroom/intent` |
| energies, optimizer, summarization, retrieval | 8, 9, 10, 11 | `reroom/retarget` |
| flow-matching proposal + projection | 13 | `reroom/generative` |
| source parsers (oracle / noisy / MIDI) | 6, 14.3 | `reroom/perception` |
| metrics, appearance, user study | 15 | `reroom/eval` |
| figures | 14.2 | `reroom/render` |

---

## Quick start

```bash
conda create -y -n reroom python=3.11
conda activate reroom
pip install numpy scipy shapely trimesh networkx matplotlib tqdm pillow \
            imageio pyyaml rtree torch torchvision open_clip_torch

# no dataset needed: procedural scenes exercise the whole pipeline
python -c "
from reroom.data.procedural import generate_scene
from reroom.intent.relations import build_scene_graph
from reroom.intent.motifs import build_motifs
from reroom.geom.deform import deform_room
from reroom.retarget.optimizer import retarget
from reroom.eval.metrics import evaluate
import numpy as np
s = generate_scene('living_room', seed=2)
g = build_motifs(build_scene_graph(s))
target = deform_room(s.room, 4, np.random.default_rng(0)).room   # L-shaped
r = retarget(g, target)
print(evaluate(g, r.scene))
"
```

### With 3D-FRONT

```bash
# 1. scene JSONs -> ReRoom scenes (real floor polygons, not bounding boxes)
python scripts/build_3dfront.py --front <3D-FRONT/*.json dir> \
    --bboxes future_bboxes.json --categories future_categories.json \
    --out data/processed

# 2. asset sizes from the 3D-FUTURE meshes
python scripts/build_future_bboxes.py <3D-FUTURE-model dirs> future_bboxes.json

# 3. corpus statistics, priors, elasticity
python scripts/analyze_corpus.py --corpus data/processed --out outputs/corpus_stats.json
python scripts/build_priors.py   --corpus data/processed --out outputs/priors
python scripts/fit_elasticity.py --corpus data/processed --out outputs/elasticity

# 4. the generative proposal (optional; the optimizer alone is the first milestone)
python scripts/train_flow.py --corpus data/processed --out outputs/flow

# 5. experiments
python experiments/exp1_oracle.py    --corpus data/processed
python experiments/exp2_geometry.py  --corpus data/processed
python experiments/exp3_image.py     --corpus data/processed
python experiments/exp4_realworld.py --corpus data/processed
python experiments/ablations.py      --corpus data/processed
```

---

## Design decisions worth knowing

**Floor polygons are parsed from the `Floor` meshes, not from bounding boxes.**
The standard preprocessed 3D-FRONT subsets replace each room with its bounding
rectangle, which is why the literature reports that 3D-FRONT is "mostly
rectangular". Parsing the raw meshes recovers the real outline; see
`outputs/corpus_stats.json` for what that changes.

**The exact energy and the differentiable surrogate are kept in sync.**
`reroom/retarget/energy.py` implements (19)–(25) twice: once on shapely areas
(honest, used for reporting and for choosing among candidates) and once as a
differentiable surrogate with a signed-distance field for the floor polygon and
SAT penetration depth for boxes (fast, used for Adam refinement over batched
random restarts). Both use the *same* definition of a relation's gap, otherwise
the optimizer chases an artefact of the metric.

**The intent-aware initialisation competes with the optimiser's output.**
Motifs are placed as rigid units and wall objects snap to the matched target
wall; that layout is scored by the exact energy alongside the refined
candidates, so the solver can never return something worse than where it
started.

**Summarization is motif-level.** Deleting a dining table while keeping six
chairs is geometrically legal and semantically absurd, so `k_{m_k}` (27) selects
whole motifs and *structured* pruning thins repeats (four chairs to two) inside
the ones that stay.

**Corridor clearance between motifs.** Two objects from different motifs 20 cm
apart create a gap nobody can walk through. Cross-motif pairs are asked for a
person-width corridor, which is what keeps the reachable-area ratio from
collapsing in dense rooms.

**The evaluation metric does not assume the hypothesis.** `S_rel` (42) compares
relations by direction, orientation, a scale-free distance *ratio* and contact
preservation, so neither "preserve metres" nor "preserve proportions" is
privileged. The elasticity-aware variant `S_rel_elastic` is reported separately
and always uses the *prior* elasticity model, so every method is scored by the
same yardstick.

**Perception and retargeting are measured separately.** Experiment 3 sweeps a
calibrated perception-noise budget to trace the whole degradation curve;
a real parser (MIDI) is then a single measured point on it. The MIDI adapter
raises rather than silently substituting a different parser when its outputs are
missing.

**The user study asks two questions, not one.** "Which looks like the
reference?" and "which is a better room?" are recorded separately (15.3);
collapsing them makes the study uninterpretable.

---

## Layout

```
reroom/
  core/       scene representation, category taxonomy and functional priors
  geom/       polygon ops, curriculum deformation (12), free-space analysis
  data/       3D-FRONT parser, asset bank, SAGE loader, procedural generator
  intent/     relation vocabulary (12), motifs (13), elasticity (8,9,45), importance (26)
  retarget/   energies (19-25), optimizer, summarization (27), population (29),
              style-aware retrieval (30), baselines (44)
  generative/ object tokens, flow-matching model (34-36), training, sampling (37)
  perception/ source-parser interface (10), noisy oracle, MIDI adapter
  eval/       metrics (40-43), appearance, two-question user study
  render/     top-down floor plans and 3D box renders
experiments/  the four experiments and the ablation grid
scripts/      dataset construction, priors, elasticity, flow training
```

## Scope

Following section 18: single room; bedroom, living room, dining room and
library; furniture as editable object instances. Photorealistic rendering,
lighting, exact product identification and multi-room consistency are out of
scope.

---

## Results (full tables in `outputs/REPORT.md`)

Everything below is measured on **held-out** 3D-FRONT rooms (house-disjoint
split), with target rooms generated by the section-12 curriculum.

### The corpus is more irregular than the literature assumes

16,597 rooms parsed from raw `Floor` meshes: **45.6 % are exact rectangles,
54.2 % have a reflex vertex, 30 % have convexity below 0.92**. The usual claim
that 3D-FRONT is "mostly rectangular" is a property of the preprocessing, not of
the data. By contrast, 100 % of sampled SAGE-10k rooms *are* axis-aligned
rectangles — which is why the plan is right to use SAGE for object diversity
only.

### Relation elasticity is real, and smaller than hoped

Fitted on 524,435 relation instances by regressing `log d` on `log gamma`:

| relation | fitted alpha (raw) | n |
|---|---|---|
| dining chair ↔ dining table | 0.047 – 0.061 | 11.7 k – 13.2 k |
| double bed ↔ nightstand | 0.171 | 7.9 k |
| sofa ↔ coffee table | 0.129 | 2.5 k |
| sofa ↔ TV stand | 0.224 | 2.0 k |
| double bed ↔ wardrobe | 0.626 – 0.867 | 0.4 k – 5.0 k |

The predicted ordering holds: body-scale relations sit near 0.05–0.17,
across-room relations at 0.6–0.87. But as an *optimizer input* it barely moves
the aggregate — it reduces error on the relations it targets by ~9–12 % and
leaves rigid ones untouched, which is precisely the intended behaviour and a
much smaller lever than motif-rigid initialisation or constraint projection.
That negative result is reported in full rather than buried.

### Experiment 1 — relation-aware vs direct scaling (the go/no-go milestone)

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | S_rel ↑ | S_motif ↑ | legality ↑ | score ↑ |
|---|---|---|---|---|---|---|---|
| *source scenes in their own room* | 0.022 | 0.060 | 0.269 | 1.000 | 1.000 | 0.689 | 0.814 |
| direct scaling | 0.088 | 0.058 | 0.287 | 0.962 | 1.000 | 0.636 | 0.770 |
| affine fit | 0.111 | 0.057 | 0.296 | 0.963 | 1.000 | 0.614 | 0.756 |
| target-only synthesis | 0.005 | 0.001 | 0.107 | 0.672 | 0.816 | 0.889 | 0.804 |
| relation-aware | 0.013 | 0.016 | 0.192 | 0.870 | 0.979 | 0.796 | 0.842 |
| + summarization | 0.006 | 0.002 | 0.154 | 0.766 | 0.919 | 0.842 | 0.831 |
| + substitution (full) | 0.005 | 0.002 | 0.151 | 0.765 | 0.913 | 0.845 | 0.833 |

The milestone is met. Note the first row: the *reference designs themselves*
carry 6 % collision area and 27 % clearance violation, so ReRoom's outputs are
more physically valid than the rooms they came from.

Under strong shrinkage (target < 0.75× source area) the gap widens sharply —
direct scaling reaches legality 0.39 against ReRoom's 0.88, and the full method
(with substitution) beats the summarization-only variant, which is the case
substitution exists for.

### Experiment 3 — perception and retargeting separate cleanly

Sweeping a calibrated source-parser noise budget from perfect to severe costs
design preservation (S_rel 0.690 → 0.263, S_motif 0.819 → 0.488) while leaving
physical legality untouched (0.902 → 0.936). A worse reading of the reference
gives a worse *design*, not a broken room.

### Ablations

| removed component | score | vs full |
|---|---|---|
| full pipeline | 0.836 | — |
| no motif-rigid init | 0.823 | −0.013 |
| no constraint projection | 0.826 | −0.010 |
| prior instead of fitted elasticity | 0.835 | −0.001 |
| alpha = 0 (no elasticity) | 0.836 | −0.000 |
| flow proposal, unprojected | 0.750 | −0.086 |
| flow proposal + projection | 0.833 | −0.003 |

Constraint projection is worth **+0.083** to the generative proposal (legality
0.624 → 0.800), which is the clearest support for the two-stage design of
eq. (37). The flow proposal preserves relations markedly better than the
optimizer alone (S_rel 0.823 vs 0.769, S_motif 0.952 vs 0.919) at somewhat lower
legality — the two are complementary, not redundant.

### Style-aware retrieval (eq. 30), on real 3D-FUTURE assets

7,989 assets with CLIP image embeddings. Keeping the reference asset costs a
mean log-size error of 0.249:

| weighting | size error ↓ | CLIP similarity ↑ |
|---|---|---|
| size only (`lambda_f = 0`) | 0.063 | 0.693 |
| balanced | 0.111 | 0.814 |
| appearance only (`lambda_s = 0`) | 0.344 | 0.863 |

The balanced objective takes most of the available size correction while giving
up little appearance similarity — a genuinely smaller sofa that still looks like
the reference.

### Experiment 3 — a real parser, measured

MIDI-3D (`VAST-AI/MIDI-3D`) was run for real, not stubbed. That needed a
textured renderer first: 3D-FRONT rooms rebuilt from their actual 3D-FUTURE
meshes into photograph-like views with exact instance masks, so what is measured
is the parser's 3D reasoning and not its segmentation. Scored on the same nine
rooms as its oracle:

| source graph | S_rel ↑ | S_motif ↑ | retention | legality ↑ |
|---|---|---|---|---|
| ground-truth graph | 0.757 | 0.884 | 0.904 | 0.796 |
| simulated: medium noise | 0.577 | 0.782 | 0.829 | 0.814 |
| simulated: severe noise | 0.262 | 0.528 | 0.627 | 0.836 |
| **MIDI-3D, measured** | **0.263** | **0.498** | **0.621** | **0.862** |

A current single-image parser sits *at* the severe end of the simulated sweep —
the plan's top listed risk (§20), measured rather than assumed. After
gauge alignment (one similarity per room, which a single image genuinely cannot
fix and which ReRoom is invariant to) MIDI's median object-centre error is
0.48 m and its mean log-size error 0.41. Legality barely moves across the whole
range: perception costs design fidelity, not usable rooms.

Two bugs found on the way there are worth recording, because both were silent:
a camera placed *inside* a piece of furniture (one chair filled 82 % of the
frame and the dining table got no mask at all), and a missing mirror in the
frame convention — without it a least-squares rotation still fits tolerably and
every room comes out plausibly arranged but turned by some scene-specific angle.
Fixing the mirror cut the median post-alignment residual from 0.709 m to
0.287 m.

### Appearance similarity, on real assets

Wired into `evaluate()` and reported in experiment 4 against a bank of 11,970
real 3D-FUTURE assets with CLIP image embeddings. ReRoom substitutes about five
objects per room and the replacements keep a CLIP similarity of 0.84 to the
originals. It is deliberately kept out of the headline `score`: the plan is
right that a global CLIP-style similarity is dominated by colour and by the
largest object.

### Experiment 4 on real photographs

Ten real room captures (ScanNet, Matterport3D, BlendSwap and photorealistic
renders, from MIDI's released example data) run through the identical path:
photograph → MIDI → design-intent graph → the six prescribed target floors.
None of them has ground truth, so all three unknowns are resolved the way a
deployed system must — categories from CLIP zero-shot over ReRoom's vocabulary,
metric scale anchored on the best-constrained object category present, room
outline inferred from the reconstructed footprints. Each is a stated
assumption rather than a hidden one.

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | S_rel ↑ | S_motif ↑ | legality ↑ | score ↑ |
|---|---|---|---|---|---|---|---|
| direct scaling | 0.034 | 0.051 | 0.291 | 0.977 | 1.000 | 0.659 | 0.789 |
| **ReRoom, full** | **0.002** | **0.004** | **0.076** | **0.744** | **0.901** | **0.919** | **0.859** |

The result holds on real input. Asset substitution cannot fire here — a
photographed object has no source asset id to substitute *from* — so the
appearance column is vacuously 1.0 and is not evidence of anything.

```bash
python scripts/run_midi_photos.py --pairs <dirs of *_rgb.png/*_seg.png> \
    --out outputs/midi_photos --midi-root <MIDI-3D checkout>
python scripts/photos_to_reroom.py --raw outputs/midi_photos \
    --out outputs/photo_scenes
python experiments/exp4_realworld.py --cases outputs/photo_scenes \
    --out outputs/exp4_photos
```

### What is genuinely still open

* **Multi-view reconstruction (GenRecon).** Code, conda environment, CUDA
  extensions, an attention backend and all 13.7 GB of checkpoints are in place,
  and `scripts/prepare_genrecon_input.py` generates its `Sage_gt` inputs. It
  stops on one gated dependency: `facebook/dinov3-vitl16-pretrain-lvd1689m` is
  `gated: manual`, so it needs a Hugging Face account that has accepted the
  licence, plus a token on this machine. Two third-party mirrors were checked
  and neither serves the weights (404). Beyond that, GenRecon reconstructs scene
  *geometry* rather than object instances, so a segmentation stage the plan does
  not specify would still be needed to reach a design-intent graph.
* **Comparison baselines from other papers.** SAGE's code is public
  (`NVlabs/sage`) and has not been run; no public CHOrD release was found, which
  matches the plan's own caveat; the 2025 size-aware retargeting paper has not
  been checked.
* **The human study.** `outputs/exp4/study/study.html` is a complete, ready-to-run
  two-question instrument over 24 randomised trials with an answer key and a
  scorer (`reroom.eval.userstudy.score_responses`, with Wilson intervals). No
  responses have been collected, so no human numbers are claimed.
* **A larger real-photograph set.** The ten captures above come from MIDI's
  released examples. More would need photographs with clear licensing; the
  pipeline takes any `*_rgb.png` / `*_seg.png` pair.

### Reproducing the parser experiment

```bash
# 1. render reference rooms from real 3D-FUTURE meshes (RGB + exact masks)
python scripts/render_references.py --corpus data/processed \
    --front <3D-FRONT json dir> --future <3D-FUTURE-model dirs> \
    --out outputs/references --n 48

# 2. run MIDI-3D in its own environment
python scripts/run_midi.py --refs outputs/references --out outputs/midi_raw \
    --midi-root <MIDI-3D checkout>

# 3. convert to ReRoom source scenes (gauge alignment + residual report)
python scripts/midi_to_reroom.py --raw outputs/midi_raw \
    --refs outputs/references --out outputs/midi

# 4. place it on the perception curve, scored on exactly those rooms
python experiments/exp3_image.py --corpus data/processed --scene-ids midi \
    --midi-dir outputs/midi --out outputs/exp3_midi
```
