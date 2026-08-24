# ReRoom: Reference-Guided Retargeting of Interior Design into New Floor Geometry

Given photographs of a room somebody likes and a target floor plan of a different size, aspect ratio
or shape, produce an editable 3D furniture arrangement that is physically placeable in the target
and preserves the reference's furniture composition, spatial relations, design motifs and style.

The working hypothesis is that a room's identity does not live in absolute coordinates but in a
hierarchy of relations with different physical rigidity — a nightstand beside a bed is fixed by the
human body, a sofa facing a television may stretch with the room. Retargeting is therefore joint
*placement, selection and substitution* under hard geometric constraints, not an affine map.

**Page:** <https://gino6178.github.io/project4/> — `index.html` is the paper, `experiments.html`
the full tables behind it. Every number was produced by the code here, on one machine, under one
protocol, and includes the results that did not work.

```
index.html experiments.html assets/   the page. Serving it is a git push; there is no build step.
code/                                 the method, the experiments and the tests
data/                                 the measured results, small enough to keep in a repository
README.md                             this
```

## What the numbers say

On 1 000 held-out retargetings, normalised-coordinate scaling leaves 8.8 % of furniture area outside
the target room and 5.8 % in collision; this leaves 0.38 % and 0.57 %. Legality rises 0.636 → 0.880.
When the target is *smaller* — the case the method exists for — the coordinate map collapses to
0.472 legality while this holds 0.858.

Against PhyScene (CVPR 2024) run here on the same rooms, same object vocabulary and one evaluator:
3.5× fewer colliding objects, 5× fewer objects outside the floor plan, at the same object count —
and a loss on free-space connectivity that is diagnosed rather than tuned away.

## Running it

```bash
python -m pytest tests/            # 30 checks, no dataset needed
bash scripts/run_all.sh            # the full pipeline, given the datasets below
```

`code/` expects 3D-FRONT and 3D-FUTURE, which require registration with their authors and are not
redistributed here; neither are the derived asset banks, trained weights or renders. `data/` holds
the measured results so the tables on the page can be checked without re-running anything.

## What is not claimed

No method in the study Pareto-dominates another — ignoring the reference wins every physical metric,
copying it wins every preservation metric, and both are useless. The claim is a position on that
frontier. The human study that would adjudicate the trade is implemented and has no participants
yet. The closest prior method is paywalled with no code, so there are no numbers against the one
system that most threatens the novelty claim. `data/report.md` states each of these at length.
