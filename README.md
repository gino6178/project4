# ReRoom: Reference-Guided Retargeting of Interior Design into New Floor Geometry

Given photographs of a room somebody likes and a target floor plan of a different size, aspect ratio
or shape, produce an editable 3D furniture arrangement that is physically placeable in the target
and preserves the reference's furniture composition, spatial relations, design motifs and style.

The working hypothesis is that a room's identity does not live in absolute coordinates but in a
hierarchy of relations with different physical rigidity — a nightstand beside a bed is fixed by the
human body, a sofa facing a television may stretch with the room. Retargeting is therefore joint
*placement, selection and substitution* under hard geometric constraints, not an affine map.

**Paper (current, v0.2):** <https://gino6178.github.io/project4/v0.2.html>. Every number on it is
recomputed from the v0.2 model and inference path; `paper.html` is the superseded v0.1 version, kept
unchanged for the record, and `index.html` / `experiments.html` document the earlier shipped method.

```
index.html experiments.html assets/   the page. Serving it is a git push; there is no build step.
code/                                 the method, the experiments and the tests
data/                                 the measured results, small enough to keep in a repository
README.md                             this
```

## What the numbers say (v0.2)

On 40 held-out references retargeted into a different real room, against the affine warp of the
reference: relation retention on surviving objects 0.866 (affine 0.934), collision 4.9 % (8.4 %),
furniture area outside the room 0.17 % (12.9 %), doorways blocked 7.3 % (37.7 %). On 192 reshapes of
the references' own rooms: 0.880 (0.943), 6.7 % (9.0 %), 1.1 % (14.2 %), 13.6 % (45.5 %). The losses —
relations given up to satisfy capacity, walkable area on PhyScene's suite, a 2.7 s projection — are
reported on the page next to the gains.

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
