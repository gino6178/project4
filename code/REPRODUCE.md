# Reproducing the ReRoom paper

Every number in the paper comes from a script in `scripts/`. This file maps
**paper table → command**, and lists what each command needs.

Experiment *records* (logs, result JSONs) are deliberately not committed —
running the scripts regenerates them.

---

## 1. Environment

```bash
conda create -n reroom python=3.11 -y && conda activate reroom
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
```

Dev environment: Python 3.11.15, torch 2.6.0+cu124, single NVIDIA L40 for
inference, 4×L40 for training.

Every script assumes it is run **from the repository root** (they resolve paths
relative to the working directory).

## 2. Data

The pipeline runs end-to-end **without any dataset** on procedural scenes (see
`README.md` → Quick start), which is enough to exercise the code. Reproducing the
paper's numbers needs 3D-FRONT + 3D-FUTURE:

```bash
python scripts/build_3dfront.py        # scene JSONs -> ReRoom scenes (real floor polygons)
python scripts/build_future_bboxes.py  # asset sizes from 3D-FUTURE meshes
python scripts/build_priors.py         # corpus statistics, category priors, elasticity
```

Products land in `data/processed/` and `outputs/priors/` (both git-ignored).

## 3. Checkpoints

| Path | What it is | How to produce |
|---|---|---|
| `outputs/priors/assets_future.pkl` | asset bank (retrieval/substitution) | `scripts/build_priors.py` |
| `outputs/elasticity/neural.pt` | learned relation elasticity \(\alpha\) | `scripts/fit_elasticity.py` |
| `outputs/flow_bfresh/flow_best.pt` | **main model** — hierarchical flow | `scripts/train_flow.py` |
| `outputs/flow_twin/flow_best.pt` | matched-protocol global-coordinate twin (Table 4) | `scripts/train_twin.py` |
| `outputs/flow_proj/flow.pt` | train-through \(\Pi_\theta\) fine-tune (§7 negative result) | `scripts/train_projfinetune.py` |
| `outputs/flow_gw/flow.pt` | Gromov–Wasserstein loss fine-tune (§7 negative result) | `scripts/train_gwfinetune.py` |

Training notes: 180 epochs, batch 192, lr 3e-4, EMA 0.999 (full hyper-parameters
in the paper's Implementation Details table). **Set `workers=0`** if `/dev/shm` is
small or shared — the dataset caches in memory instead, avoiding a mid-run
`Bus error` from DataLoader workers.

## 4. Paper tables → commands

All evaluation is on the frozen held-out test split; statistics are
reference-clustered Wilcoxon (aggregate the correlated sizes within each
reference first, then test over the independent references).

| Table | Content | Command |
|---|---|---|
| 1 | Projection ablation, benign/convex regime | `python scripts/eval_proj.py` |
| 2 | Projection ablation, **non-convex** failure regime | `python scripts/eval_hardproj.py` |
| 3 | Baseline comparison vs affine warp | `python scripts/eval_baseline.py` |
| 4 | Matched-protocol twin, anisotropic vs uniform | `python scripts/eval_learned.py --global_ outputs/flow_twin/flow_best.pt --aniso` (drop `--aniso` for the uniform regime) |
| 5 | LEGO-Net cross-lineage baseline | see `baselines_legonet/README.md` (separate py3.7/torch-1.12 env) |
| 6 | PhyScene external metric suite | `python scripts/compare_physcene.py --cache <PhyScene preprocessed dir> --room-type living_room --limit 120` |
| 7 | Relational-mass selection (Stage 1) | `python scripts/eval_relselect.py` |
| 8 | Component ablation matrix | `python scripts/eval_ablation.py` |
| 9 | Robustness to corrupted intent graphs | `python scripts/eval_graphnoise.py` |
| 10 | Challenging floorplans (L / T / trapezoid / 1:4) | `python scripts/eval_floorplans.py` |
| 11 | Inference-latency breakdown | `python scripts/eval_latency.py` |
| 12 | Real cross-pairing OOD | `python scripts/eval_crosspair.py` |
| §5.2 note | Affine's \(S_\text{motif}\) / metric drift (measurement validity) | `python scripts/eval_motifvalidity.py` |
| §7 | FSD (reported as non-discriminative) | `python scripts/eval_fsd.py` |

Each script prints its table plus the paired and reference-clustered tests, and
finishes with a `DONE_*` marker.

## 5. Notes on variance

* Sampling draws `k=16` candidates; seeds are fixed inside each script, so a
  rerun on the same checkpoint reproduces the reported means. Retraining a
  checkpoint reproduces the *effects*, not the third decimal.
* Small effects (\(\Delta S_\text{rel}\approx 0.01\text{–}0.05\)) rest on
  \(n{=}12\) independent references; treat single-reference differences as noise.
* Two documented **negative results** are reproducible the same way: folding
  \(\Pi_\theta\) into training (`train_projfinetune.py`) and the GW relational
  loss (`train_gwfinetune.py`) both leave the deployed metrics unchanged.

## 6. Layout

```
reroom/            library — geometry, intent graph, retargeting, generative flow
  intent/          relations, motifs, elasticity
  retarget/        optimizer, regularity, diffproj (Pi_theta), gwselect
  generative/      tokens, model (DiT), train, sample, guidance, xscene
scripts/           training + evaluation entry points (one per paper table)
baselines_legonet/ LEGO-Net cross-lineage baseline (separate env)
experiments/       older experiment drivers
tests/             unit tests
webapp/            interactive viewer
```
