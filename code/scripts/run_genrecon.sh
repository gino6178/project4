#!/usr/bin/env bash
# GenRecon multi-view reconstruction over the prepared ReRoom inputs.
#
# GenRecon's image tower is DINOv3 ViT-L/16, which Meta licence-gates on the
# Hub.  This machine has no accepted licence, so at the user's explicit
# instruction the weights come from an ungated third-party mirror of the same
# checkpoint (camenduru/dinov3-vitl16-pretrain-lvd1689m), verified to load as
# `DINOv3ViTModel` with the expected 303.1 M parameters and kept at $DINOV3.
# The upstream configs are patched in place, with `.orig` copies beside them.
#
# Everything else is already in place: the conda env `genrecon`, its CUDA
# extensions, the xformers attention backend, the three checkpoints, and the
# Sage_gt inputs written by scripts/prepare_genrecon_input.py.
set -eo pipefail
# `set -u` is deliberately off: the env's own cuda-nvcc activation hook reads
# NVCC_PREPEND_FLAGS before setting it, and would abort the script on entry

GENRECON=${GENRECON:-/home/gino/project/genrecon}
INPUT=${INPUT:-/home/gino/data/reroom/genrecon_input}
# GenRecon resolves each stage's *training* config as
# `dirname(dirname(ckpt))/config.json`, so the three released checkpoints have
# to sit in that run-directory shape rather than flat.  `genrecon_runs` is that
# layout: one directory per stage, holding the matching config from
# `configs/gen/` and a `ckpts/` symlink to the downloaded weights.
CKPT=${CKPT:-/home/gino/data/reroom/genrecon_runs}
OUT=${OUT:-outputs/genrecon_out}
VIEWS=${VIEWS:-24}
DINOV3=${DINOV3:-/home/gino/data/reroom/dinov3-vitl16}

CONDA_BASE=${CONDA_BASE:-/home/gino/miniconda3}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate genrecon
export ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers SPCONV_ALGO=native
# GPU 0 on this box usually has a long-running ComfyUI on it; pick the free one
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ ! -f "$DINOV3/model.safetensors" ]]; then
  echo "DINOv3 weights missing at $DINOV3 — see the note above." >&2
  exit 1
fi
python - <<PY || exit 1
from transformers import DINOv3ViTModel
m = DINOv3ViTModel.from_pretrained("$DINOV3")
n = sum(p.numel() for p in m.parameters())
assert 3.0e8 < n < 3.1e8, f"unexpected DINOv3 size: {n}"
print(f"DINOv3 ViT-L/16 ready ({n/1e6:.1f} M parameters)")
PY

cd "$GENRECON"
for d in "$INPUT"/renders_room/*/; do
  sid=$(basename "$d")
  if [[ -f "$OUT/$sid/mesh.ply" ]]; then
    echo "=== $sid (already reconstructed, skipping) ==="
    continue
  fi
  echo "=== $sid ==="
  python reconstruct_scene.py --mode Sage_gt --path "$d" \
      --output_path "$OUT/$sid" \
      --ss_ckpt "$CKPT/ss/ckpts/sparse_structure.pt" \
      --shape_ckpt "$CKPT/shape/ckpts/shape_slat.pt" \
      --tex_ckpt "$CKPT/tex/ckpts/texture_slat.pt" \
      --num_imgs_per_scene "$VIEWS"
  # the fused mesh is all the ReRoom adapter reads; the chunk tensors are
  # several hundred megabytes per room of pipeline scratch
  rm -f "$OUT/$sid/chunk_inputs.pt" "$OUT/$sid/to_glb_inputs.pt"
done
echo "reconstructions -> $OUT"
