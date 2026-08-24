#!/usr/bin/env bash
# Full pipeline: corpus -> priors -> elasticity -> flow -> experiments -> report.
# Every stage is skippable; pass a stage name to run just that one.
set -euo pipefail
PY=${PY:-python}
CORPUS=${CORPUS:-data/processed}
OUT=${OUT:-outputs}
WORKERS=${WORKERS:-16}
STAGE=${1:-all}

run() { echo; echo "=== $* ==="; "$@"; }

if [[ "$STAGE" == "all" || "$STAGE" == "priors" ]]; then
  run $PY scripts/analyze_corpus.py --corpus "$CORPUS" --out "$OUT/corpus_stats.json"
  run $PY scripts/build_priors.py   --corpus "$CORPUS" --out "$OUT/priors"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "elasticity" ]]; then
  run $PY scripts/fit_elasticity.py --corpus "$CORPUS" --out "$OUT/elasticity" \
      --device "${DEVICE:-cpu}"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "flow" ]]; then
  run $PY scripts/train_flow.py --corpus "$CORPUS" --out "$OUT/flow" \
      --device "${DEVICE:-cpu}" --epochs "${FLOW_EPOCHS:-24}"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "experiments" ]]; then
  run $PY experiments/exp1_oracle.py    --corpus "$CORPUS" --out "$OUT/exp1" \
      --workers "$WORKERS" --scenes "${N1:-200}"
  run $PY experiments/exp2_geometry.py  --corpus "$CORPUS" --out "$OUT/exp2" \
      --workers "$WORKERS" --scenes "${N2:-120}"
  run $PY experiments/exp3_image.py     --corpus "$CORPUS" --out "$OUT/exp3" \
      --workers "$WORKERS" --scenes "${N3:-100}"
  run $PY experiments/exp4_realworld.py --corpus "$CORPUS" --out "$OUT/exp4" \
      --n-cases "${N4:-30}"
  run $PY experiments/ablations.py      --corpus "$CORPUS" --out "$OUT/ablations" \
      --workers "$WORKERS" --scenes "${N5:-150}"
fi
run $PY scripts/make_report.py --out "$OUT"
echo; echo "report -> $OUT/REPORT.md and $OUT/report.html"
