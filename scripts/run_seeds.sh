#!/usr/bin/env bash
# Train and evaluate several seeds one after another.
#   SEEDS="17 23 41" WORKERS=14 bash scripts/run_seeds.sh [extra training options]
# Extra options are passed to scripts/train.py, e.g. the baseline:
#   --continuation closed-loop --inputs all --overrides
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"17 23 41"}
WORKERS=${WORKERS:-$(nproc)}
OUT=${OUT:-runs}
PYTHON=${PYTHON:-python}
mkdir -p "$OUT"
for seed in $SEEDS; do
    run="$OUT/seed$seed"
    "$PYTHON" scripts/train.py --out "$run" --seed "$seed" --workers "$WORKERS" "$@" 2>&1 | tee -a "$run.log"
    "$PYTHON" scripts/evaluate.py "$run/state.json" --out "$run/evaluation" --workers "$WORKERS" 2>&1 | tee -a "$run.log"
done
