#!/usr/bin/env bash
# Whole study (PREREGISTRATION.md). Every stage is resumable: finished work is skipped, interrupted work continues.
#
#   WORKERS=14 bash scripts/run_all.sh                          # primary study, 298-geometry library
#   LIBRARY=F8D2000 WORKERS=14 bash scripts/run_all.sh          # sensitivity analysis, F8D2000 library
#   STAGES="train" VARIANTS="full" SEEDS="17 23" bash scripts/run_all.sh
#
# Stages: test region pi0 train eval report. Results: runs/<library>/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
LIBRARY=${LIBRARY:-L298}
STAGES=${STAGES:-"test region pi0 train eval report"}
VARIANTS=${VARIANTS:-"full dist fixed mean"}
SEEDS=${SEEDS:-"17 23 41 59 83"}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=$LIBRARY
OUT=runs/$LIBRARY
LOGS=$OUT/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== $(stamp) library $LIBRARY, workers $WORKERS"
for stage in $STAGES; do
  echo "=== $(stamp) stage $stage"
  case $stage in
    test)   $PY -m pytest -q 2>&1 | tee "$LOGS/test.log" ;;
    region) if [ -f "$OUT/region.json" ]; then echo "$OUT/region.json exists (preregistered targets)"; else
              $PY -m target_rl.region --workers "$WORKERS" 2>&1 | tee "$LOGS/region.log"; fi ;;
    pi0)    $PY -m target_rl.train --pi0 --workers "$WORKERS" 2>&1 | tee "$LOGS/pi0.log" ;;
    train)  for v in $VARIANTS; do for s in $SEEDS; do
              echo "--- $(stamp) train $v seed $s"
              $PY -m target_rl.train --variant "$v" --seed "$s" --workers "$WORKERS" 2>&1 | tee -a "$LOGS/train_${v}_s${s}.log"
            done; done ;;
    eval)   $PY -m target_rl.evaluate --methods all --workers "$WORKERS" 2>&1 | tee -a "$LOGS/eval.log" ;;
    report) $PY -m target_rl.report 2>&1 | tee "$LOGS/report.log" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "=== $(stamp) done"
