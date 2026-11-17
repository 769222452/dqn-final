#!/usr/bin/env bash
# Whole study (PREREGISTRATION.md). Every stage is resumable: finished work is skipped, interrupted work continues.
#
#   WORKERS=14 bash scripts/run_all.sh                         # all stages
#   STAGES="train" VARIANTS="full" SEEDS="17 23" bash scripts/run_all.sh
#
# Stages: test region pi0 train eval report
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
STAGES=${STAGES:-"test region pi0 train eval report"}
VARIANTS=${VARIANTS:-"full dist fixed mean"}
SEEDS=${SEEDS:-"17 23 41 59 83"}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs/logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for stage in $STAGES; do
  echo "=== $(stamp) stage $stage"
  case $stage in
    test)   $PY -m pytest -q 2>&1 | tee runs/logs/test.log ;;
    region) if [ -f runs/region.json ]; then echo "runs/region.json exists"; else
              $PY -m target_rl.region --workers "$WORKERS" 2>&1 | tee runs/logs/region.log; fi ;;
    pi0)    $PY -m target_rl.train --pi0 --workers "$WORKERS" 2>&1 | tee runs/logs/pi0.log ;;
    train)  for v in $VARIANTS; do for s in $SEEDS; do
              echo "--- $(stamp) train $v seed $s"
              $PY -m target_rl.train --variant "$v" --seed "$s" --workers "$WORKERS" 2>&1 | tee -a "runs/logs/train_${v}_s${s}.log"
            done; done ;;
    eval)   $PY -m target_rl.evaluate --methods all --workers "$WORKERS" 2>&1 | tee -a runs/logs/eval.log ;;
    report) $PY -m target_rl.report 2>&1 | tee runs/logs/report.log ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "=== $(stamp) done"
