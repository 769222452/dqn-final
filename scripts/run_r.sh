#!/usr/bin/env bash
# Study R (PREREGISTRATION.md amendment 7): the policy flies with one coarse prediction every 48 s as input.
# Trains rel (morphing) and relfixed (fixed geometry) for each seed, then evaluates on validation and dev.
#   WORKERS=5 bash scripts/run_r.sh            (resumable: run the same command again to continue)
#   STAGES=report bash scripts/run_r.sh        (only evaluate the current policies and write the report)
# Output: runs/r/F8D2000B/R_REPORT.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
SEEDS=${SEEDS:-"17 23"}
STAGES=${STAGES:-"test train report"}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=F8D2000B
LOGS=runs/r/F8D2000B/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for s in $STAGES; do
  echo "=== $(stamp) study R: $s"
  case $s in
    test)   $PY -m pytest -q 2>&1 | tee "$LOGS/test.log" ;;
    train)  for seed in $SEEDS; do for v in rel relfixed; do
              echo "=== $(stamp) train $v seed $seed"
              TARGET_RL_CONFIG=configs/study_r.yaml $PY -m target_rl.train --variant "$v" --seed "$seed" \
                --workers "$WORKERS" 2>&1 | tee -a "$LOGS/train_${v}_s$seed.log"
            done; done ;;
    report) TARGET_RL_CONFIG=configs/study_r.yaml $PY -m target_rl.study_r --seeds "$SEEDS" \
              --workers "$WORKERS" 2>&1 | tee "$LOGS/report.log" ;;
  esac
done
echo "=== $(stamp) study R done: runs/r/F8D2000B/R_REPORT.md"
