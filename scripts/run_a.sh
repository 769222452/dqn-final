#!/usr/bin/env bash
# Study A (PREREGISTRATION.md amendment 6): one seed on 300 training cases, then validation/dev evaluation.
#   WORKERS=5 bash scripts/run_a.sh          (resumable: run the same command again to continue)
#   STAGES=report bash scripts/run_a.sh      (only evaluate the current policy and write the report)
# Output: runs/a/F8D2000B/A_REPORT.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
SEED=${SEED:-17}
STAGES=${STAGES:-"test train report"}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=F8D2000B
LOGS=runs/a/F8D2000B/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for s in $STAGES; do
  echo "=== $(stamp) study A: $s"
  case $s in
    test)   $PY -m pytest -q 2>&1 | tee "$LOGS/test.log" ;;
    train)  TARGET_RL_CONFIG=configs/study_a.yaml $PY -m target_rl.train --variant pcguided --seed "$SEED" \
              --workers "$WORKERS" 2>&1 | tee -a "$LOGS/train_pcguided_s$SEED.log" ;;
    report) TARGET_RL_CONFIG=configs/study_a.yaml $PY -m target_rl.study_a --seeds "$SEED" \
              --workers "$WORKERS" 2>&1 | tee "$LOGS/report.log" ;;
  esac
done
echo "=== $(stamp) study A done: runs/a/F8D2000B/A_REPORT.md"
