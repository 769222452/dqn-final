#!/usr/bin/env bash
# Round-2 pilot (PREREGISTRATION.md amendment 5, step 2): one seed, 20 iterations, evaluated on the PILOT scenarios
# (seeds.dev x T1-T6), never on the test scenarios. Decides whether the full round-2 run is worth starting.
#
#   WORKERS=5 bash scripts/pilot_r2.sh
# Output: runs/r2/F8D2000B/PILOT.md (training progress and pilot hit rates). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=F8D2000B
LOGS=runs/r2/F8D2000B/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== $(stamp) pilot: tests"
$PY -m pytest -q 2>&1 | tee "$LOGS/pilot_test.log"
echo "=== $(stamp) pilot: initial-policy tables (full)"
$PY -m target_rl.train --pi0 --variants full --workers "$WORKERS" 2>&1 | tee "$LOGS/pilot_pi0.log"
echo "=== $(stamp) pilot: training full seed 17, 20 iterations"
$PY -m target_rl.train --variant full --seed 17 --iterations 20 --workers "$WORKERS" 2>&1 | tee -a "$LOGS/train_full_s17.log"
echo "=== $(stamp) pilot: evaluation on the pilot scenarios"
$PY -m target_rl.evaluate --methods lookup,pc_fair,pc_strong,rl:full:17,r1:full:17 --cases dev --workers "$WORKERS" 2>&1 | tee "$LOGS/pilot_eval.log"
$PY -m target_rl.pilot 2>&1 | tee "$LOGS/pilot_summary.log"
echo "=== $(stamp) pilot done: runs/r2/F8D2000B/PILOT.md"
