#!/usr/bin/env bash
# One library of the study (EXPERIMENTS.md, PREREGISTRATION.md). Every stage is resumable.
#
#   WORKERS=14 bash scripts/run_all.sh                     # main library F8D2000B, full plan
#   LIBRARY=F8D2000 WORKERS=14 bash scripts/run_all.sh     # a sensitivity library, reduced plan
#   STAGES="train" VARIANTS="full" SEEDS="17 23" bash scripts/run_all.sh
#
# Stages: test region pi0 train eval generalize report
# Plan (PLAN=auto chooses by library):
#   main         variants full dist fixed mean; eval: all methods at L2; generalize: core methods at L0 L1 L3
#   sensitivity  variant full;                  eval: core methods at L2; no generalization
# The whole paper (all libraries, then tables and figures): scripts/run_paper.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
LIBRARY=${LIBRARY:-F8D2000B}
PLAN=${PLAN:-auto}
if [ "$PLAN" = auto ]; then
  if [ "$LIBRARY" = F8D2000B ]; then PLAN=main; else PLAN=sensitivity; fi
fi
if [ "$PLAN" = main ]; then
  DEF_VARIANTS="full dist fixed mean"; DEF_EVAL=all; DEF_LEVELS="L0 L1 L3"
else
  DEF_VARIANTS="full"; DEF_EVAL=core; DEF_LEVELS=""
fi
STAGES=${STAGES:-"test region pi0 train eval generalize report"}
VARIANTS=${VARIANTS:-$DEF_VARIANTS}
SEEDS=${SEEDS:-"17 23 41 59 83"}
EVAL_METHODS=${EVAL_METHODS:-$DEF_EVAL}
LEVELS=${LEVELS-$DEF_LEVELS}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=$LIBRARY
OUT=runs/$LIBRARY
LOGS=$OUT/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== $(stamp) library $LIBRARY, plan $PLAN, workers $WORKERS"
for stage in $STAGES; do
  echo "=== $(stamp) stage $stage"
  case $stage in
    test)       $PY -m pytest -q 2>&1 | tee "$LOGS/test.log" ;;
    region)     if [ -f "$OUT/region.json" ]; then echo "$OUT/region.json exists (preregistered targets)"; else
                  $PY -m target_rl.region --workers "$WORKERS" 2>&1 | tee "$LOGS/region.log"; fi ;;
    pi0)        $PY -m target_rl.train --pi0 --workers "$WORKERS" 2>&1 | tee "$LOGS/pi0.log" ;;
    train)      for v in $VARIANTS; do for s in $SEEDS; do
                  echo "--- $(stamp) train $v seed $s"
                  $PY -m target_rl.train --variant "$v" --seed "$s" --workers "$WORKERS" 2>&1 | tee -a "$LOGS/train_${v}_s${s}.log"
                done; done ;;
    eval)       $PY -m target_rl.evaluate --methods "$EVAL_METHODS" --level L2 --workers "$WORKERS" 2>&1 | tee -a "$LOGS/eval_L2.log" ;;
    generalize) for lv in $LEVELS; do
                  $PY -m target_rl.evaluate --methods core --level "$lv" --workers "$WORKERS" 2>&1 | tee -a "$LOGS/eval_${lv}.log"
                done ;;
    report)     $PY -m target_rl.report 2>&1 | tee "$LOGS/report.log" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "=== $(stamp) done ($LIBRARY)"
