#!/usr/bin/env bash
# Disturbance study, fast-model stage: training runs of experiments/perturb/PREREGISTRATION.md (sections 6 and 8).
#   VARIANTS="full" LEVEL=L2 bash scripts/run_perturb.sh
# Variants: full | nodist (h,V,gamma,t only) | nohist (no rho_prev) | mean (mean score) | untuned (pi0 r=0.80)
# Resumable: a run directory with state.json continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
LEVEL=${LEVEL:-L2}
SEEDS=${SEEDS:-"17 23 41 59 83"}
VARIANTS=${VARIANTS:-full}
WORKERS=${WORKERS:-4}
BASE=${BASE:-experiments/perturb/baselines.json}
OUT=${OUT:-runs/perturb}
mkdir -p "$OUT"
for v in $VARIANTS; do
  case $v in
    full)    args=() ;;
    nodist)  args=(--inputs h,V,gamma,t) ;;
    nohist)  args=(--inputs h,V,gamma,t,rho,q,kL,kD) ;;
    mean)    args=(--score mean) ;;
    untuned) args=(--initial untuned) ;;
    *) echo "unknown variant $v"; exit 1 ;;
  esac
  for s in $SEEDS; do
    d="$OUT/${LEVEL}_${v}_s${s}"
    echo "=== $d"
    PYTHONPATH=src $PY -m rollout_improvement.scenario_training --out "$d" --level "$LEVEL" --seed "$s" \
      --baselines "$BASE" --workers "$WORKERS" "${args[@]}" 2>&1 | tee -a "$d.log"
  done
done
