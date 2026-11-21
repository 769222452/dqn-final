#!/usr/bin/env bash
# v2 full process (PREREGISTRATION_V2.md): reliable task domain of morphing vs fixed geometry, and what learning adds.
#
#   WORKERS=5 bash scripts/run_v2.sh                       # test, train, eval (validation + domain map), report
#   QUICK=1 WORKERS=4 bash scripts/run_v2.sh               # pipeline check in about 15 minutes (tiny sizes, separate folder)
#   STAGES="eval report" bash scripts/run_v2.sh            # only evaluate and report with the policies trained so far
#   RUN_TEST=1 STAGES="testset" bash scripts/run_v2.sh     # the test set: ONCE, only after the validation decision
#
# Resumable: run the same command again to continue (finished training iterations and evaluations are skipped).
# Output: runs/v2/F8D2000B/V2_REPORT_validation.md, V2_domain_map.png, train/<variant>_s<seed>/state.json, study_v2/<set>/*.json
# Environment: WORKERS (default: all cores), SEEDS (default "17 23"), ITER_CAP (training iterations per call, default: the limit),
# LIBRARY (default F8D2000B), PY (python executable, default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
SEEDS=${SEEDS:-"17 23"}
STAGES=${STAGES:-"test train eval report"}
QUICK=${QUICK:-0}
RUN_TEST=${RUN_TEST:-0}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=${LIBRARY:-F8D2000B}
export TARGET_RL_CONFIG=${TARGET_RL_CONFIG:-configs/v2.yaml}
export PYTHONIOENCODING=utf-8      # reports contain symbols a GBK console cannot print; the files are UTF-8
if [ "$QUICK" = 1 ]; then
  mkdir -p runs
  $PY - <<'PYEOF'
import yaml
c = yaml.safe_load(open('configs/v2.yaml', encoding='utf-8'))
c['experiment']['tag'] = 'v2quick'
c['seeds'].update(train=[1100, 1105], validation=[1600, 1601], domain=[1600, 1600], test=[2300, 2300], training_runs=[17])
c['training'].update(iterations=1, states=3, scenarios_per_iteration=2, no_acceptance_iterations=20)
c['targets']['v2'].update(test_per_stratum=1, grid={'D': [1900.0, 2560.0, 220.0], 'V': [1050.0, 1850.0, 267.0]})
c['experiment']['bootstrap'] = 500
yaml.safe_dump(c, open('runs/v2quick_config.yaml', 'w', encoding='utf-8'), sort_keys=False)
PYEOF
  export TARGET_RL_CONFIG=runs/v2quick_config.yaml
  SEEDS=17
fi
TAG=$($PY -c "from target_rl import config; print(config.load()['experiment']['tag'])")
OUT=runs/$TAG/$TARGET_RL_LIBRARY
LOGS=$OUT/logs
mkdir -p "$LOGS"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ITER_ARG=""
[ -n "${ITER_CAP:-}" ] && ITER_ARG="--iterations $ITER_CAP"
for s in $STAGES; do
  echo "=== $(stamp) v2: $s (tag $TAG, library $TARGET_RL_LIBRARY, seeds $SEEDS, workers $WORKERS)"
  case $s in
    test)    TARGET_RL_CONFIG="" $PY -m pytest -q 2>&1 | tee "$LOGS/test.log" ;;      # the tests assume the default configuration
    train)   for seed in $SEEDS; do for v in rel relfixed; do
               echo "=== $(stamp) train $v seed $seed"
               $PY -m target_rl.train --variant "$v" --seed "$seed" --workers "$WORKERS" $ITER_ARG 2>&1 | tee -a "$LOGS/train_${v}_s$seed.log"
             done; done ;;
    eval)    $PY -m target_rl.study_v2 --workers "$WORKERS" --seeds "$SEEDS" --which validation 2>&1 | tee "$LOGS/eval_validation.log" ;;
    report)  $PY -m target_rl.study_v2 --workers "$WORKERS" --seeds "$SEEDS" --which validation --report-only 2>&1 | tee "$LOGS/report.log" ;;
    testset) if [ "$RUN_TEST" != 1 ]; then echo "The test set is run once, after the validation decision: set RUN_TEST=1"; exit 2; fi
             $PY -m target_rl.study_v2 --workers "$WORKERS" --seeds "$SEEDS" --which test --no-domain 2>&1 | tee "$LOGS/eval_test.log" ;;
    *)       echo "unknown stage $s"; exit 2 ;;
  esac
done
echo "=== $(stamp) v2 done: $OUT/V2_REPORT_validation.md"
