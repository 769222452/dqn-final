#!/usr/bin/env bash
# v3 full process (README_V3.md): pure reinforcement learning under model mismatch, flight ends at the target speed.
#
#   WORKERS=5 bash scripts/run_v3.sh                       # test, region (skipped: committed), train, eval, report
#   TARGET_RL_CONFIG=configs/v3b.yaml WORKERS=5 bash scripts/run_v3.sh    # v3b: held-out acceptance, guard with lift estimate
#   QUICK=1 WORKERS=4 bash scripts/run_v3.sh               # pipeline check in about 15 minutes (tiny sizes, separate folder)
#   STAGES="eval report" bash scripts/run_v3.sh            # only evaluate and report with the policies trained so far
#   RUN_TEST=1 STAGES="testset" bash scripts/run_v3.sh     # the test set: ONCE, only after the validation results are recorded
#
# Resumable: run the same command again to continue (finished training iterations and evaluations are skipped).
# Output: runs/v3/F8D2000B/V3_REPORT_validation.md, V3_domain_map.png, train/<variant>_s<seed>/state.json, study_v2/<set>/*.json
# Environment: WORKERS (default: all cores), SEEDS (default "17"), VARIANTS (default "rel"; "rel relfixed" adds fixed geometry),
# ITER_CAP (training iterations per call, default: the limit), LIBRARY (default F8D2000B), PY (python executable, default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
WORKERS=${WORKERS:-$(nproc)}
SEEDS=${SEEDS:-"17"}
VARIANTS=${VARIANTS:-"rel"}
STAGES=${STAGES:-"test region train eval report"}
QUICK=${QUICK:-0}
RUN_TEST=${RUN_TEST:-0}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TARGET_RL_LIBRARY=${LIBRARY:-F8D2000B}
export TARGET_RL_CONFIG=${TARGET_RL_CONFIG:-configs/v3.yaml}
export PYTHONIOENCODING=utf-8      # reports contain symbols a GBK console cannot print; the files are UTF-8
if [ "$QUICK" = 1 ]; then
  mkdir -p runs
  $PY - <<'PYEOF'
import os, shutil, yaml
c = yaml.safe_load(open(os.environ['TARGET_RL_CONFIG'], encoding='utf-8'))
c['experiment']['tag'] += 'quick'
c['seeds'].update(train=[1100, 1105], validation=[1600, 1601], domain=[1600, 1600], test=[2300, 2300], training_runs=[17])
if 'acceptance' in c['seeds']:
    c['seeds']['acceptance'] = [1400, 1403]
c['training'].update(iterations=1, states=3, scenarios_per_iteration=2, no_acceptance_iterations=20)
g = c['targets']['v2']['grid']
c['targets']['v2'].update(test_per_stratum=1, grid={'D': [g['D'][0], g['D'][1], 3*g['D'][2]], 'V': [g['V'][0], g['V'][1], 3*g['V'][2]]})
c['experiment']['bootstrap'] = 500
yaml.safe_dump(c, open('runs/v3quick_config.yaml', 'w', encoding='utf-8'), sort_keys=False)
lib = os.environ['TARGET_RL_LIBRARY']
tag = c['experiment']['tag']
os.makedirs(f'runs/{tag}/{lib}', exist_ok=True)
shutil.copy(f'runs/v3/{lib}/region.json', f'runs/{tag}/{lib}/region.json')   # the committed region of the full run
PYEOF
  export TARGET_RL_CONFIG=runs/v3quick_config.yaml
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
  echo "=== $(stamp) v3: $s (tag $TAG, library $TARGET_RL_LIBRARY, seeds $SEEDS, workers $WORKERS)"
  case $s in
    test)    TARGET_RL_CONFIG="" $PY -m pytest -q 2>&1 | tee "$LOGS/test.log" ;;      # the tests assume the default configuration
    region)  REG=$($PY -c "from target_rl import config; print(config.region_path())")
             if [ -f "$REG" ]; then echo "region exists: $REG (delete it to recompute, about 25 minutes)"
             else $PY -m target_rl.region --workers "$WORKERS" 2>&1 | tee "$LOGS/region.log"; fi ;;
    train)   for seed in $SEEDS; do for v in $VARIANTS; do
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
echo "=== $(stamp) v3 done: $OUT/${TAG^^}_REPORT_validation.md"
