#!/usr/bin/env bash
# v4c full process (README_V4C.md): target arrival on the certified chain with the F8D2000B library, disturbances, model
# mismatch; pure reinforcement learning, no comparison methods.
#
#   JOBS=4 bash scripts/run_v4c.sh                    # all stages in order (each stage skips what is already done)
#   STAGES="train eval" JOBS=4 bash scripts/run_v4c.sh
#
# Stages: snapshot contexts labels fit qualify install region train eval report
# Every stage resumes (labels skip labelled ids, training continues from state.json, evaluations are cached).
# Use at most one job per physical core: the corrector (2 s) and look-ahead (30 s) have wall-clock budgets.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
JOBS=${JOBS:-$(( $(nproc) ))}
STAGES=${STAGES:-"snapshot contexts labels fit qualify install region train eval report"}
SEEDS=${SEEDS:-17}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
export TARGET_RL_CONFIG=${TARGET_RL_CONFIG:-configs/v4c.yaml} TARGET_RL_LIBRARY=F8D2000B CF_RUNTIME=configs/runtime_v4c.json
OUT=runs/v4c/F8D2000B
SUR=$OUT/surrogate
MODEL=data/surrogate_v4c/controller.npz
mkdir -p "$SUR" "$OUT/logs"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for s in $STAGES; do
  echo "=== $(stamp) v4c: $s (jobs $JOBS)"
  case $s in
    snapshot) [ -f data/geometry_f8d2000b/manifest.json ] || $PY scripts/v4c_build_snapshot.py ;;
    contexts) [ -f "$SUR/contexts.jsonl" ] || $PY scripts/v4c_collect_contexts.py --flights 64 --jobs "$JOBS" \
                --out "$SUR/contexts.jsonl" 2>&1 | tee "$SUR/collect.log" ;;
    labels)   $PY -B scripts/aoa_generate_labels.py --objective aligned --jobs "$JOBS" --contexts "$SUR/contexts.jsonl" \
                --out "$SUR/labels.jsonl" 2>&1 | tee -a "$SUR/labels.log" ;;
    fit)      [ -f "$MODEL" ] || $PY -B scripts/aoa_fit_surrogate.py --objective aligned --labels "$SUR/labels.jsonl" \
                --out "$MODEL" 2>&1 | tee "$SUR/fit.log" ;;
    qualify)  [ -f data/lookahead_v4c/AB_metrics.json ] || $PY -B scripts/aoa_qualify_surrogate.py --objective aligned \
                --model "$MODEL" --labels "$SUR/labels.jsonl" 2>&1 | tee "$SUR/qualify.log" ;;
    install)  $PY - <<'PY'
import json
meta = json.load(open('data/surrogate_v4c/controller.json'))
rt = json.load(open('configs/runtime_v4c.json'))
rt['model_sha256'] = meta['model_sha256']
json.dump(rt, open('configs/runtime_v4c.json', 'w'), indent=2)
q = json.load(open('data/lookahead_v4c/AB_metrics.json'))
print('installed surrogate', meta['model_sha256'][:12], '| P1 gate passed:', q.get('P1_physical_gate_passed'), '| status:', q.get('status'))
if not q.get('P1_physical_gate_passed'):
    raise SystemExit('qualification failed: the look-ahead guard refuses this surrogate')
PY
              ;;
    region)   [ -f "$OUT/region.json" ] || $PY -m target_rl.region --workers "$JOBS" 2>&1 | tee "$OUT/logs/region.log" ;;
    train)    for seed in $SEEDS; do
                $PY -m target_rl.train --variant rel --seed "$seed" --workers "$JOBS" 2>&1 | tee -a "$OUT/logs/train_rel_s$seed.log"
              done ;;
    eval)     $PY -m target_rl.study_v2 --workers "$JOBS" --seeds "$SEEDS" --which validation 2>&1 | tee "$OUT/logs/eval_validation.log" ;;
    report)   $PY -m target_rl.study_v2 --workers "$JOBS" --seeds "$SEEDS" --which validation --report-only 2>&1 | tee "$OUT/logs/report.log" ;;
    *)        echo "unknown stage $s"; exit 2 ;;
  esac
done
echo "=== $(stamp) v4c done: $OUT/V4C_REPORT_validation.md"
