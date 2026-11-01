#!/usr/bin/env bash
# AoA-sawtooth experiments (see experiments/aoa_fix/README.md). Stages in order:
#
#   bash scripts/run_aoa_experiments.sh test        # unit tests + bit-exact V16 reference flight
#   bash scripts/run_aoa_experiments.sh smoke       # 4-decision run of each objective/inner loop (~3 min)
#   bash scripts/run_aoa_experiments.sh gate        # [true OCP] 4 nominal 1200 s flights (~1 h on 4 cores)
#   bash scripts/run_aoa_experiments.sh surrogate   # teacher labels -> fit r0 -> qualify -> DAgger -> final
#                                                   #   fit + qualification (~3 CPU-h; 45 min on 4 cores)
#   bash scripts/run_aoa_experiments.sh heldout     # [surrogate] 41 states x 8 policies x 2 objectives = 656
#                                                   #   flights (~1-2 CPU-min each)
#   bash scripts/run_aoa_experiments.sh fidelity    # [true OCP] 24 flights: surrogate-vs-teacher check
#   bash scripts/run_aoa_experiments.sh report      # tables + verdicts -> $OUT/report.md, report.json
#   bash scripts/run_aoa_experiments.sh use-local   # instead of 'surrogate': install the locally fitted models
#   bash scripts/run_aoa_experiments.sh all         # test, gate, report, gate check, surrogate (USE_LOCAL=1:
#                                                   #   use-local), heldout, fidelity, report - in this order
# Optional, only if H2 fails (learned gains do not transfer to the consistent objective):
#   bash scripts/run_aoa_experiments.sh retrain     # unchanged V16 training (train.py, 80 iterations) of the
#                                                   #   7 seeds under the consistent objective (~1700 CPU-h,
#                                                   #   as V16: SEEDS, WORKERS=8 per seed)
#   bash scripts/run_aoa_experiments.sh retrained   # [surrogate] held-out-3 flights of the retrained policies
#
# OUT (default runs/aoa), PYTHON (default python3), JOBS (default: CPUs - 1),
# MAX_DECISIONS (default 400; e.g. 2 for a dry run of the flight stages).
# Every stage resumes: finished flights (summary.json) and finished labels are skipped, and fitted or
# qualified models are kept unless deleted. An unfinished flight folder is moved to <folder>.attempt<N>.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
OUT=${OUT:-runs/aoa}
JOBS=${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
STAGE=${1:-}
LAB=$OUT/surrogate
R0=data/surrogate_consistent_r0/controller.npz
FINAL=data/surrogate_consistent/controller.npz

one() {   # inner objective surrogate policy dh dv folder
  local inner=$1 objective=$2 surrogate=$3 policy=$4 dh=$5 dv=$6 folder=$7
  if [ -f "$folder/summary.json" ]; then echo "skip $folder"; return 0; fi
  if [ -e "$folder" ]; then
    local n=1; while [ -e "$folder.attempt$n" ]; do n=$((n+1)); done
    mv "$folder" "$folder.attempt$n"
  fi
  mkdir -p "$(dirname "$folder")"
  local args=(--inner "$inner" --objective "$objective" --out "$folder" --max-decisions "${MAX_DECISIONS:-400}")
  if [ "$surrogate" != "-" ]; then args+=(--surrogate "$surrogate"); fi
  if [ "$policy" = "const0.025" ]; then args+=(--cl 0.025)
  elif [ "${policy#retrained_}" != "$policy" ]; then args+=(--chain "$OUT/retrain/${policy#retrained_}/state.json")
  else args+=(--chain "experiments/aoa_fix/policies/final_${policy}.json"); fi
  if [ "$dh" != "0.0" ] || [ "$dv" != "0.0" ]; then args+=(--offset "$dh" "$dv"); fi
  "$PYTHON" -B scripts/aoa_gate_experiment.py "${args[@]}" > "$folder.log" 2>&1 \
    && echo "done $folder" || echo "FAILED $folder (see $folder.log)"
}
export -f one
export PYTHON MAX_DECISIONS OUT

flights() {   # stage
  mkdir -p "$OUT"
  "$PYTHON" -B scripts/analysis/aoa_jobs.py "$1" "$OUT" > "$OUT/$1.jobs"
  echo "$1: $(wc -l < "$OUT/$1.jobs") flights, $JOBS in parallel"
  xargs -P "$JOBS" -L 1 bash -c 'one "$@"' _ < "$OUT/$1.jobs" | tee -a "$OUT/$1.progress"
}

fit_and_qualify() {   # model labels...
  local model=$1; shift
  mkdir -p "$(dirname "$model")"
  if [ ! -f "$model" ]; then
    "$PYTHON" -B scripts/aoa_fit_surrogate.py --objective consistent --labels "$@" --out "$model" \
      | tee "${model%.npz}.fit.log"
  fi
  local qdir; qdir=$(dirname "$(dirname "$model")")/lookahead_$(basename "$(dirname "$model")" | sed 's/^surrogate_//')
  if [ ! -f "$qdir/AB_metrics.json" ]; then
    "$PYTHON" -B scripts/aoa_qualify_surrogate.py --objective consistent --model "$model" --labels "$@" \
      | tee "$qdir.qualify.log" || { echo "QUALIFICATION FAILED for $model (see $qdir/AB_metrics.json)"; exit 1; }
  fi
}

gate_check() {   # held-out runs only if the report's gate decision says so (README section 3 and 8)
  "$PYTHON" -B scripts/analysis/aoa_report.py "$OUT" > /dev/null
  "$PYTHON" -B - "$OUT/report.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1])).get('gate', {})
missing = [p for p in ('const0.025', 'seed101') if p not in gate]
if missing:
    sys.exit(f'gate check: no gate verdict for {missing}; run the gate stage first')
for p, v in gate.items():
    print(f"gate {p}: G1={v['G1_sawtooth_removed']} G2={v['G2_feasible']} G3 declared (two-sided)="
          f"{v['G3_range_kept']} G3_no_loss={v['G3_no_loss']} range delta={v['range_delta_km']:+.3f} km "
          f"-> PROCEED_TO_HELDOUT={v['PROCEED_TO_HELDOUT']}")
if not all(v['PROCEED_TO_HELDOUT'] for v in gate.values()):
    sys.exit('gate check: PROCEED_TO_HELDOUT is false; held-out stage not run')
PY
}

case "$STAGE" in
  all)
    bash "$0" test
    bash "$0" gate
    gate_check
    if [ "${USE_LOCAL:-0}" = 1 ]; then bash "$0" use-local; else bash "$0" surrogate; fi
    bash "$0" heldout
    bash "$0" fidelity
    bash "$0" report ;;
  test)
    "$PYTHON" -B -m pytest -p no:cacheprovider -q ;;
  smoke)
    mkdir -p "$OUT/smoke"
    for o in v16 consistent; do
      rm -rf "$OUT/smoke/ocp_$o"
      "$PYTHON" -B scripts/aoa_gate_experiment.py --objective "$o" --cl 0.025 --max-decisions 4 --out "$OUT/smoke/ocp_$o"
    done
    rm -rf "$OUT/smoke/surrogate_v16"
    "$PYTHON" -B scripts/aoa_gate_experiment.py --inner surrogate --objective v16 --cl 0.025 --max-decisions 4 \
      --out "$OUT/smoke/surrogate_v16"
    "$PYTHON" -B scripts/aoa_generate_labels.py --objective consistent --limit 2 --jobs 2 \
      --contexts experiments/aoa_fix/surrogate/contexts.jsonl --out "$OUT/smoke/labels.jsonl" ;;
  retrain)
    [ -f "data/lookahead_consistent/AB_metrics.json" ] || { echo "run the surrogate stage first"; exit 1; }
    mkdir -p "$OUT/retrain"
    for s in ${SEEDS:-17 23 41 59 83 97 101}; do
      AOA_OBJECTIVE=consistent AOA_SURROGATE=$FINAL "$PYTHON" -u scripts/train.py --out "$OUT/retrain/seed$s" \
        --seed "$s" --workers "${WORKERS:-8}" --iterations 80 > "$OUT/retrain/seed$s.log" 2>&1 &
    done
    wait; sha256sum "$OUT"/retrain/seed*/state.json | tee "$OUT/retrain/state.sha256" ;;
  gate|heldout|fidelity|heldout-ocp|retrained)
    if [ "$STAGE" = heldout ]; then
      [ -f "data/lookahead_consistent/AB_metrics.json" ] || { echo "run the surrogate stage first (no qualified $FINAL)"; exit 1; }
      gate_check
    fi
    flights "$STAGE" ;;
  surrogate)
    mkdir -p "$LAB"
    "$PYTHON" -B scripts/aoa_generate_labels.py --objective consistent --jobs "$JOBS" \
      --contexts experiments/aoa_fix/surrogate/contexts.jsonl --out "$LAB/labels.jsonl" | tee -a "$LAB/labels.log"
    fit_and_qualify "$R0" "$LAB/labels.jsonl"
    flights dagger
    "$PYTHON" -B scripts/aoa_dagger_contexts.py "$OUT/dagger" --out "$LAB/dagger_contexts.jsonl"
    "$PYTHON" -B scripts/aoa_generate_labels.py --objective consistent --jobs "$JOBS" \
      --contexts "$LAB/dagger_contexts.jsonl" --out "$LAB/dagger_labels.jsonl" | tee -a "$LAB/labels.log"
    fit_and_qualify "$FINAL" "$LAB/labels.jsonl" "$LAB/dagger_labels.jsonl" ;;
  use-local)   # alternative to the surrogate stage: install the surrogates fitted and qualified locally
    for d in surrogate_consistent_r0 lookahead_consistent_r0 surrogate_consistent lookahead_consistent; do
      if [ -e "data/$d" ]; then echo "data/$d exists, not overwritten"; else
        cp -r "experiments/aoa_fix/surrogate_local/models/$d" "data/$d"; echo "installed data/$d"; fi
    done ;;
  report)
    "$PYTHON" -B scripts/analysis/aoa_report.py "$OUT" ;;
  *)
    sed -n 2,27p "$0"; exit 2 ;;
esac
