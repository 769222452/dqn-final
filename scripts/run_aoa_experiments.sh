#!/usr/bin/env bash
# AoA-sawtooth experiments (see experiments/aoa_fix/README.md).
#
#   bash scripts/run_aoa_experiments.sh test            # unit tests + bit-exact V16 reference flight
#   bash scripts/run_aoa_experiments.sh smoke           # 4-decision run of each objective (~2 min)
#   JOBS=8 bash scripts/run_aoa_experiments.sh gate     # 4 nominal 1200 s flights
#   JOBS=48 bash scripts/run_aoa_experiments.sh heldout # 41 states x 8 policies x 2 objectives = 656 flights
#   bash scripts/run_aoa_experiments.sh report          # tables + verdict -> $OUT/report.md, report.json
#
# OUT (default runs/aoa), PYTHON (default python3), JOBS (default: CPUs - 1),
# MAX_DECISIONS (default 400; set e.g. 2 for a dry run of the whole pipeline).
# Finished flights (summary.json present) are skipped, so a stage can be re-run to resume.
# An unfinished flight folder is moved to <folder>.attempt<N> before it is re-run (evidence kept).
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
OUT=${OUT:-runs/aoa}
JOBS=${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
STAGE=${1:-}

one() {   # objective policy dh dv folder
  local objective=$1 policy=$2 dh=$3 dv=$4 folder=$5
  if [ -f "$folder/summary.json" ]; then echo "skip $folder"; return 0; fi
  if [ -e "$folder" ]; then
    local n=1; while [ -e "$folder.attempt$n" ]; do n=$((n+1)); done
    mv "$folder" "$folder.attempt$n"
  fi
  mkdir -p "$(dirname "$folder")"
  local args=(--objective "$objective" --out "$folder" --max-decisions "${MAX_DECISIONS:-400}")
  if [ "$policy" = "const0.025" ]; then args+=(--cl 0.025)
  else args+=(--chain "experiments/aoa_fix/policies/final_${policy}.json"); fi
  if [ "$dh" != "0.0" ] || [ "$dv" != "0.0" ]; then args+=(--offset "$dh" "$dv"); fi
  "$PYTHON" -B scripts/aoa_gate_experiment.py "${args[@]}" > "$folder.log" 2>&1 \
    && echo "done $folder" || echo "FAILED $folder (see $folder.log)"
}
export -f one
export PYTHON MAX_DECISIONS

case "$STAGE" in
  test)
    "$PYTHON" -B -m pytest -p no:cacheprovider -q ;;
  smoke)
    for o in v16 consistent; do
      rm -rf "$OUT/smoke/$o"; mkdir -p "$OUT/smoke"
      "$PYTHON" -B scripts/aoa_gate_experiment.py --objective "$o" --cl 0.025 --max-decisions 4 --out "$OUT/smoke/$o"
    done ;;
  gate|heldout)
    mkdir -p "$OUT"
    "$PYTHON" -B scripts/analysis/aoa_jobs.py "$STAGE" "$OUT" > "$OUT/$STAGE.jobs"
    echo "$(wc -l < "$OUT/$STAGE.jobs") flights, $JOBS in parallel"
    xargs -P "$JOBS" -L 1 bash -c 'one "$@"' _ < "$OUT/$STAGE.jobs" | tee -a "$OUT/$STAGE.progress" ;;
  report)
    "$PYTHON" -B scripts/analysis/aoa_report.py "$OUT" ;;
  *)
    sed -n 2,12p "$0"; exit 2 ;;
esac
