#!/usr/bin/env bash
# Design-point decision experiment (experiments/aero_fidelity/README.md). Stages, in order:
#
#   preflight   check MATLAB (+ Optimization, Symbolic Math toolboxes), SU2_CFD, mpirun, gmsh, Python packages
#   family      120 generator calls with the unmodified upstream_matlab core (SLICES parallel MATLAB processes)
#   check       watertight/volume audit and the declared selection rule -> $OUT/family_selected/select.json
#   cfd-mesh    surfaces, gmsh meshes and SU2 inputs: medium for all cases, coarse+fine for the grid check
#   cfd-run     SU2 Euler cases, JOBS cases in parallel with NP MPI ranks each (resumable)
#   collect     equal-lift comparison and the declared decision -> $OUT/results/decision.json
#   all         preflight, family, check, cfd-mesh, cfd-run, collect
#
# Environment: OUT (default runs/aero_fidelity), PYTHON (python3), MATLAB (matlab), SLICES (4), NP (7),
# JOBS (2), GRID_CASE (default: member FAM_MD6 at Mach 6, alpha 4).  A 14-core machine: NP=7 JOBS=2.
# Every stage stops at the first problem and prints it; nothing is substituted silently.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
MATLAB=${MATLAB:-matlab}
OUT=${OUT:-runs/aero_fidelity}
SLICES=${SLICES:-4}
NP=${NP:-7}
JOBS=${JOBS:-2}
ACTIVE="$PWD/upstream_matlab/source/matlab/active"
FAM="$OUT/family"
SEL="$OUT/family_selected"
CFD="$OUT/cfd"
STAGE=${1:-}

die() { echo "STOP: $*" >&2; exit 1; }

preflight() {
  local bad=0
  (cd upstream_matlab && "$PYTHON" verify_snapshot.py >/dev/null 2>&1) || { echo "FAIL upstream_matlab SHA-256 verification"; bad=1; }
  if command -v "$MATLAB" >/dev/null 2>&1; then
    local v; v=$("$MATLAB" -batch "disp(strjoin(cellfun(@(x) x, {ver().Name}, 'UniformOutput', false), '|'))" 2>&1 | tail -1) || true
    echo "$v" | grep -q "Optimization Toolbox" || { echo "FAIL MATLAB without Optimization Toolbox"; bad=1; }
    echo "$v" | grep -q "Symbolic Math Toolbox" || { echo "FAIL MATLAB without Symbolic Math Toolbox"; bad=1; }
  else
    echo "FAIL matlab not found (set MATLAB=...). The family stage can instead be run on another machine: "
    echo "     python3 experiments/aero_fidelity/family/family_jobs.py --out $FAM, then in MATLAB"
    echo "     make_family('$FAM/jobs.json', '<repo>/upstream_matlab/source/matlab/active', '$FAM'); copy $FAM back here."
    bad=1
  fi
  command -v SU2_CFD >/dev/null 2>&1 || { echo "FAIL SU2_CFD not found (conda install -c conda-forge su2, or build SU2 >= 7)"; bad=1; }
  command -v mpirun >/dev/null 2>&1 || { echo "FAIL mpirun not found"; bad=1; }
  "$PYTHON" -c "import gmsh, scipy, numpy, yaml, casadi" 2>/dev/null || { echo "FAIL python packages gmsh/scipy/numpy/yaml/casadi (pip install -r requirements.txt gmsh)"; bad=1; }
  echo "cores: $(nproc)  planned: NP=$NP x JOBS=$JOBS"
  [ "$bad" = 0 ] && echo "preflight OK" || die "preflight failed"
}

family() {
  mkdir -p "$FAM"
  "$PYTHON" experiments/aero_fidelity/family/family_jobs.py --out "$FAM" --slices "$SLICES"
  local pids=() k
  for k in $(seq 0 $((SLICES-1))); do
    "$MATLAB" -batch "addpath('$PWD/experiments/aero_fidelity/family'); make_family('$PWD/$FAM/jobs_$k.json', '$ACTIVE', '$PWD/$FAM')" \
      > "$FAM/matlab_$k.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || die "a MATLAB process failed (see $FAM/matlab_*.log); rerun 'family' to resume"
  echo "generated: $(grep -l '"status":"GENERATED"' "$FAM"/FAM_*/status.json 2>/dev/null | wc -l) of 120"
}

check() {
  "$PYTHON" experiments/aero_fidelity/family/check_family.py "$FAM" --out "$SEL" || die "family selection failed (see $SEL/family_audit.json)"
}

cfd_mesh() {
  [ -f "$SEL/select.json" ] || die "run check first"
  local common=(--geometry-dir "$SEL" --select "$SEL/select.json" --out "$CFD")
  "$PYTHON" scripts/cfd/make_su2_cases.py "${common[@]}" --level medium --machs 4 6 8 --alphas 2 4 6
  # grid check: one case on the two neighbouring levels
  local g=${GRID_GEOMETRY:-FAM_MD6}
  "$PYTHON" scripts/cfd/make_su2_cases.py "${common[@]}" --level coarse --geometries "$g" --machs 6 --alphas 4
  "$PYTHON" scripts/cfd/make_su2_cases.py "${common[@]}" --level fine --geometries "$g" --machs 6 --alphas 4
}

run_case() {   # case directory relative to $CFD
  local c=$1
  [ -f "$CFD/$c/done" ] && { echo "skip $c"; return 0; }
  (cd "$CFD/$c" && mpirun -np "$NP" SU2_CFD case.cfg > su2.log 2>&1 && test -s history.csv && touch done) \
    && echo "done $c" || echo "FAILED $c"
}
export -f run_case

cfd_run() {
  export CFD NP
  local lvl
  for lvl in medium coarse fine; do
    [ -f "$CFD/cases_$lvl.txt" ] || continue
    echo "level $lvl: $(wc -l < "$CFD/cases_$lvl.txt") cases, $JOBS in parallel x $NP ranks"
    xargs -P "$JOBS" -I{} bash -c 'run_case {}' < "$CFD/cases_$lvl.txt" | tee -a "$CFD/progress.log"
  done
  ! grep -q '^FAILED' "$CFD/progress.log" || echo "WARNING: failed cases are listed in $CFD/progress.log; collect excludes them"
}

collect() {
  mkdir -p "$OUT/results"
  "$PYTHON" scripts/cfd/collect_su2.py "$CFD" --out "$OUT/results/decision.json" | tee "$OUT/results/decision.txt"
  cp "$SEL/family_audit.json" "$OUT/results/" 2>/dev/null || true
}

case "$STAGE" in
  preflight) preflight ;;
  family) family ;;
  check) check ;;
  cfd-mesh) cfd_mesh ;;
  cfd-run) cfd_run ;;
  collect) collect ;;
  all) preflight; family; check; cfd_mesh; cfd_run; collect ;;
  *) sed -n 2,19p "$0"; exit 2 ;;
esac
