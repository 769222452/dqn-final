#!/usr/bin/env bash
set -euo pipefail
OPS=/hy-tmp/aero_operations_4d211b1
REPO=/hy-tmp/dqn-aero
cd "$OPS"
printf '%s\n' '2d32493504a11bdcd060af0da06eda9fdb13531917b8c66fcad4da406a8b521b  verified_git_source.tar.gz' | sha256sum -c -
test ! -e "$REPO/.git"
tar -xzf verified_git_source.tar.gz -C "$REPO"
cd "$REPO"
test "$(git rev-parse HEAD)" = 4d211b1b9cc282bdf7716c6e03ff5c5156c9d4a7
test "$(git branch --show-current)" = aoa-fix
test -z "$(git status --porcelain --untracked-files=no)"
cmp requirements.txt "$OPS/requirements_frozen.txt"
export PATH="$REPO/.venv/bin:/hy-tmp/aero_su2/bin:$PATH"
export NP=7 JOBS=2 SLICES=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
{
  date -Is
  git rev-parse HEAD
  git branch --show-current
  git status --short
  uname -a
  cat /etc/os-release
  nproc
  lscpu
  free -h
  df -h /hy-tmp /
  python --version
  python -m pip freeze
  mpirun --version
  python -c "import json,glob; p=glob.glob('/hy-tmp/aero_su2/conda-meta/su2-*.json'); print('SU2_PACKAGE',[(json.load(open(f))['version'],f) for f in p])"
  sha256sum /hy-tmp/aero_su2/bin/SU2_CFD
} > "$OPS/environment.txt"
python -m pip freeze > "$OPS/pip_freeze.txt"
python -m pytest -q tests/test_aero_fidelity.py 2>&1 | tee "$OPS/pytest.log"
printf 'PASS\n' > "$OPS/tests_passed"
set +e
bash experiments/aero_fidelity/run_decision.sh preflight > "$OPS/preflight.log" 2>&1
rc=$?
set -e
cat "$OPS/preflight.log"
printf '%s\n' "$rc" > "$OPS/preflight_exit_code.txt"
echo "PREFLIGHT_EXIT=$rc; local licensed MATLAB fallback permitted by CODEX.md section 3."
