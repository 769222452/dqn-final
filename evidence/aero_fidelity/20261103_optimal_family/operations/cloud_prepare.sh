#!/usr/bin/env bash
set -euo pipefail
OPS=/hy-tmp/aero_opt_operations_20261103
REPO=/hy-tmp/dqn-aero-opt-f248c4f
exec > >(tee "$OPS/cloud_prepare.log") 2>&1
cd "$REPO"
test "$(git rev-parse HEAD)" = f248c4f7da03a48335cbe1dfca4e2b6e0e58c7da
test -z "$(git status --porcelain --untracked-files=no)"
git rev-parse HEAD | tee "$OPS/FROZEN_COMMIT.txt"
export PATH=/hy-tmp/dqn-aero/.venv/bin:/hy-tmp/aero_su2/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
(cd upstream_matlab && python verify_snapshot.py) | tee "$OPS/upstream_before.log"
command -v SU2_CFD
command -v mpirun
python -c 'import gmsh, scipy, numpy, yaml, casadi; print("gmsh", gmsh.__version__)'
python -m pytest -q -p no:cacheprovider tests/test_aero_fidelity.py | tee "$OPS/pytest.log"
echo PASS > "$OPS/tests_passed"
{ date -Is; python --version; python -m pip freeze; nproc; lscpu; free -h; df -h /hy-tmp; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us /sys/fs/cgroup/memory/memory.limit_in_bytes; mpirun --version; ls /hy-tmp/aero_su2/conda-meta/su2-*.json; } > "$OPS/environment.txt" 2>&1
test -z "$(git status --porcelain --untracked-files=no)"
