#!/usr/bin/env bash
set -euo pipefail
OPS=/hy-tmp/aero_operations_v5_20261102
REPO=/hy-tmp/dqn-aero-v5-dc39a7f
exec > >(tee "$OPS/cloud_prepare.log") 2>&1
cd "$REPO"
test "$(git rev-parse HEAD)" = dc39a7f9e7c3d78f8bc66c660866907759362c24
test -z "$(git status --porcelain --untracked-files=no)"
git rev-parse HEAD | tee "$OPS/FROZEN_COMMIT.txt"
export PATH=/hy-tmp/dqn-aero/.venv/bin:/hy-tmp/aero_su2/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export NP=7 JOBS=2 BASE_MARKER=1 OUT=runs/aero_fidelity_v5
(cd upstream_matlab && python verify_snapshot.py) | tee "$OPS/upstream_before.log"
command -v SU2_CFD
command -v mpirun
python -c 'import gmsh, scipy, numpy, yaml, casadi; print("gmsh", gmsh.__version__)'
python -m pytest -q -p no:cacheprovider tests/test_aero_fidelity.py | tee "$OPS/pytest.log"
echo PASS > "$OPS/tests_passed"
{ date -Is; python --version; python -m pip freeze; nproc; lscpu; free -h; df -h /hy-tmp; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us /sys/fs/cgroup/memory/memory.limit_in_bytes; mpirun --version; ls /hy-tmp/aero_su2/conda-meta/su2-*.json; } > "$OPS/environment.txt" 2>&1
python - <<'PY'
from pathlib import Path
import hashlib,json,shutil,datetime
source=Path('/hy-tmp/dqn-aero-v4-01589db/runs/aero_fidelity_v4/family_selected')
dest=Path('runs/aero_fidelity_v5/family_selected')
assert not dest.exists()
assert json.loads((source/'family_audit.json').read_text())['status']=='SELECTED'
def hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
before=hashes(source)
assert len(list(source.rglob('*.mat')))==3
shutil.copytree(source,dest)
assert hashes(dest)==before==hashes(source)
record=dict(source=str(source),destination=str(dest),sha256=before,all_identical=True,mat_count=3,copied_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
Path('/hy-tmp/aero_operations_v5_20261102/family_copy_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
test -z "$(git status --porcelain --untracked-files=no)"
