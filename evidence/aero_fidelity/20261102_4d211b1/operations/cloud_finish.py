"""Operations-only wrapper: audit transferred family with frozen code, then package evidence."""
from pathlib import Path
import collections
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile

REPO=Path('/hy-tmp/dqn-aero')
OPS=Path('/hy-tmp/aero_operations_4d211b1')
SHA='4d211b1b9cc282bdf7716c6e03ff5c5156c9d4a7'
OUT=REPO/'runs/aero_fidelity'

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
archive=OPS/'family_generated.tar.gz'
assert digest(archive)==sys.argv[1], 'Uploaded family SHA-256 mismatch'
assert not (OUT/'family').exists(), 'Existing cloud family: inspect before resuming; do not overwrite'
with tarfile.open(archive,'r:gz') as tf:
    for m in tf.getmembers():
        dest=(OUT/m.name).resolve()
        assert dest.is_relative_to(OUT.resolve()) and not m.issym() and not m.islnk(), m.name
    tf.extractall(OUT)
manifest=json.loads((OUT/'family/TRANSFER_SHA256.json').read_text())
for name,h in manifest.items():
    assert digest(OUT/'family'/name)==h, name
statuses=[json.loads(p.read_text()) for p in sorted((OUT/'family').glob('FAM_*/status.json'))]
assert len(statuses)==120, f'Expected 120 status records, found {len(statuses)}'
env=os.environ.copy()
env.update(NP='7',JOBS='2',SLICES='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
env['PATH']=str(REPO/'.venv/bin')+':/hy-tmp/aero_su2/bin:'+env['PATH']
start=time.time()
with (OPS/'check.log').open('w') as log:
    result=subprocess.run(['bash','experiments/aero_fidelity/run_decision.sh','check'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
print((OPS/'check.log').read_text(),flush=True)
audit_path=OUT/'family_selected/family_audit.json'
assert audit_path.is_file(), f'check failed without audit, exit {result.returncode}'
audit=json.loads(audit_path.read_text())
if result.returncode:
    assert audit.get('status')=='NO_ADMISSIBLE_FAMILY', f'Unexpected check failure: {audit}'
summary={
    'commit':SHA,'branch':'aoa-fix','completed_stages':['test','preflight','family','check'],
    'preflight_cloud':'MATLAB absent; other prerequisites installed and checked',
    'family_execution':'Local licensed MATLAB R2023a, as allowed by CODEX.md; unmodified make_family.m, 4 processes',
    'family_counts':dict(collections.Counter(s['status'] for s in statuses)),
    'family_wall_s':json.loads((OUT/'family/FAMILY_RUN.json').read_text())['wall_s'],
    'check_exit_code':result.returncode,'check_wall_s':time.time()-start,
    'selection_status':audit.get('status'),'cfd_executed':False,
    'failures':[s for s in statuses if s['status']!='GENERATED'],
}
(OUT/'RUN_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'RUN_SUMMARY.md').write_text(
    '# Aerodynamic family experiment: preflight / family / check\n\n'
    f'Commit: `{SHA}`; branch: `aoa-fix`.\n\n'
    'Python tests passed before generation. Cloud MATLAB is absent; family generation used the documented '
    'other-machine fallback with licensed local MATLAB R2023a and Optimization / Symbolic Math toolboxes. '
    'Cloud SU2 8.5.0 and Open MPI 5.0.11 installed from conda-forge. NP=7, JOBS=2, SLICES=4.\n\n'
    f'Family status counts: {summary["family_counts"]}. Wall time: {summary["family_wall_s"]:.3f} s.\n\n'
    f'Selection: `{summary["selection_status"]}`; check wall time {summary["check_wall_s"]:.3f} s. '
    'See family_selected/family_audit.json for the unmodified selection results and full per-job audit.\n\n'
    'No cfd-mesh, cfd-run or collect was executed. All failed generation records are retained.\n',encoding='utf-8')
metadata=OUT/'operations'
metadata.mkdir(exist_ok=False)
for name in ['environment.txt','pip_freeze.txt','pytest.log','preflight.log','preflight_initial.log','preflight_exit_code.txt','check.log',
             'apt_install.log','su2_install.log','python_install.log','cloud_prepare.sh','cloud_finish.py']:
    (metadata/name).write_bytes((OPS/name).read_bytes())
subprocess.run([str(REPO/'.venv/bin/python'),'verify_snapshot.py'],cwd=REPO/'upstream_matlab',check=True,stdout=(metadata/'upstream_verification.log').open('w'))
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
files=sorted(p for p in OUT.rglob('*') if p.is_file())
hashes={p.relative_to(OUT).as_posix():digest(p) for p in files}
(OUT/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in hashes.items()),encoding='utf-8')
delivery=OPS/'aero_family_4d211b1_results.zip'
assert not delivery.exists(), 'Delivery archive already exists: inspect before replacing'
with zipfile.ZipFile(delivery,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): z.write(p,arcname=p.relative_to(OUT).as_posix())
record={'zip':str(delivery),'sha256':digest(delivery),'bytes':delivery.stat().st_size,'summary':summary}
(OPS/'delivery.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'delivery':str(delivery),'sha256':record['sha256'],'counts':summary['family_counts'],'selection_status':summary['selection_status']}),flush=True)
