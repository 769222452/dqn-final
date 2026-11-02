"""Invoke a frozen v5 stage once and record timing. No scientific source edits."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
REPO=Path('/hy-tmp/dqn-aero-v5-dc39a7f')
OPS=Path('/hy-tmp/aero_operations_v5_20261102')
SHA='dc39a7f9e7c3d78f8bc66c660866907759362c24'
stage=sys.argv[1]
assert stage in ('cfd-mesh','cfd-run','collect')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
assert (OPS/'tests_passed').read_text().strip()=='PASS'
out=REPO/'runs/aero_fidelity_v5'
assert json.loads((out/'family_selected/family_audit.json').read_text())['status']=='SELECTED'
assert json.loads((OPS/'family_copy_verification.json').read_text())['all_identical']
previous={'cfd-mesh':None,'cfd-run':'cfd-mesh','collect':'cfd-run'}[stage]
if previous:
    prior=json.loads((OPS/(previous+'_state.json')).read_text())
    assert prior['status']=='COMPLETED' and prior['exit_code']==0,prior
if stage=='cfd-run':
    assert json.loads((OPS/'mesh_verification.json').read_text())['markers_valid']
with (OPS/(stage+'.started')).open('x') as f: f.write(str(os.getpid()))
env=os.environ.copy()
env.update(NP='7',JOBS='2',BASE_MARKER='1',OUT='runs/aero_fidelity_v5',PYTHONDONTWRITEBYTECODE='1',
           OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',
           OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1',PYTHONUNBUFFERED='1')
env['PATH']='/hy-tmp/dqn-aero/.venv/bin:/hy-tmp/aero_su2/bin:'+env['PATH']
record=dict(stage=stage,status='RUNNING',commit=SHA,NP=7,JOBS=2,BASE_MARKER=1,
            started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=['bash','experiments/aero_fidelity/run_decision.sh',stage])
def save(): (OPS/(stage+'_state.json')).write_text(json.dumps(record,indent=2)+'\n')
save(); tic=time.monotonic()
with (OPS/(stage+'.log')).open('x') as log:
    p=subprocess.run(record['command'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(status='COMPLETED' if p.returncode==0 else 'FAILED',exit_code=p.returncode,
              wall_s=time.monotonic()-tic,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
if stage=='cfd-run':
    progress=out/'cfd/progress.log'
    record['failed_cases']=[x.split(' ',1)[1] for x in progress.read_text().splitlines() if x.startswith('FAILED ')] if progress.exists() else []
save(); print(json.dumps(record),flush=True)
sys.exit(p.returncode)
