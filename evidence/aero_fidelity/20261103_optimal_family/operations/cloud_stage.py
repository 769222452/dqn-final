"""Operational runner only. Invoke each frozen stage once; never alter or retry scientific cases."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
REPO=Path('/hy-tmp/dqn-aero-opt-f248c4f')
OPS=Path('/hy-tmp/aero_opt_operations_20261103')
SHA='f248c4f7da03a48335cbe1dfca4e2b6e0e58c7da'
stage=sys.argv[1]
assert stage in ('check-a','check-b','evaluate','cfd-mesh','cfd-run','confirm')
record=dict(stage=stage,commit=SHA,status='STARTING',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
def save():
    temp=OPS/(stage+'_state.tmp'); temp.write_text(json.dumps(record,indent=2)+'\n'); temp.replace(OPS/(stage+'_state.json'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def passed(s):
    r=json.loads((OPS/(s+'_state.json')).read_text()); assert r['status']=='COMPLETED' and r['exit_code']==0,r
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SHA
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    assert (OPS/'tests_passed').read_text().strip()=='PASS'
    previous={'check-a':None,'check-b':'check-a','evaluate':'check-b','cfd-mesh':'evaluate','cfd-run':'cfd-mesh','confirm':'cfd-run'}[stage]
    if previous: passed(previous)
    env=os.environ.copy(); env.pop('MP_PW',None)
    env.update(NP='7',JOBS='2',SLICES='4',BASE_MARKER='1',ALPHAS='2 4 6 8',GRID_CHECK='0',PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',
               OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    env['PATH']='/hy-tmp/dqn-aero/.venv/bin:/hy-tmp/aero_su2/bin:'+env['PATH']
    if stage.startswith('check-'):
        letter=stage[-1]; out='runs/aero_opt_'+letter
        receipt=json.loads((OPS/f'import_{letter}.json').read_text()); assert receipt['verified']
        assert not (REPO/out/'family_selected').exists()
        cmd=['bash','experiments/aero_fidelity/run_decision.sh','check']
    elif stage in ('evaluate','confirm'):
        out='runs/aero_opt/results'
        audit=json.loads((REPO/'runs/aero_opt_b/family_selected/family_audit.json').read_text()); assert audit['status']=='MAP_ONLY'
        cmd=['/hy-tmp/dqn-aero/.venv/bin/python','-B','experiments/aero_fidelity/optimum/evaluate_optimum.py',
             'runs/aero_opt_b/family','--audit','runs/aero_opt_b/family_selected/family_audit.json','--out',out]
        if stage=='confirm':
            assert (OPS/'cfd_convergence.json').is_file(),'Inspect all CFD logs and record convergence before confirmation'
            cmd+=['--confirm-cfd','runs/aero_opt/cfd_root/cfd']
            assert not (REPO/out/'euler_confirmation.json').exists()
        else: assert not (REPO/out).exists()
    else:
        decision=json.loads((REPO/'runs/aero_opt/results/optimum_decision.json').read_text()); assert decision['status']=='EVALUATED'
        out='runs/aero_opt/cfd_root'
        if stage=='cfd-mesh':
            src=REPO/'runs/aero_opt/results/confirm'; dest=REPO/out/'family_selected'
            assert not dest.exists(); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(src,dest)
            hashes={p.relative_to(src).as_posix():sha(p) for p in src.rglob('*') if p.is_file()}
            assert hashes=={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
            (OPS/'confirm_copy.json').write_text(json.dumps(hashes,indent=2)+'\n')
        else:
            assert json.loads((OPS/'mesh_verification.json').read_text())['valid']
            assert not list((REPO/out/'cfd').rglob('su2.log')),'Existing CFD attempt; no automatic duplicate or retry'
        cmd=['bash','experiments/aero_fidelity/run_decision.sh',stage]
    env['OUT']=out
    with (OPS/(stage+'.started')).open('x') as f: f.write(str(os.getpid()))
    record.update(status='RUNNING',command=cmd,OUT=out,NP=7,JOBS=2,SLICES=4,BASE_MARKER=1,ALPHAS=[2,4,6,8],GRID_CHECK=0); save()
    tic=time.monotonic()
    with (OPS/(stage+'.log')).open('x') as log:
        p=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='COMPLETED' if p.returncode==0 else 'FAILED',exit_code=p.returncode,wall_s=time.monotonic()-tic,
                  finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()); save()
    if stage.startswith('check-') and p.returncode==0:
        audit=json.loads((REPO/out/'family_selected/family_audit.json').read_text())
        record['audit_summary']={k:audit[k] for k in ('jobs','generated','valid','outcomes','status')}; save()
        assert audit['status']=='MAP_ONLY'
    if stage=='cfd-run':
        progress=REPO/out/'cfd/progress.log'
        record['failed_cases']=[x[7:] for x in progress.read_text().splitlines() if x.startswith('FAILED ')] if progress.exists() else ['missing progress.log']; save()
    print(json.dumps(record),flush=True)
    return p.returncode
try: sys.exit(main())
except Exception as exc:
    record.update(status='FAILED',error=f'{type(exc).__name__}: {exc}'); save(); traceback.print_exc(); raise
