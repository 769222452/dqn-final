"""Read-only monitoring, including logs and actual residual/Cauchy evidence."""
from pathlib import Path
import csv,datetime,json,subprocess
OPS=Path('/hy-tmp/aero_operations_v5_20261102')
OUT=Path('/hy-tmp/dqn-aero-v5-dc39a7f/runs/aero_fidelity_v5')
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
for p in sorted(OPS.glob('*_state.json')): print(p.name,p.read_text())
for name in ('cfd-mesh','cfd-run','collect'):
    p=OPS/(name+'.log')
    if p.exists(): print(name,'last lines:', '\n'.join(p.read_text(errors='replace').splitlines()[-8:]))
records=[]
for p in sorted((OUT/'cfd').glob('*/*/M*_a*/case.json')):
    r={'case_id':p.parent.relative_to(OUT/'cfd').as_posix(),'done':(p.parent/'done').exists()}
    h=p.parent/'history.csv'; log=p.parent/'su2.log'
    if h.exists():
        try:
            rows=list(csv.reader(h.open())); row=dict(zip([x.strip().strip('"') for x in rows[0]],[float(x) for x in rows[-1]]))
            r.update(iter=row.get('Inner_Iter'),rms_rho=row.get('rms[Rho]'),CL=row.get('CL'),CD=row.get('CD'))
        except Exception as e: r['history_read_error']=repr(e)
    if log.exists():
        lines=log.read_text(errors='replace').splitlines()
        r['cauchy_CL']=any('Cauchy[CL]' in x and 'Yes' in x for x in lines)
        r['cauchy_CD']=any('Cauchy[CD]' in x and 'Yes' in x for x in lines)
    records.append(r)
print(json.dumps(dict(cases=len(records),done=sum(x['done'] for x in records),records=records),indent=2))
for cmd in (['df','-h','/hy-tmp'],['tmux','ls']): subprocess.run(cmd,check=False)
lines=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,args'],text=True).splitlines()
print('\n'.join(x for x in lines if any(s in x for s in ('SU2_CFD','make_su2_cases.py','cloud_stage.py'))))
