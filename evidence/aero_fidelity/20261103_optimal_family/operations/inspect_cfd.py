"""Capture final residual/Cauchy evidence before original evaluate_optimum --confirm-cfd."""
from pathlib import Path
import csv,json,re,subprocess
OPS=Path('/hy-tmp/aero_opt_operations_20261103'); CFD=Path('/hy-tmp/dqn-aero-opt-f248c4f/runs/aero_opt/cfd_root/cfd')
stage=json.loads((OPS/'cfd-run_state.json').read_text()); assert stage['status'] in ('COMPLETED','FAILED')
records=[]
for p in sorted(CFD.glob('*/*/M*_a*/case.json')):
    h=p.parent/'history.csv'; log=p.parent/'su2.log'; r=dict(case_id=p.parent.relative_to(CFD).as_posix(),done=(p.parent/'done').exists(),last_residual=None)
    if h.exists():
        try:
            with h.open(newline='') as f: rows=list(csv.reader(f))
            r['last_history']=dict(zip([x.strip().strip('"') for x in rows[0]],[float(x) for x in rows[-1]])); r['last_residual']=r['last_history'].get('rms[Rho]')
        except Exception as e: r['history_parse_error']=repr(e)
    text=log.read_text(errors='replace') if log.exists() else ''; lines=text.splitlines()
    r['cauchy_CL_pass']=any('Cauchy[CL]' in x and 'Yes' in x for x in lines)
    r['cauchy_CD_pass']=any('Cauchy[CD]' in x and 'Yes' in x for x in lines)
    r['cauchy_evidence']=[x for x in lines if 'Cauchy[CL]' in x or 'Cauchy[CD]' in x]
    r['density_pass']=r['last_residual'] is not None and r['last_residual'] < -5
    r['converged_by_frozen_rule']=r['done'] and (r['density_pass'] or (r['cauchy_CL_pass'] and r['cauchy_CD_pass']))
    r['solver_exit_success']='Exit Success (SU2_CFD)' in text
    r['all_criteria_satisfied']='All convergence criteria satisfied.' in text
    r['warnings']=[x for x in lines if 'Warning:' in x]
    r['nonphysical_points']=[int(x) for x in re.findall(r'there are (\d+) non-physical points',text)]
    r['nonphysical_reconstructed_states']=[int(x) for x in re.findall(r'Warning: (\d+) reconstructed states',text)]
    r['log_tail']=lines[-40:]; r['observed_case_wall_s']=None
    if r['done'] and log.exists():
        birth=int(subprocess.check_output(['stat','-c','%W',str(log)],text=True).strip())
        if birth>0: r['observed_case_wall_s']=(p.parent/'done').stat().st_mtime-birth
    records.append(r)
summary=dict(cases=len(records),done=sum(r['done'] for r in records),converged=sum(r['converged_by_frozen_rule'] for r in records),
             residual_units='log10; below -5 means density residual below 1e-5',
             timing_method='su2.log birth (integer seconds) to done; approximate, includes MPI startup/output',
             not_converged=[dict(case_id=r['case_id'],last_residual=r['last_residual']) for r in records if not r['converged_by_frozen_rule']],records=records)
with (OPS/'cfd_convergence.json').open('x') as f: json.dump(summary,f,indent=2); f.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
