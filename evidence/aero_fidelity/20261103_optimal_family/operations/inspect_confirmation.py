"""Operations-only coverage audit; preserve the frozen confirmation verbatim."""
from pathlib import Path
import json,sys
REPO=Path('/hy-tmp/dqn-aero-opt-f248c4f')
OPS=Path('/hy-tmp/aero_opt_operations_20261103')
sys.path.insert(0,str(REPO/'scripts/cfd'))
from collect_su2 import load_cases,equal_lift_table,CL_TARGETS
cases=load_cases(REPO/'runs/aero_opt/cfd_root/cfd')
rows=equal_lift_table(cases,'medium')
conv=json.loads((OPS/'cfd_convergence.json').read_text())
coverage=[]
for g in sorted({c['geometry'] for c in cases}):
    for mach in (4.,6.,8.):
        sel=[c for c in cases if c['geometry']==g and c['mach']==mach]
        bad=[r['case_id'] for r in conv['records'] if r['case_id'].startswith(g+'/medium/M'+str(int(mach))+'_') and not r['converged_by_frozen_rule']]
        for model in ('euler','wedge'):
            values=[c[model+'_CL']+c['friction_CL'] for c in sel]
            covered=sorted(r['CL_target'] for r in rows if r['geometry']==g and r['mach']==mach and r['model']==model)
            coverage.append(dict(geometry=g,mach=mach,model=model,total_CL_min=min(values),total_CL_max=max(values),covered_CL=covered,
                missing_CL=[t for t in CL_TARGETS if t not in covered],unconverged_Euler_input_cases=bad if model=='euler' else []))
result=dict(purpose='Evidence-only audit of frozen confirmation inputs; no collect stage or new CFD execution',
    raw_confirmation_includes_unconverged_histories=True,all_inputs_converged=conv['converged']==conv['cases'],
    all_requested_CL_covered=all(not r['missing_CL'] for r in coverage),coverage=coverage,raw_cases=cases,equal_lift=rows,
    interpretation='Original confirmation is preserved; fits involving unconverged Euler cases are not independently validated.')
dest=OPS/'confirmation_coverage.json'
with dest.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('raw_cases','equal_lift')},indent=2))
