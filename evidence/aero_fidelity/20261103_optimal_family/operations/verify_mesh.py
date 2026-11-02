"""Record meshes and cases without modifying the frozen experiment."""
from pathlib import Path
import collections,hashlib,json
REPO=Path('/hy-tmp/dqn-aero-opt-f248c4f'); OPS=Path('/hy-tmp/aero_opt_operations_20261103')
CFD=REPO/'runs/aero_opt/cfd_root/cfd'
decision=json.loads((REPO/'runs/aero_opt/results/optimum_decision.json').read_text())
ids=decision['euler_confirmation_ids']; assert decision['status']=='EVALUATED'
meshes=[]
for gid in ids:
    p=CFD/gid/'medium/mesh.su2'; info=json.loads((p.parent/'mesh_info.json').read_text())
    markers=[]; cells=collections.Counter(); left=0; nmark=0
    with p.open() as f:
        for line in f:
            s=line.strip()
            if left: cells[int(s.split()[0])]+=1; left-=1
            if s.startswith('NELEM='): left=int(s.split('=',1)[1])
            if s.startswith('NMARK='): nmark=int(s.split('=',1)[1])
            if s.startswith('MARKER_TAG='): markers.append(s.split('=',1)[1].strip())
    valid=set(markers)=={'body','base','farfield'} and len(markers)==nmark==3 and info['base_surfaces']>=1
    meshes.append(dict(geometry=gid,markers=markers,base_surfaces=info['base_surfaces'],tetrahedra=cells[10],element_types=dict(cells),valid=valid,
                       mesh_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mesh_info=info))
cases=[]
for p in sorted(CFD.glob('*/*/M*_a*/case.json')):
    meta=json.loads(p.read_text()); cfg=''.join((p.parent/'case.cfg').read_text().split())
    valid=meta['base_marker'] and meta['geometry'] in ids and meta['level']=='medium' and meta['mach'] in (4,6,8) and meta['alpha_deg'] in (2,4,6,8)
    valid=valid and 'MARKER_EULER=(body,base)' in cfg and 'MARKER_MONITORING=(body)' in cfg
    cases.append(dict(case_id=p.parent.relative_to(CFD).as_posix(),geometry=meta['geometry'],mach=meta['mach'],alpha_deg=meta['alpha_deg'],valid=valid))
expected={(g,m,a) for g in ids for m in (4,6,8) for a in (2,4,6,8)}
valid=(all(x['valid'] for x in meshes+cases) and len(cases)==len(expected) and
       {(x['geometry'],x['mach'],x['alpha_deg']) for x in cases}==expected and
       not (CFD/'cases_coarse.txt').exists() and not (CFD/'cases_fine.txt').exists())
result=dict(valid=valid,mesh_count=len(meshes),case_count=len(cases),expected_case_count=len(expected),meshes=meshes,cases=cases,GRID_CHECK=0)
with (OPS/'mesh_verification.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
print(json.dumps(result,indent=2)); assert valid
