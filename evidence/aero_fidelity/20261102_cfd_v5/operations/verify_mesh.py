"""Read-only verification of every v5 mesh against the retained v4 mesh."""
from pathlib import Path
import hashlib,json,collections
V5=Path('/hy-tmp/dqn-aero-v5-dc39a7f/runs/aero_fidelity_v5/cfd')
V4=Path('/hy-tmp/dqn-aero-v4-01589db/runs/aero_fidelity_v4/cfd')
OPS=Path('/hy-tmp/aero_operations_v5_20261102')
acceptance=json.loads((OPS/'accepted_base_deviation.json').read_text())
assert acceptance['accepted_by_user'] and acceptance['path']=='FAM_MD4/medium/mesh.su2'
def inspect(p):
    markers=[]; nmark=None; nelem=None; cells=collections.Counter(); left=0
    interior=hashlib.sha256()
    with p.open() as f:
        for line in f:
            s=line.strip()
            if s.startswith('MARKER_TAG='): markers.append(s.split('=',1)[1].strip())
            if s.startswith('NMARK='): nmark=int(s.split('=',1)[1])
            if nmark is None: interior.update(line.encode())
            if left:
                cells[int(s.split()[0])]+=1; left-=1
            if s.startswith('NELEM='): nelem=int(s.split('=',1)[1]); left=nelem
    return dict(markers=markers,nmark=nmark,elements=nelem,tetrahedra=cells[10],element_types=dict(cells),interior_sha256=interior.hexdigest(),mesh_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
meshes=[]
for p in sorted(V5.rglob('mesh.su2')):
    relative=p.relative_to(V5)
    new=inspect(p); old=inspect(V4/relative)
    info=json.loads((p.parent/'mesh_info.json').read_text())
    base_count_ok=info['base_surfaces']==1 or (relative.as_posix()==acceptance['path'] and info['base_surfaces']==2)
    meshes.append(dict(path=relative.as_posix(),v5=new,v4=old,mesh_info=info,base_surfaces_is_one=info['base_surfaces']==1,accepted_exception=info['base_surfaces']!=1 and base_count_ok,
        markers_valid=set(new['markers'])=={'body','base','farfield'} and new['nmark']==3 and len(new['markers'])==3 and base_count_ok,
        tetrahedra_identical=new['tetrahedra']==old['tetrahedra'],interior_identical=new['interior_sha256']==old['interior_sha256']))
assert len(meshes)==5,len(meshes)
cases=list(V5.glob('*/*/M*_a*/case.json')); assert len(cases)==29,len(cases)
case_settings=[]
for case in cases:
    meta=json.loads(case.read_text()); cfg=(case.parent/'case.cfg').read_text()
    assert meta['base_marker'] is True
    compact=''.join(cfg.split())
    assert 'MARKER_EULER=(body,base)' in compact
    assert 'MARKER_MONITORING=(body)' in compact
    previous=V4/case.relative_to(V5)
    old_cfg=(previous.parent/'case.cfg').read_text()
    same=cfg.replace('( body, base )','( body )')==old_cfg
    assert same,case
    old_meta=json.loads(previous.read_text())
    same_physics=all(meta[k]==old_meta[k] for k in ('geometry','level','mach','alpha_deg','freestream','surface'))
    assert same_physics,case
    case_settings.append(dict(case_id=case.parent.relative_to(V5).as_posix(),cfg_identical_except_two_marker_lines=same,physical_inputs_identical=same_physics))
result=dict(mesh_count=len(meshes),case_count=len(cases),markers_valid=all(m['markers_valid'] for m in meshes),all_base_surfaces_are_one=all(m['base_surfaces_is_one'] for m in meshes),accepted_deviation=acceptance,tetrahedra_all_identical=all(m['tetrahedra_identical'] for m in meshes),interiors_all_identical=all(m['interior_identical'] for m in meshes),meshes=meshes,case_settings=case_settings)
with (OPS/'mesh_verification.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
print(json.dumps(result,indent=2))
assert result['markers_valid']
