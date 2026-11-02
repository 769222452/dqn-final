"""Read-only examination of the frozen MD4 base marker; no mesh or parameter edits."""
from pathlib import Path
import json,hashlib
import numpy as np
base=Path('/hy-tmp/dqn-aero-v5-dc39a7f/runs/aero_fidelity_v5/cfd/FAM_MD4/medium')
p=base/'mesh.su2'
points={}; marker=None; triangles={}; interior=hashlib.sha256()
with p.open() as f:
    for line in f:
        s=line.strip()
        if s.startswith('NPOIN='):
            for _ in range(int(s.split('=')[1])):
                row=f.readline().split(); points[int(row[-1])]=np.array([float(v) for v in row[:3]])
        elif s.startswith('MARKER_TAG='): marker=s.split('=')[1].strip()
        elif s.startswith('MARKER_ELEMS='):
            triangles[marker]=[]
            for _ in range(int(s.split('=')[1])):
                row=[int(x) for x in f.readline().split()]; assert row[0]==5
                triangles[marker].append(row[1:4])
t=np.array([[points[i] for i in tri] for tri in triangles['base']])
cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])/2
parent={i:i for tri in triangles['base'] for i in tri}
def find(i):
    while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
    return i
for tri in triangles['base']:
    for i in tri[1:]: parent[find(i)]=find(tri[0])
def internal_hash(path):
    h=hashlib.sha256()
    for line in path.open():
        if line.startswith('NMARK='): break
        h.update(line.encode())
    return h.hexdigest()
old=Path('/hy-tmp/dqn-aero-v4-01589db/runs/aero_fidelity_v4/cfd/FAM_MD4/medium/mesh.su2')
report=dict(mesh_info=json.loads((base/'mesh_info.json').read_text()),markers={m:len(ts) for m,ts in triangles.items()},
    base_x_min=float(t[:,:,0].min()),base_x_max=float(t[:,:,0].max()),base_area=float(np.linalg.norm(cross,axis=1).sum()),
    base_connected_components=len({find(i) for i in parent}),base_normal_x_positive=int((cross[:,0]>0).sum()),base_normal_x_negative=int((cross[:,0]<0).sum()),
    volume_mesh_identical=internal_hash(p)==internal_hash(old),
    observation='Two elementary base surfaces are grouped in one physical base marker; recorded without changing any geometry or setting.')
dest=Path('/hy-tmp/aero_operations_v5_20261102/base_review.json')
with dest.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
print(json.dumps(report,indent=2))
