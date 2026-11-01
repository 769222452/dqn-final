"""Half-model (y>=0, symmetry plane) tetrahedral Euler mesh of one V6 wedge/waverider STEP for SU2.
usage: local_cfd_mesh.py <geometry_dir> <out_dir> <level: coarse|medium|fine>
STEP is in mm; the mesh is written in metres. Markers: BODY, BASE (CAD-only base cap), SYMMETRY, FARFIELD.
"""
import sys, json
from pathlib import Path
import gmsh

gdir, odir, level = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
odir.mkdir(parents=True, exist_ok=True)
# surface size on the body, size inside the shock layer box, far-field size
P = {'coarse': (0.04, 0.14, 1.0), 'medium': (0.02, 0.08, 0.8), 'fine': (0.02, 0.05, 0.8)}[level]
import os
if os.environ.get('MESH_SIZES'):
    P = tuple(float(v) for v in os.environ['MESH_SIZES'].split(','))
hs, hbox, hfar = P
X0, X1, Y1, Z0, Z1 = -2.5, 11.0, 4.0, -5.5, 3.5

gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 1)
gmsh.model.add('v6')
body = gmsh.model.occ.importShapes(str(gdir / 'geometry.step'))
body = [t for t in body if t[0] == 3]
gmsh.model.occ.dilate(body, 0, 0, 0, 0.001, 0.001, 0.001)   # mm -> m
box = gmsh.model.occ.addBox(X0, 0, Z0, X1 - X0, Y1, Z1 - Z0)
out, _ = gmsh.model.occ.cut([(3, box)], body, removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()
vols = [t for d, t in gmsh.model.getEntities(3)]
assert len(vols) == 1, vols
tol = 1e-6
far, sym, base, bod = [], [], [], []
for d, t in gmsh.model.getEntities(2):
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(d, t)
    if abs(x0 - X0) < tol and abs(x1 - X0) < tol or abs(x0 - X1) < tol and abs(x1 - X1) < tol \
       or abs(y0 - Y1) < tol and abs(y1 - Y1) < tol or abs(z0 - Z0) < tol and abs(z1 - Z0) < tol \
       or abs(z0 - Z1) < tol and abs(z1 - Z1) < tol:
        far.append(t)
    elif abs(y0) < tol and abs(y1) < tol:
        sym.append(t)
    elif abs(x0 - 5.0) < 1e-5 and abs(x1 - 5.0) < 1e-5:
        base.append(t)
    else:
        bod.append(t)
print('surfaces far/sym/base/body', far, sym, base, bod)
assert far and sym and base and bod
gmsh.model.addPhysicalGroup(2, bod, 1, 'BODY')
gmsh.model.addPhysicalGroup(2, base, 2, 'BASE')
gmsh.model.addPhysicalGroup(2, sym, 3, 'SYMMETRY')
gmsh.model.addPhysicalGroup(2, far, 4, 'FARFIELD')
gmsh.model.addPhysicalGroup(3, vols, 5, 'FLUID')

# size fields: fine near the body, refined box around the shock layer, coarse far away
f = gmsh.model.mesh.field
f.add('Distance', 1); f.setNumbers(1, 'SurfacesList', bod + base); f.setNumber(1, 'Sampling', 200)
f.add('Threshold', 2); f.setNumber(2, 'InField', 1); f.setNumber(2, 'SizeMin', hs); f.setNumber(2, 'SizeMax', hfar)
f.setNumber(2, 'DistMin', 0.02); f.setNumber(2, 'DistMax', 3.0)
f.add('Box', 3); f.setNumber(3, 'VIn', hbox); f.setNumber(3, 'VOut', hfar)
f.setNumber(3, 'XMin', -0.6); f.setNumber(3, 'XMax', 6.5); f.setNumber(3, 'YMin', 0); f.setNumber(3, 'YMax', 2.0)
f.setNumber(3, 'ZMin', -3.0); f.setNumber(3, 'ZMax', 1.0); f.setNumber(3, 'Thickness', 1.0)
f.add('Min', 4); f.setNumbers(4, 'FieldsList', [2, 3]); f.setAsBackgroundMesh(4)
gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
gmsh.option.setNumber('Mesh.Algorithm3D', 10)  # 10 = HXT; fine level needs classic Delaunay (HXT failed to recover facets)
gmsh.option.setNumber('General.NumThreads', 6)
gmsh.option.setNumber('Mesh.Optimize', 1)
gmsh.option.setNumber('Mesh.Algorithm', int(os.environ.get('MESH_2D', '6')))
gmsh.option.setNumber('Mesh.Smoothing', int(os.environ.get('MESH_SMOOTH', '1')))
gmsh.model.mesh.generate(2)
MESHED = False
for alg in (10, 1, 4):
    try:
        gmsh.option.setNumber('Mesh.Algorithm3D', alg); gmsh.model.mesh.generate(3); assert len(gmsh.model.mesh.getElements(3)[1]) and sum(len(e) for e in gmsh.model.mesh.getElements(3)[1]) > 0, 'no tets'; MESHED = True; print('3D algorithm used', alg); break
    except Exception as e:
        print('3D algorithm', alg, 'failed:', e)
assert MESHED
gmsh.option.setNumber('Mesh.Format', 42)  # su2
gmsh.write(str(odir / 'mesh.su2'))
nn = len(gmsh.model.mesh.getNodes()[0])
ne = sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])
# areas for the report (half model)
area = {n: sum(gmsh.model.occ.getMass(2, t) for t in lst) for n, lst in (('BODY', bod), ('BASE', base))}
json.dump({'level': level, 'hs': hs, 'hbox': hbox, 'hfar': hfar, 'nodes': nn, 'tets': ne, 'half_area_m2': area,
           'domain_m': [X0, X1, 0, Y1, Z0, Z1]}, open(odir / 'mesh_info.json', 'w'), indent=1)
print('nodes', nn, 'tets', ne, area)
gmsh.finalize()
