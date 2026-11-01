"""SU2 Euler cases for the 3D off-design check of the waverider database (experiments/aero_fidelity).

The aerodynamic snapshot is a strip shock-expansion model: every surface strip is a 2D wedge, with no
spanwise flow, so it cannot represent leading-edge shock detachment and spill-over off the design Mach. A
small Euler campaign tells whether the full 3D flow ranks the geometries differently, i.e. whether
waveriders designed for a lower Mach beat the high-design-Mach shapes there. That is the physical basis for
morphing. This script builds, for each case:

- surface.stl: closed full body (mirrored half body + planar base), body axes x aft, z up;
- mesh.su2: tetrahedral volume mesh (gmsh) of a box around the body; markers 'body' and 'farfield';
- case.cfg: SU2 Euler setup (free stream at the altitude where q = Q_PA for that Mach).

    python scripts/cfd/make_su2_cases.py --out runs/cfd --level coarse          # 24 cases
    python scripts/cfd/make_su2_cases.py --out runs/cfd --level medium --geometries S2_MD12_dd296f618e

Run on the server (SU2 >= 7, MPI): experiments/aero_fidelity/run_decision.sh runs each cases_<level>.txt in
parallel (NP MPI ranks per case); then python scripts/cfd/collect_su2.py runs/cfd.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight.friction import GAMMA, R_AIR, SREF  # noqa: E402
from certified_flight.mission import Mission  # noqa: E402

SNAPSHOT = ROOT/'data/geometry_subset'
# Best (P2 + friction L/D) member of each design-Mach family, chosen by experiments/aero_fidelity/select_snapshot_equal_volume.json
DEFAULT_GEOMETRIES = None
MACHS = (4., 6., 8.)
ALPHAS_DEG = (2., 4., 6.)    # three alphas per (geometry, Mach): equal-lift comparison by a quadratic CD(CL)
Q_PA = 50e3
LEVELS = dict(coarse=dict(body=.06, edge=.02, far=1.2), medium=dict(body=.035, edge=.01, far=1.), fine=dict(body=.02, edge=.006, far=.8))


def _ear_clip(poly):
    """Triangulate a simple polygon [n, 2] (counter-clockwise or clockwise) by ear clipping."""
    pts = list(range(len(poly)))
    area = .5*np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-np.roll(poly[:, 0], -1)*poly[:, 1])
    sign = 1. if area > 0 else -1.
    tris = []

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    guard = 0
    while len(pts) > 3 and guard < 100000:
        guard += 1
        for k in range(len(pts)):
            i, j, l = pts[k-1], pts[k], pts[(k+1) % len(pts)]
            a, b, c = poly[i], poly[j], poly[l]
            if sign*cross(a, b, c) <= 1e-14:
                continue
            inside = False
            for m in pts:
                if m in (i, j, l):
                    continue
                p = poly[m]
                if sign*cross(a, b, p) > 0 and sign*cross(b, c, p) > 0 and sign*cross(c, a, p) > 0:
                    inside = True
                    break
            if not inside:
                tris.append((i, j, l))
                pts.pop(k)
                break
        else:
            raise RuntimeError('ear clipping failed (non-simple base polygon)')
    tris.append(tuple(pts))
    return tris


def _inside(poly, pts):
    """Even-odd ray casting: which of pts [m, 2] lie inside the simple polygon poly [n, 2]."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None], poly[:, 1][None]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2-x1)*(y-y1)/np.where(y2 == y1, 1e-300, y2-y1)+x1)
    return crosses.sum(axis=1) % 2 == 1


def _base_triangles(poly):
    """Triangulate the planar base polygon [n, 2]: Delaunay of the boundary points, keeping the triangles
    whose centroid is inside the polygon (well-shaped triangles; ear clipping gives long slivers). Falls back
    to ear clipping if the result does not tile the polygon."""
    from scipy.spatial import Delaunay
    tri = Delaunay(poly).simplices
    inside = _inside(poly, poly[tri].mean(axis=1))
    tri = tri[inside]
    area_poly = abs(.5*np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-np.roll(poly[:, 0], -1)*poly[:, 1]))
    a = poly[tri]
    area_tri = np.abs(.5*((a[:, 1, 0]-a[:, 0, 0])*(a[:, 2, 1]-a[:, 0, 1])-(a[:, 2, 0]-a[:, 0, 0])*(a[:, 1, 1]-a[:, 0, 1]))).sum()
    if abs(area_tri/area_poly-1) < 1e-6:
        return [tuple(t) for t in tri]
    return _ear_clip(poly)


def closed_surface(path):
    """(vertices [n, 3], triangles [m, 3]) of the closed full body in body axes (x aft, y right, z up)."""
    z = loadmat(path)
    flip = np.array([1., -1., -1.])
    lo = np.stack([z['wav_'+c] for c in 'xyz'], -1)*flip
    up = np.stack([z['wav_up_'+c] for c in 'xyz'], -1)*flip
    up[:, 0] = lo[:, 0]                  # shared leading edge (identical to 1e-15 in the file)
    lo[-1, :, 1] = up[-1, :, 1] = 0.     # centreline exactly on the symmetry plane
    V, T = [], []

    def grid(P, reverse):
        base = sum(len(v) for v in V)
        V.append(P.reshape(-1, 3))
        n1, n2 = P.shape[:2]
        idx = lambda i, j: base+i*n2+j
        for i in range(n1-1):
            for j in range(n2-1):
                q = (idx(i, j), idx(i+1, j), idx(i+1, j+1), idx(i, j+1))
                t1, t2 = (q[0], q[1], q[2]), (q[0], q[2], q[3])
                T.extend([t1[::-1], t2[::-1]] if reverse else [t1, t2])

    mirror = np.array([1., -1., 1.])
    for P, rev in ((lo, False), (up, True)):
        grid(P, rev)
        grid(P*mirror, not rev)
    # planar base x = x_max: loop = lower base curve (tip -> centre -> other tip) + upper curve back
    lb, ub = lo[:, -1], up[:, -1]                              # rows: tip (0) ... centre (-1)
    loop = np.concatenate([lb, (lb*mirror)[::-1][1:], (ub*mirror)[1:], ub[::-1][1:-1]])
    keep = np.r_[True, np.linalg.norm(np.diff(loop[:, 1:], axis=0), axis=1) > 1e-10]
    loop = loop[keep]
    base = sum(len(v) for v in V)
    V.append(loop)
    for t in _base_triangles(loop[:, 1:]):
        T.append(tuple(base+k for k in t))
    V = np.concatenate(V)
    T = np.asarray(T)
    # merge coincident vertices (leading edge, centreline, collapsed tip) and drop degenerate triangles
    from scipy.spatial import cKDTree
    parent = np.arange(len(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in cKDTree(V).query_pairs(1e-9):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    root = np.array([find(i) for i in range(len(V))])
    keep, inverse = np.unique(root, return_inverse=True)
    V = V[keep]
    T = inverse[T]
    # only triangles with a repeated vertex are degenerate; thin slivers near the nose are kept (they close
    # the surface)
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    # orient outward: positive enclosed volume
    vol = np.sum(np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])))/6
    if vol < 0:
        T = T[:, ::-1]
    return V, T


def check_closed(T):
    """Every edge must be shared by exactly two triangles (watertight, manifold)."""
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return dict(edges=int(len(counts)), open_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum()))


def write_stl(path, V, T):
    with open(path, 'w') as f:
        f.write('solid body\n')
        for t in T:
            a, b, c = V[t]
            n = np.cross(b-a, c-a)
            n /= max(np.linalg.norm(n), 1e-300)
            f.write(f' facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n  outer loop\n')
            for p in (a, b, c):
                f.write(f'   vertex {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n')
            f.write('  endloop\n endfacet\n')
        f.write('endsolid body\n')


EDGE_SCALES = (1., 1.5, 2., 3.)      # retried in this order when the volume mesher fails or returns no elements


def mesh(stl, out, level):
    """Volume mesh; returns dict(cells, edge_scale, algorithm, remesh_surface).

    First choice: gmsh re-meshes the surface to the size fields (resolution follows the level). If it fails (gmsh
    can produce overlapping facets on thin patches near the wing tip), the validated STL triangulation is used
    as the surface mesh and only the volume is meshed. Frontal 3D (algorithm 4) is not used: it crashes gmsh on
    these surfaces. What was actually used is returned and stored next to the mesh."""
    errors = []
    for remesh in (True, False):
        for scale in EDGE_SCALES if remesh else (1.,):
            for algorithm in (10, 1):                             # HXT, Delaunay
                try:
                    info = _mesh_once(stl, out, level, scale, algorithm, remesh)
                except Exception as exc:                          # noqa: BLE001 - gmsh raises bare Exception
                    errors.append(f'remesh={remesh}, edge x{scale:g}, algorithm {algorithm}: {exc}')
                    continue
                if info['cells'] > 0:
                    return info
                errors.append(f'remesh={remesh}, edge x{scale:g}, algorithm {algorithm}: no tetrahedra')
    raise RuntimeError('volume meshing failed for '+str(stl)+':\n  '+'\n  '.join(errors))


def _mesh_once(stl, out, level, scale, algorithm, remesh=True):
    import gmsh
    lc = dict(LEVELS[level])
    lc['edge'] = lc['edge']*scale
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('wr')
    gmsh.merge(str(stl))
    gmsh.model.mesh.classifySurfaces(np.deg2rad(30), True, remesh, np.pi)
    if remesh:
        gmsh.model.mesh.createGeometry()
    body = [s[1] for s in gmsh.model.getEntities(2)]
    sl_body = gmsh.model.geo.addSurfaceLoop(body)
    gmsh.model.geo.synchronize()
    # farfield box (hypersonic: short upstream, longer downstream)
    pts = [(-3, -8, -8), (15, -8, -8), (15, 8, -8), (-3, 8, -8), (-3, -8, 8), (15, -8, 8), (15, 8, 8), (-3, 8, 8)]
    p = [gmsh.model.geo.addPoint(*q, lc['far']) for q in pts]
    lines = {}

    def edge(a, b):                        # oriented curve tag from vertex a to vertex b
        if (a, b) in lines:
            return lines[(a, b)]
        if (b, a) in lines:
            return -lines[(b, a)]
        lines[(a, b)] = gmsh.model.geo.addLine(p[a], p[b])
        return lines[(a, b)]
    cycles = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    surf = [gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([edge(c[k], c[(k+1) % 4]) for k in range(4)])])
            for c in cycles]
    l = {k: abs(v) for k, v in lines.items()}
    sl_far = gmsh.model.geo.addSurfaceLoop(surf)
    vol = gmsh.model.geo.addVolume([sl_far, sl_body])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, body, name='body')
    gmsh.model.addPhysicalGroup(2, surf, name='farfield')
    gmsh.model.addPhysicalGroup(3, [vol], name='fluid')
    # size field: fine on the body, finest along sharp edges (leading edge), growing outward
    gmsh.model.mesh.field.add('Distance', 1)
    gmsh.model.mesh.field.setNumbers(1, 'SurfacesList', body)
    gmsh.model.mesh.field.add('Threshold', 2)
    gmsh.model.mesh.field.setNumber(2, 'InField', 1)
    gmsh.model.mesh.field.setNumber(2, 'SizeMin', lc['body'])
    gmsh.model.mesh.field.setNumber(2, 'SizeMax', lc['far'])
    gmsh.model.mesh.field.setNumber(2, 'DistMin', .05)
    gmsh.model.mesh.field.setNumber(2, 'DistMax', 4.)
    curves = [c[1] for c in gmsh.model.getEntities(1) if c[1] not in l.values()]
    gmsh.model.mesh.field.add('Distance', 3)
    gmsh.model.mesh.field.setNumbers(3, 'CurvesList', curves)
    gmsh.model.mesh.field.add('Threshold', 4)
    gmsh.model.mesh.field.setNumber(4, 'InField', 3)
    gmsh.model.mesh.field.setNumber(4, 'SizeMin', lc['edge'])
    gmsh.model.mesh.field.setNumber(4, 'SizeMax', lc['far'])
    gmsh.model.mesh.field.setNumber(4, 'DistMin', .02)
    gmsh.model.mesh.field.setNumber(4, 'DistMax', 1.)
    gmsh.model.mesh.field.add('Min', 5)
    gmsh.model.mesh.field.setNumbers(5, 'FieldsList', [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    for k in ('Mesh.MeshSizeExtendFromBoundary', 'Mesh.MeshSizeFromPoints', 'Mesh.MeshSizeFromCurvature'):
        gmsh.option.setNumber(k, 0)
    gmsh.option.setNumber('Mesh.Algorithm3D', algorithm)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber('Mesh.SaveAll', 0)
    gmsh.write(str(out))
    n = int(gmsh.model.mesh.getElementsByType(4)[0].size)
    gmsh.finalize()
    return dict(cells=n, edge_scale=scale, algorithm=algorithm, remesh_surface=remesh)


def freestream(mission, M, q=Q_PA):
    atm = mission.model(None).atm
    h = brentq(lambda h: .5*float(atm.density(h))*(M*float(atm.speed_of_sound(h)))**2-q, 10e3, 70e3)
    a = float(atm.speed_of_sound(h))
    T = a*a/(GAMMA*R_AIR)
    rho = float(atm.density(h))
    return dict(altitude_m=h, T_K=T, rho_kgm3=rho, p_Pa=rho*R_AIR*T)


CFG = """% SU2 Euler, waverider off-design check (generated by scripts/cfd/make_su2_cases.py)
SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= {mach}
AOA= {alpha}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= {p}
FREESTREAM_TEMPERATURE= {T}
GAMMA_VALUE= 1.4
GAS_CONSTANT= 287.05287
REF_ORIGIN_MOMENT_X= 2.5
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.9481139754931033
REF_LENGTH= 5.0
REF_AREA= {sref}
REF_DIMENSIONALIZATION= DIMENSIONAL
MARKER_EULER= ( body )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( body )
MARKER_MONITORING= ( body )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 0.5
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.2, 0.5, 20.0 )
ITER= {iters}
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_FLOW= AUSM
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_FIELD= ( LIFT, DRAG )
CONV_CAUCHY_ELEMS= 200
CONV_CAUCHY_EPS= 1E-5
CONV_RESIDUAL_MINVAL= -9
MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
TABULAR_FORMAT= CSV
CONV_FILENAME= history
OUTPUT_FILES= ( RESTART, SURFACE_CSV )
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--level', default='coarse', choices=sorted(LEVELS))
    ap.add_argument('--geometries', nargs='*', help='default: experiments/aero_fidelity/select_snapshot_equal_volume.json')
    ap.add_argument('--machs', nargs='*', type=float, default=list(MACHS))
    ap.add_argument('--alphas', nargs='*', type=float, default=list(ALPHAS_DEG))
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--no-mesh', action='store_true', help='write surfaces and configs only')
    ap.add_argument('--geometry-dir', default=str(SNAPSHOT/'geometries'),
                    help='folder with <geometry_id>/native_geometry.mat (default: the 298-geometry snapshot)')
    ap.add_argument('--select', default=str(ROOT/'experiments/aero_fidelity/select_snapshot_equal_volume.json'),
                    help='JSON with cfd_geometries (used when --geometries is not given)')
    a = ap.parse_args(argv)
    gids = a.geometries or json.loads(Path(a.select).read_text())['cfd_geometries']
    gdir_root = Path(a.geometry_dir)
    mission = Mission(ROOT, 'configs/mission.yaml')
    out = Path(a.out)
    cases = []
    for g in gids:
        gdir = out/g/a.level
        gdir.mkdir(parents=True, exist_ok=True)
        V, T = closed_surface(gdir_root/g/'native_geometry.mat')
        closed = check_closed(T)
        if closed['open_edges'] or closed['nonmanifold_edges']:
            raise RuntimeError(f'{g}: surface not watertight {closed}')
        write_stl(gdir/'surface.stl', V, T)
        if a.no_mesh:
            minfo = None
        elif (gdir/'mesh.su2').exists() and (gdir/'mesh_info.json').exists():
            minfo = json.loads((gdir/'mesh_info.json').read_text())
        else:
            minfo = mesh(gdir/'surface.stl', gdir/'mesh.su2', a.level)
            (gdir/'mesh_info.json').write_text(json.dumps(minfo))
        for M in a.machs:
            fs = freestream(mission, M)
            for al in a.alphas:
                cdir = gdir/f'M{M:g}_a{al:g}'
                cdir.mkdir(parents=True, exist_ok=True)
                (cdir/'case.cfg').write_text(CFG.format(mach=M, alpha=al, p=fs['p_Pa'], T=fs['T_K'], sref=SREF,
                                                         iters=a.iters, mesh='../mesh.su2'))
                meta = dict(geometry=g, geometry_mat=str(gdir_root/g/'native_geometry.mat'), level=a.level, mach=M,
                            alpha_deg=al, freestream=fs, surface=closed, mesh=minfo)
                (cdir/'case.json').write_text(json.dumps(meta, indent=1))
                cases.append(str(cdir.relative_to(out)))
        print(json.dumps(dict(geometry=g, level=a.level, triangles=int(len(T)), mesh=minfo, **closed)), flush=True)
    lst = out/f'cases_{a.level}.txt'
    lst.write_text(''.join(c+'\n' for c in cases))
    print(f'{len(cases)} cases -> {lst}')


if __name__ == '__main__':
    main()
