"""Build a real B-rep solid (OpenCascade) from a native teacher MAT grid, write STEP (mm), re-import and check.

Run with .venv_cad (cadquery-ocp).  Geometry frame = aero body frame used by offdesign_aero_v7:
  x downstream, z up, y_b = -y_native, z_b = -z_native (native frame diag(1,-1,-1)), program length unit 1 m -> STEP in mm (x1000).
Full body = native half model + its y-mirror.  Skins are interpolating cubic B-spline surfaces through the native grid nodes (99 x 50).
The native audited geometry has an OPEN base; for a CAD solid a ruled base cap between the two skin trailing-edge curves is added and
recorded as CAD_ONLY_BASE_CAP (the aero model treats the base separately and never uses this cap).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
from OCP.gp import gp_Pnt
from OCP.collections import Array2_gp_Pnt
from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCP.GeomFill import GeomFill
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS
from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopExp import TopExp
from OCP.ShapeFix import ShapeFix_Solid, ShapeFix_Shape
from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRep import BRep_Tool
from OCP.GeomAbs import GeomAbs_C2

SCALE = 1000.0   # program metre -> STEP millimetre


def load_native(mat_path):
    S = sio.loadmat(str(mat_path))
    lo = np.stack([S["wav_x"], S["wav_y"], S["wav_z"]], -1).astype(float)
    up = np.stack([S["wav_up_x"], S["wav_up_y"], S["wav_up_z"]], -1).astype(float)
    return lo, up


def body_full_grid(P):
    """native half grid [ni,nj,3] -> body frame full grid [2ni-1,nj,3] (tip(y<0) ... symmetry ... tip(y>0))."""
    B = P * np.array([1.0, -1.0, -1.0])
    M = B * np.array([1.0, -1.0, 1.0])
    return np.concatenate([B[:-1], M[::-1]], axis=0)


def mesh_volume(lo, up):
    """Divergence-theorem volume (m^3) of the closed polyhedron: lower+upper quad skins + flat-quad base strip, full body (independent of OCC)."""
    L, U = body_full_grid(lo), body_full_grid(up)
    vol = 0.0
    def quad_vol(g, outward_sign):
        v = 0.0
        p1 = g[:-1, :-1]; p2 = g[1:, :-1]; p3 = g[1:, 1:]; p4 = g[:-1, 1:]
        for a, b, c in ((p1, p2, p3), (p1, p3, p4)):
            v += outward_sign * np.einsum("ijk,ijk->", a, np.cross(b, c)) / 6.0
        return v
    # choose sign by checking positivity at the end
    base = np.concatenate([L[:, -1], U[::-1, -1]], axis=0)   # TE loop (lower forward, upper back) used for cap polygon
    vl = quad_vol(L, +1); vu = quad_vol(U, -1)
    # cap: strip between lower and upper TE curves
    a, b = L[:-1, -1], L[1:, -1]; c, d = U[1:, -1], U[:-1, -1]
    vc = 0.0
    for p, q, r in ((a, b, c), (a, c, d)):
        vc += np.einsum("ik,ik->", p, np.cross(q, r)) / 6.0
    cands = []
    for s in (1.0, -1.0):
        cands.append(s * (vl + vu + vc))
    return max(cands, key=abs) if False else abs(vl + vu + vc), (vl, vu, vc)


def _halves(P):
    """native half grid -> body-frame left (y<=0) and right (y>=0) grids [ni,nj,3]; symmetry row forced to y=0 exactly (native value is ~7e-17 m)."""
    B = P * np.array([1.0, -1.0, -1.0]); B = B.copy(); B[-1, :, 1] = 0.0
    return B, (B * np.array([1.0, -1.0, 1.0]))[::-1]


def _surface(grid_mm):
    from OCP.Approx import Approx_ParametrizationType
    nu, nv, _ = grid_mm.shape
    arr = Array2_gp_Pnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            arr.SetValue(i + 1, j + 1, gp_Pnt(*grid_mm[i, j]))
    g = GeomAPI_PointsToBSplineSurface()
    g.Interpolate(arr, Approx_ParametrizationType.Approx_IsoParametric)   # cubic C2 B-spline passing exactly through every native node
    return g.Surface()


def build_solid(lo, up, sew_tol=0.01):
    faces = []
    for k, (a, b) in enumerate(zip(_halves(lo), _halves(up))):       # k=0 left half, k=1 right half
        sl, su = _surface(a * SCALE), _surface(b * SCALE)
        faces += [BRepBuilderAPI_MakeFace(sl, 1e-6).Face(), BRepBuilderAPI_MakeFace(su, 1e-6).Face()]
        cl = sl.VIso(sl.Bounds()[3]); cu = su.VIso(su.Bounds()[3])    # trailing-edge iso curves of lower / upper skin
        faces.append(BRepBuilderAPI_MakeFace(GeomFill.Surface_s(cl, cu), 1e-6).Face())   # CAD_ONLY_BASE_CAP
    sew = BRepBuilderAPI_Sewing(sew_tol)
    for f in faces: sew.Add(f)
    sew.Perform()
    shp = sew.SewedShape()
    fe = []
    for k in range(1, sew.NbFreeEdges() + 1):
        e = TopoDS.Edge(sew.FreeEdge(k)); bb_ = Bnd_Box(); BRepBndLib.AddOptimal_s(e, bb_, False, False)
        a_, b_ = bb_.CornerMin(), bb_.CornerMax(); fe.append([a_.X(), a_.Y(), a_.Z(), b_.X(), b_.Y(), b_.Z()])
    info = {"free_edge_boxes_mm": fe, "free_edges_after_sewing": sew.NbFreeEdges(), "multiple_edges": sew.NbMultipleEdges(),
            "degenerated": sew.NbDegeneratedShapes(), "sewing_tolerance_mm": sew_tol}
    if shp.ShapeType() == TopAbs_SHELL:
        shell = TopoDS.Shell(shp)
    else:
        ex = TopExp_Explorer(shp, TopAbs_SHELL)
        if not ex.More(): raise RuntimeError("sewing produced no shell")
        shell = TopoDS.Shell(ex.Current())
    ms = BRepBuilderAPI_MakeSolid(shell); solid = ms.Solid()
    fx = ShapeFix_Solid(solid); fx.Perform(); solid = TopoDS.Solid(fx.Solid())
    return solid, info


def props(shape):
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(shape, g)
    s = GProp_GProps(); BRepGProp.SurfaceProperties_s(shape, s)
    return g.Mass(), s.Mass()


def topology(shape):
    n = {}
    for nm, t in (("faces", TopAbs_FACE), ("edges", TopAbs_EDGE), ("vertices", TopAbs_VERTEX)):
        ex = TopExp_Explorer(shape, t); c = 0
        while ex.More(): c += 1; ex.Next()
        n[nm] = c
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_WIRE
    fb = ShapeAnalysis_FreeBounds(shape, 1e-3, False, False); w = 0
    for comp in (fb.GetClosedWires(), fb.GetOpenWires()):
        e = TopExp_Explorer(comp, TopAbs_WIRE)
        while e.More(): w += 1; e.Next()
    n["free_boundary_wires"] = w          # 0 => every edge is shared by two faces (watertight shell)
    return n


def bbox(shape):
    b = Bnd_Box(); BRepBndLib.AddOptimal_s(shape, b, False, False); lo_, hi_ = b.CornerMin(), b.CornerMax(); x0, y0, z0, x1, y1, z1 = lo_.X(), lo_.Y(), lo_.Z(), hi_.X(), hi_.Y(), hi_.Z()
    return [x0, y0, z0, x1, y1, z1]


def write_step(shape, path):
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs)
    st = w.Write(str(path))
    return st == IFSelect_RetDone


def read_step(path):
    r = STEPControl_Reader(); st = r.ReadFile(str(path))
    if st != IFSelect_RetDone: raise RuntimeError("STEP read failed")
    r.TransferRoots(); return r.OneShape()


def deviation(shape, lo, up, nsamp=300, seed=0):
    """max distance (mm) from sampled original grid nodes (body frame) to the underlying re-imported B-spline face surfaces (nearest face)."""
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    surfs = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        surfs.append(BRep_Tool.Surface_s(TopoDS.Face(ex.Current()))); ex.Next()
    rng = np.random.default_rng(seed); pts = []
    for P in (lo, up):
        G = body_full_grid(P).reshape(-1, 3) * SCALE
        pts.append(G[rng.choice(len(G), nsamp // 2, replace=False)])
    pts = np.concatenate(pts); dmax = 0.0
    for p in pts:
        pp = gp_Pnt(*p); dmin = min(GeomAPI_ProjectPointOnSurf(pp, s).LowerDistance() for s in surfs)
        dmax = max(dmax, dmin)
    return dmax

def process(geometry_id, mat_path, out_dir, inventory_volume=None, check_deviation=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); lo, up = load_native(mat_path)
    rec = {"geometry_id": geometry_id, "length_unit": "mm (program unit 1 m x 1000)", "coordinate_frame": "aero body frame: x downstream, y span (full body symmetric about y=0), z up; alpha>0 nose-up with V=(cos a,0,sin a)"}
    vm, parts = mesh_volume(lo, up); rec["mesh_volume_m3"] = vm
    if inventory_volume is not None: rec["inventory_closed_volume_m3"] = inventory_volume
    try:
        solid, bi = build_solid(lo, up); rec.update(bi)
        valid = BRepCheck_Analyzer(solid).IsValid(); vol_mm3, area_mm2 = props(solid)
        rec.update({"brep_valid": bool(valid), "cad_volume_m3": abs(vol_mm3) / SCALE**3, "cad_area_m2": area_mm2 / SCALE**2,
                    "topology": topology(solid), "shape_type": "SOLID", "bbox_mm": bbox(solid)})
        step = out_dir / "geometry.step"; ok = write_step(solid, step); rec["step_written"] = bool(ok and step.is_file())
        rs = read_step(step)
        rv, ra = props(rs); st = rs.ShapeType()
        rec["reimport"] = {"shape_type": "SOLID" if st == TopAbs_SOLID else str(st), "brep_valid": bool(BRepCheck_Analyzer(rs).IsValid()),
                           "volume_m3": abs(rv) / SCALE**3, "area_m2": ra / SCALE**2, "topology": topology(rs), "bbox_mm": bbox(rs)}
        rec["reimport"]["volume_rel_diff_vs_mesh"] = abs(rec["reimport"]["volume_m3"] / vm - 1)
        rec["reimport"]["volume_rel_diff_vs_exported"] = abs(rec["reimport"]["volume_m3"] / rec["cad_volume_m3"] - 1)
        if check_deviation: rec["reimport"]["max_node_deviation_mm"] = deviation(rs, lo, up)
        rec["cad_export_status"] = "PASS" if (valid and rec["reimport"]["brep_valid"] and bi["free_edges_after_sewing"] == 0) else "PARTIAL"
        rec["step_bytes"] = step.stat().st_size
    except Exception as e:  # noqa
        rec["cad_export_status"] = "CAD_EXPORT_BLOCKED"; rec["error"] = repr(e)
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


if __name__ == "__main__":
    mat, gid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    print(json.dumps(process(gid, mat, out), indent=1, default=float))
