"""CFD availability check + preparation of Euler validation inputs (nothing is solved here).

Writes results/v6_aero_physics_validation_v1/CFD_AVAILABILITY.json and, per geometry,
cfd_inputs/<geom>/{<geom>_half.stl, <geom>_full.stl, mesh_<geom>.geo, su2_<case>.cfg}.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

import common as C
from p0_geometry import closed_half_hull, manifold_report, signed_volume, weld

CFD = C.OUT / "cfd_inputs"
TOOLS = {"SU2_CFD": ["SU2_CFD", "SU2_CFD.exe"], "OpenFOAM": ["rhoCentralFoam", "simpleFoam", "foamExec"],
         "gmsh": ["gmsh", "gmsh.exe"], "ANSYS_Fluent": ["fluent", "fluent.exe"], "STAR-CCM+": ["starccm+", "starccm+.exe"],
         "Cart3D": ["flowCart", "cubes"], "mpiexec": ["mpiexec", "mpiexec.exe"]}
SEARCH_DIRS = [r"D:\star", r"C:\Program Files\ANSYS Inc", r"D:\ANSYS Inc", r"C:\Program Files\SU2", r"D:\SU2",
               r"C:\OpenFOAM", r"D:\OpenFOAM", r"C:\Program Files\gmsh", r"D:\gmsh", r"C:\Program Files\Siemens"]


def probe():
    out = {"executables_on_PATH": {k: {n: shutil.which(n) for n in v} for k, v in TOOLS.items()}}
    out["install_dirs"] = {d: (os.path.isdir(d), len(os.listdir(d)) if os.path.isdir(d) else None) for d in SEARCH_DIRS}
    out["env"] = {k: os.environ.get(k) for k in ("AWP_ROOT231", "AWP_ROOT222", "SU2_RUN", "WM_PROJECT_DIR", "CDLMD_LICENSE_FILE")}
    try:
        r = subprocess.run(["wsl", "-l", "-v"], capture_output=True, timeout=30)
        txt = r.stdout.decode("utf-16-le", errors="ignore") or r.stdout.decode(errors="ignore")
        out["wsl_list"] = txt.replace("\x00", "").strip()
        r2 = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "echo", "ok"], capture_output=True, timeout=60)
        out["wsl_exec"] = {"rc": r2.returncode,
                           "stderr": (r2.stderr.decode("utf-16-le", errors="ignore") or r2.stderr.decode(errors="ignore"))
                           .replace("\x00", "").strip()[:400],
                           "stdout": r2.stdout.decode(errors="ignore").strip()[:200]}
    except Exception as e:  # noqa: BLE001
        out["wsl_error"] = repr(e)
    pk = {}
    for name in ("gmsh", "pyvista", "meshio", "trimesh", "SU2", "pyfoam"):
        try:
            __import__(name)
            pk[name] = True
        except Exception:  # noqa: BLE001
            pk[name] = False
    out["python_packages"] = pk
    out["matlab"] = {"exe": str(C.MATLAB), "exists": C.MATLAB.exists(), "pde_toolbox_note": "MATLAB has no compressible Euler solver usable here"}
    du = {d: shutil.disk_usage(d)._asdict() for d in ("C:\\", "D:\\")}
    out["disk_free_GB"] = {k: round(v["free"] / 2 ** 30, 1) for k, v in du.items()}
    out["cpu"] = {"logical": os.cpu_count()}
    return out


# ----------------------------------------------------------------------------- watertight STL
def body_frame(V):
    return V * np.array([1.0, -1.0, -1.0])


def write_stl(path, V, F, name):
    with open(path, "w", encoding="ascii") as f:
        f.write(f"solid {name}\n")
        for a, b, c in F:
            n = np.cross(V[b] - V[a], V[c] - V[a])
            n = n / (np.linalg.norm(n) or 1)
            f.write(f" facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n  outer loop\n")
            for p in (V[a], V[b], V[c]):
                f.write(f"   vertex {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")


def watertight(gid):
    S = C.load_mat(C.GEOMS[gid]["mat"])
    lo, up = C.lower_upper(S)
    p = closed_half_hull(lo, up)
    half_open = np.concatenate([p["lower"], p["upper_reversed"], p["base"], p["tip"]])  # symmetry plane left open
    half_closed = np.concatenate([half_open, p["symmetry"]])
    Vh, Fh = weld(half_closed)
    if signed_volume(Vh[Fh]) < 0:
        Fh = Fh[:, ::-1]
    rep_h = manifold_report(Vh, Fh)
    mir = half_open.copy()
    mir[..., 1] *= -1
    full = np.concatenate([half_open, mir[:, ::-1]])
    Vf, Ff = weld(full)
    if signed_volume(Vf[Ff]) < 0:
        Ff = Ff[:, ::-1]
    rep_f = manifold_report(Vf, Ff)
    d = CFD / C.SHORT[gid]
    d.mkdir(parents=True, exist_ok=True)
    # STL in the Teacher body frame (x downstream, y span, z up); body = diag(1,-1,-1) * native, which
    # flips orientation twice (det = +1) so outward normals are preserved
    Vh_b, Vf_b = body_frame(Vh), body_frame(Vf)
    write_stl(d / f"{C.SHORT[gid]}_half_closed.stl", Vh_b, Fh, C.SHORT[gid] + "_half")
    write_stl(d / f"{C.SHORT[gid]}_full.stl", Vf_b, Ff, C.SHORT[gid] + "_full")
    rep_h["volume_full_equiv_m3"] = 2 * abs(rep_h["signed_volume"])
    rep_f["volume_m3"] = abs(rep_f["signed_volume"])
    return {"half_closed_with_symmetry_face": rep_h, "full_mirrored": rep_f,
            "frame": "Teacher body frame: x downstream, y span, z up (body = diag(1,-1,-1) * native)",
            "files": [str((d / f"{C.SHORT[gid]}_half_closed.stl").relative_to(C.ROOT)),
                      str((d / f"{C.SHORT[gid]}_full.stl").relative_to(C.ROOT))]}


def su2_cfg(case, M, alpha, p, T, half=True):
    return f"""% SU2 v8 Euler validation case {case} (PREPARED, NOT RUN)
SOLVER= EULER
MATH_PROBLEM= DIRECT
FLUID_MODEL= STANDARD_AIR
GAMMA_VALUE= 1.4
GAS_CONSTANT= 287.05287
MACH_NUMBER= {M:.6f}
AOA= {alpha:.6f}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= {p:.6f}
FREESTREAM_TEMPERATURE= {T:.4f}
REF_ORIGIN_MOMENT_X= 0.0
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 5.0
REF_AREA= {C.SREF / (2 if half else 1):.9f}
REF_DIMENSIONALIZATION= DIMENSIONAL
MARKER_EULER= ( body )
MARKER_SYM= ( symmetry )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( body )
MARKER_MONITORING= ( body )
NUM_METHOD_GRAD= GREEN_GAUSS
CONV_NUM_METHOD_FLOW= AUSMPLUSUP2
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
CFL_NUMBER= 1.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 1.0, 50.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
ITER= 20000
CONV_FIELD= DRAG
CONV_CAUCHY_ELEMS= 200
CONV_CAUCHY_EPS= 1E-5
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_CSV)
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)
"""


def gmsh_geo(gid):
    s = C.SHORT[gid]
    return f"""// gmsh 4.x half-model Euler mesh for {s} (PREPARED, NOT RUN). Body frame, metres.
// The body STL is closed on the symmetry plane y=0; the domain is the half box y>=0 minus the body.
Merge "{s}_half_closed.stl";
ClassifySurfaces{{40*Pi/180, 1, 1, Pi}};
CreateGeometry;
body() = Surface{{:}};
Box(100) = {{-2.5, 0, -6, 12.5, 7, 12}};      // x in [-2.5, 10], y in [0, 7], z in [-6, 6]
Surface Loop(200) = {{body()}};
Volume(300) = {{200}};
BooleanDifference(400) = {{ Volume{{100}}; Delete; }}{{ Volume{{300}}; Delete; }};
Mesh.MeshSizeMin = 0.004;     // leading edge / tip
Mesh.MeshSizeMax = 0.6;       // far field
Field[1] = Distance; Field[1].SurfacesList = {{body()}};
Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = 0.01; Field[2].SizeMax = 0.6; Field[2].DistMin = 0.02; Field[2].DistMax = 3;
Background Field = 2;
Mesh.Algorithm3D = 10;        // HXT
// Physical groups must be re-assigned after the boolean: body / symmetry (y=0 faces) / farfield.
Physical Surface("body") = {{body()}};
Physical Volume("fluid") = {{400}};
"""


def main():
    av = probe()
    av["utc"] = C.utc()
    geo = {}
    cases = C.read_csv(C.OUT / "VALIDATION_CASE_MATRIX.csv")
    for gid in C.ORDER:
        geo[gid] = watertight(gid)
        d = CFD / C.SHORT[gid]
        (d / f"mesh_{C.SHORT[gid]}.geo").write_text(gmsh_geo(gid), encoding="utf-8")
        for c in cases:
            if c["geometry_id"] != gid or c["alpha_deg"] in ("", None):
                continue
            if c["kind"] == "DESIGN_POINT":
                M, a, p, T = float(c["Mach"]), float(c["alpha_deg"]), C.P0_DESIGN, C.P0_DESIGN / (C.R0_DESIGN * 287.05287)
            elif float(c["CL_target"]) == 0.045:
                M, a = float(c["Mach"]), float(c["alpha_deg"])
                p = 2 * float(c["q_pa"]) / (1.4 * M ** 2)
                T = float(c["T_inf_K"])
            else:
                continue
            (d / f"su2_{c['case_id']}.cfg").write_text(su2_cfg(c["case_id"], M, a, p, T), encoding="utf-8")
        print(gid, geo[gid]["half_closed_with_symmetry_face"]["watertight"], geo[gid]["full_mirrored"]["watertight"],
              round(geo[gid]["full_mirrored"]["volume_m3"], 4))
    solver = any(v for k, d in av["executables_on_PATH"].items() if k in ("SU2_CFD", "OpenFOAM", "ANSYS_Fluent", "STAR-CCM+", "Cart3D")
                 for v in d.values())
    n_cases = sum(1 for _ in (CFD).rglob("su2_*.cfg"))
    av.update({
        "real_cfd_solver_available": bool(solver),
        "CFD_STATUS": "NOT_RUN" if not solver else "AVAILABLE",
        "reason": ("No compressible CFD solver (SU2, OpenFOAM, Fluent, STAR-CCM+, Cart3D) and no mesher (gmsh) is installed; "
                   "D:\\star is empty; WSL2 cannot start (HCS_E_SERVICE_NOT_AVAILABLE: virtual machine platform disabled); "
                   "C: has too little free space for an install. No CFD result exists or is implied in this round."),
        "prepared_inputs": {"geometry": geo, "su2_configs": n_cases, "gmsh_templates": len(C.ORDER),
                            "dir": str(CFD.relative_to(C.ROOT))},
        "resource_estimate": {
            "model": "3-D inviscid Euler, half model, unstructured tetra/prism, AUSM+up2 2nd order, implicit",
            "cells_per_case": "3e6 (coarse, shock-capturing) to 8e6 (medium); a grid-convergence triplet needs ~1.5e6/4e6/1e7",
            "memory_GB_per_case": "SU2 ~1.2-1.5 kB/cell -> 4-12 GB (this PC has 15.9 GB: medium grids are at the limit)",
            "cpu_core_hours_per_case": "~10-40 (3e6-8e6 cells, 3000-8000 iterations, ~3e-6 s per cell-iteration-core)",
            "wall_hours_per_case_on_6_cores": "~2-7",
            "minimum_campaign": {"cases": n_cases, "wall_hours": f"~{2 * n_cases}-{7 * n_cases}",
                                 "content": "4 design points + 4 geometries x 3 Mach at CL=0.045"},
            "disk_GB": "~1-3 per case (restart + volume output); install on D: (C: has ~1 GB free)",
            "enablers": ["install SU2 v8 Windows binaries (MPI build needs MS-MPI, already present: MSMPI_BIN set)",
                         "install gmsh 4.x Windows binary", "or enable the Windows 'Virtual Machine Platform' feature so WSL2 Ubuntu can run SU2/OpenFOAM"]},
    })
    C.dump(C.OUT / "CFD_AVAILABILITY.json", av)


if __name__ == "__main__":
    main()
