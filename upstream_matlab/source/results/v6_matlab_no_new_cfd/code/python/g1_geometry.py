"""Geometry branches for the v7 study. STRICT_ORIGINAL = existing validated G42->MD9.873 path nodes (unchanged files). AREA_MORPH_EXPLORATORY = volume-preserving
spanwise map of G42 (RELAXES the fixed span / planform assumption):  x'=x, y'=lam*y, z'=z_ref(x)+(z-z_ref(x))/lam  (same map on the lower AND upper surfaces)."""
import sys, json, hashlib
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(str(Path(__file__).resolve().parents[4])); OUT = ROOT/"results/v6_matlab_no_new_cfd"
sys.path.insert(0, str(ROOT/"tools/v6_aero_physics_validation"))
import numpy as np, pandas as pd
from scipy.io import savemat
from scipy.interpolate import PchipInterpolator
import common as C1, p0_geometry as P0
G42_MAT = ROOT/"results/v5_geometry_audit/cases/design_42/native_geometry.mat"
S0 = C1.load_mat(G42_MAT); LO0, UP0 = C1.lower_upper(S0)
# z_ref(x): mid-thickness of G42 on the symmetry plane (row i=49), fixed once and for all
xc = LO0[-1, :, 0]; zc = 0.5*(LO0[-1, :, 2]+UP0[-1, :, 2]); ZREF = PchipInterpolator(xc, zc)
LAM_NODES = [0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10]; LAM_INITIAL = [0.90, 0.95, 1.00, 1.05, 1.10]
def lam_map(lam):
    def f(P):
        Q = P.copy(); Q[..., 1] = lam*P[..., 1]; zr = ZREF(P[..., 0]); Q[..., 2] = zr+(P[..., 2]-zr)/lam; return Q
    return f(LO0), f(UP0)
def sha(lo, up): return hashlib.sha256(np.ascontiguousarray(lo).tobytes()+np.ascontiguousarray(up).tobytes()).hexdigest()
V0 = P0.oriented_volume(LO0, UP0)["volume_full"]
def metrics(lo, up):
    S = {"wav_x": lo[..., 0], "wav_y": lo[..., 1], "wav_z": lo[..., 2], "wav_up_x": up[..., 0], "wav_up_y": up[..., 1], "wav_up_z": up[..., 2], "inputs10": S0["inputs10"]}
    xy = P0.xy_grid_metrics(lo, up); ov = P0.oriented_volume(lo, up); bf = P0.body_frame_checks(S); bs = P0.base_face(lo, up); Pn = C1.R.panels(S)
    T = lo[..., 2]-up[..., 2]
    # penetration: grid thickness and xy-resampled thickness; normal flips: panel normals vs body frame
    d2 = np.abs(np.diff(lo[..., 2], 2, axis=1)).max()
    return dict(closed_volume=ov["volume_full"], volume_ratio_G42=ov["volume_full"]/V0, watertight=bool(ov["welded_watertight"]), euler=ov["euler_characteristic"], planform_area=xy["planform_full_xy_grid"],
                wetted_area=float(Pn["A"].sum()), base_area=bs["base_area_full_exact"], thickness_max=xy["thickness_max_xy"], thickness_min_xy=xy["thickness_min_xy"], grid_T_min=float(T.min()), neg_thickness_area_frac=xy["negative_thickness_area_fraction"],
                flipped_lower=bf["body_lower_panels_with_upward_normal"], flipped_upper=bf["body_upper_panels_with_downward_normal"], span_half=float(np.abs(lo[..., 1]).max()), max_chordwise_2nd_diff=float(d2),
                valid=bool(ov["welded_watertight"] and ov["euler_characteristic"] == 2 and xy["thickness_min_xy"] > -1e-9 and xy["negative_thickness_area_fraction"] == 0 and bf["body_lower_panels_with_upward_normal"] == 0 and bf["body_upper_panels_with_downward_normal"] == 0))
rows = []
d = OUT/"geometry"; d.mkdir(exist_ok=True)
for lam in LAM_NODES:
    lo, up = lam_map(lam); gid = f"AREA_lam{lam:.3f}"
    if lam == 1.0: lo, up = LO0.copy(), UP0.copy(); gid = "AREA_lam1.000"
    mat = d/f"{gid}.mat"; savemat(mat, {"wav_x": lo[..., 0], "wav_y": lo[..., 1], "wav_z": lo[..., 2], "wav_up_x": up[..., 0], "wav_up_y": up[..., 1], "wav_up_z": up[..., 2], "inputs10": S0["inputs10"]})
    m = metrics(lo, up); rows.append(dict(branch="AREA_MORPH_EXPLORATORY", geometry_id=gid, param="lambda", value=lam, mat_path=str(mat), geometry_hash=sha(lo, up), initial_node=lam in LAM_INITIAL, **m))
pp = pd.read_csv(ROOT/"results/v6_48h_multifidelity_morph_rl/MORPH_PATH_GEOMETRY.csv"); pp = pp[pp.path == "G42__S2_MD9.873_4cfddf7fac"].reset_index(drop=True)
for r in pp.itertuples():
    S = C1.load_mat((ROOT / r.native_geometry_mat).resolve()); lo, up = C1.lower_upper(S); gid = f"STRICT_xi{r.xi:.4f}"
    m = metrics(lo, up); rows.append(dict(branch="STRICT_ORIGINAL", geometry_id=gid, param="xi", value=r.xi, mat_path=str((ROOT / r.native_geometry_mat).resolve()), geometry_hash=sha(lo, up), initial_node=r.xi in (0, 0.25, 0.5, 0.75, 1.0), source_candidate=r.candidate_id, design_mach_label=r.Ma_design, **m))
df = pd.DataFrame(rows); df.to_csv(OUT/"GEOMETRY_METADATA.csv", index=False)
print(df[["branch", "geometry_id", "valid", "volume_ratio_G42", "planform_area", "wetted_area", "thickness_max", "thickness_min_xy", "span_half", "base_area"]].round(4).to_string())
mp = np.array([[x, ZREF(x)] for x in (2.5,)]); print("z_ref(2.5) native =", float(ZREF(2.5)))
