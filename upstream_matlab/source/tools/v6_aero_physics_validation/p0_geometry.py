"""P0: geometry identity, independent metric audit, validation case matrix and baseline Teacher results.

    .venv\\Scripts\\python.exe -B tools\\v6_aero_physics_validation\\p0_geometry.py
"""
from __future__ import annotations

import importlib.util
from collections import Counter

import numpy as np
from scipy.interpolate import LinearNDInterpolator

import common as C

MATRIX_MACH = (5.5, 6.5, 7.5)
MATRIX_CL = (0.030, 0.045, 0.055)


# ----------------------------------------------------------------------------- identity
def stored_identity(gid):
    g = C.GEOMS[gid]
    hashes = {r["geometry_hash"] for r in C.read_csv(g["labels"]) if r["geometry_id"] == g["label_id"]}
    inv = {}
    for p in (C.ROOT / "results/v5_multimach_geometry_db_v2/valid_geometry_inventory.csv",):
        for r in C.read_csv(p):
            if r["geometry_id"] == gid:
                inv = {"inventory_geometry_hash": r["geometry_hash"], "inventory_mat_sha256": r["mat_sha256"],
                       "inventory_source": "results/v5_multimach_geometry_db_v2/valid_geometry_inventory.csv"}
    if gid == "G42":
        a = __import__("json").loads((C.ROOT / "results/v5_geometry_audit/cases/design_42/audit.json")
                                     .read_text(encoding="utf-8"))
        inv = {"inventory_geometry_hash": a["geometry_hash"], "inventory_mat_sha256": a["mat_sha256"],
               "inventory_source": "results/v5_geometry_audit/cases/design_42/audit.json"}
    if gid.startswith("S2_"):
        for r in C.read_csv(C.ROOT / "results/v6_mission_geometry_stage2/geometry_candidates.csv"):
            if r["candidate_id"] == gid:
                inv = {"inventory_geometry_hash": r["geometry_hash"],
                       "inventory_source": "results/v6_mission_geometry_stage2/geometry_candidates.csv"}
        a = __import__("json").loads((C.GEOMS[gid]["mat"].parent / "audit.json").read_text(encoding="utf-8"))
        if "mat_sha256" in a:
            inv["inventory_mat_sha256"] = a["mat_sha256"]
            inv["inventory_source"] += " + case audit.json"
    return hashes, inv


def identity(gid):
    g = C.GEOMS[gid]
    S = C.load_mat(g["mat"])
    lo, up = C.lower_upper(S)
    inp = C.inputs10(gid)
    h = C.native_hash(lo, up, inp)
    label_hashes, inv = stored_identity(gid)
    rec = {"geometry_id": gid, "mat": str(g["mat"].relative_to(C.ROOT)), "mat_sha256": C.sha_file(g["mat"]),
           "inputs10": inp, "inputs10_in_mat": S["inputs10"] is not None,
           "recomputed_geometry_hash": h, "label_geometry_hashes": sorted(label_hashes), **inv}
    rec["hash_matches_labels"] = label_hashes == {h}
    if inv:
        rec["hash_matches_inventory"] = inv["inventory_geometry_hash"] == h
        if "inventory_mat_sha256" in inv:
            rec["mat_sha_matches_inventory"] = inv["inventory_mat_sha256"] == rec["mat_sha256"]
    return rec, S


# ----------------------------------------------------------------------------- metrics
def v2_metrics(mat):
    spec = importlib.util.spec_from_file_location(
        "v2gm", C.ROOT / "results/v5_multimach_geometry_db_v2/code/geometry_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.metrics(mat), m


def tris(grid, flip=False):
    m, n = grid.shape[:2]
    ids = np.arange(m * n).reshape(m, n)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    if flip:
        f = f[:, ::-1]
    return grid.reshape(-1, 3)[f]


def strip(p, q):
    """Triangles joining two polylines p[k] -> q[k] (same length), consistently ordered."""
    t = []
    for k in range(len(p) - 1):
        t.append([p[k], p[k + 1], q[k + 1]])
        t.append([p[k], q[k + 1], q[k]])
    return np.asarray(t)


def signed_volume(T, origin=(0, 0, 0)):
    T = T - np.asarray(origin, float)
    return float(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6)


def closed_half_hull(lo, up):
    """Lower + reversed upper + base strip + tip strip + symmetry strip; returns triangle sets and a closure test."""
    parts = {"lower": tris(lo), "upper_reversed": tris(up, flip=True),
             "base": strip(lo[:, -1], up[:, -1]),
             "tip": strip(up[0, :], lo[0, :]),
             "symmetry": strip(up[-1, :], lo[-1, :])}
    return parts


def oriented_volume(lo, up):
    parts = closed_half_hull(lo, up)
    best = None
    # orientation of the three closure strips is fixed by trying both and keeping the translation-invariant one
    for sb in (1, -1):
        for st in (1, -1):
            for ss in (1, -1):
                T = [parts["lower"], parts["upper_reversed"]]
                for s, k in ((sb, "base"), (st, "tip"), (ss, "symmetry")):
                    T.append(parts[k] if s > 0 else parts[k][:, ::-1])
                T = np.concatenate(T)
                v0 = signed_volume(T)
                v1 = signed_volume(T, origin=(3.0, -2.0, 5.0))
                err = abs(v0 - v1)
                if best is None or err < best[0]:
                    best = (err, v0, v1, (sb, st, ss), T)
    err, v0, v1, signs, T = best
    V, F = weld(T)
    mr = manifold_report(V, F)
    return {"volume_half": abs(v0), "volume_full": 2 * abs(v0), "translation_invariance_abs_err": err,
            "closure_strip_signs": signs, "welded_watertight": mr["watertight"],
            "edges_not_shared_by_exactly_2": mr["edges_not_shared_by_exactly_2"], "euler_characteristic": mr["euler_characteristic"],
            "triangles_after_weld": mr["triangles"]}


def weld(T, tol=1e-9):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    P = T.reshape(-1, 3)
    pairs = cKDTree(P).query_pairs(tol, output_type="ndarray")
    n = len(P)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    _, idx, inv = np.unique(lab, return_index=True, return_inverse=True)
    V = P[idx]
    F = inv.reshape(-1, 3)
    good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[good]
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) / 2
    F = F[area > 1e-14]
    return V, F


def manifold_report(V, F):
    und = Counter()
    dirc = Counter()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            und[(min(a, b), max(a, b))] += 1
            dirc[(a, b)] += 1
    nonman = sum(1 for n in und.values() if n != 2)
    bad_orient = sum(1 for (a, b), n in dirc.items() if n != 1 or dirc.get((b, a), 0) != 1)
    vol = signed_volume(V[F])
    return {"vertices": int(len(V)), "triangles": int(len(F)), "edges_not_shared_by_exactly_2": int(nonman),
            "directed_edges_without_opposite_pair": int(bad_orient), "watertight": nonman == 0 and bad_orient == 0,
            "signed_volume": vol, "euler_characteristic": int(len(V) - len(und) + len(F))}


def v2_style_volume(lo, up):
    """V2 formula: lower and upper triangulated with the same index orientation + base cap."""
    T = np.concatenate([tris(lo), tris(up), strip(lo[:, -1], up[:, -1])])
    return {"volume_full": 2 * abs(signed_volume(T)),
            "volume_full_origin_shift": 2 * abs(signed_volume(T, origin=(3.0, -2.0, 5.0)))}


def xy_grid_metrics(lo, up, nx=600, ny=300):
    """Thickness z_lower - z_upper on a common (x, y) grid (native z points toward the shock)."""
    fl = LinearNDInterpolator(lo[..., :2].reshape(-1, 2), lo[..., 2].ravel())
    fu = LinearNDInterpolator(up[..., :2].reshape(-1, 2), up[..., 2].ravel())
    x = np.linspace(lo[..., 0].min(), 5.0, nx)
    y = np.linspace(0.0, max(lo[..., 1].max(), up[..., 1].max()), ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    zl, zu = fl(X, Y), fu(X, Y)
    ok = np.isfinite(zl) & np.isfinite(zu)
    t = np.where(ok, zl - zu, np.nan)
    dA = (x[1] - x[0]) * (y[1] - y[0])
    tb = t[-1]
    return {"planform_full_xy_grid": 2 * ok.sum() * dA,
            "volume_full_xy_grid": 2 * np.nansum(t) * dA,
            "thickness_max_xy": float(np.nanmax(t)), "thickness_min_xy": float(np.nanmin(t)),
            "thickness_mean_xy": float(np.nanmean(t)),
            "negative_thickness_area_fraction": float(np.mean(t[ok] < -1e-9)),
            "thickness_max_location_xy": [float(X.ravel()[np.nanargmax(t)]), float(Y.ravel()[np.nanargmax(t)])],
            "base_thickness_max_xy": float(np.nanmax(tb)) if np.isfinite(tb).any() else None,
            "grid": [nx, ny]}


def veg1_strip_volume(lo, up):
    """Re-implementation of tools/v5_physics_realism_audit/geometry_props.py::surface_props volume and planform."""
    xl, yl, zl = lo[..., 0], lo[..., 1], lo[..., 2]
    zu = up[..., 2]
    t = zl - zu
    tq = (t[:-1, :-1] + t[1:, :-1] + t[1:, 1:] + t[:-1, 1:]) / 4
    ax = np.abs(np.cross(np.stack([xl[1:, 1:] - xl[:-1, :-1], yl[1:, 1:] - yl[:-1, :-1]], -1),
                         np.stack([xl[1:, :-1] - xl[:-1, 1:], yl[1:, :-1] - yl[:-1, 1:]], -1))) / 2
    L = float(xl.max())
    s_plan = 2 * abs(np.trapz(L - xl[:, 0], yl[:, 0]))
    thick_base = zl[:, -1] - zu[:, -1]
    s_base = 2 * abs(np.trapz(thick_base, yl[:, -1]))
    return {"volume_full_veg1_strip": float(2 * (tq * ax).sum()), "planform_full_veg1_le": float(s_plan),
            "base_area_full_veg1": float(s_base)}


def base_face(lo, up):
    """Exact area of the open base polygon (half) between the lower and upper trailing curves."""
    T = strip(lo[:, -1], up[:, -1])
    a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum() / 2
    xs = np.concatenate([lo[:, -1, 0], up[:, -1, 0]])
    return {"base_area_full_exact": 2 * float(a), "base_x_spread": float(xs.max() - xs.min())}


def tip_face(lo, up):
    T = strip(up[0, :], lo[0, :])
    a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum() / 2
    return {"tip_open_face_area_half": float(a), "lower_row0_collapsed": bool(np.ptp(lo[0], axis=0).max() < 1e-12),
            "upper_row0_length": float(np.linalg.norm(up[0, -1] - up[0, 0]))}


def body_frame_checks(S):
    P = C.R.panels(S)
    n_lower_up = int(np.sum((P["surface"] == "lower") & (P["n"][:, 2] > 1e-9)))
    n_upper_down = int(np.sum((P["surface"] == "upper") & (P["n"][:, 2] < -1e-9)))
    return {"body_lower_panels_with_upward_normal": n_lower_up, "body_upper_panels_with_downward_normal": n_upper_down,
            "panels_total_full": int(len(P["A"]))}


def index_alignment(lo, up):
    return {"max_abs_dx_same_index": float(np.abs(lo[1:, :, 0] - up[1:, :, 0]).max()),
            "max_abs_dy_same_index": float(np.abs(lo[1:, :, 1] - up[1:, :, 1]).max()),
            "leading_edge_max_gap": float(np.linalg.norm(lo[:, 0] - up[:, 0], axis=1).max())}


# ----------------------------------------------------------------------------- mission conditions
def mission_conditions():
    rows = C.read_csv(C.ROOT / "results/v6_mission_geometry_stage2/mission_design_conditions.csv")
    M = np.array([float(r["Mach"]) for r in rows])
    out = {}
    for m in MATRIX_MACH:
        w = np.exp(-((M - m) / 0.25) ** 2)
        out[m] = {k: float(np.sum(w * np.array([float(r[k]) for r in rows])) / w.sum())
                  for k in ("h_m", "V_mps", "q_pa")}
    return out


def isa_T(h):
    h = float(h)
    if h <= 11000:
        return 288.15 - 0.0065 * h
    if h <= 20000:
        return 216.65
    if h <= 32000:
        return 216.65 + 0.001 * (h - 20000)
    if h <= 47000:
        return 228.65 + 0.0028 * (h - 32000)
    return 270.65


# ----------------------------------------------------------------------------- Teacher baseline
def teacher_at_cl(P, M, cl):
    a = C.alpha_for_cl(lambda x: C.teacher(P, M, x)["CL"], cl)
    if a is None:
        return None, None
    return a, C.teacher(P, M, a)


def label_alpha_for_cl(lab, M, cl):
    al = np.array(sorted(a for (m, a) in lab if m == M))
    if al.size == 0:
        return None, None, None
    c = np.array([lab[(M, a)][0] for a in al])
    d = np.array([lab[(M, a)][1] for a in al])
    s = c - cl
    k = np.flatnonzero(s[:-1] * s[1:] <= 0)
    if k.size == 0:
        return None, None, None
    k = k[0]
    w = (cl - c[k]) / (c[k + 1] - c[k])
    return float(al[k] + w * (al[k + 1] - al[k])), cl, float(d[k] + w * (d[k + 1] - d[k]))


def main():
    ident, metrics, cases, base = [], [], [], []
    cond = mission_conditions()
    for gid in C.ORDER:
        rec, S = identity(gid)
        ident.append(rec)
        lo, up = C.lower_upper(S)
        v2, _ = v2_metrics(C.GEOMS[gid]["mat"])
        m = {"geometry_id": gid, "V2_volume_full": v2["volume_full"], "V2_grid_thickness_max": v2["grid_thickness_max"],
             "V2_planform_area_full": v2["planform_area_full"], "V2_base_height_at_symmetry": v2["base_height_at_symmetry"],
             "V2_wetted_lower_half": v2["wetted_area_lower_half"], "V2_wetted_upper_half": v2["wetted_area_upper_half"],
             "V2_lower_z_range": v2["lower_z_range"], "V2_upper_z_range": v2["upper_z_range"]}
        m.update({f"repro_{k}": v for k, v in v2_style_volume(lo, up).items()})
        m.update({f"closed_{k}": v for k, v in oriented_volume(lo, up).items()})
        m.update(xy_grid_metrics(lo, up))
        m.update(veg1_strip_volume(lo, up))
        m.update(base_face(lo, up))
        m.update(tip_face(lo, up))
        m.update(index_alignment(lo, up))
        m.update(body_frame_checks(S))
        m["Sref_used_by_teacher"] = C.SREF
        m["Sref_over_own_planform"] = C.SREF / m["V2_planform_area_full"]
        m["V_over_Splan_closed"] = m["closed_volume_full"] / m["V2_planform_area_full"]
        m["V_over_Splan_V2"] = m["V2_volume_full"] / m["V2_planform_area_full"]
        m["volumetric_efficiency_closed"] = m["closed_volume_full"] ** (2 / 3) / m["V2_planform_area_full"]
        m["V2_volume_over_closed_volume"] = m["V2_volume_full"] / m["closed_volume_full"]
        metrics.append(m)

        # ---------------------------------------------------------------- case matrix + baseline
        P = C.R.panels(S)
        lab = C.labels(gid)
        Md = C.GEOMS[gid]["Ma_design"]
        des = C.teacher(P, Md, C.ALPHA_DESIGN_BODY)
        cases.append({"case_id": f"{C.SHORT[gid]}_DESIGN", "geometry_id": gid, "kind": "DESIGN_POINT", "Mach": Md,
                      "alpha_deg": C.ALPHA_DESIGN_BODY, "CL_target": None, "phase": "P1,P2,P3,P4",
                      "reachable_teacher": True, "note": "native AOA=-4 deg == body alpha +4 deg (design attitude)"})
        base.append({"case_id": f"{C.SHORT[gid]}_DESIGN", "geometry_id": gid, "Mach": Md, "CL_target": None,
                     "alpha_deg_replica": C.ALPHA_DESIGN_BODY, "CL_replica": des["CL"], "CD_replica": des["CD"],
                     "LD_replica": des["CL"] / des["CD"], "label_source": "none (replica only; Md outside label grid)"})
        for M in MATRIX_MACH:
            for cl in MATRIX_CL:
                cid = f"{C.SHORT[gid]}_M{M}_CL{cl:.3f}"
                al, _, cdl = label_alpha_for_cl(lab, M, cl)
                ar, res = teacher_at_cl(P, M, cl)
                cases.append({"case_id": cid, "geometry_id": gid, "kind": "MISSION_ISO_CL", "Mach": M,
                              "alpha_deg": ar, "CL_target": cl, "phase": "P2,P3,P4",
                              "reachable_teacher": ar is not None,
                              "h_m": cond[M]["h_m"], "V_mps": cond[M]["V_mps"], "q_pa": cond[M]["q_pa"],
                              "T_inf_K": isa_T(cond[M]["h_m"]),
                              "note": "alpha from replica brentq on [-2, 12] deg"})
                b = {"case_id": cid, "geometry_id": gid, "Mach": M, "CL_target": cl,
                     "alpha_deg_label_interp": al, "CD_label_interp": cdl,
                     "label_source": str(C.GEOMS[gid]["labels"].relative_to(C.ROOT))}
                if res is not None:
                    b.update({"alpha_deg_replica": ar, "CL_replica": res["CL"], "CD_replica": res["CD"],
                              "LD_replica": res["CL"] / res["CD"]})
                    if cdl is not None:
                        b["CD_rel_diff_replica_vs_label_interp"] = res["CD"] / cdl - 1
                base.append(b)
        # stored label rows at matrix Mach, alpha 0..12 for the raw table
        for (M, a), (cl, cd, cm, h) in sorted(lab.items()):
            if M in MATRIX_MACH and a in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
                base.append({"case_id": f"{C.SHORT[gid]}_LABEL_M{M}_A{a:g}", "geometry_id": gid, "Mach": M,
                             "alpha_deg_label": a, "CL_label": cl, "CD_label": cd, "Cm_label": cm,
                             "LD_label": cl / cd if cd else None, "label_geometry_hash": h,
                             "label_source": str(C.GEOMS[gid]["labels"].relative_to(C.ROOT))})
        print(gid, "done")

    # common reachability per (M, CL)
    for c in cases:
        if c["kind"] == "MISSION_ISO_CL":
            c["reachable_all_4"] = all(x["reachable_teacher"] for x in cases
                                       if x["kind"] == "MISSION_ISO_CL" and x["Mach"] == c["Mach"]
                                       and x["CL_target"] == c["CL_target"])
    C.dump(C.WORK / "p0_identity.json", ident)
    C.dump(C.WORK / "p0_metrics.json", metrics)
    C.dump(C.WORK / "p0_mission_conditions.json", {str(k): v for k, v in cond.items()})
    C.write_csv(C.OUT / "VALIDATION_CASE_MATRIX.csv", cases,
                ["case_id", "geometry_id", "kind", "Mach", "alpha_deg", "CL_target", "reachable_teacher",
                 "reachable_all_4", "h_m", "V_mps", "q_pa", "T_inf_K", "phase", "note"])
    C.write_csv(C.OUT / "BASELINE_TEACHER_RESULTS.csv", base,
                ["case_id", "geometry_id", "Mach", "CL_target", "alpha_deg_replica", "CL_replica", "CD_replica",
                 "LD_replica", "alpha_deg_label_interp", "CD_label_interp", "CD_rel_diff_replica_vs_label_interp",
                 "alpha_deg_label", "CL_label", "CD_label", "Cm_label", "LD_label", "label_geometry_hash",
                 "label_source"])
    for r in ident:
        print(r["geometry_id"], r["hash_matches_labels"], r.get("hash_matches_inventory"), r.get("mat_sha_matches_inventory"))
    for m in metrics:
        print(m["geometry_id"], {k: round(m[k], 4) if isinstance(m[k], float) else m[k] for k in
                                 ("V2_volume_full", "repro_volume_full", "closed_volume_full",
                                  "closed_translation_invariance_abs_err", "closed_welded_watertight", "closed_euler_characteristic",
                                  "volume_full_xy_grid", "volume_full_veg1_strip", "V2_grid_thickness_max",
                                  "thickness_max_xy", "V2_planform_area_full", "planform_full_xy_grid",
                                  "planform_full_veg1_le", "base_area_full_exact", "max_abs_dx_same_index",
                                  "max_abs_dy_same_index", "tip_open_face_area_half")})


if __name__ == "__main__":
    main()
