"""P4: off-design shock-mismatch RISK SCREENING (no aerodynamic correction, no penalty term).

For every leading-edge station i of the lower surface (one osculating plane per stored streamline):
  beta_geom  angle between the DESIGN shock surface (create_shock_functions, at the stored LE root u_fct,
             v_fct) and the ACTUAL free stream e_f = (cos a, 0, -sin a) in the native frame;
  beta_fl    weak oblique-shock angle a sharp wedge of the local first-panel incidence theta_LE would
             produce at the flight Mach (osculating-plane, 2-D);
  dbeta      (beta_fl - beta_geom) minus the same difference at the design state (removes the
             discretisation offset of the first panel), > 0: shock stands off the design surface
             (spillage around the leading edge), < 0: shock moves inside the design shock surface;
  M_n        free-stream Mach number normal to the local leading edge segment (subsonic LE if < 1);
  mu         Mach angle; beta_geom < mu means the design shock surface cannot exist as a shock.
"""
from __future__ import annotations

import numpy as np
from scipy.io import loadmat

import aeromodels as AM
import common as C

THRESH = {"HIGH": {"Mn_lt_1_frac": 0.25, "abs_dbeta_deg": 3.0, "below_mach_frac": 0.10},
          "MEDIUM": {"Mn_lt_1p3_frac": 0.25, "abs_dbeta_deg": 1.5, "offset_over_depth": 0.5}}


def shock_normal_native(sh, u, v):
    ru = np.array([C.R.L * np.ones_like(u), sh.dRy(u) * np.cos(v), sh.dRz(u) * np.sin(v)])
    rv = np.array([np.zeros_like(u), -sh.Ry(u) * np.sin(v), sh.Rz(u) * np.cos(v)])
    n = np.cross(ru.T, rv.T)
    return n / np.linalg.norm(n, axis=1)[:, None]


def ef_native(alpha):
    a = np.radians(alpha)
    return np.array([np.cos(a), 0.0, -np.sin(a)])


def screen(gid, S, P, d, sh, M, alpha, ref=None):
    lo, _ = C.lower_upper(S)
    u, v = np.asarray(d.u_fct, float), np.asarray(d.v_fct, float)
    n = shock_normal_native(sh, u, v)
    e = ef_native(alpha)
    beta_geom = np.arcsin(np.clip(np.abs(n @ e), 0, 1))  # per LE root i (0 = tip, last = centreline)
    nr, nc = S["wav_x"].shape[0] - 1, S["wav_x"].shape[1] - 1
    th = AM.incidence(P, alpha)[: nr * nc].reshape(nr, nc)
    th_le = np.r_[th[0, 0], (th[:-1, 0] + th[1:, 0]) / 2, th[-1, 0]]  # node-centred first-panel incidence
    th_le = th_le[: len(u)]
    beta_fl = AM.weak_beta(M, np.maximum(th_le, 1e-9))
    detached = ~np.isfinite(beta_fl)
    mu = np.arcsin(1 / M)
    diff = beta_fl - beta_geom
    dbeta = diff - (ref if ref is not None else 0.0)
    # leading-edge normal Mach on each LE segment
    le = lo[:, 0]
    t = np.diff(le, axis=0)
    seg = np.linalg.norm(t, axis=1)
    keep = seg > 1e-9
    t = t[keep] / seg[keep][:, None]
    mn = M * np.sqrt(np.clip(1 - (t @ e) ** 2, 0, 1))
    w = seg[keep] / seg[keep].sum()
    # shock offset at the base plane relative to local lower-surface depth below the LE
    run = np.maximum(5.0 - le[:, 0], 0)
    depth = np.abs(lo[:, -1, 2] - lo[:, 0, 2])
    offset = run * np.tan(np.nan_to_num(dbeta))
    wle = np.r_[seg, 0] + np.r_[0, seg]
    wle = wle / wle.sum()
    sel = slice(1, None)  # row 0 is the collapsed wing-tip point
    rec = {"geometry_id": gid, "Mach": M, "alpha_deg": alpha, "Ma_design": C.GEOMS[gid]["Ma_design"],
           "mach_angle_deg": float(np.degrees(mu)),
           "beta_geom_deg_min": float(np.degrees(beta_geom[sel].min())), "beta_geom_deg_max": float(np.degrees(beta_geom[sel].max())),
           "beta_flight_wedge_deg_min": float(np.degrees(np.nanmin(beta_fl[sel]))) if np.isfinite(beta_fl[sel]).any() else None,
           "beta_flight_wedge_deg_max": float(np.degrees(np.nanmax(beta_fl[sel]))) if np.isfinite(beta_fl[sel]).any() else None,
           "dbeta_deg_LE_weighted_mean": float(np.degrees(np.nansum(wle[sel] * dbeta[sel]) / wle[sel].sum())),
           "dbeta_deg_centreline": float(np.degrees(dbeta[-1])),
           "dbeta_deg_abs_max": float(np.degrees(np.nanmax(np.abs(dbeta[sel])))),
           "LE_frac_beta_geom_below_mach_angle": float((wle[sel] * (beta_geom[sel] < mu)).sum() / wle[sel].sum()),
           "LE_frac_detached_wedge": float((wle[sel] * detached[sel]).sum() / wle[sel].sum()),
           "LE_normal_mach_min": float(mn.min()), "LE_normal_mach_mean": float((w * mn).sum()),
           "LE_frac_normal_mach_lt_1": float(w[mn < 1].sum()), "LE_frac_normal_mach_lt_1p3": float(w[mn < 1.3].sum()),
           "shock_offset_at_base_m_mean": float(np.nansum(wle[sel] * offset[sel]) / wle[sel].sum()),
           "offset_over_local_depth_median": float(np.nanmedian(offset[sel] / np.maximum(depth[sel], 1e-3))),
           "theta_LE_deg_mean": float(np.degrees(np.nansum(wle[sel] * th_le[sel]) / wle[sel].sum())),
           "upper_max_incidence_deg": float(np.degrees(AM.incidence(P, alpha)[2 * nr * nc:3 * nr * nc].max()))}
    return rec, diff, beta_geom


def classify(r):
    why = []
    H, Mm = THRESH["HIGH"], THRESH["MEDIUM"]
    if r["LE_frac_normal_mach_lt_1"] > H["Mn_lt_1_frac"]:
        why.append("HIGH: subsonic LE normal Mach on >25% of LE")
    if abs(r["dbeta_deg_LE_weighted_mean"]) > H["abs_dbeta_deg"]:
        why.append("HIGH: |dbeta| > 3 deg")
    if r["LE_frac_beta_geom_below_mach_angle"] > H["below_mach_frac"]:
        why.append("HIGH: design shock below Mach angle on >10% of LE")
    if r["LE_frac_detached_wedge"] > 0:
        why.append("HIGH: detached LE wedge")
    if not why:
        if r["LE_frac_normal_mach_lt_1p3"] > Mm["Mn_lt_1p3_frac"]:
            why.append("MEDIUM: LE normal Mach < 1.3 on >25% of LE")
        if abs(r["dbeta_deg_LE_weighted_mean"]) > Mm["abs_dbeta_deg"]:
            why.append("MEDIUM: |dbeta| > 1.5 deg")
        if abs(r["offset_over_local_depth_median"]) > Mm["offset_over_depth"]:
            why.append("MEDIUM: shock offset at base > 0.5 x local depth")
    level = "HIGH" if any(w.startswith("HIGH") for w in why) else ("MEDIUM" if why else "LOW")
    return level, why


def main():
    cases = C.read_csv(C.OUT / "VALIDATION_CASE_MATRIX.csv")
    rows, checks = [], {}
    for gid in C.ORDER:
        S = C.load_mat(C.GEOMS[gid]["mat"])
        P = C.R.panels(S)
        d = loadmat(C.WORK / "regen" / f"{C.SHORT[gid]}_diag.mat", squeeze_me=True, struct_as_record=False)["flowdiag"]
        inp = C.inputs10(gid)
        sh = C.R.Shock(*inp[1:5], c1=inp[5], c2=inp[6], aoa=inp[8])
        Md = C.GEOMS[gid]["Ma_design"]
        rec_d, diff_d, bg = screen(gid, S, P, d, sh, Md, C.ALPHA_DESIGN_BODY)
        beta_mat = np.asarray(d.shock_angle, float)[:, 0]
        checks[gid] = {"max_abs_beta_geom_vs_matlab_shock_angle_deg": float(np.degrees(np.abs(bg - beta_mat).max())),
                       "design_offset_beta_wedge_minus_geom_deg_median": float(np.degrees(np.nanmedian(diff_d[1:])))}
        for c in cases:
            if c["geometry_id"] != gid or c["alpha_deg"] in ("", None):
                continue
            M, a = float(c["Mach"]), float(c["alpha_deg"])
            r, _, _ = screen(gid, S, P, d, sh, M, a, ref=diff_d)
            r["case_id"] = c["case_id"]
            r["kind"] = c["kind"]
            r["CL_target"] = c["CL_target"]
            r["alpha_source"] = "Teacher iso-CL alpha (VALIDATION_CASE_MATRIX)" if c["kind"] != "DESIGN_POINT" else "design attitude"
            r["risk_level"], r["risk_reasons"] = classify(r)
            rows.append(r)
            print(c["case_id"], r["risk_level"], round(r["dbeta_deg_LE_weighted_mean"], 2),
                  round(r["LE_normal_mach_mean"], 2), round(r["LE_frac_normal_mach_lt_1"], 2), flush=True)
    order = ["case_id", "geometry_id", "kind", "Ma_design", "Mach", "CL_target", "alpha_deg", "alpha_source", "risk_level",
             "risk_reasons"]
    C.write_csv(C.OUT / "SHOCK_MISMATCH_SCREENING.csv", rows, order + [k for k in rows[0] if k not in order])
    C.dump(C.WORK / "p4_checks.json", {"beta_geom_validation": checks, "thresholds": THRESH})
    print(checks)


if __name__ == "__main__":
    main()
