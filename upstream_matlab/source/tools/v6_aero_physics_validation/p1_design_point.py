"""P1b: design-point comparison of the generator's own design pressure field (p_wav) with the Teacher.

At the design state (M = Ma_design, native AOA = -4 deg == body alpha +4 deg) the generator has
already solved the flow it designed the lower surface for: oblique-shock jump on the prescribed shock
followed by the osculating-axisymmetric method of characteristics along each wall streamline
(qiujie_shock / qiujie_inter / qiujie_end_point_improved). p_wav is that wall pressure interpolated
onto the stored lower-surface grid; the upper surface is free-stream aligned (p_wav_up = p0).

The Teacher (offdesign_aero_v4: tangent cone + expansion similarity on panel inclination) is
evaluated on exactly the same panels. Both pressure fields are integrated with the same panel
areas/normals/Sref, so differences come from the pressure model only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

import common as C  # noqa: E402
import aeromodels as AM  # noqa: E402

REGEN = C.WORK / "regen"


def load_diag(gid):
    m = loadmat(REGEN / f"{C.SHORT[gid]}_diag.mat", squeeze_me=True, struct_as_record=False)
    return m, m["flowdiag"]


def corner_avg(a):
    return (a[:-1, :-1] + a[1:, :-1] + a[1:, 1:] + a[:-1, 1:]) / 4


def design_cp_panels(P, d, Ma):
    """Design Cp per Teacher panel: lower panels from p_wav corner averages, upper panels Cp from p_wav_up."""
    cpl = 2 * (corner_avg(np.asarray(d.p_wav, float)) / d.p0 - 1) / (C.G * Ma ** 2)
    cpu = 2 * (corner_avg(np.asarray(d.p_wav_up, float)) / d.p0 - 1) / (C.G * Ma ** 2)
    out = np.empty(len(P["A"]))
    for lab, cp in (("lower", cpl), ("upper", cpu)):
        idx = np.flatnonzero(P["surface"] == lab)
        out[idx] = cp[P["row"][idx], P["col"][idx]]
    return out


def integrate(P, cp, alpha):
    a = np.radians(alpha)
    ed, el = np.array([np.cos(a), 0, np.sin(a)]), np.array([-np.sin(a), 0, np.cos(a)])
    dF = -(cp * P["A"])[:, None] * P["n"] / C.SREF
    F = dF.sum(0)
    return float(F @ el), float(F @ ed)


def wall_flow_deflection(d):
    """Angle between local design wall velocity and the free stream (wind frame vx, vy, vz)."""
    vx, vy, vz = (np.asarray(x, float) for x in (d.vx_wav, d.vy_wav, d.vz_wav))
    return np.degrees(np.arctan2(np.hypot(vy, vz), vx))


def main():
    rows, summ = [], {}
    fig_map, axm = plt.subplots(len(C.ORDER), 3, figsize=(15, 3.6 * len(C.ORDER)))
    fig_line, axl = plt.subplots(len(C.ORDER), 3, figsize=(15, 3.4 * len(C.ORDER)))
    for gi, gid in enumerate(C.ORDER):
        S = C.load_mat(C.GEOMS[gid]["mat"])
        P = C.R.panels(S)
        m, d = load_diag(gid)
        Ma = float(d.Ma)
        assert abs(Ma - C.GEOMS[gid]["Ma_design"]) < 1e-12 and abs(float(d.AOA) + 4) < 1e-12
        t = C.teacher(P, Ma, C.ALPHA_DESIGN_BODY, detail=True)
        cp_t = t["cp"]
        cp_d = design_cp_panels(P, d, Ma)
        th = np.degrees(t["theta"])
        cp_w = AM.wedge_cp(Ma, t["theta"])["cp"]
        cp_n = AM.newtonian_cp(Ma, t["theta"], modified=True)
        cl_t, cd_t = integrate(P, cp_t, C.ALPHA_DESIGN_BODY)
        cl_d, cd_d = integrate(P, cp_d, C.ALPHA_DESIGN_BODY)
        cl_w, cd_w = integrate(P, np.where(np.isfinite(cp_w), cp_w, cp_n), C.ALPHA_DESIGN_BODY)
        assert abs(cl_t - t["CL"]) < 1e-12 and abs(cd_t - t["CD"]) < 1e-12
        cp_s = AM.strip_shock_expansion(P, t["theta"], Ma)["cp"]
        cl_s, cd_s = integrate(P, np.where(np.isfinite(cp_s), cp_s, cp_n), C.ALPHA_DESIGN_BODY)
        # first block of panels = native lower half (row/col addressable)
        nlow = (S["wav_x"].shape[0] - 1) * (S["wav_x"].shape[1] - 1)
        blk = np.arange(nlow)
        A = P["A"][blk]
        r_cp = cp_t[blk] / np.where(np.abs(cp_d[blk]) > 1e-12, cp_d[blk], np.nan)
        comp = cp_d[blk] > 1e-6
        # design shock jump directly behind the leading-edge shock (first wall point of each streamline)
        wall_p = np.asarray(d.wall_p, float)  # (2N_l+1, M)
        p_jump = wall_p[0, :]
        p_end = np.array([np.asarray(d.p_wav, float)[i, -1] for i in range(wall_p.shape[1])])
        beta_le = np.degrees(np.asarray(d.shock_angle, float)[:, 0])
        defl = wall_flow_deflection(d)
        defl_c = corner_avg(defl).ravel()
        # wedge pressure driven by the design wall-flow deflection delta instead of the panel incidence theta
        cp_wd = AM.wedge_cp(Ma, np.radians(defl_c))["cp"]
        cp_wd_full = cp_t.copy() * 0
        cp_wd_full[blk] = cp_wd
        cp_wd_full[nlow:2 * nlow] = cp_wd
        cl_wd, cd_wd = integrate(P, np.nan_to_num(cp_wd_full), C.ALPHA_DESIGN_BODY)
        s = {"geometry_id": gid, "Ma_design": Ma, "alpha_body_deg": C.ALPHA_DESIGN_BODY,
             "CL_design_pwav": cl_d, "CD_design_pwav": cd_d, "LD_design_pwav": cl_d / cd_d,
             "CL_teacher": cl_t, "CD_teacher": cd_t, "LD_teacher": cl_t / cd_t,
             "CL_wedge_same_panels": cl_w, "CD_wedge_same_panels": cd_w,
             "CL_strip_SE_same_panels": cl_s, "CD_strip_SE_same_panels": cd_s,
             "CL_ratio_wedge_over_design": cl_w / cl_d, "CL_ratio_strip_over_design": cl_s / cl_d,
             "lower_cp_ratio_wedge_area_weighted": float((np.nan_to_num(cp_w[blk]) * A).sum() / (cp_d[blk] * A).sum()),
             "lower_cp_ratio_strip_area_weighted": float((np.nan_to_num(cp_s[blk]) * A).sum() / (cp_d[blk] * A).sum()),
             "CL_ratio_teacher_over_design": cl_t / cl_d, "CD_ratio_teacher_over_design": cd_t / cd_d,
             "lower_area_weighted_cp_design": float((cp_d[blk] * A).sum() / A.sum()),
             "lower_area_weighted_cp_teacher": float((cp_t[blk] * A).sum() / A.sum()),
             "lower_cp_ratio_area_weighted": float((cp_t[blk] * A).sum() / (cp_d[blk] * A).sum()),
             "lower_cp_ratio_median": float(np.nanmedian(r_cp[comp])),
             "lower_cp_ratio_p10": float(np.nanpercentile(r_cp[comp], 10)),
             "lower_cp_ratio_p90": float(np.nanpercentile(r_cp[comp], 90)),
             "lower_area_frac_abs_err_gt_20pct": float(A[comp & (np.abs(r_cp - 1) > 0.2)].sum() / A[comp].sum()),
             "lower_teacher_expansion_area_frac": float(A[th[blk] < -1e-9].sum() / A.sum()),
             "upper_cp_design_max_abs": float(np.abs(cp_d[P["surface"] == "upper"]).max()),
             "upper_cp_teacher_max_abs": float(np.abs(cp_t[P["surface"] == "upper"]).max()),
             "le_shock_beta_deg_range": [float(beta_le[1:].min()), float(beta_le[1:].max())],
             "p_jump_over_p0_range": [float((p_jump[1:] / d.p0).min()), float((p_jump[1:] / d.p0).max())],
             "p_base_over_p0_range": [float((p_end[1:] / d.p0).min()), float((p_end[1:] / d.p0).max())],
             "teacher_theta_deg_lower_mean": float((th[blk] * A).sum() / A.sum()),
             "design_wall_deflection_deg_mean": float((defl_c * A).sum() / A.sum()),
             "theta_minus_deflection_deg_mean": float(((th[blk] - defl_c) * A).sum() / A.sum()),
             "lower_cp_ratio_wedge_on_deflection_area_weighted": float((np.nan_to_num(cp_wd) * A).sum() / (cp_d[blk] * A).sum()),
             "CL_wedge_on_design_deflection": cl_wd, "CL_ratio_wedge_on_deflection_over_design": cl_wd / cl_d}
        summ[gid] = s
        # per-streamline table (each lower row = one osculating-plane streamline)
        nr, nc = S["wav_x"].shape[0] - 1, S["wav_x"].shape[1] - 1
        cpd2, cpt2, A2 = cp_d[blk].reshape(nr, nc), cp_t[blk].reshape(nr, nc), A.reshape(nr, nc)
        th2 = th[blk].reshape(nr, nc)
        for i in range(nr):
            rows.append({"geometry_id": gid, "Ma_design": Ma, "row": i,
                         "y_mid_m": float(np.mean(corner_avg(S["wav_y"])[i])),
                         "area_half_m2": float(A2[i].sum()),
                         "cp_design_mean": float((cpd2[i] * A2[i]).sum() / A2[i].sum()),
                         "cp_teacher_mean": float((cpt2[i] * A2[i]).sum() / A2[i].sum()),
                         "cp_ratio_teacher_over_design": float((cpt2[i] * A2[i]).sum() / (cpd2[i] * A2[i]).sum())
                         if abs((cpd2[i] * A2[i]).sum()) > 1e-14 else None,
                         "cp_design_le": float(cpd2[i, 0]), "cp_teacher_le": float(cpt2[i, 0]),
                         "cp_design_te": float(cpd2[i, -1]), "cp_teacher_te": float(cpt2[i, -1]),
                         "theta_teacher_mean_deg": float((th2[i] * A2[i]).sum() / A2[i].sum()),
                         "design_wall_deflection_mean_deg": float((defl_c.reshape(nr, nc)[i] * A2[i]).sum() / A2[i].sum()),
                         "shock_beta_le_deg": float(beta_le[i + 1]),
                         "p_jump_over_p0": float(p_jump[i + 1] / d.p0)})
        # ---------------------------------------------------------------- figures
        xc, yc = corner_avg(S["wav_x"]), corner_avg(S["wav_y"])
        vmax = max(np.nanmax(cpd2), np.nanmax(cpt2))
        for k, (val, ttl) in enumerate(((cpd2, "design p_wav Cp"), (cpt2, "Teacher Cp (tangent cone)"),
                                        (cpt2 / np.where(np.abs(cpd2) > 1e-9, cpd2, np.nan), "Teacher / design"))):
            ax = axm[gi, k]
            if k < 2:
                sc = ax.scatter(xc.ravel(), yc.ravel(), c=val.ravel(), s=6, cmap="viridis", vmin=min(0, np.nanmin(val)), vmax=vmax)
            else:
                sc = ax.scatter(xc.ravel(), yc.ravel(), c=val.ravel(), s=6, cmap="RdBu_r", vmin=0, vmax=2)
            fig_map.colorbar(sc, ax=ax)
            ax.set_title(f"{C.SHORT[gid]} Md={Ma:g}: {ttl}", fontsize=9)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        for k, i in enumerate((nr - 1, nr // 2, max(2, nr // 8))):
            ax = axl[gi, k]
            ax.plot(xc[i], cpd2[i], "k-", label="design p_wav")
            ax.plot(xc[i], cpt2[i], "r--", label="Teacher cone")
            cw = AM.wedge_cp(Ma, np.radians(th2[i]))["cp"]
            ax.plot(xc[i], cw, "b:", label="tangent wedge")
            ax.plot(xc[i], cp_s[blk].reshape(nr, nc)[i], "g-.", label="strip shock-expansion")
            ax.set_title(f"{C.SHORT[gid]} row {i} (y~{np.mean(yc[i]):.2f} m)", fontsize=9)
            ax.set_xlabel("x [m]"); ax.set_ylabel("Cp")
            if k == 0:
                ax.legend(fontsize=7)
        print(gid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "geometry_id"})
    fig_map.tight_layout(); fig_map.savefig(C.FIG / "V1_design_cp_maps.png", dpi=110)
    fig_line.tight_layout(); fig_line.savefig(C.FIG / "V1_design_cp_streamlines.png", dpi=110)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(C.ORDER))
    ax.bar(x - 0.2, [summ[g]["CL_ratio_teacher_over_design"] for g in C.ORDER], 0.4, label="CL Teacher/design")
    ax.bar(x + 0.2, [summ[g]["CD_ratio_teacher_over_design"] for g in C.ORDER], 0.4, label="CD Teacher/design")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x, [C.SHORT[g] for g in C.ORDER]); ax.legend(); ax.set_title("Design point: Teacher vs generator design pressure")
    fig.tight_layout(); fig.savefig(C.FIG / "V1_design_force_ratios.png", dpi=110)
    C.write_csv(C.OUT / "DESIGN_POINT_PRESSURE_COMPARISON.csv",
                [{"level": "GEOMETRY", **{k: v for k, v in s.items()}} for s in summ.values()] +
                [{"level": "STREAMLINE_ROW", **r} for r in rows])
    C.dump(C.WORK / "p1_design_point_summary.json", summ)


if __name__ == "__main__":
    main()
