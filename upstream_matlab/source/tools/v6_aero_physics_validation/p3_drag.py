"""P3: skin-friction (Eckert reference-temperature method) and base-pressure ranges on the mission cases.

Inviscid part: Teacher (model A) and strip shock-expansion (model C) at their own iso-CL alpha (from P2).
Friction: per panel, edge state from the strip shock-expansion solution at that alpha (local Mach and
pressure; total temperature conserved), running length from the strip leading edge, Eckert T*,
Sutherland viscosity, laminar Cf* = 0.664/sqrt(Re*), turbulent Cf* = 0.0592/Re*^0.2, tau = Cf* rho* ue^2/2,
acting along the free stream projected on the panel. Base: Cp_base in {0, -1/M^2, -2/(gamma M^2)} on the
exact open-base area. Nothing is fed back into any label or Teacher.
"""
from __future__ import annotations

import json

import numpy as np

import aeromodels as AM
import common as C

RGAS = 287.05287
TW_COLD = 300.0
FRICTION = {
    "LAM_COLD": ("lam", "cold", None), "LAM_ADIABATIC": ("lam", "adiabatic", None),
    "TURB_COLD": ("turb", "cold", None), "TURB_ADIABATIC": ("turb", "adiabatic", None),
    "TRANS_RE2E6_COLD": ("trans", "cold", 2e6), "TRANS_RE5E5_COLD": ("trans", "cold", 5e5),
    "TRANS_RE1E7_COLD": ("trans", "cold", 1e7),
}
BASE = {"BASE_0": lambda M: 0.0, "BASE_M2": lambda M: -1.0 / M ** 2, "BASE_VACUUM": lambda M: -2.0 / (C.G * M ** 2)}


def sutherland(T):
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


def running_length(P):
    nc = int(P["col"].max() + 1)
    R = P["R"].reshape(-1, nc, 3)
    d = np.linalg.norm(np.diff(R, axis=1), axis=2)
    s = np.concatenate([np.zeros((R.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)
    # half of the first panel's streamwise extent as the distance of its centre from the leading edge
    first = d[:, :1] / 2 if d.shape[1] else 0
    return (s + first).ravel()


def friction(P, M, alpha, cond, strip, s, mode, wall, re_tr):
    a = np.radians(alpha)
    ed, el = np.array([np.cos(a), 0, np.sin(a)]), np.array([-np.sin(a), 0, np.cos(a)])
    Tinf = cond["T_inf_K"]
    pinf = 2 * cond["q_pa"] / (C.G * M ** 2)
    rhoinf = pinf / (RGAS * Tinf)
    uinf = M * np.sqrt(C.G * RGAS * Tinf)
    qinf = 0.5 * rhoinf * uinf ** 2
    Me = strip["M_local"]
    ok = np.isfinite(Me) & (strip["cp"] > -2 / (C.G * M ** 2) + 1e-12)
    pe = pinf * (1 + C.G * M ** 2 * np.nan_to_num(strip["cp"]) / 2)
    T0 = Tinf * (1 + AM.GM * M ** 2)
    Me_ = np.where(ok, Me, M)
    Te = T0 / (1 + AM.GM * Me_ ** 2)
    ue = Me_ * np.sqrt(C.G * RGAS * Te)
    rhoe = pe / (RGAS * Te)
    r = 0.85 if mode == "lam" else 0.89
    Taw = Te * (1 + r * AM.GM * Me_ ** 2)
    Tw = np.full_like(Te, TW_COLD) if wall == "cold" else Taw
    Ts = Te * (1 + 0.032 * Me_ ** 2 + 0.58 * (Tw / Te - 1))
    rhos = pe / (RGAS * Ts)
    mus = sutherland(Ts)
    Res = rhos * ue * s / mus
    Ree = rhoe * ue * s / sutherland(Te)
    cf_lam = 0.664 / np.sqrt(np.maximum(Res, 1.0))
    cf_turb = 0.0592 / np.maximum(Res, 1.0) ** 0.2
    if mode == "lam":
        cf = cf_lam
    elif mode == "turb":
        cf = cf_turb
    else:
        cf = np.where(Ree < re_tr, cf_lam, cf_turb)
    tau = np.where(ok, 0.5 * rhos * ue ** 2 * cf, 0.0)
    tdir = ed[None, :] - (P["n"] @ ed)[:, None] * P["n"]
    tdir /= np.maximum(np.linalg.norm(tdir, axis=1), 1e-15)[:, None]
    F = ((tau * P["A"])[:, None] * tdir).sum(0) / (qinf * C.SREF)
    return {"CDf": float(F @ ed), "CLf": float(F @ el), "Re_e_max": float(np.nanmax(np.where(ok, Ree, np.nan))),
            "Re_e_unit_per_m_inf": float(rhoinf * uinf / sutherland(Tinf)),
            "turbulent_area_frac": float(P["A"][(Ree >= re_tr) & ok].sum() / P["A"].sum()) if mode == "trans" else None,
            "Tw_mean_K": float(np.nanmean(Tw))}


def main():
    m0 = {g["geometry_id"]: g for g in json.loads((C.WORK / "p0_metrics.json").read_text(encoding="utf-8"))}
    cases = {r["case_id"]: r for r in C.read_csv(C.OUT / "VALIDATION_CASE_MATRIX.csv")}
    p2 = {(r["case_id"], r["model"]): r for r in C.read_csv(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv")}
    rows = []
    for gid in C.ORDER:
        S = C.load_mat(C.GEOMS[gid]["mat"])
        P = C.R.panels(S)
        s = running_length(P)
        Abase = m0[gid]["base_area_full_exact"]
        Swet = 2 * (m0[gid]["V2_wetted_lower_half"] + m0[gid]["V2_wetted_upper_half"])
        for cid, c in cases.items():
            if c["geometry_id"] != gid or c["kind"] != "MISSION_ISO_CL":
                continue
            M = float(c["Mach"])
            cond = {k: float(c[k]) for k in ("h_m", "V_mps", "q_pa", "T_inf_K")}
            for model in ("A_tangent_cone_teacher", "C_strip_shock_expansion"):
                r2 = p2.get((cid, model))
                if not r2 or r2.get("reachable") != "True":
                    continue
                alpha, cl, cdp = float(r2["alpha_deg"]), float(r2["CL"]), float(r2["CD"])
                th = AM.incidence(P, alpha)
                strip = AM.strip_shock_expansion(P, th, M)
                base_rec = {"case_id": cid, "geometry_id": gid, "Mach": M, "CL_target": float(c["CL_target"]),
                            "inviscid_model": model, "alpha_deg": alpha, "CL_inviscid": cl, "CD_inviscid": cdp,
                            "h_m": cond["h_m"], "q_pa": cond["q_pa"], "T_inf_K": cond["T_inf_K"],
                            "S_wet_full_m2": Swet, "A_base_full_m2": Abase,
                            "CDf_const_cf1e-3": 1e-3 * Swet / C.SREF}
                fr = {k: friction(P, M, alpha, cond, strip, s, *v) for k, v in FRICTION.items()}
                for fk, f in fr.items():
                    for bk, bf in BASE.items():
                        cdb = -bf(M) * Abase / C.SREF
                        cd = cdp + f["CDf"] + cdb
                        rows.append({**base_rec, "friction": fk, "base": bk, "CD_friction": f["CDf"],
                                     "CL_friction": f["CLf"], "CD_base": cdb, "CD_total": cd, "LD_total": cl / cd,
                                     "friction_share": f["CDf"] / cd, "base_share": cdb / cd,
                                     "Re_unit_inf_per_m": f["Re_e_unit_per_m_inf"], "Re_e_max": f["Re_e_max"],
                                     "turbulent_area_frac": f["turbulent_area_frac"], "Tw_mean_K": f["Tw_mean_K"]})
            print(gid, cid, flush=True)
    # J-like iso-CL drag ratios vs G42 and rankings per scenario
    key = {(r["case_id"], r["inviscid_model"], r["friction"], r["base"]): r for r in rows}
    for r in rows:
        g = key.get((r["case_id"].replace(C.SHORT[r["geometry_id"]], "G42", 1), r["inviscid_model"], r["friction"], r["base"]))
        if g:
            r["CD_total_over_G42"] = r["CD_total"] / g["CD_total"]
    rank = []
    for model in ("A_tangent_cone_teacher", "C_strip_shock_expansion"):
        for fk in FRICTION:
            for bk in BASE:
                ratios = {gid: [] for gid in C.ORDER}
                for r in rows:
                    if r["inviscid_model"] == model and r["friction"] == fk and r["base"] == bk and "CD_total_over_G42" in r:
                        ratios[r["geometry_id"]].append(r["CD_total_over_G42"])
                if all(ratios.values()):
                    rank.append({"inviscid_model": model, "friction": fk, "base": bk,
                                 **{f"mean_CD_over_G42_{C.SHORT[g]}": float(np.mean(v)) for g, v in ratios.items()},
                                 "n_cases": len(ratios["G42"]),
                                 "best": C.SHORT[min(ratios, key=lambda g: np.mean(ratios[g]))]})
    C.write_csv(C.OUT / "DRAG_COMPONENT_SENSITIVITY.csv", rows)
    C.write_csv(C.WORK / "p3_scenario_ranking.csv", rank)
    for r in rank:
        print(r["inviscid_model"][:1], r["friction"], r["base"], {k[-7:]: round(v, 3) for k, v in r.items() if k.startswith("mean_")}, r["best"])


if __name__ == "__main__":
    main()
