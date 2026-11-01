"""P5 synthesis: derived numbers used by REPORT_ZH / TEACHER_RELIABILITY_ASSESSMENT (read-only on all inputs).

- V1: leading-edge vs trailing-edge and inner vs outer span Teacher/design Cp ratios (STREAMLINE_ROW level).
- Off-design (Teacher): same-alpha (+4 deg), same-CL (0.045) and alpha-optimised L/D per Mach.
- Cross-Mach ranking crossings (pairwise CD sign change between Mach numbers at the same CL), per inviscid model
  and per P3 drag scenario, with the relative gap at both ends.
- G42 vs MD9.873 iso-CL CD ratio over every model / scenario.
- Free-stream state per mission Mach (p, rho, Re/m) for the unified physical definition table.
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict

import numpy as np

import common as C

RGAS = 287.05287


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def v1_location():
    out = {}
    rows = [r for r in rd(C.OUT / "DESIGN_POINT_PRESSURE_COMPARISON.csv") if r["level"] == "STREAMLINE_ROW"]
    for gid in C.ORDER:
        rr = [r for r in rows if r["geometry_id"] == gid]
        A = np.array([float(r["area_half_m2"]) for r in rr])
        y = np.array([float(r["y_mid_m"]) for r in rr])
        f = lambda k: np.array([float(r[k]) for r in rr])
        le_t, le_d, te_t, te_d = f("cp_teacher_le"), f("cp_design_le"), f("cp_teacher_te"), f("cp_design_te")
        mt, md = f("cp_teacher_mean"), f("cp_design_mean")
        inner = y <= 0.5 * y.max()
        out[gid] = {
            "le_ratio": float((A * le_t).sum() / (A * le_d).sum()),
            "te_ratio": float((A * te_t).sum() / (A * te_d).sum()),
            "inner_half_span_ratio": float((A[inner] * mt[inner]).sum() / (A[inner] * md[inner]).sum()),
            "outer_half_span_ratio": float((A[~inner] * mt[~inner]).sum() / (A[~inner] * md[~inner]).sum()),
            "row_ratio_min": float(np.nanmin(mt / md)), "row_ratio_max": float(np.nanmax(mt / md)),
            "y_at_min_ratio": float(y[np.nanargmin(mt / md)]), "y_max": float(y.max()),
        }
    return out


def offdesign():
    sw = rd(C.WORK / "p2_alpha_sweeps.csv")
    lo = rd(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv")
    p1 = json.loads((C.WORK / "p1_design_point_summary.json").read_text(encoding="utf-8"))
    out = {}
    for gid in C.ORDER:
        g = {"design": {"Mach": C.GEOMS[gid]["Ma_design"], "CL_teacher_a4": p1[gid]["CL_teacher"],
                        "CD_teacher_a4": p1[gid]["CD_teacher"], "LD_teacher_a4": p1[gid]["LD_teacher"],
                        "LD_design_pwav_a4": p1[gid]["LD_design_pwav"]}}
        for M in ("5.5", "6.5", "7.5"):
            s = [r for r in sw if r["geometry_id"] == gid and r["Mach"] == M and r["CL_A_tangent_cone_teacher"]]
            a = np.array([float(r["alpha_deg"]) for r in s])
            cl = np.array([float(r["CL_A_tangent_cone_teacher"]) for r in s])
            cd = np.array([float(r["CD_A_tangent_cone_teacher"]) for r in s])
            o = np.argsort(a)
            a, cl, cd = a[o], cl[o], cd[o]
            cl4, cd4 = float(np.interp(4.0, a, cl)), float(np.interp(4.0, a, cd))
            pos = cl > 0
            ld = cl[pos] / cd[pos]
            k = int(np.argmax(ld))
            iso = {r["CL_target"]: r for r in lo if r["geometry_id"] == gid and r["Mach"] == M
                   and r["model"].startswith("A") and r["kind"] == "MISSION_ISO_CL"}
            g[M] = {"CL_a4": cl4, "CD_a4": cd4, "LD_a4": cl4 / cd4,
                    "LDmax": float(ld[k]), "alpha_LDmax": float(a[pos][k]), "CL_at_LDmax": float(cl[pos][k]),
                    "iso_CL0.045": {"alpha": float(iso["0.045"]["alpha_deg"]), "CD": float(iso["0.045"]["CD"]),
                                    "LD": float(iso["0.045"]["LD"])}}
        out[gid] = g
    return out


def crossings_from(table):
    """table: {(scenario, CL): {Mach: {geom: CD}}} -> list of crossings."""
    res = []
    for (sc, cl), byM in table.items():
        Ms = sorted(byM, key=float)
        for gi, gj in itertools.combinations(C.ORDER, 2):
            a, b = C.SHORT[gi], C.SHORT[gj]
            gaps = {M: byM[M][a] / byM[M][b] - 1 for M in Ms if a in byM[M] and b in byM[M]}
            if len(gaps) < 2:
                continue
            signs = {np.sign(v) for v in gaps.values()}
            if len(signs) > 1:
                res.append({"scenario": sc, "CL": cl, "pair": f"{a} vs {b}", "gap_by_Mach": {k: round(v, 4) for k, v in gaps.items()},
                            "min_abs_gap_at_crossing_ends": float(min(abs(max(gaps.values())), abs(min(gaps.values()))))})
    return res


def crossings():
    lo = [r for r in rd(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv") if r["kind"] == "MISSION_ISO_CL" and r["reachable"] == "True"]
    t = defaultdict(lambda: defaultdict(dict))
    for r in lo:
        if r["model"][0] in "ABC":
            t[(r["model"], r["CL_target"])][r["Mach"]][C.SHORT[r["geometry_id"]]] = float(r["CD"])
    inv = crossings_from(t)
    d3 = rd(C.OUT / "DRAG_COMPONENT_SENSITIVITY.csv")
    t3 = defaultdict(lambda: defaultdict(dict))
    for r in d3:
        t3[(f"{r['inviscid_model'][0]}|{r['friction']}|{r['base']}", r["CL_target"])][r["Mach"]][C.SHORT[r["geometry_id"]]] = float(r["CD_total"])
    vis = crossings_from(t3)
    # base-only (no friction) variant from the same rows
    t4 = defaultdict(lambda: defaultdict(dict))
    for r in d3:
        if r["friction"] == "LAM_COLD":
            t4[(f"{r['inviscid_model'][0]}|NO_FRICTION|{r['base']}", r["CL_target"])][r["Mach"]][C.SHORT[r["geometry_id"]]] = \
                float(r["CD_inviscid"]) + float(r["CD_base"])
    base_only = crossings_from(t4)

    def summ(lst, n_total):
        big = [c for c in lst if c["min_abs_gap_at_crossing_ends"] > 0.02]
        pairs = defaultdict(int)
        for c in lst:
            pairs[c["pair"]] += 1
        return {"n_crossings": len(lst), "n_with_both_ends_gap_gt_2pct": len(big), "n_scenario_CL_tables": n_total,
                "by_pair": dict(pairs), "largest": sorted(lst, key=lambda c: -c["min_abs_gap_at_crossing_ends"])[:5]}
    return {"inviscid_models": summ(inv, len(t)), "p3_total_drag": summ(vis, len(t3)), "base_only": summ(base_only, len(t4)),
            "base_only_rank": base_only_rank(t4)}


def base_only_rank(t4):
    out = {}
    for (sc, cl), byM in t4.items():
        for M, d in byM.items():
            if len(d) == 4:
                out.setdefault(sc, defaultdict(int))[min(d, key=d.get)] += 1
    return {k: dict(v) for k, v in out.items()}


def g42_vs_md9():
    vals = defaultdict(list)
    lo = [r for r in rd(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv") if r["kind"] == "MISSION_ISO_CL" and r["reachable"] == "True"]
    by = {(r["Mach"], r["CL_target"], r["model"], C.SHORT[r["geometry_id"]]): float(r["CD"]) for r in lo}
    for (M, cl, m, g), v in by.items():
        if g == "MD9.873" and (M, cl, m, "G42") in by and m[0] in "ABC":
            vals[f"inviscid_{m[0]}"].append(v / by[(M, cl, m, "G42")])
    for r in rd(C.OUT / "DRAG_COMPONENT_SENSITIVITY.csv"):
        if C.SHORT[r["geometry_id"]] == "MD9.873":
            vals[f"total_{r['inviscid_model'][0]}"].append(float(r["CD_total_over_G42"]))
    return {k: {"n": len(v), "min": float(min(v)), "max": float(max(v)), "mean": float(np.mean(v))} for k, v in vals.items()}


def freestream():
    out = {}
    for r in rd(C.OUT / "VALIDATION_CASE_MATRIX.csv"):
        if r["kind"] == "MISSION_ISO_CL" and r["Mach"] not in out:
            M, q, T = float(r["Mach"]), float(r["q_pa"]), float(r["T_inf_K"])
            p = 2 * q / (C.G * M ** 2)
            rho = p / (RGAS * T)
            V = M * np.sqrt(C.G * RGAS * T)
            mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
            out[r["Mach"]] = {"h_m": float(r["h_m"]), "q_pa": q, "T_K": T, "p_pa": p, "rho": rho, "V_mps": V,
                              "Re_per_m": rho * V / mu, "T0_K": T * (1 + 0.2 * M ** 2)}
    return out


def main():
    res = {"v1_location": v1_location(), "offdesign_teacher": offdesign(), "crossings": crossings(),
           "G42_vs_MD9873_CD_ratio": g42_vs_md9(), "freestream": freestream()}
    (C.WORK / "p5_synthesis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["v1_location"], indent=0))
    for g, d in res["offdesign_teacher"].items():
        print(g, json.dumps(d))
    for k in ("inviscid_models", "p3_total_drag", "base_only"):
        c = res["crossings"][k]
        print(k, c["n_crossings"], c["n_with_both_ends_gap_gt_2pct"], c["n_scenario_CL_tables"], c["by_pair"])
        for x in c["largest"][:3]:
            print("   ", x)
    print("base_only_rank", res["crossings"]["base_only_rank"])
    print(json.dumps(res["G42_vs_MD9873_CD_ratio"]))
    print(json.dumps(res["freestream"]))


if __name__ == "__main__":
    main()
