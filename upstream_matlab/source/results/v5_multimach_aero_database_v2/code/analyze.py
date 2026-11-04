"""Performance analysis Q1-Q9 on the assembled V2 aero database (RAW Teacher grid labels).

Label rules come verbatim from provenance/analysis_plan.json (frozen before the build).
Skin friction enters only as a labelled what-if sensitivity, never the labels.
Writes equal_lift_drag_comparison.csv, optimal_geometry_by_mach.csv,
optimal_geometry_by_cl.csv, volume_constrained_comparison.csv, ld_summary.csv,
provenance/analysis.json and figures/*.png.
"""
import json
import math
from collections import Counter, defaultdict, deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from common import (ALPHA_DEG, FLIGHT_MACH, GEO_DB, OUT, REFERENCE, dump, fnum, load, read_csv,  # noqa: E402
                    write_csv)

PLAN = load(OUT / "provenance/analysis_plan.json")["performance"]
CL_LEVELS = PLAN["CL_levels"]
CL_USABLE = PLAN["usable_CL_min"]
USABLE = [c for c in CL_LEVELS if c >= CL_USABLE]
REL_FLOOR = 0.01
CF_WHATIF = [0.001, 0.002]
TWIN = "V2_MD15_001"
MDS = [4.0, 5.0, 6.0, 7.0, 8.0, 15.0]
MD_COLORS = {4.0: "tab:red", 5.0: "tab:orange", 6.0: "tab:green", 7.0: "tab:blue", 8.0: "tab:purple",
             15.0: "tab:brown"}
FIG = OUT / "figures"


class Curve:
    def __init__(self, CL, CD):
        self.a = np.array(ALPHA_DEG)
        self.CL, self.CD = np.asarray(CL), np.asarray(CD)
        self.mono = bool(np.all(np.diff(self.CL) > 0))
        self.p_cl = PchipInterpolator(self.a, self.CL)
        self.p_cd = PchipInterpolator(self.a, self.CD)
        self.inv = PchipInterpolator(self.CL, self.a) if self.mono else None

    def at_cl(self, cl):
        """(alpha, CD_pchip, u=|CD_pchip-CD_linear|) or None if cl outside the alpha grid range."""
        if not self.mono or cl < self.CL[0] or cl > self.CL[-1]:
            return None
        a = float(self.inv(cl))
        cd = float(self.p_cd(a))
        a_l = float(np.interp(cl, self.CL, self.a))
        cd_l = float(np.interp(a_l, self.a, self.CD))
        return a, cd, abs(cd - cd_l)

    def ld(self):
        ld = self.CL / self.CD
        i = int(np.argmax(ld))
        fa = np.linspace(self.a[0], self.a[-1], 2801)
        fcl, fcd = self.p_cl(fa), self.p_cd(fa)
        fld = fcl / fcd
        j = int(np.argmax(fld))
        u = self.CL >= CL_USABLE
        fu = fcl >= CL_USABLE
        out = {"LD_max_grid": float(ld[i]), "alpha_LDmax_grid": float(self.a[i]), "CL_at_LDmax_grid": float(self.CL[i]),
               "LD_max_pchip": float(fld[j]), "alpha_LDmax_pchip": float(fa[j]),
               "LD_max_usable_grid": float(ld[u].max()) if u.any() else math.nan,
               "alpha_LDmax_usable_grid": float(self.a[u][int(np.argmax(ld[u]))]) if u.any() else math.nan,
               "LD_max_usable_pchip": float(fld[fu].max()) if fu.any() else math.nan,
               "alpha_LDmax_usable_pchip": float(fa[fu][int(np.argmax(fld[fu]))]) if fu.any() else math.nan,
               "CL_min": float(self.CL[0]), "CL_max": float(self.CL[-1]), "CD_min": float(self.CD.min()),
               "CL_monotonic": self.mono}
        return out


def sig(d, u, ref):
    return abs(d) > 2 * u and abs(d) / ref > REL_FLOOR


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = {r["geometry_id"]: r for r in read_csv(GEO_DB / "geometry_metrics.csv") if r["geometry_id"]}
    rows = read_csv(OUT / "aero_database.csv") + read_csv(OUT / "g42_baseline.csv")
    meta, data = {}, defaultdict(dict)
    for r in rows:
        g = r["geometry_id"]
        if g not in meta:
            m = metrics[TWIN if g == "G42" else g]
            meta[g] = {"kind": r["kind"], "design_mach": fnum(r["design_mach"]), "design_space": r["design_space"],
                       "volume": fnum(r["volume_full"]), "thick": fnum(r["grid_thickness_max"]),
                       "Sw": 2 * (fnum(m["wetted_area_lower_half"]) + fnum(m["wetted_area_upper_half"]))}
        if r["status"] == "PASS":
            data[(g, float(r["flight_mach"]))][float(r["alpha_deg"])] = (fnum(r["CL"]), fnum(r["CD"]))
    curves = {}
    for k, d in data.items():
        if len(d) == len(ALPHA_DEG):
            curves[k] = Curve([d[a][0] for a in ALPHA_DEG], [d[a][1] for a in ALPHA_DEG])
    V2 = sorted(g for g in meta if g != "G42")
    g42v, g42t = meta["G42"]["volume"], meta["G42"]["thick"]
    C1 = {g for g in V2 if meta[g]["volume"] >= g42v}
    C2 = {g for g in C1 if meta[g]["thick"] >= g42t}
    cons = {"C0": set(V2), "C1": C1, "C2": C2}

    # ---------------------------------------------------- Q1 L/D per curve
    ld_rows, LD = [], {}
    for (g, M), c in sorted(curves.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        o = c.ld()
        for cf in CF_WHATIF:
            cdf = c.CD + cf * meta[g]["Sw"] / REFERENCE["area_m2"]
            o["whatif_LD_max_Cf%g" % cf] = float(np.max(c.CL / cdf))
        o.update({"geometry_id": g, "flight_mach": M, "kind": meta[g]["kind"], "design_mach": meta[g]["design_mach"],
                  "volume_ratio_to_G42": meta[g]["volume"] / g42v, "thickness_ratio_to_G42": meta[g]["thick"] / g42t})
        LD[(g, M)] = o
        ld_rows.append(o)
    write_csv(OUT / "ld_summary.csv", ld_rows,
              ["geometry_id", "kind", "design_mach", "flight_mach", "LD_max_grid", "alpha_LDmax_grid",
               "CL_at_LDmax_grid", "LD_max_pchip", "alpha_LDmax_pchip", "LD_max_usable_grid",
               "alpha_LDmax_usable_grid", "LD_max_usable_pchip", "alpha_LDmax_usable_pchip", "CL_min", "CL_max",
               "CD_min", "CL_monotonic", "whatif_LD_max_Cf0.001", "whatif_LD_max_Cf0.002", "volume_ratio_to_G42",
               "thickness_ratio_to_G42"])
    opt_mach = []
    for M in FLIGHT_MACH:
        row = {"flight_mach": M}
        g42 = LD.get(("G42", M))
        row["G42_LD_max_usable"] = g42["LD_max_usable_grid"] if g42 else math.nan
        row["G42_alpha"] = g42["alpha_LDmax_usable_grid"] if g42 else math.nan
        for cn, cs in cons.items():
            cand = [LD[(g, M)] for g in cs if (g, M) in LD and not math.isnan(LD[(g, M)]["LD_max_usable_grid"])]
            if cand:
                b = max(cand, key=lambda o: o["LD_max_usable_grid"])
                row.update({"%s_best" % cn: b["geometry_id"], "%s_design_mach" % cn: b["design_mach"],
                            "%s_LD_max_usable" % cn: b["LD_max_usable_grid"],
                            "%s_alpha" % cn: b["alpha_LDmax_usable_grid"],
                            "%s_gain_vs_G42" % cn: b["LD_max_usable_grid"] / row["G42_LD_max_usable"] - 1,
                            "%s_n_better_than_G42" % cn: sum(1 for o in cand
                                                             if o["LD_max_usable_grid"] > row["G42_LD_max_usable"]),
                            "%s_n" % cn: len(cand)})
        cand = [LD[(g, M)] for g in V2 if (g, M) in LD]
        if cand:
            b = max(cand, key=lambda o: o["LD_max_grid"])
            row.update({"C0_best_unrestricted_CL": b["geometry_id"], "C0_LD_max_any_CL": b["LD_max_grid"],
                        "C0_CL_at_LD_max_any_CL": b["CL_at_LDmax_grid"]})
        for md in MDS:
            c = [LD[(g, M)] for g in V2 if meta[g]["design_mach"] == md and (g, M) in LD
                 and not math.isnan(LD[(g, M)]["LD_max_usable_grid"])]
            if c:
                b = max(c, key=lambda o: o["LD_max_usable_grid"])
                row["best_MD%g" % md] = b["geometry_id"]
                row["best_MD%g_LD_max_usable" % md] = b["LD_max_usable_grid"]
                row["median_MD%g_LD_max_usable" % md] = float(np.median([o["LD_max_usable_grid"] for o in c]))
        opt_mach.append(row)
    write_csv(OUT / "optimal_geometry_by_mach.csv", opt_mach,
              sorted({k for r in opt_mach for k in r}, key=lambda k: (k != "flight_mach", not k.startswith("G42"),
                                                                     not k.startswith("C"), k)))

    # ---------------------------------------------------- Q3 equal-lift CD
    EL = {}
    el_rows = []
    for M in FLIGHT_MACH:
        for c in CL_LEVELS:
            ref = curves.get(("G42", M))
            rr = ref.at_cl(c) if ref else None
            for g in ["G42"] + V2:
                cv = curves.get((g, M))
                base = {"flight_mach": M, "CL_target": c, "usable_CL": c >= CL_USABLE, "geometry_id": g,
                        "kind": meta[g]["kind"], "design_mach": meta[g]["design_mach"],
                        "volume_ratio_to_G42": meta[g]["volume"] / g42v,
                        "thickness_ratio_to_G42": meta[g]["thick"] / g42t, "in_C1": g in C1, "in_C2": g in C2,
                        "G42_feasible": rr is not None}
                if cv is None:
                    el_rows.append(dict(base, feasible=False, note="curve incomplete (non-PASS label)"))
                    continue
                res = cv.at_cl(c)
                if res is None:
                    el_rows.append(dict(base, feasible=False,
                                        note="CL outside alpha grid range" if cv.mono else "CL(alpha) not monotonic"))
                    continue
                a, cd, u = res
                EL[(g, M, c)] = (cd, u, a)
                row = dict(base, feasible=True, alpha_required_deg=a, CD_pchip=cd, interp_unc_abs=u,
                           LD_at_target=c / cd, note="")
                if rr:
                    d = cd - rr[1]
                    s = sig(d, u + rr[2], rr[1])
                    row.update({"CD_G42": rr[1], "G42_interp_unc_abs": rr[2], "dCD_abs": d, "dCD_rel": d / rr[1],
                                "significant": s, "significantly_lower": s and d < 0})
                    for cf in CF_WHATIF:
                        fg = cf * meta[g]["Sw"] / REFERENCE["area_m2"]
                        f0 = cf * meta["G42"]["Sw"] / REFERENCE["area_m2"]
                        row["whatif_dCD_rel_Cf%g" % cf] = (cd + fg - rr[1] - f0) / (rr[1] + f0)
                el_rows.append(row)
    write_csv(OUT / "equal_lift_drag_comparison.csv", el_rows,
              ["flight_mach", "CL_target", "usable_CL", "geometry_id", "kind", "design_mach", "volume_ratio_to_G42",
               "thickness_ratio_to_G42", "in_C1", "in_C2", "feasible", "alpha_required_deg", "CD_pchip",
               "interp_unc_abs", "LD_at_target", "G42_feasible", "CD_G42", "G42_interp_unc_abs", "dCD_abs", "dCD_rel",
               "significant", "significantly_lower", "whatif_dCD_rel_Cf0.001", "whatif_dCD_rel_Cf0.002", "note"])

    def best(M, c, cs):
        cand = [(EL[(g, M, c)][0], g) for g in cs if (g, M, c) in EL]
        return (min(cand)[1], len(cand)) if cand else (None, 0)

    def rel_g42(g, M, c):
        r, e = EL.get(("G42", M, c)), EL.get((g, M, c))
        if not r or not e:
            return None
        d = e[0] - r[0]
        return {"dCD_rel": d / r[0], "significant": sig(d, e[1] + r[1], r[0]), "lower": d < 0}

    # ---------------------------------------------------- Q4-Q7 winners by Mach / by CL, constraints
    by_cl_rows, vc_rows = [], []
    W = {cn: {} for cn in cons}
    for M in FLIGHT_MACH:
        for c in CL_LEVELS:
            row = {"flight_mach": M, "CL_target": c, "usable_CL": c >= CL_USABLE,
                   "G42_CD": EL[("G42", M, c)][0] if ("G42", M, c) in EL else math.nan}
            for cn, cs in cons.items():
                g, n = best(M, c, cs)
                W[cn][(M, c)] = g
                rg = rel_g42(g, M, c) if g else None
                row.update({"%s_winner" % cn: g, "%s_design_mach" % cn: meta[g]["design_mach"] if g else math.nan,
                            "%s_CD" % cn: EL[(g, M, c)][0] if g else math.nan,
                            "%s_alpha" % cn: EL[(g, M, c)][2] if g else math.nan, "%s_n_feasible" % cn: n,
                            "%s_dCD_rel_vs_G42" % cn: rg["dCD_rel"] if rg else math.nan,
                            "%s_significantly_lower_than_G42" % cn: bool(rg and rg["significant"] and rg["lower"])})
                if cn == "C0" and g:
                    cand = sorted((EL[(h, M, c)][0], h) for h in cs if (h, M, c) in EL and h != g)
                    if cand:
                        h = cand[0][1]
                        d = cand[0][0] - EL[(g, M, c)][0]
                        row.update({"C0_runner_up": h, "C0_margin_rel_to_runner_up": d / EL[(g, M, c)][0],
                                    "C0_margin_significant": sig(d, EL[(g, M, c)][1] + EL[(h, M, c)][1],
                                                                 EL[(g, M, c)][0])})
                vc_rows.append({"flight_mach": M, "CL_target": c, "usable_CL": c >= CL_USABLE, "constraint": cn,
                                "constraint_definition": PLAN["constraints"][cn], "n_geometries_in_set": len(cs),
                                "n_feasible": n, "winner": g,
                                "winner_design_mach": meta[g]["design_mach"] if g else math.nan,
                                "winner_volume_ratio": meta[g]["volume"] / g42v if g else math.nan,
                                "winner_thickness_ratio": meta[g]["thick"] / g42t if g else math.nan,
                                "CD": EL[(g, M, c)][0] if g else math.nan, "CD_G42": row["G42_CD"],
                                "dCD_rel_vs_G42": rg["dCD_rel"] if rg else math.nan,
                                "significantly_lower_than_G42": bool(rg and rg["significant"] and rg["lower"]),
                                "significantly_higher_than_G42": bool(rg and rg["significant"] and not rg["lower"])})
            by_cl_rows.append(row)
    write_csv(OUT / "optimal_geometry_by_cl.csv", by_cl_rows,
              list(dict.fromkeys(k for r in by_cl_rows for k in r)))
    write_csv(OUT / "volume_constrained_comparison.csv", vc_rows)

    # NEW_GEOMETRY_BETTER_THAN_G42 (C0; twin excluded)
    better = {}
    for cn, cs in cons.items():
        sig_by_m, any_lower = defaultdict(set), False
        for (g, M, c), e in EL.items():
            if g in ("G42", TWIN) or g not in cs or c < CL_USABLE:
                continue
            rg = rel_g42(g, M, c)
            if rg and rg["lower"]:
                any_lower = True
                if rg["significant"]:
                    sig_by_m[M].add(c)
        bm = sorted(M for M, s in sig_by_m.items() if len(s) >= 2)
        better[cn] = {"label": "YES" if bm else ("INCONCLUSIVE" if any_lower else "NO"), "machs_with_>=2_sig_CL": bm,
                      "sig_CL_levels_by_mach": {"%g" % M: sorted(s) for M, s in sorted(sig_by_m.items())}}
    # best gain vs G42 per (M, c) under each constraint
    gains = {cn: {"%g@%g" % k: (rel_g42(g, *k) or {}).get("dCD_rel") for k, g in W[cn].items() if g}
             for cn in cons}

    # Q5 switching by Mach at each CL;  Q6 switching by CL at each Mach
    sw_mach = {}
    for c in USABLE:
        ws = [W["C0"].get((M, c)) for M in FLIGHT_MACH]
        sw_mach["%g" % c] = {"winners": dict(zip(["%g" % M for M in FLIGHT_MACH], ws)),
                             "n_distinct": len({w for w in ws if w}),
                             "design_mach_of_winners": sorted({meta[w]["design_mach"] for w in ws if w})}
    sw_cl = {}
    for M in FLIGHT_MACH:
        ws = [W["C0"].get((M, c)) for c in USABLE]
        sw_cl["%g" % M] = {"winners": dict(zip(["%g" % c for c in USABLE], ws)), "n_distinct": len({w for w in ws if w})}

    # CROSS_MACH_ADVANTAGE: per usable c, fixed_c over 13 Machs vs per-Mach envelope
    cross = {}
    for c in USABLE:
        ms_ref = [M for M in FLIGHT_MACH if ("G42", M, c) in EL]
        feas = [g for g in V2 if all((g, M, c) in EL for M in FLIGHT_MACH)]
        if not feas or not ms_ref:
            cross["%g" % c] = {"evaluable": False, "n_fixed_candidates": len(feas), "G42_feasible_machs": len(ms_ref)}
            continue
        fx = min(feas, key=lambda g: np.mean([EL[(g, M, c)][0] / EL[("G42", M, c)][0] for M in ms_ref]))
        ws = {M: W["C0"][(M, c)] for M in FLIGHT_MACH}
        sm = []
        per = {}
        for M in FLIGHT_MACH:
            e, f = EL[(ws[M], M, c)], EL[(fx, M, c)]
            d = e[0] - f[0]
            s = sig(d, e[1] + f[1], f[0]) and d < 0
            per["%g" % M] = {"winner": ws[M], "dCD_rel_vs_fixed": d / f[0], "significant": s}
            if s:
                sm.append(M)
        switch = len(set(ws.values())) > 1
        cross["%g" % c] = {"evaluable": True, "fixed": fx, "fixed_design_mach": meta[fx]["design_mach"],
                           "fixed_mean_CD_over_G42": float(np.mean([EL[(fx, M, c)][0] / EL[("G42", M, c)][0]
                                                                    for M in ms_ref])),
                           "G42_feasible_machs": len(ms_ref), "n_fixed_candidates": len(feas),
                           "winner_switches_with_mach": switch, "n_distinct_winners": len(set(ws.values())),
                           "significant_machs": sm, "per_mach": per,
                           "label": "YES" if switch and len(sm) >= 2 else ("INCONCLUSIVE" if switch else "NO")}
    ev = [v for v in cross.values() if v["evaluable"]]
    lab_cross = ("YES" if any(v["label"] == "YES" for v in ev) else
                 "INCONCLUSIVE" if any(v["label"] == "INCONCLUSIVE" for v in ev) or not ev else "NO")

    # Q9 BETTER_THAN_BEST_FIXED_GEOMETRY over S = usable (M, c) with G42 feasible
    S = [(M, c) for M in FLIGHT_MACH for c in USABLE if ("G42", M, c) in EL]
    feasS = [g for g in V2 if all((g, M, c) in EL for M, c in S)]
    bf = None
    env_vs_fixed = {}
    if feasS:
        bf = min(feasS, key=lambda g: np.mean([EL[(g, M, c)][0] / EL[("G42", M, c)][0] for M, c in S]))
        for M, c in S:
            w = W["C0"][(M, c)]
            e, f = EL[(w, M, c)], EL[(bf, M, c)]
            d = e[0] - f[0]
            env_vs_fixed[(M, c)] = {"winner": w, "rel": d / f[0], "significant": sig(d, e[1] + f[1], f[0]) and d < 0}
    nsig = sum(v["significant"] for v in env_vs_fixed.values())
    frac = nsig / len(S) if S else 0.0
    lab_fixed = ("INCONCLUSIVE" if not feasS else "YES" if frac >= 0.10 else "INCONCLUSIVE" if nsig > 0 else "NO")
    fixed_info = {"states": len(S), "fixed_candidates_feasible_on_all_states": len(feasS), "best_fixed": bf,
                  "best_fixed_design_mach": meta[bf]["design_mach"] if bf else None,
                  "best_fixed_mean_CD_over_G42": float(np.mean([EL[(bf, M, c)][0] / EL[("G42", M, c)][0]
                                                               for M, c in S])) if bf else None,
                  "G42_rank_note": "G42 itself is CD/CD_G42 = 1 on S by construction",
                  "envelope_significantly_lower_states": nsig, "fraction": frac,
                  "envelope_gain_vs_fixed_median": float(np.median([v["rel"] for v in env_vs_fixed.values()]))
                  if env_vs_fixed else None,
                  "envelope_gain_vs_fixed_min": float(min(v["rel"] for v in env_vs_fixed.values()))
                  if env_vs_fixed else None,
                  "label": lab_fixed}
    by_cl = {}
    for c in USABLE:
        ks = [(M, c) for M in FLIGHT_MACH if (M, c) in env_vs_fixed]
        g42r = [rel_g42(W["C0"][(M, c)], M, c) for M in FLIGHT_MACH if W["C0"].get((M, c))]
        g42r = [r for r in g42r if r]
        by_cl["%g" % c] = {
            "states": len(ks),
            "env_vs_fixed_median_rel": float(np.median([env_vs_fixed[k]["rel"] for k in ks])) if ks else None,
            "env_vs_fixed_min_rel": float(min(env_vs_fixed[k]["rel"] for k in ks)) if ks else None,
            "env_vs_fixed_significant_states": sum(env_vs_fixed[k]["significant"] for k in ks),
            "env_vs_G42_median_rel": float(np.median([r["dCD_rel"] for r in g42r])) if g42r else None,
            "env_vs_G42_min_rel": float(min(r["dCD_rel"] for r in g42r)) if g42r else None,
            "env_vs_G42_significant_lower_machs": sum(1 for r in g42r if r["significant"] and r["lower"])}
    fixed_info["by_CL"] = by_cl
    # same comparison with friction what-if (sensitivity only)
    fr = {}
    for cf in CF_WHATIF:
        def cdf(g, M, c):
            return EL[(g, M, c)][0] + cf * meta[g]["Sw"] / REFERENCE["area_m2"]
        if not feasS:
            continue
        bff = min(feasS, key=lambda g: np.mean([cdf(g, M, c) / cdf("G42", M, c) for M, c in S]))
        gains_f = []
        for M, c in S:
            cand = [g for g in V2 if (g, M, c) in EL]
            w = min(cand, key=lambda g: cdf(g, M, c))
            gains_f.append(cdf(w, M, c) / cdf(bff, M, c) - 1)
        g42g = []
        for M, c in S:
            cand = [g for g in V2 if (g, M, c) in EL and g != TWIN]
            w = min(cand, key=lambda g: cdf(g, M, c))
            g42g.append(cdf(w, M, c) / cdf("G42", M, c) - 1)
        fr["Cf%g" % cf] = {"best_fixed": bff, "envelope_vs_fixed_median_rel": float(np.median(gains_f)),
                           "states_envelope_>1pct_below_fixed": int(sum(1 for x in gains_f if x < -REL_FLOOR)),
                           "envelope_vs_G42_median_rel": float(np.median(g42g)),
                           "states_envelope_>1pct_below_G42": int(sum(1 for x in g42g if x < -REL_FLOOR))}

    # ---------------------------------------------------- Q8 reachability in the morph graph
    inv = {r["candidate_id"]: r["geometry_id"] for r in read_csv(GEO_DB / "valid_geometry_inventory.csv")}
    adj = defaultdict(set)
    chains = defaultdict(list)
    for r in read_csv(GEO_DB / "morph_path_validation.csv"):
        if r["classification"] == "VALID" and r["candidate_id"] in inv:
            chains[r["edge_id"]].append((float(r["rho"]), inv[r["candidate_id"]], r["from_geometry_id"],
                                         r["to_geometry_id"]))
    for e, lst in chains.items():
        lst.sort()
        seq = [lst[0][2]] + [x[1] for x in lst] + [lst[0][3]]
        for a, b in zip(seq, seq[1:]):
            adj[a].add(b)
            adj[b].add(a)

    def hops(a, b):
        if a == b:
            return 0
        seen, q = {a: 0}, deque([a])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen[y] = seen[x] + 1
                    if y == b:
                        return seen[y]
                    q.append(y)
        return None

    comp = {}
    for g in V2:
        if g in comp:
            continue
        q, cid = deque([g]), len(set(comp.values()))
        comp[g] = cid
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in comp:
                    comp[y] = cid
                    q.append(y)
    groups = defaultdict(list)
    for g in V2:
        groups[comp[g]].append(g)
    reach = {"components": {str(k): {"size": len(v), "design_mach": sorted({meta[g]["design_mach"] for g in v})}
                            for k, v in groups.items()},
             "G42_component": comp.get(TWIN), "G42_note": "G42 is represented by its hash twin V2_MD15_001",
             "per_CL": {}}
    for c in USABLE:
        ws = [W["C0"].get((M, c)) for M in FLIGHT_MACH]
        cs = {comp[w] for w in ws if w}
        path = [hops(a, b) for a, b in zip(ws, ws[1:]) if a and b]
        comp_env = {}
        for k, gs in groups.items():
            vals = []
            for M in FLIGHT_MACH:
                cand = [EL[(g, M, c)][0] for g in gs if (g, M, c) in EL]
                if not cand or ("G42", M, c) not in EL:
                    break
                vals.append(min(cand) / EL[("G42", M, c)][0])
            if len(vals) == len(FLIGHT_MACH):
                comp_env[k] = float(np.mean(vals))
        bk = min(comp_env, key=comp_env.get) if comp_env else None
        full = [EL[(W["C0"][(M, c)], M, c)][0] / EL[("G42", M, c)][0] for M in FLIGHT_MACH
                if W["C0"].get((M, c)) and ("G42", M, c) in EL]
        reach["per_CL"]["%g" % c] = {
            "winner_components": sorted(cs), "all_winners_in_one_component": len(cs) == 1,
            "hops_between_consecutive_mach_winners": path,
            "best_single_component": bk,
            "best_single_component_design_mach": reach["components"][str(bk)]["design_mach"] if bk is not None else None,
            "best_single_component_mean_CD_over_G42": comp_env.get(bk),
            "unrestricted_envelope_mean_CD_over_G42": float(np.mean(full)) if full else None,
            "G42_component_mean_CD_over_G42": comp_env.get(comp.get(TWIN)),
            "component_mean_CD_over_G42": {str(k): v for k, v in sorted(comp_env.items())}}

    # ---------------------------------------------------- Q2 design-Mach group effect
    group = {}
    for M in FLIGHT_MACH:
        for c in (0.03, 0.05, 0.08, 0.10):
            for md in MDS:
                vals = [rel_g42(g, M, c) for g in V2 if meta[g]["design_mach"] == md and meta[g]["kind"] == "design"]
                vals = [v["dCD_rel"] for v in vals if v]
                if vals:
                    group["MD%g@M%g@CL%g" % (md, M, c)] = {"best": float(min(vals)), "median": float(np.median(vals)),
                                                         "n": len(vals)}
    group_winner = {}
    for M in FLIGHT_MACH:
        for c in (0.03, 0.05, 0.08, 0.10):
            cand = [(group[k]["best"], md) for md in MDS for k in ["MD%g@M%g@CL%g" % (md, M, c)] if k in group]
            if cand:
                group_winner["M%g@CL%g" % (M, c)] = min(cand)[1]

    labels = {"NEW_GEOMETRY_BETTER_THAN_G42": better["C0"]["label"], "CROSS_MACH_ADVANTAGE": lab_cross,
              "BETTER_THAN_BEST_FIXED_GEOMETRY": lab_fixed}
    out = {"labels": labels, "constraint_sets": {k: len(v) for k, v in cons.items()},
           "curves_complete": len(curves), "curves_nonmonotonic": sum(1 for c in curves.values() if not c.mono),
           "better_than_G42": better, "best_gain_vs_G42": gains, "switch_by_mach": sw_mach, "switch_by_cl": sw_cl,
           "cross_mach": cross, "best_fixed": fixed_info,
           "envelope_vs_fixed": {"%g@%g" % k: v for k, v in env_vs_fixed.items()},
           "friction_whatif": fr, "reachability": reach, "design_mach_group": group,
           "design_mach_group_winner": group_winner,
           "opt_by_mach": opt_mach,
           "interp_unc_rel_median": float(np.median([e[1] / e[0] for e in EL.values()])) if EL else None,
           "interp_unc_rel_p95": float(np.percentile([e[1] / e[0] for e in EL.values()], 95)) if EL else None}
    dump(OUT / "provenance/analysis.json", out)
    figures(curves, LD, EL, W, meta, opt_mach, env_vs_fixed, bf, rel_g42, V2)
    print(json.dumps(labels), json.dumps(fixed_info))


def figures(curves, LD, EL, W, meta, opt_mach, env_vs_fixed, bf, rel_g42, V2):
    Ms = FLIGHT_MACH
    # 1 L/D max vs Mach
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for md in MDS:
        y = [r.get("best_MD%g_LD_max_usable" % md, math.nan) for r in opt_mach]
        ax[0].plot(Ms, y, "-o", ms=3, color=MD_COLORS[md], label="best of Ma_design=%g" % md)
    ax[0].plot(Ms, [r["G42_LD_max_usable"] for r in opt_mach], "k--", lw=2.5, label="G42")
    ax[0].set(xlabel="flight Mach", ylabel="max L/D (CL>=%.2f, pressure only)" % CL_USABLE,
              title="Q1/Q2: best usable L/D per design-Mach group")
    ax[0].legend(fontsize=8)
    for cn, st in (("C0", "-o"), ("C1", "-s"), ("C2", "-^")):
        ax[1].plot(Ms, [100 * r.get("%s_gain_vs_G42" % cn, math.nan) for r in opt_mach], st, label=cn)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="flight Mach", ylabel="best usable L/D gain vs G42 (%)",
              title="Q7: L/D gain by constraint (C1 vol>=G42, C2 vol&thick>=G42)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ldmax_vs_mach.png", dpi=130)
    plt.close(fig)
    # 2 envelope dCD heatmap per constraint
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, cn in zip(axs, ("C0", "C1", "C2")):
        Z = np.full((len(CL_LEVELS), len(Ms)), np.nan)
        for i, c in enumerate(CL_LEVELS):
            for j, M in enumerate(Ms):
                g = W[cn].get((M, c))
                r = rel_g42(g, M, c) if g else None
                if r:
                    Z[i, j] = 100 * r["dCD_rel"]
                    if r["significant"]:
                        ax.text(j, i, "*", ha="center", va="center", fontsize=8)
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30)
        ax.set_xticks(range(len(Ms)))
        ax.set_xticklabels(["%g" % M for M in Ms], fontsize=7)
        ax.set_yticks(range(len(CL_LEVELS)))
        ax.set_yticklabels(["%g" % c for c in CL_LEVELS])
        ax.set(xlabel="flight Mach", ylabel="CL", title="%s envelope CD vs G42 (%%), * significant" % cn)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_equal_lift_envelope_vs_g42.png", dpi=130)
    plt.close(fig)
    # 3 winner map
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, c in enumerate(CL_LEVELS):
        for j, M in enumerate(Ms):
            g = W["C0"].get((M, c))
            if g:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=MD_COLORS[meta[g]["design_mach"]], alpha=0.55))
                ax.text(j, i, g.replace("V2M_", "M:").replace("V2_", ""), ha="center", va="center", fontsize=5.5)
    ax.set_xlim(-0.5, len(Ms) - 0.5)
    ax.set_ylim(-0.5, len(CL_LEVELS) - 0.5)
    ax.set_xticks(range(len(Ms)))
    ax.set_xticklabels(["%g" % M for M in Ms])
    ax.set_yticks(range(len(CL_LEVELS)))
    ax.set_yticklabels(["%g" % c for c in CL_LEVELS])
    for md, col in MD_COLORS.items():
        ax.plot([], [], "s", color=col, alpha=0.55, label="Ma_design=%g" % md)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set(xlabel="flight Mach", ylabel="CL", title="Q5/Q6: lowest-CD geometry per (Mach, CL), C0")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_winner_map.png", dpi=140)
    plt.close(fig)
    # 4 envelope vs best fixed
    if env_vs_fixed:
        fig, ax = plt.subplots(figsize=(12, 5))
        Z = np.full((len(CL_LEVELS), len(Ms)), np.nan)
        for (M, c), v in env_vs_fixed.items():
            Z[CL_LEVELS.index(c), Ms.index(M)] = 100 * v["rel"]
            if v["significant"]:
                ax.text(Ms.index(M), CL_LEVELS.index(c), "*", ha="center", va="center")
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(Ms)))
        ax.set_xticklabels(["%g" % M for M in Ms])
        ax.set_yticks(range(len(CL_LEVELS)))
        ax.set_yticklabels(["%g" % c for c in CL_LEVELS])
        ax.set(xlabel="flight Mach", ylabel="CL",
               title="Q9: per-state envelope CD vs best fixed %s (%%), * significant" % bf)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIG / "fig4_envelope_vs_best_fixed.png", dpi=130)
        plt.close(fig)
    # 5 polars
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, M in zip(axs, (4.0, 6.0, 8.0, 9.25)):
        show = ["G42"] + sorted({W["C0"].get((M, c)) for c in (0.03, 0.05, 0.10)} - {None})
        for g in show:
            cv = curves.get((g, M))
            if cv is None:
                continue
            ax.plot(cv.CD, cv.CL, "k--" if g == "G42" else "-o", ms=3, lw=2 if g == "G42" else 1, label=g)
        ax.set(xlabel="CD (pressure)", ylabel="CL", title="polars at M=%g" % M)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_polars.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
