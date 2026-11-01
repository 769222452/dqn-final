"""P2: low-order inviscid model sensitivity on the mission case matrix and at the design points.

Each model (A Teacher tangent cone, B tangent wedge, C strip shock-expansion, N modified Newtonian) is
swept over alpha = -2..12 deg (0.5 deg) at M = 5.5 / 6.5 / 7.5 on the identical panels; iso-CL alpha is
found per model by PCHIP on the sweep followed by an exact re-evaluation and one secant refinement.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

import aeromodels as AM  # noqa: E402
import common as C  # noqa: E402
from p0_geometry import MATRIX_CL, MATRIX_MACH  # noqa: E402

MODELS = ("A_tangent_cone_teacher", "B_tangent_wedge", "C_strip_shock_expansion", "N_modified_newtonian")
ALPHAS = np.arange(-2.0, 12.0001, 0.5)


def evaluate(P, M, a, model=None):
    """All models, or one model (the Teacher replica is only called when needed: it dominates run time)."""
    teach = model in (None, MODELS[0])
    res, _ = AM.all_models(P, M, a, C.SREF, C.teacher if teach else None)
    return res if model is None else res[model]


def sweep(P, M, lab):
    """Models B/C/N on the 0.5 deg grid; model A from the stored Teacher labels (integer alpha, PASS rows),
    which the replica reproduces to ~1e-10 (results/v6_matlab_deep_audit_v1/analysis/replica_validation.json)."""
    tab = {m: (ALPHAS, np.zeros((len(ALPHAS), 3))) for m in MODELS[1:]}
    for k, a in enumerate(ALPHAS):
        r = AM.all_models(P, M, float(a), C.SREF, None)[0]
        for m in MODELS[1:]:
            tab[m][1][k] = r[m]
    al = np.array(sorted(a for (mm, a) in lab if mm == M))
    tab[MODELS[0]] = (al, np.array([[lab[(M, a)][0], lab[(M, a)][1], 0.0] for a in al]))
    return tab


def solve_alpha(P, M, tab, model, cl):
    al, arr = tab[model]
    c = arr[:, 0]
    if not (c.min() <= cl <= c.max()):
        return None
    good = np.r_[True, np.diff(c) > 0]
    f = PchipInterpolator(c[good], al[good])
    a0 = float(f(cl))
    r0 = evaluate(P, M, a0, model)
    da = 0.05
    r1 = evaluate(P, M, a0 + da, model)
    slope = (r1[0] - r0[0]) / da
    a1 = a0 + (cl - r0[0]) / slope
    r = evaluate(P, M, a1, model)
    return a1, r


def main():
    rows, sweeps = [], []
    design = {}
    for gid in C.ORDER:
        S = C.load_mat(C.GEOMS[gid]["mat"])
        P = C.R.panels(S)
        Md = C.GEOMS[gid]["Ma_design"]
        rd = evaluate(P, Md, C.ALPHA_DESIGN_BODY)
        design[gid] = rd
        for m in MODELS:
            rows.append({"case_id": f"{C.SHORT[gid]}_DESIGN", "geometry_id": gid, "kind": "DESIGN_POINT_FIXED_ALPHA",
                         "Mach": Md, "CL_target": None, "model": m, "alpha_deg": C.ALPHA_DESIGN_BODY,
                         "CL": rd[m][0], "CD": rd[m][1], "LD": rd[m][0] / rd[m][1] if rd[m][1] else None,
                         "detached_area_fraction": rd[m][2]})
        lab = C.labels(gid)
        for M in MATRIX_MACH:
            tab = sweep(P, M, lab)
            ta = {float(a): k for k, a in enumerate(tab[MODELS[0]][0])}
            for k, a in enumerate(ALPHAS):
                rec = {"geometry_id": gid, "Mach": M, "alpha_deg": float(a)}
                for m in MODELS[1:]:
                    rec.update({f"CL_{m}": tab[m][1][k, 0], f"CD_{m}": tab[m][1][k, 1], f"detached_{m}": tab[m][1][k, 2]})
                if float(a) in ta:
                    j = ta[float(a)]
                    rec.update({f"CL_{MODELS[0]}": tab[MODELS[0]][1][j, 0], f"CD_{MODELS[0]}": tab[MODELS[0]][1][j, 1]})
                sweeps.append(rec)
            for cl in MATRIX_CL:
                for m in MODELS:
                    sol = solve_alpha(P, M, tab, m, cl)
                    rec = {"case_id": f"{C.SHORT[gid]}_M{M}_CL{cl:.3f}", "geometry_id": gid, "kind": "MISSION_ISO_CL",
                           "Mach": M, "CL_target": cl, "model": m}
                    if sol is None:
                        rec["reachable"] = False
                    else:
                        a, r = sol
                        rec.update({"reachable": True, "alpha_deg": a, "CL": r[0], "CD": r[1], "LD": r[0] / r[1],
                                    "detached_area_fraction": r[2]})
                    rows.append(rec)
            print(gid, M, "done", flush=True)
    # ratios to Teacher and geometry ranking per (M, CL) and model
    by = {}
    for r in rows:
        by.setdefault((r["case_id"], r["model"]), r)
    for r in rows:
        t = by.get((r["case_id"], MODELS[0]))
        if t and r.get("CD") and t.get("CD"):
            r["CD_over_teacher"] = r["CD"] / t["CD"]
            r["alpha_minus_teacher_deg"] = r["alpha_deg"] - t["alpha_deg"]
        g = by.get((r["case_id"].replace(C.SHORT[r["geometry_id"]], "G42", 1), r["model"]))
        if g and r.get("CD") and g.get("CD") and r["kind"] == "MISSION_ISO_CL":
            r["CD_over_G42_same_model"] = r["CD"] / g["CD"]
    rank_rows = []
    for M in MATRIX_MACH:
        for cl in MATRIX_CL:
            ranks = {}
            for m in MODELS:
                vals = [(gid, by.get((f"{C.SHORT[gid]}_M{M}_CL{cl:.3f}", m), {}).get("CD")) for gid in C.ORDER]
                if any(v is None for _, v in vals):
                    continue
                order = [g for g, _ in sorted(vals, key=lambda x: x[1])]
                ranks[m] = [order.index(g) + 1 for g in C.ORDER]
                rank_rows.append({"Mach": M, "CL_target": cl, "model": m,
                                  "ranking_best_to_worst": [C.SHORT[g] for g in order],
                                  **{f"CD_{C.SHORT[g]}": v for g, v in vals}})
            for m in MODELS[1:]:
                if m in ranks and MODELS[0] in ranks:
                    rank_rows.append({"Mach": M, "CL_target": cl, "model": f"agreement {m} vs Teacher",
                                      "spearman": spearmanr(ranks[MODELS[0]], ranks[m]).statistic,
                                      "kendall": kendalltau(ranks[MODELS[0]], ranks[m]).statistic,
                                      "same_best": ranks[MODELS[0]].index(1) == ranks[m].index(1),
                                      "identical_order": ranks[MODELS[0]] == ranks[m]})
    C.write_csv(C.OUT / "LOW_ORDER_MODEL_COMPARISON.csv", rows,
                ["case_id", "geometry_id", "kind", "Mach", "CL_target", "model", "reachable", "alpha_deg", "CL", "CD",
                 "LD", "CD_over_teacher", "alpha_minus_teacher_deg", "CD_over_G42_same_model", "detached_area_fraction"])
    C.write_csv(C.WORK / "p2_ranking.csv", rank_rows)
    C.write_csv(C.WORK / "p2_alpha_sweeps.csv", sweeps)
    # figures: CL-alpha and drag polar per model at M = 6.5
    fig, ax = plt.subplots(2, len(C.ORDER), figsize=(4.2 * len(C.ORDER), 7.5))
    for gi, gid in enumerate(C.ORDER):
        for m, st in zip(MODELS, ("r-o", "b:", "g-.", "k--")):
            sw = [s for s in sweeps if s["geometry_id"] == gid and s["Mach"] == 6.5 and f"CL_{m}" in s]
            a = [s["alpha_deg"] for s in sw]
            ax[0, gi].plot(a, [s[f"CL_{m}"] for s in sw], st, ms=3, label=m.split("_", 1)[1])
            ax[1, gi].plot([s[f"CD_{m}"] for s in sw], [s[f"CL_{m}"] for s in sw], st)
        ax[0, gi].set_title(f"{C.SHORT[gid]}  M6.5  CL(alpha)"); ax[0, gi].set_xlabel("alpha [deg]")
        ax[1, gi].set_title("CL vs CD (inviscid)"); ax[1, gi].set_xlabel("CD")
        for y in MATRIX_CL:
            ax[0, gi].axhline(y, color="0.8", lw=0.6)
        ax[0, gi].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIG / "V2_model_polars_M6.5.png", dpi=110)
    print("written")


if __name__ == "__main__":
    main()
