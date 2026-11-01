"""Stage 5: nearby / cross-Mach / equal-lift tables from the real wedge-model evaluations of the selected unique geometries."""
from __future__ import annotations
import collections, json, math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
from common import OUT, cfg, read_csv, write_csv, MODEL_VERSION

c = cfg(); sel = {r["geometry_id"]: r for r in read_csv(OUT / "screening/selected_unique_geometries.csv")}
pool = {r["geometry_id"]: r for r in read_csv(OUT / "screening/candidate_pool.csv")}
res = read_csv(OUT / "work/selected_results.csv")
R = {}
for r in res:
    CL, CD = float(r["CL"]) if r["CL"] != "" else float("nan"), float(r["CD_pressure"]) if r["CD_pressure"] != "" else float("nan")
    ok = r["model_valid"] in ("1", "true", "True") and math.isfinite(CL) and math.isfinite(CD) and CD > 0
    R[(r["geometry_id"], round(float(r["M_flight"]), 6), round(float(r["alpha_deg"]), 6))] = (CL, CD, ok, r["invalid_reasons"] or r["error"])
def get(g, M, a): return R[(g, round(M, 6), round(a, 6))]
def row_vals(g, M, a):
    CL, CD, ok, why = get(g, M, a); return CL, CD, (CL / CD if ok else float("nan")), ("OK" if ok else "INVALID:" + (why or "nonfinite"))

# ---- nearby (center + 4 neighbours for every condition in which the geometry was selected) ----
nb = c["nearby"]; near = []
for gid, g in sel.items():
    for cid in g["conditions"].split(";"):
        Mc = float(cid[1:5]); ac = float(cid.split("_A")[1]); c0 = row_vals(gid, Mc, ac)
        for name, dM, da in (("CENTER", 0, 0), ("M-", -nb["dM"], 0), ("M+", nb["dM"], 0), ("ALPHA-", 0, -nb["dalpha_deg"]), ("ALPHA+", 0, nb["dalpha_deg"])):
            v = row_vals(gid, Mc + dM, ac + da)
            near.append({"geometry_id": gid, "geometry_hash": g["geometry_hash"], "Ma_design": g["Ma_design"], "condition_id": cid, "offset": name,
                         "M_center": Mc, "alpha_center": ac, "M_query": Mc + dM, "alpha_query": ac + da, "CL": v[0], "CD_pressure": v[1], "L_over_D": v[2],
                         "delta_CL": v[0] - c0[0], "delta_CD": v[1] - c0[1], "delta_L_over_D": v[2] - c0[2], "aero_status": v[3], "aero_model": MODEL_VERSION,
                         "note": "delta = change vs the centre flight condition (includes the effect of the flight Mach/alpha change itself, not a design-Mach mismatch loss)"})
write_csv(OUT / "offdesign/nearby_performance.csv", near)

# ---- common cross-Mach table ----
cm = []
for gid, g in sel.items():
    for M in c["cross_mach"]["M_grid"]:
        for a in c["cross_mach"]["alpha_deg_grid"]:
            v = row_vals(gid, M, a)
            cm.append({"geometry_id": gid, "Ma_design": g["Ma_design"], "M_flight": M, "alpha_flight": a, "CL": v[0], "CD_pressure": v[1], "L_over_D": v[2],
                       "aero_model": MODEL_VERSION, "geometry_hash": g["geometry_hash"], "aero_status": v[3]})
write_csv(OUT / "offdesign/cross_mach_performance.csv", cm)

# ---- equal-lift ----
Ad = np.array(c["equal_lift"]["alpha_dense_deg"], float); eq = []
for gid, g in sel.items():
    for M in c["equal_lift"]["M_grid"]:
        v = [row_vals(gid, M, a) for a in Ad]
        CL = np.array([x[0] for x in v]); CD = np.array([x[1] for x in v]); ok = all(x[3] == "OK" for x in v)
        mono = bool(ok and np.all(np.diff(CL) > 0))
        for tgt in c["equal_lift"]["CL_targets"]:
            rec = {"geometry_id": gid, "geometry_hash": g["geometry_hash"], "Ma_design": g["Ma_design"], "M_flight": M, "CL_target": tgt}
            if not mono: rec.update({"status": "NON_MONOTONIC_OR_INVALID_CL_ALPHA", "alpha_at_target": "", "CD_at_target": "", "L_over_D_at_target": ""})
            elif not (CL[0] <= tgt <= CL[-1]): rec.update({"status": "TARGET_OUTSIDE_COMPUTED_ALPHA_RANGE", "alpha_at_target": "", "CD_at_target": "", "L_over_D_at_target": ""})
            else:
                a_t = float(PchipInterpolator(CL, Ad)(tgt)); cd_t = float(PchipInterpolator(Ad, CD)(a_t))
                rec.update({"status": "OK", "alpha_at_target": a_t, "CD_at_target": cd_t, "L_over_D_at_target": tgt / cd_t})
            eq.append(rec)
# rank by CD at each (M, target)
grp = collections.defaultdict(list)
for r in eq:
    if r["status"] == "OK": grp[(r["M_flight"], r["CL_target"])].append(r)
for k, L in grp.items():
    for i, r in enumerate(sorted(L, key=lambda r: (r["CD_at_target"], r["geometry_id"])), 1): r["rank_equal_lift_CD"] = i
write_csv(OUT / "offdesign/equal_lift_comparison.csv", eq, fields=["geometry_id", "geometry_hash", "Ma_design", "M_flight", "CL_target", "status", "alpha_at_target", "CD_at_target", "L_over_D_at_target", "rank_equal_lift_CD"])

# ---- consistency of fixed-alpha L/D ranking vs equal-lift CD ranking among the selected set ----
cons = []
for (M, tgt), L in sorted(grp.items()):
    ids = [r["geometry_id"] for r in L]
    for a in (4, 6, 8, 10):
        ld = [row_vals(g, M, a)[2] for g in ids]
        if len(ids) > 3 and all(math.isfinite(x) for x in ld):
            rho = spearmanr([r["CD_at_target"] for r in L], [-x for x in ld]).statistic
            top_eq = min(L, key=lambda r: r["CD_at_target"])["geometry_id"]; top_ld = ids[int(np.argmax(ld))]
            cons.append({"M_flight": M, "CL_target": tgt, "alpha_for_fixed_alpha_LD": a, "n": len(ids), "spearman": rho, "same_top1": top_eq == top_ld, "top1_equal_lift": top_eq, "top1_fixed_alpha_LD": top_ld})
write_csv(OUT / "offdesign/ranking_consistency.csv", cons)
sp = [r["spearman"] for r in cons]
out = {"nearby_rows": len(near), "cross_rows": len(cm), "equal_lift_rows": len(eq), "equal_lift_ok": sum(r["status"] == "OK" for r in eq),
       "invalid_cross": sum(r["aero_status"] != "OK" for r in cm), "invalid_nearby": sum(r["aero_status"] != "OK" for r in near),
       "consistency_cases": len(cons), "spearman_median": float(np.median(sp)) if sp else None, "spearman_min": float(np.min(sp)) if sp else None,
       "same_top1_fraction": float(np.mean([r["same_top1"] for r in cons])) if cons else None}
(OUT / "work/offdesign_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8"); print(json.dumps(out, indent=1))
