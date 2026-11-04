"""Stage 3: join wedge results with pool metadata, apply the frozen gates, Top5 per condition, baseline and diversity controls."""
from __future__ import annotations
import collections, json, math
import numpy as np
from common import OUT, cfg, read_csv, write_csv, MODEL_VERSION

c = cfg(); gates = c["ranking"]["gates"]; CLMIN = gates["lift_floor_CL_min"]
pool = {r["geometry_id"]: r for r in read_csv(OUT / "screening/candidate_pool.csv")}
res = read_csv(OUT / "work/pool_results.csv")
assert len(res) == len(pool) * c["flight_grid"]["n_conditions"], (len(res), len(pool))


def cond_id(M, a): return f"M{M:.2f}_A{a:05.2f}"


def fnum(x):
    try: return float(x)
    except Exception: return float("nan")


rows = []
for r in res:
    p = pool[r["geometry_id"]]; M = float(r["M_flight"]); a = float(r["alpha_deg"])
    CL, CD = fnum(r["CL"]), fnum(r["CD_pressure"]); ok = r["model_valid"] in ("1", "true", "True") and math.isfinite(CL) and math.isfinite(CD) and CD > 0
    status = "OK" if ok else ("INVALID:" + (r["invalid_reasons"] or r["error"] or "nonfinite_or_nonpositive_CD"))
    vol_pass = float(p["volume_ratio_G42"]) >= 0.85 and float(p["thickness_ratio_G42"]) >= 0.5
    flags = []
    if ok and CL < CLMIN: flags.append("LOW_LIFT")
    if ok and CL <= 0: flags.append("NONPOSITIVE_LIFT")
    if not vol_pass: flags.append("FAILS_EXPLORATORY_VOLUME_SCREEN")
    rows.append({"condition_id": cond_id(M, a), "M_flight": M, "alpha_flight_deg": a, "geometry_id": r["geometry_id"],
                 "geometry_hash": p["geometry_hash"], "Ma_design": p["Ma_design"], "CL": CL, "CD_pressure": CD,
                 "L_over_D": CL / CD if ok else float("nan"), "aero_status": status, "model_version": MODEL_VERSION,
                 "detached_area_fraction": r["detached_area_fraction"], "geometry_validity": p["geometry_validity"],
                 "volume_ratio_G42": p["volume_ratio_G42"], "thickness_ratio_G42": p["thickness_ratio_G42"],
                 "flags": ";".join(flags),
                 "eligible": bool(ok and CL >= CLMIN and vol_pass and p["geometry_validity"] == "VALID")})
write_csv(OUT / "screening/all_candidate_scores.csv", rows)

by_cond = collections.defaultdict(list)
for r in rows: by_cond[r["condition_id"]].append(r)
FEATS = ["Ma_design", "volume_ratio_G42", "thickness_ratio_G42", "a1", "a2", "b1", "b2", "d1", "a_x"]
allp = list(pool.values()); sd = {k: (np.std([float(p[k]) for p in allp]) or 1.0) for k in FEATS}


def vec(g): return np.array([float(pool[g][k]) / sd[k] for k in FEATS])


top_rows, ctrl_rows = [], []
summary_cond = []
for cid in sorted(by_cond):
    L = by_cond[cid]; el = sorted([r for r in L if r["eligible"]], key=lambda r: (-r["L_over_D"], r["CD_pressure"], r["geometry_id"]))
    top = el[:5]
    for k, r in enumerate(top, 1):
        p = pool[r["geometry_id"]]
        top_rows.append({"condition_id": cid, "M_flight": r["M_flight"], "alpha_flight_deg": r["alpha_flight_deg"], "rank": k, "geometry_id": r["geometry_id"],
                         "geometry_hash": r["geometry_hash"], "Ma_design": r["Ma_design"], "AOA_design_prog": p["AOA_design_prog"],
                         "a1": p["a1"], "a2": p["a2"], "b1": p["b1"], "b2": p["b2"], "c1": p["c1"], "c2": p["c2"], "d1": p["d1"], "a_x": p["a_x"],
                         "CL": r["CL"], "CD_pressure": r["CD_pressure"], "L_over_D": r["L_over_D"], "volume": p["closed_volume_full"],
                         "volume_ratio_G42": p["volume_ratio_G42"], "geometry_validity": r["geometry_validity"], "ranking_model": "TANGENT_WEDGE_SHOCK_EXPANSION",
                         "n_eligible_in_condition": len(el)})
    topids = {r["geometry_id"] for r in top}
    g42 = next(r for r in L if r["geometry_id"] == "G42")
    base_role = "BASELINE_ALREADY_IN_TOP5" if "G42" in topids else "BASELINE"
    rank_in = next((i + 1 for i, r in enumerate(el) if r["geometry_id"] == "G42"), "")
    # diversity: max-min normalised distance to the Top5 among eligible, non-Top5, non-G42 geometries
    div = None
    cand = [r for r in el if r["geometry_id"] not in topids and r["geometry_id"] != "G42"]
    if top and cand:
        T = np.array([vec(r["geometry_id"]) for r in top])
        best = max(cand, key=lambda r: (float(np.min(np.linalg.norm(T - vec(r["geometry_id"]), axis=1))), r["geometry_id"]))
        div = (best, float(np.min(np.linalg.norm(T - vec(best["geometry_id"]), axis=1))), next(i + 1 for i, r in enumerate(el) if r is best))
    for role, r, extra in (("BASELINE", g42, {"role_note": base_role, "rank_among_eligible": rank_in}),
                           ("DIVERSITY", div[0] if div else None, {"role_note": "max-min distance to Top5", "min_distance_to_top5": div[1] if div else "", "rank_among_eligible": div[2] if div else ""})):
        if r is None: continue
        if role == "BASELINE" and "G42" in topids: continue
        p = pool[r["geometry_id"]]
        ctrl_rows.append({"condition_id": cid, "M_flight": r["M_flight"], "alpha_flight_deg": r["alpha_flight_deg"], "role": role, "geometry_id": r["geometry_id"],
                          "geometry_hash": r["geometry_hash"], "Ma_design": r["Ma_design"], "CL": r["CL"], "CD_pressure": r["CD_pressure"], "L_over_D": r["L_over_D"],
                          "aero_status": r["aero_status"], "flags": r["flags"], "eligible": r["eligible"], **extra})
    summary_cond.append({"condition_id": cid, "n_eligible": len(el), "n_top": len(top), "n_invalid": sum(r["aero_status"] != "OK" for r in L),
                         "n_low_lift": sum("LOW_LIFT" in r["flags"] for r in L)})
write_csv(OUT / "screening/top5_by_condition.csv", top_rows)
write_csv(OUT / "screening/baseline_diversity.csv", ctrl_rows)
write_csv(OUT / "work/condition_summary.csv", summary_cond)

# selected unique geometries (Top5 + controls), with roles and associated conditions
sel = collections.OrderedDict()
def add(g, role, cid):
    d = sel.setdefault(g, {"geometry_id": g, "geometry_hash": pool[g]["geometry_hash"], "Ma_design": pool[g]["Ma_design"], "mat_path": pool[g]["mat_path"],
                           "roles": set(), "conditions": [], "n_top5": 0, "n_baseline": 0, "n_diversity": 0})
    d["roles"].add(role); d["conditions"].append((cid, role))
    d["n_top5"] += role.startswith("TOP"); d["n_baseline"] += role == "BASELINE"; d["n_diversity"] += role == "DIVERSITY"
for r in top_rows: add(r["geometry_id"], f"TOP{r['rank']}", r["condition_id"])
for r in ctrl_rows: add(r["geometry_id"], r["role"], r["condition_id"])
out = [{"geometry_id": d["geometry_id"], "geometry_hash": d["geometry_hash"], "Ma_design": d["Ma_design"], "mat_path": d["mat_path"],
        "roles": "|".join(sorted(d["roles"])), "n_top5_conditions": d["n_top5"], "n_baseline_conditions": d["n_baseline"], "n_diversity_conditions": d["n_diversity"],
        "n_conditions": len({c_ for c_, _ in d["conditions"]}), "conditions": ";".join(sorted({c_ for c_, _ in d["conditions"]}))} for d in sel.values()]
out.sort(key=lambda r: (-r["n_top5_conditions"], r["geometry_id"]))
write_csv(OUT / "screening/selected_unique_geometries.csv", out)
print(json.dumps({"conditions": len(by_cond), "top5_rows": len(top_rows), "control_rows": len(ctrl_rows), "selected_unique": len(out),
                  "conditions_with_lt5": [s["condition_id"] for s in summary_cond if s["n_top"] < 5],
                  "invalid_evals": sum(r["aero_status"] != "OK" for r in rows), "eligible_evals": sum(r["eligible"] for r in rows)}, indent=1))
