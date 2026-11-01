"""Stage 9: numbers quoted in REPORT_ZH.md (all computed from the CSV tables)."""
import collections, json
import numpy as np
from common import OUT, read_csv

top = read_csv(OUT / "screening/top5_by_condition.csv"); sel = read_csv(OUT / "screening/selected_unique_geometries.csv")
near = read_csv(OUT / "offdesign/nearby_performance.csv"); cross = read_csv(OUT / "offdesign/cross_mach_performance.csv")
eq = read_csv(OUT / "offdesign/equal_lift_comparison.csv"); cons = read_csv(OUT / "offdesign/ranking_consistency.csv")
pool = read_csv(OUT / "screening/candidate_pool.csv"); ctrl = read_csv(OUT / "screening/baseline_diversity.csv")
f = {}
f["pool_unique_valid"] = len(pool); f["selected_unique"] = len(sel)
f["top5_design_mach_bins"] = dict(sorted(collections.Counter(int(round(float(r["Ma_design"]))) for r in top).items()))
f["top5_ge12_fraction"] = float(np.mean([float(r["Ma_design"]) >= 12 for r in top])); f["top1_ge12_fraction"] = float(np.mean([float(r["Ma_design"]) >= 12 for r in top if r["rank"] == "1"]))
f["top5_design_mach_lt8_count"] = sum(float(r["Ma_design"]) < 8 for r in top)
f["top1_unique"] = len({r["geometry_id"] for r in top if r["rank"] == "1"}); f["top1_ids"] = collections.Counter(r["geometry_id"] for r in top if r["rank"] == "1").most_common()
f["top5_count_by_geometry"] = collections.Counter(r["geometry_id"] for r in top).most_common(12)
f["g42_in_top5_conditions"] = sorted({r["condition_id"] for r in top if r["geometry_id"] == "G42"})
g42 = {(r["M_flight"], r["alpha_flight_deg"]): r for r in ctrl if r["geometry_id"] == "G42" and r["role"] == "BASELINE"}
f["G42_rank_among_eligible_range"] = [min(int(r["rank_among_eligible"]) for r in ctrl if r["role"] == "BASELINE" and r["rank_among_eligible"]), max(int(r["rank_among_eligible"]) for r in ctrl if r["role"] == "BASELINE" and r["rank_among_eligible"])] if any(r["rank_among_eligible"] for r in ctrl if r["role"] == "BASELINE") else None
f["G42_flagged_not_eligible_conditions"] = sum(1 for r in ctrl if r["role"] == "BASELINE" and not r["rank_among_eligible"])
f["top1_LD_range"] = [min(float(r["L_over_D"]) for r in top if r["rank"] == "1"), max(float(r["L_over_D"]) for r in top if r["rank"] == "1")]
f["top1_over_top5_spread_median_pct"] = float(np.median([100 * (float(a["L_over_D"]) / float(b["L_over_D"]) - 1) for cid in {r["condition_id"] for r in top} for a in [next(r for r in top if r["condition_id"] == cid and r["rank"] == "1")] for b in [next(r for r in top if r["condition_id"] == cid and r["rank"] == "5")]]))
d = collections.defaultdict(list)
for r in near:
    if r["offset"] == "CENTER": continue
    cl0 = next(x for x in near if x["geometry_id"] == r["geometry_id"] and x["condition_id"] == r["condition_id"] and x["offset"] == "CENTER")
    d[r["offset"]].append((float(r["delta_CL"]) / float(cl0["CL"]) * 100, float(r["delta_CD"]) / float(cl0["CD_pressure"]) * 100, float(r["delta_L_over_D"]) / float(cl0["L_over_D"]) * 100))
f["nearby_pct_change_median_and_maxabs"] = {k: {"dCL_med": float(np.median([x[0] for x in v])), "dCL_maxabs": float(np.max(np.abs([x[0] for x in v]))), "dCD_med": float(np.median([x[1] for x in v])), "dCD_maxabs": float(np.max(np.abs([x[1] for x in v]))),
                                                 "dLD_med": float(np.median([x[2] for x in v])), "dLD_maxabs": float(np.max(np.abs([x[2] for x in v])))} for k, v in d.items()}
# cross-Mach: which design-Mach family leads at each (M, alpha); do orderings change with M?
best = {}
for r in cross:
    k = (float(r["M_flight"]), float(r["alpha_flight"])); b = best.get(k)
    if b is None or float(r["L_over_D"]) > float(b["L_over_D"]): best[k] = r
f["cross_best_geometry_changes_with_M_at_alpha"] = {a: [(m, best[(m, a)]["geometry_id"]) for m in sorted({k[0] for k in best})][::4] for a in (4.0, 7.0, 10.0)}
cross_pairs = 0; flips = 0
ids = [s["geometry_id"] for s in sel]
for a in (5.0, 7.0, 9.0):
    Ms = sorted({float(r["M_flight"]) for r in cross}); LD = {(r["geometry_id"], float(r["M_flight"])): float(r["L_over_D"]) for r in cross if float(r["alpha_flight"]) == a}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = [np.sign(LD[(ids[i], m)] - LD[(ids[j], m)]) for m in Ms]; cross_pairs += 1; flips += len(set(s)) > 1
f["pairwise_crossings_over_flight_mach_(alpha 5,7,9)"] = {"pairs": cross_pairs, "pairs_with_order_change": flips}
# families
by_fam = collections.defaultdict(list)
for r in cross:
    if float(r["alpha_flight"]) == 7.0: by_fam[(round(float(r["Ma_design"])), float(r["M_flight"]))].append(float(r["L_over_D"]))
f["family_max_LD_alpha7"] = {str(k[0]): {} for k in by_fam}
for (fm, m), v in by_fam.items(): f["family_max_LD_alpha7"][str(fm)][str(m)] = max(v)
f["equal_lift_status_counts"] = dict(collections.Counter(r["status"] for r in eq))
sp = [float(r["spearman"]) for r in cons]; f["ranking_consistency"] = {"cases": len(cons), "spearman_median": float(np.median(sp)), "spearman_min": float(np.min(sp)), "same_top1_fraction": float(np.mean([r["same_top1"] == "True" for r in cons]))}
f["ranking_consistency_by_alpha"] = {a: float(np.mean([r["same_top1"] == "True" for r in cons if r["alpha_for_fixed_alpha_LD"] == a])) for a in ("4", "6", "8", "10")}
f["CFD_candidates_by_top5_frequency"] = [(s["geometry_id"], s["Ma_design"][:6], s["n_top5_conditions"], s["roles"]) for s in sel[:10]]
(OUT / "work/report_facts.json").write_text(json.dumps(f, indent=1, default=str), encoding="utf-8"); print(json.dumps(f, indent=1, default=str))
