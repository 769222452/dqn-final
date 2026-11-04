"""P0b: recompute a closed, consistently oriented volume for every scored V2 / stage-2 geometry and
re-evaluate the stage-2 research scenarios (EXPLORATORY / CONSERVATIVE) with it.

Read-only: nothing in stage 2 or V2 is changed; the output is a side table for the metric audit.
"""
from __future__ import annotations

import numpy as np

import common as C
from p0_geometry import closed_half_hull, signed_volume

SCEN = {"CONSERVATIVE": (1.0, 1.0), "EXPLORATORY": (0.85, 0.5)}


def closed_volume(mat):
    lo, up = C.lower_upper(C.load_mat(mat))
    p = closed_half_hull(lo, up)
    T = np.concatenate([p["lower"], p["upper_reversed"], p["base"], p["tip"], p["symmetry"]])
    v0, v1 = signed_volume(T), signed_volume(T, origin=(3.0, -2.0, 5.0))
    return 2 * abs(v0), abs(v0 - v1), float((lo[..., 2] - up[..., 2]).max())


def main():
    mats = {}
    for r in C.read_csv(C.ROOT / "results/v5_multimach_geometry_db_v2/valid_geometry_inventory.csv"):
        mats[r["geometry_id"]] = C.ROOT / r["mat_path"].replace("\\", "/")
    for r in C.read_csv(C.ROOT / "results/v6_mission_geometry_stage2/geometry_candidates.csv"):
        if r.get("native_geometry_mat") and r["classification"] == "VALID":
            mats[r["candidate_id"]] = C.ROOT / r["native_geometry_mat"].replace("\\", "/")
    mats["G42"] = C.GEOMS["G42"]["mat"]
    scored = []
    for r in C.read_csv(C.ROOT / "results/v6_mission_geometry_stage2/v2_mission_scores.csv"):
        scored.append({"geometry_id": r["geometry_id"], "source": "v2_mission_scores", "Ma_design": r["Ma_design"],
                       "volume_full_stored": r["volume_full"], "grid_thickness_max": r["grid_thickness_max"],
                       "J_ALL": r["J_ALL"], "stored_EXPLORATORY": r["scenario_EXPLORATORY"],
                       "stored_CONSERVATIVE": r["scenario_CONSERVATIVE"]})
    for r in C.read_csv(C.ROOT / "results/v6_mission_geometry_stage2/geometry_candidates.csv"):
        if r["classification"] == "VALID":
            scored.append({"geometry_id": r["candidate_id"], "source": "stage2_geometry_candidates",
                           "Ma_design": r["Ma_design"], "volume_full_stored": r["volume_full"],
                           "grid_thickness_max": r["grid_thickness_max"], "J_ALL": r["J_ALL"],
                           "stored_EXPLORATORY": r["scenario_EXPLORATORY"],
                           "stored_CONSERVATIVE": r["scenario_CONSERVATIVE"]})
    gv, gerr, gt = closed_volume(mats["G42"])
    rows = []
    for s in scored:
        p = mats.get(s["geometry_id"])
        if p is None or not p.exists():
            s["closed_volume_full"] = None
            rows.append(s)
            continue
        v, err, t = closed_volume(p)
        s.update({"closed_volume_full": v, "closure_translation_err": err,
                  "closed_volume_ratio_G42": v / gv, "stored_volume_ratio_G42": float(s["volume_full_stored"]) / 5.330427254474266,
                  "thickness_ratio_G42": t / gt})
        for k, (vmin, tmin) in SCEN.items():
            s[f"corrected_{k}"] = bool(v / gv >= vmin - 1e-12 and t / gt >= tmin - 1e-12)
        rows.append(s)
    C.write_csv(C.WORK / "p0b_volume_rescreen.csv", rows)
    ok = [r for r in rows if r.get("closed_volume_full") is not None and r["J_ALL"] not in ("", None)]
    summ = {"G42_closed_volume_full": gv, "n_scored_with_mat": len(ok)}
    for k in SCEN:
        st = [r for r in ok if str(r[f"stored_{k}"]) == "True"]
        co = [r for r in ok if r[f"corrected_{k}"]]
        best = lambda L: min(L, key=lambda r: float(r["J_ALL"])) if L else None
        bs, bc = best(st), best(co)
        summ[k] = {"n_pass_stored_metric": len(st), "n_pass_closed_volume": len(co),
                   "best_stored": bs and {"id": bs["geometry_id"], "J_ALL": float(bs["J_ALL"]),
                                          "closed_ratio": bs["closed_volume_ratio_G42"]},
                   "best_closed": bc and {"id": bc["geometry_id"], "J_ALL": float(bc["J_ALL"]),
                                          "closed_ratio": bc["closed_volume_ratio_G42"],
                                          "thickness_ratio": bc["thickness_ratio_G42"]}}
    r2 = np.corrcoef([r["stored_volume_ratio_G42"] for r in ok], [r["closed_volume_ratio_G42"] for r in ok])[0, 1]
    summ["pearson_stored_vs_closed_ratio"] = float(r2)
    summ["max_closure_err"] = max(r["closure_translation_err"] for r in ok)
    C.dump(C.WORK / "p0b_volume_rescreen_summary.json", summ)
    print(summ)


if __name__ == "__main__":
    main()
