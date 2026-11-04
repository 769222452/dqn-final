"""Assemble the stage-2 deliverable tables from the ledgers (no new computation)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import teacher_pool
from common import (ACTIVE, AERO_WORKER_SHA, FIXED_INPUTS, MODEL_VERSION, OFFDESIGN_SHA, OUT, PARAMS, dump, load,
                    read_csv, read_jsonl, sha, utc, write_csv)


def all_candidate_records():
    import geometry
    import search
    recs = {}
    for p in sorted((OUT / "candidates").glob("*_scored.csv")):
        for r in read_csv(p):
            recs[r["candidate_id"]] = r
    extra = []
    for d in sorted((OUT / "cases").iterdir()):
        cj = d / "candidate.json"
        if cj.exists() and d.name not in recs:
            extra.append(load(cj))
    if extra:
        for r in search.score_new([geometry.summary(c) for c in extra]):
            src = load(Path(r["case_dir"]) / "candidate.json")
            r["round"] = "path_or_pilot" if src.get("source") != "g42_regression" else "regression"
            r["path"] = src.get("path")
            r["s"] = src.get("s")
            recs[r["candidate_id"]] = r
    return list(recs.values())


def candidates_csv():
    rows = all_candidate_records()
    keep = ["candidate_id", "round", "source", "path", "s", "Ma_design", *PARAMS, "design_space",
            "generation_status", "generation_error", "classification", "failure_reason", "geometry_hash",
            "volume_full", "grid_thickness_max", "volume_ratio_G42", "thickness_ratio_G42",
            "scenario_CONSERVATIVE", "scenario_EXPLORATORY", "teacher_labels_pass",
            "teacher_labels_missing_or_failed", "J_ALL", "J_M7-8", "J_M6-7", "J_M5-6", "aero_evidence",
            "span", "length", "planform_area_full", "volumetric_efficiency", "native_geometry_mat"]
    write_csv(OUT / "geometry_candidates.csv", rows, keep)
    return rows


def manifest(rows):
    committed = teacher_pool.committed()
    by_hash = {}
    for (h, m, a), p in committed.items():
        by_hash.setdefault(h, [0, 0])[0 if p["status"] == "PASS" else 1] += 1
    v2_hashes = {r["geometry_hash"] for r in read_csv(OUT / "v2_mission_scores.csv")}
    items = []
    twins = []
    for r in rows:
        if r.get("classification") != "VALID" or r.get("round") == "regression":
            continue
        if r["geometry_hash"] in v2_hashes:  # path endpoints regenerated natively (G42, V2_MD8_009)
            twins.append({"candidate_id": r["candidate_id"], "geometry_hash": r["geometry_hash"]})
            continue
        case = Path(r["case_dir"]) if r.get("case_dir") else None
        audit = load(case / "audit.json") if case and (case / "audit.json").exists() else {}
        c = load(case / "candidate.json") if case else {}
        items.append({"candidate_id": r["candidate_id"], "round": r.get("round"), "source": r.get("source"),
                      "Ma_design": float(r["Ma_design"]), "z": {p: float(r[p]) for p in PARAMS},
                      "inputs10": c.get("inputs10"), "design_space": r.get("design_space"),
                      "geometry_hash": r["geometry_hash"], "mat_sha256": audit.get("mat_sha256"),
                      "native_geometry_mat": r.get("native_geometry_mat"),
                      "volume_full": r.get("volume_full"), "grid_thickness_max": r.get("grid_thickness_max"),
                      "teacher_labels": {"pass": by_hash.get(r["geometry_hash"], [0, 0])[0],
                                         "fail": by_hash.get(r["geometry_hash"], [0, 0])[1]},
                      "J_ALL": r.get("J_ALL") or None})
    m = {"utc": utc(), "n_new_valid_geometries": len(items),
         "generator": {"wrapper": "tools/geometry_audit/run_geometry_audit_batch.m (unmodified)",
                       "core": "matlab/active/build_geometry_core.m (unmodified)",
                       "fixed_inputs": FIXED_INPUTS, "audit": "tools/geometry_audit/scan.py::audit (frozen)",
                       "metrics": "results/v5_multimach_geometry_db_v2/code/geometry_metrics.py (unmodified)"},
         "teacher": {"model_version": MODEL_VERSION, "offdesign_aero_v4_sha256": OFFDESIGN_SHA,
                     "aero_worker_sha256": AERO_WORKER_SHA,
                     "active_hash_check": "bridge.guard_v6_teacher (matlab/active == V5 manifest)"},
         "regenerated_v2_twins_excluded": twins,
         "geometries": items}
    dump(OUT / "new_geometry_manifest.json", m)
    return m


def labels_csv():
    rows = []
    for r in read_jsonl(OUT / "teacher/ledger.jsonl"):
        if r.get("outcome") != "COMMITTED":
            continue
        for p in r["points"]:
            rows.append({"geometry_id": r["geometry_id"], "geometry_hash": r["geometry_hash"],
                         "flight_mach": r["M_flight"], "alpha_deg": p["alpha_deg"], "status": p["status"],
                         "CL": p.get("CL"), "CD": p.get("CD"), "Cm": p.get("Cm"),
                         "L_over_D": (p["CL"] / p["CD"]) if p.get("CL") is not None and p.get("CD") else None,
                         "model_version": p.get("model_version"), "error_id": p.get("error_id"),
                         "request_id": r["request_id"], "worker": r["worker"], "completed_utc": r["completed_utc"],
                         "evidence": "NEW_TEACHER"})
    write_csv(OUT / "aero_pilot_labels.csv", rows)
    return rows


def morph_paths():
    recs = [load(p) for p in sorted((OUT / "morph").glob("*.json"))]
    s = load(OUT / "morph_paths_summary.json") if (OUT / "morph_paths_summary.json").exists() else {}
    dump(OUT / "morph_paths.json", {"utc": utc(), "plan": s.get("plan"), "paths": recs,
                                    "rules": {"intermediates": "native MATLAB generation at every node; "
                                                               "no MAT coordinate interpolation",
                                              "cross_design_mach": "allowed only if every node is VALID "
                                                                   "(checked, not assumed)",
                                              "aero_between_nodes": "linear blend of Teacher node tables; "
                                                                    "midpoint Teacher residual reported"}})
    return recs


def _passes():
    a = {p.stem: load(p) for p in sorted((OUT / "pilot").glob("*.json"))}
    b = {p.stem: load(p) for p in sorted((OUT / "pilot_B").glob("*.json"))} if (OUT / "pilot_B").exists() else {}
    best = {}
    for k in sorted(set(a) | set(b)):
        cand = [(x.get("range_m") or -np.inf, tag, x) for tag, x in (("A", a.get(k)), ("B", b.get(k))) if x]
        r, tag, x = max(cand, key=lambda c: c[0])
        best[k] = dict(x, best_pass=tag, range_pass_A_km=(a.get(k, {}).get("range_m") or np.nan) / 1e3,
                       range_pass_B_km=(b.get(k, {}).get("range_m") or np.nan) / 1e3)
    return best


def pilot_json():
    raw = load(OUT / "pilot_results_raw.json")
    res = _passes()
    ep = load(OUT / "endpoints.json")
    g42 = res.get("fixed__G42", {}).get("range_m")
    table = []
    for k, v in sorted(res.items()):
        row = {"job": k, "status": v.get("status"), "kind": v.get("kind"), "used_by": v.get("spec", {}).get("used_by"),
               "best_pass": v.get("best_pass"), "range_pass_A_km": v.get("range_pass_A_km"),
               "range_pass_B_km": v.get("range_pass_B_km"),
               "range_km": v.get("range_m", np.nan) / 1e3 if v.get("range_m") else None,
               "delta_vs_G42_km": (v["range_m"] - g42) / 1e3 if v.get("range_m") and g42 else None,
               "t_switch_s": v.get("t_switch_s"), "evaluations": v.get("evaluations"), "wall_s": v.get("wall_s"),
               "fine_disagreement_max": v.get("fine_disagreement_max"),
               "drag_work_MJ": v.get("flight", {}).get("drag_work_j", np.nan) / 1e6 if v.get("flight") else None,
               "q_margin_pa": v.get("flight", {}).get("min_margins", {}).get("q_max_pa") if v.get("flight") else None,
               "trajectory": v.get("trajectory_summary"), "path_status": v.get("spec", {}).get("path_status"),
               "solver": v.get("solver")}
        table.append(row)
    comp = {}
    for scen, s in ep["scenarios"].items():
        A, C, S = s["A_high"]["id"], s["C_low_mid"]["id"], s["S_best_single"]["id"]
        fixed = {g: res.get(f"fixed__{g}", {}).get("range_m") for g in {A, C, S, "G42"}}
        fx = {g: v for g, v in fixed.items() if v}
        bestfixed = max(fx, key=fx.get) if fx else None
        name = f"{A}__{C}"
        step = res.get(f"step__{name}", {}).get("range_m")
        morph = res.get(f"morph__{name}", {}).get("range_m")
        morphg = res.get(f"morphg42__{name}", {}).get("range_m") or res.get(f"g42start__{A}", {}).get("range_m")
        comp[scen] = {"A_high": A, "C_low_mid": C, "best_single": S,
                      "fixed_range_km": {g: (v / 1e3 if v else None) for g, v in fixed.items()},
                      "best_fixed": bestfixed,
                      "B_ideal_selection_km": step / 1e3 if step else None,
                      "C_executable_morph_family_start_km": morph / 1e3 if morph else None,
                      "C_executable_morph_G42_start_km": morphg / 1e3 if morphg else None,
                      "morph_minus_best_fixed_km": (morph - fx[bestfixed]) / 1e3 if morph and bestfixed else None,
                      "ideal_minus_best_fixed_km": (step - fx[bestfixed]) / 1e3 if step and bestfixed else None}
    out = {"utc": utc(), "budget_s_per_job": raw["budget_s_per_job"], "jobs": table, "skipped": raw["skipped"],
           "comparison": comp,
           "caveats": ["best-found local results (SLSQP iteration/time limited); differences below the "
                       "optimiser noise (~10-30 km in stage 1 between rounds) are not significant",
                       "aero = PCHIP through Teacher labels on the MISSION grid M4.5-8.5, alpha -2..12",
                       "morph aero between native nodes = linear blend of node tables",
                       "same mass, reference area, limits and initial state for every geometry (stage-1 rules)"]}
    dump(OUT / "fixed_vs_morph_pilot.json", out)
    return out


if __name__ == "__main__":
    rows = candidates_csv()
    m = manifest(rows)
    labs = labels_csv()
    morph_paths()
    if (OUT / "pilot_results_raw.json").exists():
        pilot_json()
    print(json.dumps({"candidates": len(rows), "new_valid": m["n_new_valid_geometries"], "labels": len(labs)}))
