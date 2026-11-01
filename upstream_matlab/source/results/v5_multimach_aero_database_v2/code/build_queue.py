"""Build the resumable task list, verify pilot labels for reuse, and freeze the
hold-out / analysis plan BEFORE any new Teacher call.

Outputs (state/ and provenance/):
  state/tasks.json            every request (MAIN / BASELINE / HOLDOUT), static
  state/reused_labels.jsonl   pilot labels that passed all six reuse conditions
  provenance/reuse_audit.json per-condition counts and every rejection
  provenance/analysis_plan.json  interpolation method, tolerances, label rules (frozen)
"""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

from common import (ALPHA_DEG, FLIGHT_MACH, G42_HASH, G42_MAT, GEO_DB, MODEL_OPTIONS, MODEL_VERSION, OFFDESIGN_SHA,
                    OUT, PILOT, PILOT_WORKER_SHA, REFERENCE, STATE, dump, load, read_csv, read_jsonl, request_id, sha)

HOLDOUT_PER_GROUP = 8
HOLDOUT_OFF_MACH = [3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.125]
HOLDOUT_ALPHA_AT_OFF_MACH = [0.0, 4.0, 8.0, 12.0, 1.5, 4.5, 7.5, 10.5]
HOLDOUT_OFF_ALPHA = [-1.5, 0.5, 2.5, 3.5, 5.5, 6.5, 8.5, 9.5, 11.5]


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / "tasks.json").exists():
        raise SystemExit("tasks.json exists; the queue is frozen (delete only by explicit decision)")
    ident = load(OUT / "provenance/identity_check.json")
    assert ident["IDENTITY_OK"], "identity check failed"
    geo = {r["geometry_id"]: r for r in read_csv(OUT / "provenance/geometry_identity.csv")}
    inv = {r["geometry_id"]: r for r in read_csv(GEO_DB / "valid_geometry_inventory.csv")}
    ids = [g for g in geo if g not in ident["geometries"]["quarantined"]]

    # ---------------- reuse audit (pilot)
    pid = load(PILOT / "state/identity.json")
    pbase = load(PILOT / "provenance/baseline_before.json")["hashed"]
    cond_source = (pid["offdesign_aero_v4_sha256"] == OFFDESIGN_SHA and pid["aero_worker_sha256"] == PILOT_WORKER_SHA
                   and pid["reference"] == REFERENCE and pid["model_options"] == MODEL_OPTIONS
                   and all(pbase.get(f"matlab/active/{k}") == v for k, v in ident["active_m_files"].items()))
    sel = {r["geometry_id"]: r for r in read_csv(PILOT / "selection.csv")}
    recs, bad = read_jsonl(PILOT / "state/ledger.jsonl")
    audit = {"pilot_ledger_records": len(recs), "pilot_ledger_bad_lines": bad,
             "condition_3_source_options_sref_gas": cond_source, "rejections": defaultdict(int),
             "rejected_examples": []}
    by_req = defaultdict(list)
    for rec in recs:
        by_req[rec["request_id"]].append(rec)
    reused = []
    for rid, rs in by_req.items():
        if rid.endswith("__REPEAT"):
            continue
        rec = rs[0]
        g = rec["geometry_id"]
        target = "G42" if g == "G42" else g

        def reject(why):
            audit["rejections"][why] += 1
            if len(audit["rejected_examples"]) < 30:
                audit["rejected_examples"].append({"request": rid, "why": why})

        if not cond_source:
            reject("C3_source_mismatch"); continue
        s = sel.get(g)
        if g == "G42":
            ok1 = s and s["geometry_hash"] == G42_HASH
            ok2 = s and s["mat_sha256"] == sha(G42_MAT) and s["mat_path"] == str(G42_MAT)
        else:
            gi = geo.get(g)
            ok1 = bool(s and gi and s["geometry_hash"] == gi["inventory_geometry_hash"] == gi["recomputed_geometry_hash"])
            ok2 = bool(s and gi and s["mat_sha256"] == gi["native_mat_sha256"]
                       and s["mat_path"].replace("\\", "/").endswith(inv[g]["native_geometry_path"]))
        if not ok1:
            reject("C1_id_or_hash"); continue
        if not ok2:
            reject("C2_mat_identity"); continue
        if len(rs) > 1 and any(json.dumps(r["response"], sort_keys=True) != json.dumps(rec["response"], sort_keys=True) for r in rs):
            reject("C6_duplicate_conflict"); continue
        M = float(rec["M_flight"])
        if M not in FLIGHT_MACH:
            reject("C4_mach_off_grid"); continue
        pts = rec["response"].get("points") or []
        pts = [pts] if isinstance(pts, dict) else pts
        seen = set()
        for p in pts:
            a = float(p["alpha_deg"])
            if a not in ALPHA_DEG:
                reject("C4_alpha_off_grid"); continue
            if a in seen:
                reject("C6_duplicate_alpha"); continue
            seen.add(a)
            CL, CD = p.get("CL"), p.get("CD")
            if p.get("status") != "PASS" or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (CL, CD)):
                reject("C5_not_pass_or_nonfinite"); continue
            if p.get("model_version") != MODEL_VERSION:
                reject("C3_model_version"); continue
            reused.append({"geometry_id": target, "flight_mach": M, "alpha_deg": a, "CL": CL, "CD": CD,
                           "Cm": p.get("Cm"), "status": "PASS", "error_code": "", "kernel_s": p.get("kernel_s"),
                           "model_version": p.get("model_version"), "n_cone": p.get("n_cone"),
                           "n_expansion": p.get("n_expansion"), "n_freestream": p.get("n_freestream"),
                           "origin": "REUSED_VERIFIED_TEACHER", "source_request": f"pilot:{rid}",
                           "source_completed_utc": rec.get("completed_utc")})
    audit["rejections"] = dict(audit["rejections"])
    audit["reused_labels"] = len(reused)
    audit["reused_by_geometry"] = len({r["geometry_id"] for r in reused})
    keyset = defaultdict(set)
    for r in reused:
        keyset[(r["geometry_id"], r["flight_mach"])].add(r["alpha_deg"])
    with open(STATE / "reused_labels.jsonl", "w", encoding="utf-8") as f:
        for r in reused:
            f.write(json.dumps(r) + "\n")

    # ---------------- tasks
    def interleave(gs):
        by = defaultdict(list)
        for g in gs:
            by[float(inv[g]["Ma_design"])].append(g)
        out = []
        while any(by.values()):
            for md in sorted(by):
                if by[md]:
                    out.append(by[md].pop(0))
        return out

    design = interleave(sorted(g for g in ids if g.startswith("V2_")))
    morph = interleave(sorted(g for g in ids if g.startswith("V2M_")))
    tasks = []
    for M in FLIGHT_MACH:
        miss = [a for a in ALPHA_DEG if a not in keyset[("G42", M)]]
        if miss:
            tasks.append({"request_id": request_id("G42", M, "BASELINE"), "kind": "BASELINE", "geometry_id": "G42",
                          "geometry_mat": str(G42_MAT), "geometry_hash": G42_HASH, "M_flight": M, "alpha_deg": miss})
    for g in design + morph:
        mat = str(GEO_DB / inv[g]["native_geometry_path"])
        for M in FLIGHT_MACH:
            miss = [a for a in ALPHA_DEG if a not in keyset[(g, M)]]
            if miss:
                tasks.append({"request_id": request_id(g, M), "kind": "MAIN", "geometry_id": g, "geometry_mat": mat,
                              "geometry_hash": inv[g]["geometry_hash"], "M_flight": M, "alpha_deg": miss})

    # ---------------- hold-out set (pre-registered, deterministic)
    groups = defaultdict(list)
    for g in ids:
        groups[float(inv[g]["Ma_design"])].append(g)
    hold = ["G42"]
    for md in sorted(groups):
        hold += sorted(groups[md], key=lambda g: hashlib.sha256(g.encode()).hexdigest())[:HOLDOUT_PER_GROUP]
    hold_tasks = []
    for i, g in enumerate(hold):
        mat = str(G42_MAT) if g == "G42" else str(GEO_DB / inv[g]["native_geometry_path"])
        gh = G42_HASH if g == "G42" else inv[g]["geometry_hash"]
        for j in range(3):
            M = HOLDOUT_OFF_MACH[(3 * i + j) % len(HOLDOUT_OFF_MACH)]
            hold_tasks.append({"request_id": request_id(g, M, "HOLDOUT"), "kind": "HOLDOUT", "geometry_id": g,
                               "geometry_mat": mat, "geometry_hash": gh, "M_flight": M,
                               "alpha_deg": HOLDOUT_ALPHA_AT_OFF_MACH})
        Mg = FLIGHT_MACH[i % len(FLIGHT_MACH)]
        hold_tasks.append({"request_id": request_id(g, Mg, "HOLDOUT"), "kind": "HOLDOUT", "geometry_id": g,
                           "geometry_mat": mat, "geometry_hash": gh, "M_flight": Mg, "alpha_deg": HOLDOUT_OFF_ALPHA})
    all_tasks = tasks + hold_tasks
    assert len({t["request_id"] for t in all_tasks}) == len(all_tasks), "duplicate request ids"
    dump(STATE / "tasks.json", {"created_utc": utc(), "n_tasks": len(all_tasks),
                                "n_main_points": sum(len(t["alpha_deg"]) for t in tasks if t["kind"] == "MAIN"),
                                "n_baseline_points": sum(len(t["alpha_deg"]) for t in tasks if t["kind"] == "BASELINE"),
                                "n_holdout_points": sum(len(t["alpha_deg"]) for t in hold_tasks),
                                "tasks": all_tasks})
    audit["target_labels_main"] = len(ids) * len(FLIGHT_MACH) * len(ALPHA_DEG)
    audit["reused_main"] = sum(1 for r in reused if r["geometry_id"] != "G42")
    audit["reused_G42"] = sum(1 for r in reused if r["geometry_id"] == "G42")
    audit["new_main_points_needed"] = sum(len(t["alpha_deg"]) for t in tasks if t["kind"] == "MAIN")
    audit["V2_MD15_001_note"] = ("same geometry_hash as G42 but different geometry_id and MAT sha (regenerated file); "
                                 "per reuse rule 1-2 its labels are computed fresh, never copied from G42")
    dump(OUT / "provenance/reuse_audit.json", audit)

    plan = {
        "frozen_utc": utc(), "frozen_before_any_new_teacher_call": True,
        "holdout": {
            "geometries": hold, "selection": f"G42 + first {HOLDOUT_PER_GROUP} per design-Mach group by sha256(geometry_id)",
            "off_grid_mach": HOLDOUT_OFF_MACH, "alpha_at_off_grid_mach": HOLDOUT_ALPHA_AT_OFF_MACH,
            "off_grid_alpha_at_grid_mach": HOLDOUT_OFF_ALPHA, "never_part_of_database": True,
            "interpolant": "per geometry: PCHIP over the 15 grid alphas at each of the 13 grid Machs, then PCHIP over Mach",
            "tolerance": {"CL_abs": 5e-4, "CL_rel": 0.01, "CD_abs": 2e-5, "CD_rel": 0.02,
                          "rule": "point passes if |dCL| <= CL_abs + CL_rel*|CL| AND |dCD| <= CD_abs + CD_rel*CD"},
            "PASS_if": "overall pass fraction >= 0.95 AND each subtype (mach_only, alpha_only, both_off) >= 0.90 "
                       "AND not-evaluable (missing/failed neighbours or hold-out FAIL) <= 5 %",
            "subtypes": {"mach_only": "off-grid Mach, grid alpha", "both_off": "off-grid Mach, off-grid alpha",
                         "alpha_only": "grid Mach, off-grid alpha"}},
        "performance": {
            "CL_levels": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
            "usable_CL_min": 0.02,
            "equal_lift_CD": "PCHIP inverse alpha(CL) on the grid curve (CL strictly increasing), then PCHIP CD(alpha); "
                             "uncertainty u = |PCHIP - linear|",
            "significance": "|dCD| > 2*(u_geometry + u_reference) AND |dCD|/CD_reference > 0.01",
            "constraints": {"C0": "none", "C1": "volume_full >= G42 volume",
                            "C2": "volume_full >= G42 volume AND grid_thickness_max >= G42 grid_thickness_max"},
            "excluded_from_better_than_G42": ["V2_MD15_001 (identical geometry_hash to G42)"],
            "labels": {
                "NEW_GEOMETRY_BETTER_THAN_G42": "YES if some Mach has >=2 usable CL levels where some V2 geometry is "
                                                "significantly lower CD than G42 (C0); INCONCLUSIVE if only non-significant "
                                                "lower CD; NO otherwise",
                "CROSS_MACH_ADVANTAGE": "per usable CL level c: fixed_c = geometry feasible at c at all 13 Machs with lowest "
                                        "mean CD/CD_G42; YES if for some c the per-Mach winner changes with Mach AND the "
                                        "per-Mach envelope is significantly below fixed_c at >=2 Machs; INCONCLUSIVE if "
                                        "switching but <2 significant Machs; NO otherwise",
                "BETTER_THAN_BEST_FIXED_GEOMETRY": "states S = usable (M, CL) where G42 is feasible; best fixed = V2 geometry "
                                                   "feasible on all S with lowest mean CD/CD_G42; YES if the per-state envelope "
                                                   "is significantly lower at >=10 % of S; INCONCLUSIVE if >0 and <10 %; NO if 0",
                "FULL_AERO_DATABASE_V2": "COMPLETE if every target label has a final Teacher status (PASS or recorded "
                                         "Teacher FAIL) and no task remains; PARTIAL if tasks remain or infra failures are "
                                         "unresolved; FAILED if integrity fails",
                "TEACHER_LABEL_INTEGRITY": "PASS if identity, hashes, uniqueness, finiteness, grid membership and G42 "
                                           "regression (V1 and pilot within 1e-9 absolute) all pass",
                "READY_FOR_CONTROL_INTEGRATION": "YES only if COMPLETE, integrity PASS and interpolation PASS"}},
    }
    dump(OUT / "provenance/analysis_plan.json", plan)
    print(json.dumps({k: v for k, v in audit.items() if k != "rejected_examples"}, indent=1))
    print("tasks", len(all_tasks), "main", len(tasks), "holdout", len(hold_tasks),
          "holdout points", sum(len(t["alpha_deg"]) for t in hold_tasks))


if __name__ == "__main__":
    main()
