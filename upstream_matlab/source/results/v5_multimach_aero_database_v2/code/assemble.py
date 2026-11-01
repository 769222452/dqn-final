"""Assemble the V2 aero database from reused pilot labels + committed ledger records,
then run integrity, G42-regression and curve checks.

Writes: aero_database.csv (348 x 13 x 15 target labels), g42_baseline.csv (195),
holdout_labels.csv (never part of the database), geometry_aero_coverage.csv,
failed_teacher_labels.csv, provenance/integrity.json, provenance/curve_checks.json,
provenance/curve_flags.csv. The manifest is written by report.py after all outputs exist.
"""
import json
import math
from collections import Counter, defaultdict

import numpy as np

from common import (ACTIVE, ALPHA_DEG, FLIGHT_MACH, G42_HASH, G42_MAT, GEO_DB, MODEL_VERSION, OFFDESIGN_SHA, OUT,
                    PILOT, PILOT_WORKER_SHA, STATE, V1_DB, CODE, dump, fnum, load, read_csv, read_jsonl, sha,
                    write_csv)

TOL_REGRESSION = 1e-9
DB_COLS = ["geometry_id", "kind", "design_mach", "design_space", "geometry_hash", "flight_mach", "alpha_deg", "CL",
           "CD", "Cm", "L_over_D", "status", "error_code", "error_message", "origin", "source_request", "worker",
           "model_version", "n_cone", "n_expansion", "n_freestream", "kernel_s", "completed_utc", "volume_full",
           "grid_thickness_max", "planform_area_full"]


def pts_of(resp):
    p = (resp or {}).get("points") or []
    return [p] if isinstance(p, dict) else list(p)


def finite(*v):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def label_from_point(p, base, origin, src, rec):
    CL, CD = p.get("CL"), p.get("CD")
    st = p.get("status")
    if st == "PASS" and not finite(CL, CD):
        st = "FAIL_NONFINITE"
    row = dict(base)
    row.update({"CL": CL if finite(CL) else math.nan, "CD": CD if finite(CD) else math.nan,
                "Cm": p.get("Cm") if finite(p.get("Cm")) else math.nan,
                "L_over_D": CL / CD if st == "PASS" and CD and CD > 0 else math.nan, "status": st,
                "error_code": p.get("error_id", "") or "", "error_message": (p.get("error_message", "") or "")[:300],
                "origin": origin, "source_request": src, "worker": rec.get("worker", ""),
                "model_version": p.get("model_version", ""), "n_cone": p.get("n_cone"),
                "n_expansion": p.get("n_expansion"), "n_freestream": p.get("n_freestream"),
                "kernel_s": p.get("kernel_s"), "completed_utc": rec.get("completed_utc", "")})
    return row


def main():
    tasks = load(STATE / "tasks.json")["tasks"]
    task_by = {t["request_id"]: t for t in tasks}
    inv = {r["geometry_id"]: r for r in read_csv(GEO_DB / "valid_geometry_inventory.csv")}
    metrics = {r["geometry_id"]: r for r in read_csv(GEO_DB / "geometry_metrics.csv") if r["geometry_id"]}
    ident = load(OUT / "provenance/identity_check.json")
    gid_rows = {r["geometry_id"]: r for r in read_csv(OUT / "provenance/geometry_identity.csv")}
    ids = sorted(g for g in gid_rows if g not in ident["geometries"]["quarantined"])
    twin = "V2_MD15_001"

    def meta(g):
        m = metrics[twin if g == "G42" else g]
        return {"geometry_id": g, "kind": "G42_BASELINE" if g == "G42" else ("design" if g.startswith("V2_") else "morph"),
                "design_mach": 15.0 if g == "G42" else fnum(inv[g]["Ma_design"]),
                "design_space": "FROZEN" if g == "G42" else inv[g]["design_space"],
                "geometry_hash": G42_HASH if g == "G42" else inv[g]["geometry_hash"],
                "volume_full": fnum(m["volume_full"]), "grid_thickness_max": fnum(m["grid_thickness_max"]),
                "planform_area_full": fnum(m["planform_area_full"])}

    metas = {g: meta(g) for g in ids + ["G42"]}

    # ------------------------------------------------ ledger
    recs, bad = read_jsonl(STATE / "ledger.jsonl")
    committed, dup_committed, retries = {}, Counter(), Counter()
    for r in recs:
        if r.get("outcome") == "COMMITTED":
            if r["request_id"] in committed:
                dup_committed[r["request_id"]] += 1
                continue
            committed[r["request_id"]] = r
        else:
            retries[r.get("reason", "")[:60]] += 1
    reused, rbad = read_jsonl(STATE / "reused_labels.jsonl")

    labels = {}          # (g, M, a) -> row  (main + G42)
    holdout = []
    collisions = []
    for r in reused:
        k = (r["geometry_id"], float(r["flight_mach"]), float(r["alpha_deg"]))
        base = dict(metas[r["geometry_id"]], flight_mach=k[1], alpha_deg=k[2])
        p = {kk: r.get(kk) for kk in ("CL", "CD", "Cm", "status", "model_version", "n_cone", "n_expansion",
                                       "n_freestream", "kernel_s")}
        p["error_id"] = r.get("error_code", "")
        row = label_from_point(p, base, "REUSED_VERIFIED_TEACHER", r["source_request"],
                               {"worker": "pilot", "completed_utc": r.get("source_completed_utc", "")})
        if k in labels:
            collisions.append(k)
        labels[k] = row
    resp_id_mismatch = 0
    for rid, rec in committed.items():
        t = task_by.get(rid)
        if t is None:
            collisions.append(("unknown_request", rid))
            continue
        resp = rec.get("response") or {}
        if resp.get("request_id") not in (None, rid):
            resp_id_mismatch += 1
        g, M = t["geometry_id"], float(t["M_flight"])
        pts = {float(p["alpha_deg"]): p for p in pts_of(resp)}
        for a in t["alpha_deg"]:
            a = float(a)
            base = dict(metas[g], flight_mach=M, alpha_deg=a)
            p = pts.get(a)
            if p is None:
                p = {"status": "INFRA_FAILED" if rec.get("reason") else "MISSING_IN_RESPONSE",
                     "error_id": rec.get("reason", "")[:120], "error_message": str(resp.get("message", ""))}
            elif rec.get("reason") and p.get("status") != "PASS":
                p = dict(p, status="INFRA_FAILED")
            row = label_from_point(p, base, "NEW_TEACHER", rid, rec)
            if t["kind"] == "HOLDOUT":
                holdout.append(row)
                continue
            k = (g, M, a)
            if k in labels:
                collisions.append(k)
                continue
            labels[k] = row

    main_rows, g42_rows, missing = [], [], []
    for g in ids + ["G42"]:
        for M in FLIGHT_MACH:
            for a in ALPHA_DEG:
                row = labels.get((g, M, a))
                if row is None:
                    row = dict(metas[g], flight_mach=M, alpha_deg=a, CL=math.nan, CD=math.nan, Cm=math.nan,
                               L_over_D=math.nan, status="NOT_RUN", origin="", source_request="")
                    missing.append((g, M, a))
                (g42_rows if g == "G42" else main_rows).append(row)
    write_csv(OUT / "aero_database.csv", main_rows, DB_COLS)
    write_csv(OUT / "g42_baseline.csv", g42_rows, DB_COLS)
    write_csv(OUT / "holdout_labels.csv", holdout, DB_COLS)
    failed = [r for r in main_rows + g42_rows if r["status"] != "PASS"]
    write_csv(OUT / "failed_teacher_labels.csv", failed,
              ["geometry_id", "kind", "design_mach", "flight_mach", "alpha_deg", "status", "error_code",
               "error_message", "origin", "source_request", "worker"])

    # ------------------------------------------------ integrity
    st = Counter(r["status"] for r in main_rows)
    origin = Counter(r["origin"] for r in main_rows)
    keys = [(r["geometry_id"], r["flight_mach"], r["alpha_deg"]) for r in main_rows]
    pass_rows = [r for r in main_rows + g42_rows + holdout if r["status"] == "PASS"]
    mat_ok, mat_bad = 0, []
    for g in ids:
        p = GEO_DB / inv[g]["native_geometry_path"]
        if sha(p) == gid_rows[g]["native_mat_sha256"]:
            mat_ok += 1
        else:
            mat_bad.append(g)
    g42_mat_ok = sha(G42_MAT) == ident["G42"]["mat_sha256"] if "G42" in ident and "mat_sha256" in ident["G42"] else None
    active_now = {k: sha(ACTIVE / k) for k in ident["active_m_files"]}
    task_hash_ok = all(t["geometry_hash"] == (G42_HASH if t["geometry_id"] == "G42" else
                                              gid_rows[t["geometry_id"]]["recomputed_geometry_hash"]) for t in tasks)
    checks = {
        "target_labels": len(ids) * len(FLIGHT_MACH) * len(ALPHA_DEG), "rows_written": len(main_rows),
        "unique_keys": len(set(keys)) == len(keys), "label_collisions": len(collisions),
        "status_counts": dict(st), "origin_counts": dict(origin),
        "g42_status_counts": dict(Counter(r["status"] for r in g42_rows)),
        "holdout_status_counts": dict(Counter(r["status"] for r in holdout)),
        "not_run": len(missing), "ledger_records": len(recs), "ledger_bad_lines": bad,
        "reused_bad_lines": rbad, "committed_requests": len(committed), "tasks": len(tasks),
        "tasks_uncommitted": len([t for t in tasks if t["request_id"] not in committed]),
        "duplicate_committed_records_ignored": sum(dup_committed.values()),
        "retry_records_by_reason": dict(retries), "response_request_id_mismatch": resp_id_mismatch,
        "pass_all_finite": all(finite(r["CL"], r["CD"]) for r in pass_rows),
        "pass_all_CD_positive": all(r["CD"] > 0 for r in pass_rows),
        "pass_model_version_ok": all(r["model_version"] == MODEL_VERSION for r in pass_rows),
        "grid_membership_ok": all(r["flight_mach"] in FLIGHT_MACH and r["alpha_deg"] in ALPHA_DEG
                                  for r in main_rows + g42_rows),
        "task_geometry_hash_matches_recomputed": task_hash_ok,
        "native_mat_sha_unchanged": mat_ok == len(ids), "native_mat_sha_changed": mat_bad,
        "g42_mat_sha_unchanged": g42_mat_ok,
        "offdesign_sha_unchanged": sha(ACTIVE / "offdesign_aero_v4.m") == OFFDESIGN_SHA,
        "active_m_files_unchanged": active_now == ident["active_m_files"],
        "aero_worker_is_pilot_worker": sha(CODE / "aero_worker.m") == PILOT_WORKER_SHA,
    }

    # ------------------------------------------------ G42 regression (V1, pilot, twin)
    g42 = {(r["flight_mach"], r["alpha_deg"]): r for r in g42_rows if r["status"] == "PASS"}
    v1 = [r for r in read_csv(V1_DB / "teacher_aero_database_v1.csv")
          if r["geometry_id"] == "G42" and r["teacher_status"] == "PASS"]
    v1d = []
    for r in v1:
        k = (float(r["M"]), float(r["alpha_deg"]))
        if k in g42:
            v1d.append(max(abs(float(r["CL_teacher"]) - g42[k]["CL"]), abs(float(r["CD_teacher"]) - g42[k]["CD"])))
    precs, _ = read_jsonl(PILOT / "state/ledger.jsonl")
    pd_ = []
    for rec in precs:
        if rec["geometry_id"] != "G42":
            continue
        for p in pts_of(rec["response"]):
            k = (float(rec["M_flight"]), float(p["alpha_deg"]))
            if k in g42 and p.get("status") == "PASS":
                pd_.append(max(abs(p["CL"] - g42[k]["CL"]), abs(p["CD"] - g42[k]["CD"])))
    new_g42 = [k for k, r in g42.items() if r["origin"] == "NEW_TEACHER"]
    tw = {(r["flight_mach"], r["alpha_deg"]): r for r in main_rows if r["geometry_id"] == twin and r["status"] == "PASS"}
    twd = [max(abs(tw[k]["CL"] - g42[k]["CL"]), abs(tw[k]["CD"] - g42[k]["CD"])) for k in tw if k in g42]
    reg = {"tolerance_abs": TOL_REGRESSION,
           "v1": {"points": len(v1d), "max_abs_diff": max(v1d) if v1d else None,
                  "pass": bool(v1d) and max(v1d) <= TOL_REGRESSION},
           "pilot": {"points": len(pd_), "max_abs_diff": max(pd_) if pd_ else None,
                     "pass": bool(pd_) and max(pd_) <= TOL_REGRESSION},
           "twin_V2_MD15_001_vs_G42": {"points": len(twd), "max_abs_diff": max(twd) if twd else None,
                                       "bitwise_identical": bool(twd) and max(twd) == 0.0,
                                       "pass": bool(twd) and max(twd) <= TOL_REGRESSION,
                                       "note": "same geometry_hash, different MAT file, computed fresh in this build"},
           "g42_points_pass": len(g42), "g42_points_new_teacher": len(new_g42)}
    checks["g42_regression"] = reg
    checks["TEACHER_LABEL_INTEGRITY"] = "PASS" if all([
        checks["unique_keys"], checks["label_collisions"] == 0, checks["pass_all_finite"],
        checks["pass_all_CD_positive"], checks["pass_model_version_ok"], checks["grid_membership_ok"],
        checks["task_geometry_hash_matches_recomputed"], checks["native_mat_sha_unchanged"],
        checks["g42_mat_sha_unchanged"] is not False, checks["offdesign_sha_unchanged"],
        checks["active_m_files_unchanged"], checks["aero_worker_is_pilot_worker"],
        checks["response_request_id_mismatch"] == 0, not bad, not rbad,
        reg["v1"]["pass"], reg["pilot"]["pass"], reg["twin_V2_MD15_001_vs_G42"]["pass"]]) else "FAIL"
    infra = st.get("INFRA_FAILED", 0) + st.get("MISSING_IN_RESPONSE", 0)
    if checks["TEACHER_LABEL_INTEGRITY"] != "PASS":
        full = "FAILED"
    elif not missing and checks["tasks_uncommitted"] == 0 and infra == 0:
        full = "COMPLETE"
    else:
        full = "PARTIAL"
    checks["FULL_AERO_DATABASE_V2"] = full
    dump(OUT / "provenance/integrity.json", checks)

    # ------------------------------------------------ curve checks
    curves = defaultdict(dict)
    for r in main_rows + g42_rows:
        curves[(r["geometry_id"], r["flight_mach"])][r["alpha_deg"]] = r
    flags = []
    cc = Counter()
    ld_tab = defaultdict(dict)
    for (g, M), d in curves.items():
        rs = [d[a] for a in ALPHA_DEG]
        if not all(r["status"] == "PASS" for r in rs):
            cc["incomplete"] += 1
            continue
        cc["complete"] += 1
        CL = np.array([r["CL"] for r in rs])
        CD = np.array([r["CD"] for r in rs])
        if not np.all(np.diff(CL) > 0):
            cc["CL_not_monotonic"] += 1
            flags.append({"geometry_id": g, "flight_mach": M, "flag": "CL_not_strictly_increasing"})
        if np.argmin(CD) not in range(0, 6):
            cc["CD_min_alpha_gt_3"] += 1
            flags.append({"geometry_id": g, "flight_mach": M, "flag": "CD_min_at_alpha>%g" % ALPHA_DEG[5]})
        if not np.all(np.diff(CD[np.argmin(CD):]) > 0):
            cc["CD_not_increasing_above_min"] += 1
            flags.append({"geometry_id": g, "flight_mach": M, "flag": "CD_not_increasing_above_min"})
        beta = math.sqrt(M * M - 1)
        pos = CL > 0.005
        nb = int(np.sum(CL[pos] / CD[pos] > 4.0 / (CL[pos] * beta)))
        cc["flat_plate_reference_exceedances"] += nb
        ld_tab[g][M] = (CL, CD)
    # Mach smoothness: robust outlier test on second differences across Mach at each alpha
    mach_out = 0
    for g, dm in ld_tab.items():
        if len(dm) != len(FLIGHT_MACH):
            continue
        Ms = np.array(FLIGHT_MACH)
        for qi, q in enumerate(("CL", "CD")):
            Y = np.array([dm[M][qi] for M in FLIGHT_MACH])  # 13 x 15
            for j, a in enumerate(ALPHA_DEG):
                y = Y[:, j]
                lin = np.interp(Ms[1:-1], Ms[[0, -1]], y[[0, -1]])
                r = y[1:-1] - (y[:-2] * (Ms[2:] - Ms[1:-1]) + y[2:] * (Ms[1:-1] - Ms[:-2])) / (Ms[2:] - Ms[:-2])
                scale = max(np.median(np.abs(y)), 1e-6)
                bad_i = np.where(np.abs(r) > 0.05 * scale + 5 * np.median(np.abs(r)) + 1e-9)[0]
                for i in bad_i:
                    mach_out += 1
                    if len(flags) < 5000:
                        flags.append({"geometry_id": g, "flight_mach": FLIGHT_MACH[i + 1],
                                      "flag": "%s_mach_kink_alpha_%g" % (q, a)})
                del lin
    cc["mach_kink_flags"] = mach_out
    write_csv(OUT / "provenance/curve_flags.csv", flags, ["geometry_id", "flight_mach", "flag"])
    dump(OUT / "provenance/curve_checks.json", {
        "counts": dict(cc),
        "notes": {"CL_not_monotonic": "inverse CL->alpha not defined; curve excluded from equal-lift analysis",
                  "flat_plate_reference_exceedances": "informational: linear thin-plate L/D bound 4/(CL*beta); "
                                                      "exceedance is not a violation for these models (pilot finding)",
                  "mach_kink_flags": "second difference across Mach > 5 % of median |y| + 5x median residual; "
                                     "informational smoothness screen"}})

    # ------------------------------------------------ coverage
    cov = []
    by_g = defaultdict(list)
    for r in main_rows + g42_rows:
        by_g[r["geometry_id"]].append(r)
    for g in ids + ["G42"]:
        rs = by_g[g]
        c = Counter(r["status"] for r in rs)
        o = Counter(r["origin"] for r in rs if r["origin"])
        comp = sum(1 for M in FLIGHT_MACH if all(curves[(g, M)][a]["status"] == "PASS" for a in ALPHA_DEG))
        cov.append({**{k: metas[g][k] for k in ("geometry_id", "kind", "design_mach", "design_space", "geometry_hash",
                                                "volume_full", "grid_thickness_max")},
                    "labels_target": len(rs), "PASS": c.get("PASS", 0),
                    "FAIL": sum(v for k, v in c.items() if k != "PASS" and k != "NOT_RUN"),
                    "NOT_RUN": c.get("NOT_RUN", 0), "REUSED_VERIFIED_TEACHER": o.get("REUSED_VERIFIED_TEACHER", 0),
                    "NEW_TEACHER": o.get("NEW_TEACHER", 0), "complete_mach_curves": comp,
                    "complete_all_13_mach": comp == len(FLIGHT_MACH),
                    "in_holdout_set": g in load(OUT / "provenance/analysis_plan.json")["holdout"]["geometries"]})
    write_csv(OUT / "geometry_aero_coverage.csv", cov)
    print(json.dumps({k: checks[k] for k in ("status_counts", "origin_counts", "not_run", "tasks_uncommitted",
                                             "TEACHER_LABEL_INTEGRITY", "FULL_AERO_DATABASE_V2")}))
    print(json.dumps(reg))
    print(json.dumps(dict(cc)))


if __name__ == "__main__":
    main()
