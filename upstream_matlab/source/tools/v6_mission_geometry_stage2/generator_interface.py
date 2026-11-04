"""Stage-2 Generator interface (database lookup, no learned model).

    query(Mach, CL_des, CD_des=None, current_geometry_state=None) ->
        {target_geometry, reachable_path, matching_error, feasibility_status, evidence}

Database = every native geometry with Teacher labels: stage-2 morph-path nodes (MISSION grid),
stage-2 search candidates (SCREEN grid), V2 geometries (V2 grid restricted to the MISSION grid).
CL is matched exactly by choosing alpha on the PCHIP curve through real Teacher labels; the
reported CD is that interpolated value (never a new label). Reachability is only claimed along
validated stage-2 morph paths; the actuator moves at 0.1 path/s.

current_geometry_state: {"path": <name>, "s": <float>}  or  {"geometry_hash": <hash>}.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np

import morph
import teacher_pool
from aerotab import DomainExit, Table, labels_from_ledger, read_v2_labels
from common import (ACTUATOR_RATE_PATH_PER_S, MISSION_ALPHA, MISSION_MACH, OUT, SCREEN_ALPHA, SCREEN_MACH, dump,
                    load, read_csv, utc)

VALID_PATH = ("VALIDATED", "VALID_GEOMETRY_AERO_RESIDUAL_ABOVE_TOL")


@lru_cache(maxsize=1)
def database():
    committed = teacher_pool.committed()
    db = {}
    lab, meta = read_v2_labels(machs=MISSION_MACH)
    for g, l in lab.items():
        try:
            db[meta[g]["geometry_hash"]] = {"id": g, "origin": "V2", "table": Table(g, l, MISSION_MACH, MISSION_ALPHA),
                                            "evidence": "V2_TEACHER"}
        except KeyError:
            pass
    for p in sorted((OUT / "candidates").glob("*_scored.csv")):
        for r in read_csv(p):
            h = r.get("geometry_hash")
            if r.get("classification") != "VALID" or not h or h in db:
                continue
            l = labels_from_ledger(committed, h)
            for grid in ((MISSION_MACH, MISSION_ALPHA), (SCREEN_MACH, SCREEN_ALPHA)):
                try:
                    db[h] = {"id": r["candidate_id"], "origin": r.get("round"), "table": Table(h, l, *grid),
                             "evidence": "NEW_TEACHER"}
                    break
                except KeyError:
                    continue
    paths = {}
    for p in sorted((OUT / "morph").glob("*.json")):
        rec = load(p)
        paths[rec["path"]] = rec
        for n in rec["nodes"]:
            h = n.get("geometry_hash")
            if h and n.get("teacher_mission_grid_complete") and h not in db:
                db[h] = {"id": n["candidate_id"], "origin": f"path:{rec['path']}",
                         "table": Table(h, morph.mission_labels(h, committed), MISSION_MACH, MISSION_ALPHA),
                         "evidence": "NEW_TEACHER"}
    return db, paths


def _eval(entry, mach, cl):
    try:
        r = entry["table"].iso_cl(mach, cl)
    except DomainExit:
        return None
    return r


def query(Mach, CL_des, CD_des=None, current_geometry_state=None):
    db, paths = database()
    scored = []
    for h, e in db.items():
        r = _eval(e, Mach, CL_des)
        if r is not None:
            scored.append((h, e, r))
    if not scored:
        return {"feasibility_status": "CL_NOT_ACHIEVABLE_OR_OUT_OF_DOMAIN", "target_geometry": None,
                "reachable_path": None, "matching_error": None}
    key = (lambda x: x[2][1]) if CD_des is None else (lambda x: abs(x[2][1] - CD_des))
    best_any = min(scored, key=key)

    reachable = []
    cur = current_geometry_state or {}
    for name, rec in paths.items():
        if rec["status"] not in VALID_PATH:
            continue
        s_cur = None
        if "path" in cur and cur["path"] == name:
            s_cur = float(cur["s"])
        elif "geometry_hash" in cur:
            s_cur = next((n["s"] for n in rec["nodes"] if n["geometry_hash"] == cur["geometry_hash"]), None)
        if s_cur is None:
            continue
        for n in rec["nodes"]:
            e = db.get(n["geometry_hash"])
            r = _eval(e, Mach, CL_des) if e else None
            if r is not None:
                reachable.append((n["geometry_hash"], e, r, name, s_cur, n["s"]))
    if "geometry_hash" in cur and cur["geometry_hash"] in db:
        r = _eval(db[cur["geometry_hash"]], Mach, CL_des)
        if r is not None:
            reachable.append((cur["geometry_hash"], db[cur["geometry_hash"]], r, None, None, None))

    def pack(h, e, r):
        return {"geometry_hash": h, "geometry_id": e["id"], "origin": e["origin"], "evidence": e["evidence"],
                "alpha_deg": r[0], "CL": CL_des, "CD": r[1], "L_over_D": CL_des / r[1]}

    out = {"query": {"Mach": Mach, "CL_des": CL_des, "CD_des": CD_des,
                     "current_geometry_state": current_geometry_state},
           "best_in_database": pack(*best_any)}
    if reachable:
        h, e, r, name, s0, s1 = min(reachable, key=lambda x: key(x[:3]))
        out["target_geometry"] = pack(h, e, r)
        out["reachable_path"] = (None if name is None else
                                 {"path": name, "s_from": s0, "s_to": s1,
                                  "transition_time_s": abs(s1 - s0) / ACTUATOR_RATE_PATH_PER_S})
        out["feasibility_status"] = "FEASIBLE_REACHABLE"
    else:
        out["target_geometry"] = pack(*best_any)
        out["reachable_path"] = None
        out["feasibility_status"] = "FEASIBLE_BUT_NO_VALIDATED_PATH" if cur else "FEASIBLE_NO_STATE_GIVEN"
    t = out["target_geometry"]
    out["matching_error"] = ({"CL_abs": 0.0, "CD_rel_to_best_in_database": t["CD"] / out["best_in_database"]["CD"] - 1}
                             if CD_des is None else {"CL_abs": 0.0, "CD_rel": (t["CD"] - CD_des) / CD_des})
    out["note"] = "CD from PCHIP through Teacher labels; not a new Teacher evaluation"
    return out


def demo():
    from aerotab import read_v2_labels as _r
    _, meta = _r(["G42"])
    g42 = meta["G42"]["geometry_hash"]
    db, paths = database()
    qs = []
    for M, CL in ((7.5, 0.045), (6.5, 0.031), (5.5, 0.028)):
        qs.append(query(M, CL, None, {"geometry_hash": g42}))
        for name, rec in paths.items():
            if rec["status"] in VALID_PATH and not name.startswith("G42__"):
                qs.append(query(M, CL, None, {"path": name, "s": 0.0}))
        cd_g42 = db[g42]["table"].iso_cl(M, CL)[1]
        qs.append(query(M, CL, 0.97 * cd_g42, {"geometry_hash": g42}))  # ask for 3 % less drag than G42
    res = {"utc": utc(), "database_size": len(db), "paths": list(paths), "queries": qs,
           "interface": "query(Mach, CL_des, CD_des=None, current_geometry_state=None)"}
    dump(OUT / "generator_interface_demo.json", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--mach", type=float)
    ap.add_argument("--cl", type=float)
    ap.add_argument("--cd", type=float)
    ap.add_argument("--state", type=str, help='JSON, e.g. {"path": "X__Y", "s": 0}')
    a = ap.parse_args()
    if a.demo:
        r = demo()
        print(json.dumps({"database_size": r["database_size"], "n_queries": len(r["queries"])}))
    else:
        print(json.dumps(query(a.mach, a.cl, a.cd, json.loads(a.state) if a.state else None), indent=1, default=float))
