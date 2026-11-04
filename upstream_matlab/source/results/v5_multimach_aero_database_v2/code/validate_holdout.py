"""Pre-registered hold-out interpolation validation (provenance/analysis_plan.json).

Per geometry: PCHIP over the 15 grid alphas at each of the 13 grid Machs, then PCHIP
over Mach. Hold-out Teacher labels (holdout_labels.csv) never enter the database.
RAW Teacher labels and INTERPOLATED values are reported in separate columns.
"""
import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import PchipInterpolator

from common import ALPHA_DEG, FLIGHT_MACH, OUT, dump, fnum, load, read_csv, write_csv


def main():
    plan = load(OUT / "provenance/analysis_plan.json")["holdout"]
    tol = plan["tolerance"]
    db = defaultdict(dict)
    for f in ("aero_database.csv", "g42_baseline.csv"):
        for r in read_csv(OUT / f):
            if r["status"] == "PASS":
                db[r["geometry_id"]][(float(r["flight_mach"]), float(r["alpha_deg"]))] = (fnum(r["CL"]), fnum(r["CD"]))
    hold = read_csv(OUT / "holdout_labels.csv")
    tasks = [t for t in load(OUT / "state/tasks.json")["tasks"] if t["kind"] == "HOLDOUT"]
    planned = {(t["geometry_id"], float(t["M_flight"]), float(a)) for t in tasks for a in t["alpha_deg"]}
    got = {(r["geometry_id"], float(r["flight_mach"]), float(r["alpha_deg"])): r for r in hold}
    Ms = np.array(FLIGHT_MACH)
    As = np.array(ALPHA_DEG)
    rows = []
    for k in sorted(planned):
        g, M, a = k
        sub = ("alpha_only" if M in FLIGHT_MACH else ("mach_only" if a in ALPHA_DEG else "both_off"))
        r = got.get(k)
        row = {"geometry_id": g, "flight_mach": M, "alpha_deg": a, "subtype": sub,
               "RAW_status": r["status"] if r else "NOT_RUN", "RAW_CL": math.nan, "RAW_CD": math.nan,
               "INTERP_CL": math.nan, "INTERP_CD": math.nan, "evaluable": False, "pass": False, "note": ""}
        if not r or r["status"] != "PASS":
            row["note"] = "hold-out Teacher label not PASS"
            rows.append(row)
            continue
        row["RAW_CL"], row["RAW_CD"] = fnum(r["CL"]), fnum(r["CD"])
        d = db.get(g, {})
        if not all((m, al) in d for m in FLIGHT_MACH for al in ALPHA_DEG):
            row["note"] = "database grid for this geometry incomplete"
            rows.append(row)
            continue
        vals = []
        for qi in (0, 1):
            per_m = [float(PchipInterpolator(As, [d[(m, al)][qi] for al in ALPHA_DEG])(a)) for m in FLIGHT_MACH]
            vals.append(float(PchipInterpolator(Ms, per_m)(M)))
        cl_i, cd_i = vals
        dcl, dcd = cl_i - row["RAW_CL"], cd_i - row["RAW_CD"]
        ok = (abs(dcl) <= tol["CL_abs"] + tol["CL_rel"] * abs(row["RAW_CL"])
              and abs(dcd) <= tol["CD_abs"] + tol["CD_rel"] * row["RAW_CD"])
        row.update({"INTERP_CL": cl_i, "INTERP_CD": cd_i, "dCL": dcl, "dCD": dcd,
                    "dCD_rel": dcd / row["RAW_CD"], "evaluable": True, "pass": ok})
        rows.append(row)
    write_csv(OUT / "holdout_validation.csv", rows,
              ["geometry_id", "flight_mach", "alpha_deg", "subtype", "RAW_status", "RAW_CL", "RAW_CD", "INTERP_CL",
               "INTERP_CD", "dCL", "dCD", "dCD_rel", "evaluable", "pass", "note"])
    ev = [r for r in rows if r["evaluable"]]
    res = {"plan": plan, "planned_points": len(planned), "teacher_points_received": len(got),
           "teacher_status": dict(Counter(r["RAW_status"] for r in rows)),
           "evaluable": len(ev), "not_evaluable": len(rows) - len(ev),
           "not_evaluable_fraction": (len(rows) - len(ev)) / len(rows) if rows else 1.0,
           "overall_pass_fraction": sum(r["pass"] for r in ev) / len(ev) if ev else None, "by_subtype": {}}
    for s in ("mach_only", "alpha_only", "both_off"):
        e = [r for r in ev if r["subtype"] == s]
        res["by_subtype"][s] = {
            "n": len(e), "pass_fraction": sum(r["pass"] for r in e) / len(e) if e else None,
            "median_abs_dCL": float(np.median([abs(r["dCL"]) for r in e])) if e else None,
            "p95_abs_dCL": float(np.percentile([abs(r["dCL"]) for r in e], 95)) if e else None,
            "median_abs_dCD_rel": float(np.median([abs(r["dCD_rel"]) for r in e])) if e else None,
            "p95_abs_dCD_rel": float(np.percentile([abs(r["dCD_rel"]) for r in e], 95)) if e else None,
            "max_abs_dCD_rel": float(np.max([abs(r["dCD_rel"]) for r in e])) if e else None}
    fails = sorted([r for r in ev if not r["pass"]], key=lambda r: -abs(r["dCD_rel"]))
    res["worst_failures"] = [{k: r[k] for k in ("geometry_id", "flight_mach", "alpha_deg", "subtype", "RAW_CL",
                                                "INTERP_CL", "RAW_CD", "INTERP_CD", "dCD_rel")} for r in fails[:20]]
    by_alpha = defaultdict(list)
    for r in ev:
        by_alpha[r["alpha_deg"]].append(r["pass"])
    res["pass_fraction_by_alpha"] = {"%g" % a: sum(v) / len(v) for a, v in sorted(by_alpha.items())}
    ok = (res["overall_pass_fraction"] is not None and res["overall_pass_fraction"] >= 0.95
          and all(v["pass_fraction"] is not None and v["pass_fraction"] >= 0.90 for v in res["by_subtype"].values())
          and res["not_evaluable_fraction"] <= 0.05)
    res["DATABASE_INTERPOLATION"] = "PASS" if ok else "FAIL"
    dump(OUT / "interpolation_validation.json", res)
    print(json.dumps({k: res[k] for k in ("evaluable", "not_evaluable", "overall_pass_fraction",
                                          "DATABASE_INTERPOLATION")}), json.dumps(res["by_subtype"]))


if __name__ == "__main__":
    main()
