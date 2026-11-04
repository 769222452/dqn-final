"""Stage 0: shock-angle x design-Mach feasibility scan.

Purpose: decide, with real MATLAB evidence, whether the frozen optimizer bounds
(a1,a2,b1,b2 in [3,8] deg) admit any geometry at Ma_design = 4..8, and if not,
how far the shock angles must grow. The scan scales G42's four shock angles by
a factor f; f=1 is exactly the frozen design point. Angles above 8 deg are
outside the historical bounds and are labelled as such.
"""
from __future__ import annotations

import json
import subprocess
import time
from math import asin, degrees

import mmg
from mmg import ACTIVE, G42_Z, MATLAB, OUT, PARAMS, DesignSpace, dump

MA_SCAN = (4.0, 5.0, 6.0, 7.0, 8.0, 15.0)
SCALES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
HERE = mmg.Path(__file__).resolve().parent


def main():
    space = DesignSpace()
    space.verify()
    out_dir = OUT / "stage0_feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    a1, a2, b1, b2, d1, ax = G42_Z
    jobs, meta = [], []
    for ma in MA_SCAN:
        for f in SCALES:
            z = (a1 * f, a2 * f, b1 * f, b2 * f, d1, ax)
            pid = f"MD{ma:g}_F{f:g}"
            jobs.append({"probe_id": pid, "inputs10": space.inputs(ma, z)})
            meta.append({"probe_id": pid, "Ma_design": ma, "angle_scale": f,
                         "z": list(z), "within_frozen_bounds": space.in_bounds(z),
                         "mach_angle_deg": degrees(asin(1.0 / ma)),
                         "min_shock_angle_deg": min(z[:4]), "max_shock_angle_deg": max(z[:4])})
    job_file = out_dir / "probe_jobs.json"
    result_file = out_dir / "probe_results.json"
    dump(job_file, jobs)
    dump(out_dir / "probe_plan.json", {"utc": mmg.now(), "G42_z": list(G42_Z),
                                       "parameter_names": list(PARAMS),
                                       "Ma_design_scan": list(MA_SCAN), "angle_scales": list(SCALES),
                                       "frozen_bounds": {"lower": space.lower.tolist(),
                                                         "upper": space.upper.tolist()},
                                       "probes": meta})
    code = (f"addpath('{HERE.as_posix()}'); "
            f"probe_feasibility('{job_file.as_posix()}','{ACTIVE.as_posix()}','{result_file.as_posix()}');")
    log = (out_dir / "matlab_probe.log").open("w", encoding="utf-8")
    start = time.monotonic()
    proc = subprocess.Popen([str(MATLAB), "-batch", code], cwd=str(out_dir),
                            stdout=log, stderr=subprocess.STDOUT)
    proc.wait()
    log.close()
    wall = time.monotonic() - start
    results = mmg.load(result_file)
    by_id = {r["probe_id"]: r for r in results}
    rows = []
    for m in meta:
        r = by_id.get(m["probe_id"], {})
        rows.append({**{k: m[k] for k in ("probe_id", "Ma_design", "angle_scale",
                                          "within_frozen_bounds", "mach_angle_deg",
                                          "min_shock_angle_deg", "max_shock_angle_deg")},
                     **{p: m["z"][i] for i, p in enumerate(PARAMS)},
                     "status": r.get("status"), "error_identifier": r.get("error_identifier"),
                     "error_message": r.get("error_message"),
                     "error_stack_top": r.get("error_stack_top"),
                     "elapsed_s": r.get("elapsed_s"),
                     "length": r.get("length"), "half_span": r.get("half_span"),
                     "nonfinite_count": r.get("nonfinite_count")})
    mmg.write_csv(out_dir / "feasibility_scan.csv", rows)
    summary = {"utc": mmg.now(), "wall_s": wall, "returncode": proc.returncode,
               "probe_count": len(meta), "completed": len(results),
               "per_Ma_design": {}}
    for ma in MA_SCAN:
        sub = [r for r in rows if r["Ma_design"] == ma]
        ok = [r for r in sub if r["status"] == "GENERATED"]
        ok_in = [r for r in ok if r["within_frozen_bounds"]]
        summary["per_Ma_design"][f"{ma:g}"] = {
            "mach_angle_deg": degrees(asin(1.0 / ma)),
            "generated": len(ok), "failed": sum(1 for r in sub if r["status"] == "GENERATION_FAILED"),
            "generated_within_frozen_bounds": len(ok_in),
            "min_successful_angle_scale": min((r["angle_scale"] for r in ok), default=None),
            "min_successful_min_shock_angle_deg": min((r["min_shock_angle_deg"] for r in ok), default=None),
            "error_identifiers": sorted({r["error_identifier"] for r in sub
                                         if r.get("error_identifier")})}
    dump(out_dir / "feasibility_scan.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
