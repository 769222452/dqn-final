"""Stage 0c: can leaving the frozen shock-angle box buy a low design Mach?

Set D  uniform shock angle A (8..35 deg) at Ma_design 4,5,6,7,8,10,12
       Angles above 8 deg are outside the frozen optimizer bounds and are
       recorded as EXTENDED, never as part of the historical design space.
Set E  extended-angle families crossed with the leading-edge Bezier parameter
       d1, plus the historical G42 angle set scaled up with its growth
       character preserved.
Set F  the same deterministic Sobol + corner sample of the frozen box that
       stage 0b ran at Ma_design 4 and 8, now at Ma_design 5, 6 and 7. Stage 0b
       showed the analytic attached-shock gate is necessary but not sufficient,
       so the frozen box at these design Machs has to be probed directly.

Everything is geometry-only through the confirmed active build_geometry_core.
"""
from __future__ import annotations

import json
import subprocess
import time
from math import asin, degrees
from pathlib import Path

import mmg
import shock_angle_model as sam
from mmg import ACTIVE, G42_Z, MATLAB, OUT, PARAMS, DesignSpace, dump

ANGLES = (8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0, 35.0)
MA_D = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
D1_SET = (0.02, 0.11606537709080691, 0.18)
HERE = Path(__file__).resolve().parent


def main():
    space = DesignSpace()
    space.verify()
    out_dir = OUT / "stage0_feasibility"
    ax = G42_Z[5]
    jobs, meta = [], []

    def add(pid, ma, z, kind, note=""):
        z = [float(v) for v in z]
        jobs.append({"probe_id": pid, "inputs10": space.inputs(ma, z)})
        meta.append({"probe_id": pid, "set": kind, "Ma_design": float(ma), "note": note, "z": z,
                     "within_frozen_bounds": space.in_bounds(z),
                     "design_space": "FROZEN" if space.in_bounds(z) else "EXTENDED",
                     "mach_angle_deg": degrees(asin(1.0 / ma)),
                     "uniform_shock_angle_deg": z[0] if len(set(z[:4])) == 1 else None,
                     "model_min_required_Ma": sam.min_required_mach(z),
                     "model_beta_min_deg": float(min(sam.leading_edge_shock_angles(z)["beta_deg"])),
                     "model_gate_satisfied": sam.predicted_first_failure(z, ma) is None})

    for a in ANGLES:
        for d1 in (D1_SET[1],):
            for ma in MA_D:
                add(f"D_A{a:g}_MD{ma:g}", ma, [a, a, a, a, d1, ax], "D",
                    "uniform shock angle sweep")
    for a in (12.0, 18.0, 26.0):
        for d1 in (D1_SET[0], D1_SET[2]):
            for ma in (4.0, 6.0, 8.0):
                add(f"E_A{a:g}_D1{d1:g}_MD{ma:g}", ma, [a, a, a, a, d1, ax], "E",
                    "uniform angle x leading-edge Bezier d1")
    # Keep the a2>a1 / b2>b1 growth character of the historical designs while
    # lifting the whole angle set out of the frozen box.
    for f in (2.0, 3.0, 4.0, 5.0, 6.0):
        for ma in (4.0, 6.0, 8.0, 10.0, 12.0):
            z = [G42_Z[0] * f, G42_Z[1] * f, G42_Z[2] * f, G42_Z[3] * f, G42_Z[4], G42_Z[5]]
            add(f"E_G42x{f:g}_MD{ma:g}", ma, z, "E", "G42 angle set scaled, growth preserved")
    from stage0b_scan import SOBOL_M, corner_sample, sobol_sample
    sob = sobol_sample(space, 2 ** SOBOL_M)
    cor = corner_sample(space)
    for ma in (5.0, 6.0, 7.0):
        for i, z in enumerate(sob):
            add(f"F_MD{ma:g}_S{i:02d}", ma, z, "F", "sobol sample of frozen box")
        for i, z in enumerate(cor):
            add(f"F_MD{ma:g}_K{i:02d}", ma, z, "F", "shock-angle box corner")

    job_file = out_dir / "probe_jobs_c.json"
    result_file = out_dir / "probe_results_c.json"
    dump(job_file, jobs)
    dump(out_dir / "probe_plan_c.json", {"utc": mmg.now(), "total": len(jobs), "probes": meta})

    code = (f"addpath('{HERE.as_posix()}'); "
            f"probe_feasibility('{job_file.as_posix()}','{ACTIVE.as_posix()}','{result_file.as_posix()}');")
    log = (out_dir / "matlab_probe_c.log").open("w", encoding="utf-8")
    start = time.monotonic()
    proc = subprocess.Popen([str(MATLAB), "-batch", code], cwd=str(out_dir),
                            stdout=log, stderr=subprocess.STDOUT)
    proc.wait()
    log.close()

    results = mmg.load(result_file)
    by_id = {r["probe_id"]: r for r in results}
    from stage0b_scan import failed_surface
    rows = []
    for m in meta:
        r = by_id.get(m["probe_id"], {})
        rows.append({**{k: m[k] for k in ("probe_id", "set", "Ma_design", "note", "design_space",
                                          "within_frozen_bounds", "mach_angle_deg",
                                          "uniform_shock_angle_deg", "model_min_required_Ma",
                                          "model_beta_min_deg", "model_gate_satisfied")},
                     **{p: m["z"][i] for i, p in enumerate(PARAMS)},
                     "status": r.get("status"), "error_identifier": r.get("error_identifier"),
                     "error_message": r.get("error_message"),
                     "failed_stream_surface": failed_surface(r.get("error_message")),
                     "elapsed_s": r.get("elapsed_s"), "length": r.get("length"),
                     "half_span": r.get("half_span")})
    mmg.write_csv(out_dir / "feasibility_scan_c.csv", rows)
    gen = [r for r in rows if r["status"] == "GENERATED"]
    summary = {"utc": mmg.now(), "wall_s": time.monotonic() - start, "returncode": proc.returncode,
               "probe_count": len(meta), "completed": len(results),
               "generated": len(gen),
               "generated_ids": [r["probe_id"] for r in gen],
               "lowest_Ma_design_generated": min((r["Ma_design"] for r in gen), default=None),
               "per_Ma_design": {}}
    for ma in sorted({r["Ma_design"] for r in rows}):
        sub = [r for r in rows if r["Ma_design"] == ma]
        g = [r for r in sub if r["status"] == "GENERATED"]
        summary["per_Ma_design"][f"{ma:g}"] = {
            "n": len(sub), "generated": len(g),
            "generated_ids": [r["probe_id"] for r in g],
            "error_messages": sorted({(r["error_message"] or "").split(":", 1)[-1].strip()
                                      for r in sub if r["status"] != "GENERATED"})}
    dump(out_dir / "feasibility_scan_c.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
