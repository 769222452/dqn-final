"""Stage 0b: wide design-Mach / design-space feasibility evidence.

Three independent probe sets, all geometry-only, all through the confirmed
active build_geometry_core:

  A  the 26 frozen VALID Pareto seeds at Ma_design 4, 6, 8
  B  a design-Mach boundary search for the pilot seeds between 8 and 15
  C  a deterministic Sobol + corner sample of the frozen 6-D box at Ma_design 4 and 8

Set A answers "do historically valid designs survive a lower design Mach".
Set B answers "what is the lowest design Mach this generator can actually do".
Set C answers "is the failure specific to the seeds or to the whole box".
"""
from __future__ import annotations

import json
import subprocess
import time
from math import asin, degrees
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import mmg
from mmg import ACTIVE, MATLAB, OUT, PARAMS, DesignSpace, dump, valid_seeds
from stage1_pilot import pick_pilot_seeds

MA_A = (4.0, 6.0, 8.0)
MA_B = (8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
MA_C = (4.0, 8.0)
SOBOL_M = 5  # 32 points
HERE = Path(__file__).resolve().parent


def sobol_sample(space, n):
    s = qmc.Sobol(d=6, scramble=True, seed=20261009)
    pts = s.random_base2(m=int(np.log2(n)))
    return qmc.scale(pts, space.lower, space.upper)


def corner_sample(space):
    """16 corners of the four shock angles with d1 and a_x at their midpoints."""
    mid_d1 = (space.lower[4] + space.upper[4]) / 2
    mid_ax = (space.lower[5] + space.upper[5]) / 2
    out = []
    for bits in range(16):
        z = [space.upper[i] if (bits >> i) & 1 else space.lower[i] for i in range(4)]
        out.append(z + [mid_d1, mid_ax])
    return np.asarray(out)


def main():
    space = DesignSpace()
    space.verify()
    out_dir = OUT / "stage0_feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = valid_seeds()
    pilot = pick_pilot_seeds(seeds)

    jobs, meta = [], []

    def add(pid, ma, z, kind, note=""):
        z = [float(v) for v in z]
        jobs.append({"probe_id": pid, "inputs10": space.inputs(ma, z)})
        meta.append({"probe_id": pid, "set": kind, "Ma_design": float(ma), "note": note,
                     "z": z, "within_frozen_bounds": space.in_bounds(z),
                     "mach_angle_deg": degrees(asin(1.0 / ma)),
                     "min_shock_angle_deg": min(z[:4]), "max_shock_angle_deg": max(z[:4])})

    for s in seeds:
        for ma in MA_A:
            add(f"A_MD{ma:g}_D{s['design_index']:02d}", ma, s["z"], "A",
                f"frozen VALID seed design_{s['design_index']:02d}")
    for s in pilot:
        for ma in MA_B:
            add(f"B_MD{ma:g}_D{s['design_index']:02d}", ma, s["z"], "B",
                f"design-Mach boundary search, seed design_{s['design_index']:02d}")
    sob = sobol_sample(space, 2 ** SOBOL_M)
    cor = corner_sample(space)
    for ma in MA_C:
        for i, z in enumerate(sob):
            add(f"C_MD{ma:g}_S{i:02d}", ma, z, "C", "sobol sample of frozen box")
        for i, z in enumerate(cor):
            add(f"C_MD{ma:g}_K{i:02d}", ma, z, "C", "shock-angle box corner")

    job_file = out_dir / "probe_jobs_b.json"
    result_file = out_dir / "probe_results_b.json"
    dump(job_file, jobs)
    dump(out_dir / "probe_plan_b.json",
         {"utc": mmg.now(), "sets": {"A": len(MA_A) * len(seeds), "B": len(MA_B) * len(pilot),
                                     "C": len(MA_C) * (len(sob) + len(cor))},
          "total": len(jobs), "frozen_bounds": {"lower": space.lower.tolist(),
                                                "upper": space.upper.tolist()},
          "probes": meta})

    code = (f"addpath('{HERE.as_posix()}'); "
            f"probe_feasibility('{job_file.as_posix()}','{ACTIVE.as_posix()}','{result_file.as_posix()}');")
    log = (out_dir / "matlab_probe_b.log").open("w", encoding="utf-8")
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
        rows.append({**{k: m[k] for k in ("probe_id", "set", "Ma_design", "note",
                                          "within_frozen_bounds", "mach_angle_deg",
                                          "min_shock_angle_deg", "max_shock_angle_deg")},
                     **{p: m["z"][i] for i, p in enumerate(PARAMS)},
                     "status": r.get("status"), "error_identifier": r.get("error_identifier"),
                     "error_message": r.get("error_message"),
                     "error_stack_top": r.get("error_stack_top"),
                     "failed_stream_surface": failed_surface(r.get("error_message")),
                     "elapsed_s": r.get("elapsed_s"), "length": r.get("length"),
                     "half_span": r.get("half_span"), "nonfinite_count": r.get("nonfinite_count")})
    mmg.write_csv(out_dir / "feasibility_scan_b.csv", rows)

    summary = {"utc": mmg.now(), "wall_s": wall, "returncode": proc.returncode,
               "probe_count": len(meta), "completed": len(results), "per_set": {}}
    for kind in ("A", "B", "C"):
        sub = [r for r in rows if r["set"] == kind]
        per_ma = {}
        for ma in sorted({r["Ma_design"] for r in sub}):
            s2 = [r for r in sub if r["Ma_design"] == ma]
            gen = [r for r in s2 if r["status"] == "GENERATED"]
            fs = [r["failed_stream_surface"] for r in s2 if r["failed_stream_surface"]]
            per_ma[f"{ma:g}"] = {
                "n": len(s2), "generated": len(gen),
                "failed": sum(1 for r in s2 if r["status"] == "GENERATION_FAILED"),
                "mach_angle_deg": degrees(asin(1.0 / ma)),
                "failed_stream_surface_min": min(fs) if fs else None,
                "failed_stream_surface_max": max(fs) if fs else None,
                "error_messages": sorted({(r["error_message"] or "").split(":", 1)[-1].strip()
                                          for r in s2 if r["status"] != "GENERATED"})}
        summary["per_set"][kind] = per_ma
    dump(out_dir / "feasibility_scan_b.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def failed_surface(msg):
    if not msg:
        return None
    import re
    m = re.search(r"Stream surface (\d+)", msg)
    return int(m[1]) if m else None


if __name__ == "__main__":
    main()
