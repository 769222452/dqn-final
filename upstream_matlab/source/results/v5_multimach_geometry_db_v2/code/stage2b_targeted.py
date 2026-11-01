"""Stage 2b: targeted sampling inside the measured generable-and-valid region.

Wave 1 (294 real candidates, 84 generated, 24 VALID) produced two measured
rules that this wave exploits:

  * generation needs a root shock angle above a design-Mach dependent floor
    (about 12 deg at Ma 4, 10 deg at Ma 5, 8 deg at Ma 6-7, 3-5 deg at Ma 8+);
  * validity needs a small leading-edge Bezier control point: every one of the
    24 VALID geometries has d1 <= 0.109, and identical angle sets flip from
    VALID at d1 = 0.02 to INVALID at d1 = 0.116.

The windows below are therefore per design Mach. They are search windows, not
a redefinition of the design space: the recorded bounds stay the frozen
optimizer box plus the declared extended shock-angle upper bound, and every
record still carries design_space = FROZEN or EXTENDED.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from scipy.stats import qmc

import mmg
from mmg import OUT, PARAMS, DesignSpace, dump

# per design Mach: (A, g, B, h, d1, a_x) lower / upper
WINDOWS = {
    4.0: ([12.0, 0.95, 12.0, 0.90, 0.020, 0.90], [20.0, 1.25, 20.0, 1.10, 0.090, 1.10]),
    5.0: ([10.0, 0.95, 9.0, 0.85, 0.020, 0.90], [20.0, 1.40, 20.0, 1.15, 0.090, 1.10]),
    6.0: ([8.0, 0.95, 6.0, 0.85, 0.020, 0.90], [20.0, 1.50, 20.0, 1.20, 0.100, 1.10]),
    7.0: ([5.0, 0.95, 4.0, 0.80, 0.020, 0.90], [20.0, 1.80, 20.0, 1.30, 0.100, 1.10]),
    8.0: ([3.0, 0.95, 3.0, 0.80, 0.020, 0.90], [20.0, 2.10, 20.0, 1.70, 0.110, 1.10]),
    15.0: ([3.0, 0.95, 3.0, 0.80, 0.020, 0.90], [20.0, 2.10, 20.0, 1.70, 0.120, 1.10]),
}


def family_to_z(p):
    a, g, b, h, d1, ax = p
    return [a, a * g, b, b * h, d1, ax]


def sample(space, ma, n, seed):
    lo, hi = (np.asarray(v, float) for v in WINDOWS[ma])
    eng = qmc.Sobol(d=6, scramble=True, seed=seed)
    out = []
    while len(out) < n:
        for p in qmc.scale(eng.random(max(n, 8)), lo, hi):
            z = family_to_z(p)
            if space.label(z) != "OUT_OF_RANGE":
                out.append((z, p))
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2)
    ap.add_argument("--per-mach", type=int, default=48)
    ap.add_argument("--reference-count", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    space = DesignSpace()
    space.verify()
    stage = f"stage2_wave{args.wave}"
    seen = set()
    for p in (OUT / "cases").glob("*/candidate.json"):
        c = mmg.load(p)
        seen.add((float(c["Ma_design"]), tuple(np.round(c["z"], 12))))

    cands = []
    for ma, (lo, hi) in WINDOWS.items():
        n = args.reference_count if ma == 15.0 else args.per_mach
        for z, p in sample(space, ma, n, seed=31337 + args.wave * 101 + int(ma * 10)):
            key = (ma, tuple(np.round(z, 12)))
            if key in seen:
                continue
            seen.add(key)
            cands.append({"candidate_id": mmg.candidate_id(ma, z, prefix=f"W{args.wave}"),
                          "Ma_design": ma, "z": z, "source": f"targeted_window_wave{args.wave}",
                          "seed_design_index": None, "path_id": None, "path_k": None,
                          "family_params": {"A": p[0], "g": p[1], "B": p[2], "h": p[3],
                                            "d1": p[4], "a_x": p[5]}})
    for i, c in enumerate(cands, start=1):
        c["job_index"] = i
        c["design_space"] = space.label(c["z"])
        c["case_dir"] = str(OUT / "cases" / c["candidate_id"])
        c["inputs10"] = space.inputs(c["Ma_design"], c["z"])
    dump(OUT / "generation" / f"{stage}_candidates.json", cands)
    dump(OUT / "generation" / f"{stage}_windows.json",
         {"utc": mmg.now(), "family": ["A", "g", "B", "h", "d1", "a_x"],
          "z_from_family": "z = (A, A*g, B, B*h, d1, a_x)",
          "windows": {f"{k:g}": {"lower": v[0], "upper": v[1]} for k, v in WINDOWS.items()},
          "basis": "wave 1 measurement: generation angle floor per design Mach, validity d1 <= 0.109"})
    print(f"[{stage}] {len(cands)} candidates {dict(Counter(c['Ma_design'] for c in cands))}",
          flush=True)

    mmg.generate(space, cands, stage)
    results = mmg.audit_parallel(cands, workers=args.workers)
    rows = [mmg.audit_row(r) for r in results]
    mmg.write_csv(OUT / "generation" / f"{stage}_audit.csv", rows, mmg.AUDIT_COLUMNS)

    gen_status = {}
    for c in cands:
        p = mmg.Path(c["case_dir"]) / "generation_status.json"
        gen_status[c["candidate_id"]] = mmg.load(p)["status"] if p.exists() else "NOT_RUN"
    summary = {"utc": mmg.now(), "stage": stage, "candidates": len(cands),
               "generation_counts": dict(Counter(gen_status.values())),
               "audit_counts": dict(Counter(r["classification"] for r in results)),
               "per_Ma_design": {}}
    for ma in WINDOWS:
        sub = [r for r in results if r["Ma_design"] == ma]
        summary["per_Ma_design"][f"{ma:g}"] = {
            "candidates": sum(1 for c in cands if c["Ma_design"] == ma),
            "counts": dict(Counter(r["classification"] for r in sub)),
            "valid_ids": [r["candidate_id"] for r in sub if r["classification"] == "VALID"]}
    dump(OUT / "generation" / f"{stage}_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_Ma_design"},
                     ensure_ascii=False))
    print(json.dumps({k: v["counts"] for k, v in summary["per_Ma_design"].items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
