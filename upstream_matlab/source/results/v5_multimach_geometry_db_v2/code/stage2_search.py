"""Stage 2: search for geometries that both generate and pass the frozen audit.

Stage 0 and stage 1 established the trade-off that governs this database:

  * a low design Mach needs large, nearly uniform shock angles, otherwise the
    osculating MOC characteristic net folds (V5:GeometryDomain);
  * a valid closed body needs streamwise growth of the shock cross-section
    (a2 > a1), otherwise the upper free-stream surface cuts through the lower
    MOC surface (audit INVALID, ~200 intersecting triangle pairs).

The candidates are therefore drawn from a growth-parameterised family

    z = (A, A*g, B, B*h, d1, a_x)

with A, B the root shock angles and g, h the streamwise growth ratios, which
spans both extremes continuously. Every candidate is generated for real and
audited with the unmodified frozen rules; failures are kept, never discarded.

    python stage2_search.py --wave 1 --per-mach 40
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from scipy.stats import qmc

import mmg
from mmg import MA_DESIGN_GRID, MA_DESIGN_REFERENCE, OUT, PARAMS, DesignSpace, dump

# (A, g, B, h, d1, a_x)
FAMILY_LOWER = np.array([3.0, 0.90, 3.0, 0.70, 0.02, 0.90])
FAMILY_UPPER = np.array([20.0, 1.80, 20.0, 1.30, 0.18, 1.10])
MA_TARGETS = tuple(MA_DESIGN_GRID) + (MA_DESIGN_REFERENCE,)


def family_to_z(p):
    a, g, b, h, d1, ax = p
    return [a, a * g, b, b * h, d1, ax]


def sample(space, n, seed):
    """Deterministic Sobol sample of the family, rejected onto the extended box."""
    eng = qmc.Sobol(d=6, scramble=True, seed=seed)
    out, tries = [], 0
    while len(out) < n and tries < 200:
        tries += 1
        pts = qmc.scale(eng.random(n), FAMILY_LOWER, FAMILY_UPPER)
        for p in pts:
            z = family_to_z(p)
            if space.label(z) != "OUT_OF_RANGE":
                out.append((z, p))
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError("SAMPLER_COULD_NOT_FILL_WAVE")
    return out


def anchors(space):
    """Points stage 0 proved generable at a low design Mach, re-run here with a
    stored native geometry so that their audit classification is known."""
    out = []
    rows = mmg.read_csv(OUT / "stage0_feasibility" / "feasibility_scan_b.csv")
    rows += mmg.read_csv(OUT / "stage0_feasibility" / "feasibility_scan_c.csv")
    for r in rows:
        if r["status"] != "GENERATED":
            continue
        ma = float(r["Ma_design"])
        if ma not in MA_TARGETS:
            continue
        z = [float(r[p]) for p in PARAMS]
        if space.label(z) == "OUT_OF_RANGE":
            continue
        out.append((ma, z, r["probe_id"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--per-mach", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    space = DesignSpace()
    space.verify()
    stage = f"stage2_wave{args.wave}"
    cands, used = [], set()

    if args.wave == 1:
        for ma, z, pid in anchors(space):
            key = (ma, tuple(np.round(z, 12)))
            if key in used:
                continue
            used.add(key)
            cands.append({"candidate_id": mmg.candidate_id(ma, z, prefix="ANCH"), "Ma_design": ma,
                          "z": z, "source": "stage0_generable_anchor", "seed_design_index": None,
                          "path_id": None, "path_k": None, "origin_probe_id": pid})
    for ma in MA_TARGETS:
        for i, (z, p) in enumerate(sample(space, args.per_mach, seed=20261009 + args.wave * 97)):
            key = (ma, tuple(np.round(z, 12)))
            if key in used:
                continue
            used.add(key)
            cands.append({"candidate_id": mmg.candidate_id(ma, z, prefix=f"W{args.wave}"),
                          "Ma_design": ma, "z": z, "source": f"family_sobol_wave{args.wave}",
                          "seed_design_index": None, "path_id": None, "path_k": None,
                          "family_params": {"A": p[0], "g": p[1], "B": p[2], "h": p[3],
                                            "d1": p[4], "a_x": p[5]}})
    for i, c in enumerate(cands, start=1):
        c["job_index"] = i
        c["design_space"] = space.label(c["z"])
        c["case_dir"] = str(OUT / "cases" / c["candidate_id"])
        c["inputs10"] = space.inputs(c["Ma_design"], c["z"])
    dump(OUT / "generation" / f"{stage}_candidates.json", cands)
    print(f"[{stage}] {len(cands)} candidates "
          f"({dict(Counter(c['Ma_design'] for c in cands))})", flush=True)

    if not args.audit_only:
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
    for ma in MA_TARGETS:
        sub = [r for r in results if r["Ma_design"] == ma]
        sub_c = [c for c in cands if c["Ma_design"] == ma]
        valid = [r for r in sub if r["classification"] == "VALID"]
        summary["per_Ma_design"][f"{ma:g}"] = {
            "candidates": len(sub_c),
            "generated": sum(1 for c in sub_c if gen_status[c["candidate_id"]] == "GENERATED"),
            "audited": len(sub), "counts": dict(Counter(r["classification"] for r in sub)),
            "valid_ids": [r["candidate_id"] for r in valid]}
    dump(OUT / "generation" / f"{stage}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
