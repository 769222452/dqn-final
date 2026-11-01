"""Stage 1: G42@Md15 regression + real-geometry pilot in the measured feasible region.

Nothing downstream runs unless the G42 regression reproduces the frozen
geometry hash exactly. The pilot then generates and fully audits real native
geometries at every target design Mach, using uniform shock-angle families
that stage 0 proved generable, so that the VALID rate (not just the generation
rate) is known before the main sampling budget is committed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import mmg
from mmg import (G42_DESIGN_INDEX, G42_GEOMETRY_HASH, G42_MAT_SHA256, G42_Z, OLD_AUDIT, OUT,
                 PARAMS, DesignSpace, dump, valid_seeds)

# Uniform shock-angle families proved generable by stage 0 for each design Mach.
PILOT_ANGLES = {4.0: (12.0, 15.0, 18.0, 20.0),
                5.0: (10.0, 12.0, 15.0, 18.0),
                6.0: (8.0, 10.0, 12.0, 15.0),
                7.0: (8.0, 10.0, 12.0, 15.0),
                8.0: (8.0, 10.0, 12.0, 15.0),
                15.0: (8.0, 12.0)}
PILOT_D1 = 0.11606537709080691
PILOT_AX = 0.9944782795122264


def build_candidates(space):
    cands = [{"candidate_id": "REG_G42_MD15", "Ma_design": space.historical_ma_design,
              "z": list(G42_Z), "source": "regression", "seed_design_index": G42_DESIGN_INDEX,
              "path_id": None, "path_k": None}]
    for ma, angles in PILOT_ANGLES.items():
        for a in angles:
            z = [a, a, a, a, PILOT_D1, PILOT_AX]
            cands.append({"candidate_id": f"PILOT_MD{ma:g}_A{a:g}", "Ma_design": ma, "z": z,
                          "source": "pilot_uniform_angle", "seed_design_index": None,
                          "path_id": None, "path_k": None})
    for i, c in enumerate(cands, start=1):
        c["job_index"] = i
        c["design_space"] = space.label(c["z"])
        if c["design_space"] == "OUT_OF_RANGE":
            raise RuntimeError("DESIGN_PARAMETERS_OUT_OF_RANGE:" + c["candidate_id"])
        c["case_dir"] = str(OUT / "cases" / c["candidate_id"])
        c["inputs10"] = space.inputs(c["Ma_design"], c["z"])
    return cands


def regression_report(candidate, result):
    old_mat = OLD_AUDIT / "cases" / f"design_{G42_DESIGN_INDEX:02d}" / "native_geometry.mat"
    new_mat = Path(candidate["case_dir"]) / "native_geometry.mat"
    rep = {"candidate_id": candidate["candidate_id"], "inputs10": candidate["inputs10"],
           "reference_mat": str(old_mat), "reference_mat_sha256_recorded": G42_MAT_SHA256,
           "reference_mat_sha256_now": mmg.sha(old_mat),
           "reference_geometry_hash": G42_GEOMETRY_HASH,
           "regenerated_geometry_hash": result.get("geometry_hash"),
           "regenerated_mat_sha256": result.get("mat_sha256"),
           "generation_status": result.get("generation_status"),
           "classification": result.get("classification"),
           "classification_reference": "VALID (frozen audit design_42)"}
    if new_mat.exists():
        a, b = loadmat(old_mat), loadmat(new_mat)
        diffs = {}
        for k in ("wav_x", "wav_y", "wav_z", "wav_up_x", "wav_up_y", "wav_up_z"):
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            diffs[k] = {"shape_reference": list(x.shape), "shape_new": list(y.shape),
                        "max_abs_diff": float(np.abs(x - y).max()) if x.shape == y.shape else None}
        rep["coordinate_diff"] = diffs
        rep["max_abs_coordinate_diff"] = max(
            (v["max_abs_diff"] for v in diffs.values() if v["max_abs_diff"] is not None), default=None)
    rep["geometry_hash_identical"] = bool(result.get("geometry_hash") == G42_GEOMETRY_HASH)
    rep["mat_container_sha256_identical"] = bool(result.get("mat_sha256") == G42_MAT_SHA256)
    rep["GEOMETRY_REGRESSION"] = "PASS" if rep["geometry_hash_identical"] else "FAIL"
    rep["note"] = ("geometry_hash covers inputs10 plus the float64 native surface coordinates and "
                   "is the binding identity. The .mat container sha256 also covers MATLAB's file "
                   "header and is not expected to reproduce across sessions.")
    return rep


def main():
    space = DesignSpace()
    space.verify()
    dump(OUT / "provenance" / "design_space.json", space.spec())
    dump(OUT / "provenance" / "frozen_valid_seeds.json",
         [dict(s, z=list(s["z"])) for s in valid_seeds()])

    cands = build_candidates(space)
    dump(OUT / "generation" / "stage1_candidates.json", cands)
    mmg.generate(space, cands, "stage1")
    results = mmg.audit_candidates(cands)
    by_id = {r["candidate_id"]: r for r in results}

    reg = regression_report(cands[0], by_id["REG_G42_MD15"])
    dump(OUT / "g42_regression.json", reg)
    print(json.dumps({k: reg[k] for k in ("GEOMETRY_REGRESSION", "geometry_hash_identical",
                                          "max_abs_coordinate_diff", "classification")},
                     ensure_ascii=False))

    rows = [mmg.audit_row(r) for r in results]
    mmg.write_csv(OUT / "stage1_pilot.csv", rows, mmg.AUDIT_COLUMNS)
    pilot = [r for r in results if r["source"] == "pilot_uniform_angle"]
    summary = {"utc": mmg.now(),
               "regression": {k: reg[k] for k in ("GEOMETRY_REGRESSION", "geometry_hash_identical",
                                                  "max_abs_coordinate_diff")},
               "pilot_count": len(pilot),
               "classification_counts": dict(Counter(r["classification"] for r in pilot)),
               "per_Ma_design": {}}
    for ma in sorted({r["Ma_design"] for r in pilot}):
        sub = [r for r in pilot if r["Ma_design"] == ma]
        summary["per_Ma_design"][f"{ma:g}"] = {
            "n": len(sub), "counts": dict(Counter(r["classification"] for r in sub)),
            "by_angle": {r["candidate_id"].split("_A")[-1]:
                         {"classification": r["classification"],
                          "min_section_thickness": r.get("min_section_thickness"),
                          "surface_intersection_count": r.get("surface_intersection_count"),
                          "failure_reason": r.get("failure_reason")} for r in sub}}
    dump(OUT / "stage1_pilot.json", summary)
    print(json.dumps(summary["per_Ma_design"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
