"""Stage 3: assemble the V5 multimach geometry database V2 from real cases.

Reads every case directory produced by the generation stages, recomputes the
geometric metrics from the stored native surfaces, and writes the database
tables. Nothing is inferred: a geometry appears in the valid inventory only if
the frozen native-mesh audit classified it VALID, and every failure is kept in
failed_candidates.csv with its MATLAB error identifier.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import mmg
from mmg import OLD_AUDIT, OUT, PARAMS, DesignSpace, dump
from geometry_metrics import metrics

INVENTORY_COLUMNS = ["geometry_id", "candidate_id", "Ma_design", "design_space", "source",
                     "classification", "generation_status", "geometry_hash", "mat_sha256",
                     "audit_sha256", "mat_path", "audit_path", "native_geometry_path",
                     *PARAMS, "inputs10", "failure_reason"]
PARAMETER_COLUMNS = ["geometry_id", "Ma_design", "design_space", *PARAMS,
                     "c1", "c2", "AOA_design_prog_deg", "inputs10",
                     "shock_growth_ratio_a2_over_a1", "shock_growth_ratio_b2_over_b1",
                     "classification"]
METRIC_COLUMNS = ["geometry_id", "candidate_id", "Ma_design", "design_space", "classification",
                  "length", "span", "half_span", "aspect_ratio_span_over_length",
                  "planform_area_full", "volume_full", "volumetric_efficiency",
                  "wetted_area_lower_half", "wetted_area_upper_half",
                  "grid_thickness_min", "grid_thickness_max", "grid_thickness_mean",
                  "base_height_at_symmetry", "leading_edge_mean_sweep_deg",
                  "leading_edge_max_gap", "min_section_thickness",
                  "negative_thickness_fraction", "surface_intersection_count",
                  "self_intersection_count", "single_valued_graph", "body_transform_invariant",
                  "length_scale", "volume_closure"]
FAILED_COLUMNS = ["candidate_id", "Ma_design", "design_space", "source", *PARAMS, "inputs10",
                  "generation_status", "classification", "error_identifier", "error_message",
                  "failed_stream_surface", "failure_reason", "generation_elapsed_s"]


def failed_surface(msg):
    import re
    if not msg:
        return None
    m = re.search(r"Stream surface (\d+)", msg)
    return int(m[1]) if m else None


def collect():
    cases = []
    for cand_file in sorted((OUT / "cases").glob("*/candidate.json")):
        case = cand_file.parent
        c = mmg.load(cand_file)
        gs = case / "generation_status.json"
        au = case / "audit.json"
        c["generation_status_data"] = mmg.load(gs) if gs.exists() else None
        c["audit"] = mmg.load(au) if au.exists() else None
        c["case_dir"] = str(case)
        cases.append(c)
    return cases


def main():
    space = DesignSpace()
    space.verify()
    cases = collect()
    db_dir = OUT
    native_dir = OUT / "native_geometries"
    native_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(cases, key=lambda c: (c["Ma_design"], c["candidate_id"]))
    # Geometry ids must never move once published, so earlier assignments are
    # reused and new ones continue above the highest existing index. Morph-path
    # intermediates get their own V2M prefix so the design geometries keep a
    # contiguous V2 numbering.
    registry = {}
    counters = Counter()
    existing = OUT / "geometry_inventory.csv"
    if existing.exists():
        for r in mmg.read_csv(existing):
            if r["classification"] == "VALID" and r["geometry_id"]:
                registry[r["candidate_id"]] = r["geometry_id"]
                prefix, md, idx = r["geometry_id"].rsplit("_", 2)
                counters[(prefix, md)] = max(counters[(prefix, md)], int(idx))
    inventory, params, metric_rows, failed = [], [], [], []
    for c in order:
        au = c["audit"]
        gs = c["generation_status_data"] or {}
        case = Path(c["case_dir"])
        cls = (au or {}).get("classification") or ("GENERATION_FAILED"
                                                   if gs.get("status") != "GENERATED" else None)
        gid = ""
        if cls == "VALID":
            gid = registry.get(c["candidate_id"], "")
            if not gid:
                prefix = "V2M" if c.get("source") == "morph_intermediate" else "V2"
                key = (prefix, f"MD{c['Ma_design']:g}")
                counters[key] += 1
                gid = f"{prefix}_MD{c['Ma_design']:g}_{counters[key]:03d}"
                registry[c["candidate_id"]] = gid
        mat = case / "native_geometry.mat"
        row = {"geometry_id": gid, "candidate_id": c["candidate_id"], "Ma_design": c["Ma_design"],
               "design_space": c.get("design_space"), "source": c.get("source"),
               "classification": cls, "generation_status": gs.get("status"),
               "geometry_hash": (au or {}).get("geometry_hash"),
               "mat_sha256": (au or {}).get("mat_sha256"),
               "audit_sha256": mmg.sha(case / "audit.json") if (case / "audit.json").exists() else None,
               "mat_path": str(mat.relative_to(OUT.parent.parent)) if mat.exists() else None,
               "audit_path": str((case / "audit.json").relative_to(OUT.parent.parent))
                             if (case / "audit.json").exists() else None,
               "native_geometry_path": None,
               **{p: float(c["z"][i]) for i, p in enumerate(PARAMS)},
               "inputs10": json.dumps(c["inputs10"]),
               "failure_reason": (au or {}).get("failure_reason") or gs.get("error_message")}
        if cls == "VALID" and mat.exists():
            target = native_dir / f"{gid}.mat"
            shutil.copy2(mat, target)
            if mmg.sha(target) != mmg.sha(mat):
                raise RuntimeError("NATIVE_GEOMETRY_COPY_MISMATCH:" + gid)
            row["native_geometry_path"] = f"native_geometries/{gid}.mat"
        inventory.append(row)

        if gs.get("status") == "GENERATED" and mat.exists():
            m = metrics(mat)
            metric_rows.append({"geometry_id": gid, "candidate_id": c["candidate_id"],
                                "Ma_design": c["Ma_design"], "design_space": c.get("design_space"),
                                "classification": cls, **m,
                                **{k: (au or {}).get(k) for k in
                                   ("min_section_thickness", "negative_thickness_fraction",
                                    "surface_intersection_count", "self_intersection_count",
                                    "single_valued_graph", "body_transform_invariant",
                                    "length_scale")}})
        if cls == "VALID":
            z = c["z"]
            params.append({"geometry_id": gid, "Ma_design": c["Ma_design"],
                           "design_space": c.get("design_space"),
                           **{p: float(z[i]) for i, p in enumerate(PARAMS)},
                           "c1": space.fixed["c1"], "c2": space.fixed["c2"],
                           "AOA_design_prog_deg": space.fixed["AOA"],
                           "inputs10": json.dumps(c["inputs10"]),
                           "shock_growth_ratio_a2_over_a1": float(z[1] / z[0]),
                           "shock_growth_ratio_b2_over_b1": float(z[3] / z[2]),
                           "classification": cls})
        else:
            failed.append({"candidate_id": c["candidate_id"], "Ma_design": c["Ma_design"],
                           "design_space": c.get("design_space"), "source": c.get("source"),
                           **{p: float(c["z"][i]) for i, p in enumerate(PARAMS)},
                           "inputs10": json.dumps(c["inputs10"]),
                           "generation_status": gs.get("status"), "classification": cls,
                           "error_identifier": gs.get("error_identifier"),
                           "error_message": gs.get("error_message"),
                           "failed_stream_surface": failed_surface(gs.get("error_message")),
                           "failure_reason": (au or {}).get("failure_reason"),
                           "generation_elapsed_s": gs.get("elapsed_s")})

    mmg.write_csv(db_dir / "geometry_inventory.csv", inventory, INVENTORY_COLUMNS)
    valid = [r for r in inventory if r["classification"] == "VALID"]
    mmg.write_csv(db_dir / "valid_geometry_inventory.csv", valid, INVENTORY_COLUMNS)
    mmg.write_csv(db_dir / "geometry_parameters.csv", params, PARAMETER_COLUMNS)
    mmg.write_csv(db_dir / "geometry_metrics.csv", metric_rows, METRIC_COLUMNS)
    mmg.write_csv(db_dir / "failed_candidates.csv", failed, FAILED_COLUMNS)

    dup = Counter(r["geometry_hash"] for r in valid if r["geometry_hash"])
    manifest = {
        "schema": "v5-multimach-geometry-db-2",
        "utc": mmg.now(),
        "database_root": str(OUT),
        "candidate_count": len(inventory),
        "classification_counts": dict(Counter(r["classification"] for r in inventory)),
        "valid_count": len(valid),
        "valid_per_Ma_design": {f"{ma:g}": sum(1 for r in valid if r["Ma_design"] == ma)
                                for ma in sorted({r["Ma_design"] for r in valid})},
        "valid_per_design_space": dict(Counter(r["design_space"] for r in valid)),
        "valid_per_source": dict(Counter(r["source"] for r in valid)),
        "design_geometry_count": sum(1 for r in valid if r["geometry_id"].startswith("V2_")),
        "morph_intermediate_geometry_count": sum(1 for r in valid
                                                 if r["geometry_id"].startswith("V2M_")),
        "duplicate_geometry_hashes": {h: n for h, n in dup.items() if n > 1},
        "unique_valid_geometry_hashes": len(dup),
        "design_space": space.spec(),
        "validation": {
            "rules": "native-triangle-audit-1",
            "rules_sha256": mmg.sha(OLD_AUDIT / "classification_rules.json"),
            "audit_code": {str(p.relative_to(OUT.parent.parent)): mmg.sha(p)
                           for p in sorted(mmg.AUDIT_TOOLS.glob("*.py"))},
            "reused_unmodified": True,
        },
        "geometry_source": {
            "matlab_core": str(mmg.ACTIVE / "build_geometry_core.m"),
            "matlab_core_sha256": mmg.sha(mmg.ACTIVE / "build_geometry_core.m"),
            "batch_wrapper": str(mmg.AUDIT_TOOLS / "run_geometry_audit_batch.m"),
            "batch_wrapper_sha256": mmg.sha(mmg.AUDIT_TOOLS / "run_geometry_audit_batch.m"),
            "path_correction": ("The historical driver tools/geometry_audit/generate.py passes "
                                "teacher_matlab/active, which does not exist in this project "
                                "root. This task's independent layer passes matlab/active. The 16 "
                                "geometry source files are byte-identical between the two trees; "
                                "only the aerodynamic comptue_conicalfield.m differs and it is "
                                "never loaded by the geometry core."),
        },
        "tables": {
            "geometry_inventory.csv": "every candidate, every classification",
            "valid_geometry_inventory.csv": "VALID geometries only, with native_geometries path",
            "geometry_parameters.csv": "design vector and fixed inputs of the VALID geometries",
            "geometry_metrics.csv": "mesh-derived geometric metrics of every generated geometry",
            "failed_candidates.csv": "all non-VALID candidates with their generator error",
            "native_geometries/": "copy of the MATLAB native surface .mat of each VALID geometry",
        },
        "not_computed": ["aerodynamic coefficients", "trajectory or OCP quantities",
                         "DQN training or policy quantities", "flight-Mach tables"],
        "g42_regression": mmg.load(OUT / "g42_regression.json")["GEOMETRY_REGRESSION"]
        if (OUT / "g42_regression.json").exists() else "NOT_RUN",
    }
    dump(db_dir / "geometry_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("candidate_count", "classification_counts",
                                               "valid_count", "valid_per_Ma_design",
                                               "valid_per_design_space",
                                               "unique_valid_geometry_hashes",
                                               "duplicate_geometry_hashes", "g42_regression")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
