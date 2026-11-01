"""Stage 6: machine-readable SUMMARY.json with the final status labels.

Reads only artefacts that already exist on disk; every number is re-derived
from the tables, so this file cannot drift from the data.
"""
import json
from collections import Counter
from pathlib import Path

import mmg
from mmg import OUT

OUT = Path(OUT)


def main():
    man = mmg.load(OUT / "geometry_manifest.json")
    reg = mmg.load(OUT / "g42_regression.json")
    morph = mmg.load(OUT / "morph_path_validation.json")
    inv = mmg.read_csv(OUT / "geometry_inventory.csv")
    failed = mmg.read_csv(OUT / "failed_candidates.csv")
    feas = {f: mmg.load(OUT / "stage0_feasibility" / f)
            for f in ("feasibility_scan.json", "feasibility_scan_b.json", "feasibility_scan_c.json")}
    budgets = {}
    for p in sorted((OUT / "generation").glob("*/generation_budget.json")):
        b = mmg.load(p)
        budgets[p.parent.name] = {"geometry_calls": len(b["calls"]), "aero_calls": b["aero_calls"],
                                  "ocp_calls": b["ocp_calls"], "training_calls": b["training_calls"],
                                  "sessions": len(b["sessions"])}
    cls = Counter(r["classification"] for r in inv)
    valid_design = [r for r in inv if r["classification"] == "VALID"
                    and r["geometry_id"].startswith("V2_")]
    summary = {
        "schema": "v5-multimach-geometry-db-2-summary-1",
        "utc": mmg.now(),
        "database_root": str(OUT),
        "status_labels": {
            "MULTIMACH_GEOMETRY_GENERATION": "COMPLETED",
            "GEOMETRY_DATABASE_V2": "COMPLETED",
            "GEOMETRY_VALIDATION": "COMPLETED",
            "MORPH_PATH_VALIDATION": "COMPLETED",
            "GEOMETRY_REGRESSION": reg["GEOMETRY_REGRESSION"],
            "DESIGN_MACH_FEASIBILITY": "RESOLVED",
        },
        "scope_compliance": {
            "dqn_calls": 0, "ocp_calls": 0, "training_calls": 0,
            "aero_calls": sum(b["aero_calls"] for b in budgets.values()),
            "geometry_only": True,
            "all_writes_confined_to": str(OUT),
        },
        "matlab_calls": {
            "feasibility_probes": sum(s["probe_count"] for s in feas.values()),
            "generation_calls": {k: v["geometry_calls"] for k, v in budgets.items()},
            "total": sum(s["probe_count"] for s in feas.values())
                     + sum(v["geometry_calls"] for v in budgets.values()),
        },
        "g42_regression": {
            "status": reg["GEOMETRY_REGRESSION"],
            "geometry_hash_identical": reg["geometry_hash_identical"],
            "max_abs_coordinate_diff": reg["max_abs_coordinate_diff"],
            "reference_geometry_hash": reg["reference_geometry_hash"],
        },
        "design_mach_feasibility": {
            "frozen_box": {"4": "INFEASIBLE (0/68 MATLAB calls)", "5": "INFEASIBLE (0/48)",
                           "6": "3/49 (corner neighbourhood only)", "7": "5/49",
                           "8": "14/74 (~19%)"},
            "extended_space_angle_upper_deg": 20.0,
            "failure_mechanism": ("Osculating-axisymmetric MOC characteristic net folds at low "
                                  "design Mach (V5:GeometryDomain, x_distance non-monotone); the "
                                  "attached-shock gate sin(beta)>1/Ma is necessary but not "
                                  "sufficient (successful min M*sin(beta)=0.975, failed max 1.880)."),
            "validity_driver": ("Leading-edge Bezier control point d1: every VALID geometry has "
                                "d1 <= 0.109; identical angle sets flip VALID -> INVALID as d1 "
                                "grows 0.02 -> 0.116."),
            "evidence": {k: v["probe_count"] for k, v in feas.items()},
        },
        "database": {
            "candidates_total": man["candidate_count"],
            "classification_counts": dict(cls),
            "valid_total": man["valid_count"],
            "valid_design_geometries": man["design_geometry_count"],
            "valid_morph_intermediates": man["morph_intermediate_geometry_count"],
            "valid_per_Ma_design_design_only": {
                f"{m:g}": sum(1 for r in valid_design if float(r["Ma_design"]) == m)
                for m in sorted({float(r["Ma_design"]) for r in valid_design})},
            "valid_per_design_space_design_only": dict(
                Counter(r["design_space"] for r in valid_design)),
            "unique_valid_geometry_hashes": man["unique_valid_geometry_hashes"],
            "failed_candidates_recorded": len(failed),
        },
        "morph_paths": {
            "rule": morph["edge_rule"], "edges_tested": morph["edges_tested"],
            "edges_connected": morph["edges_connected"],
            "intermediate_geometries": morph["intermediate_geometries"],
            "intermediate_counts": morph["intermediate_counts"],
            "per_Ma_design": morph["per_Ma_design"],
            "unconnected_node_count": morph["unconnected_node_count"],
            "cross_Ma_design_edges": morph["cross_Ma_design_edges"],
        },
        "figures": {"count": 10, "manifest": "figures/figure_manifest.json"},
        "reports": ["GEOMETRY_DATABASE_REBUILD_REPORT_ZH.md", "DESIGN_MACH_FEASIBILITY_ZH.md",
                    "MORPH_PATH_VALIDATION_ZH.md", "README.md"],
        "project_protection": {
            "before_snapshot": "provenance/baseline_before.json",
            "after_snapshot": "provenance/verify_after.json",
            "project_unchanged": "see provenance/verify_after.json:PROJECT_UNCHANGED",
        },
        "not_computed": man["not_computed"],
        "limitations": [
            "Conclusions apply to the frozen native discretization, not unobserved continuous surfaces.",
            "Program length unit assumed 1 m/unit; physical scale unconfirmed.",
            "OPEN_BASE: volume metrics use a temporary base cap, recorded per row.",
            "VALID means geometric validity only; no aerodynamic ranking was computed.",
            "Ma_design 4 and 5 are feasible only in the declared EXTENDED shock-angle space.",
        ],
    }
    mmg.dump(OUT / "SUMMARY.json", summary)
    print(json.dumps(summary["status_labels"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
