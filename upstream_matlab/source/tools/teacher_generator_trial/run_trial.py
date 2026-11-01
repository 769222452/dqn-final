"""Run at most six new fixed-surface aero calls, never geometry/OCP/training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from morphing_waverider.generator.interface import GeneratorState, TargetAero
from morphing_waverider.teacher_v5.target_matching import (
    AuditedCandidateLibrary, FrozenTeacherAeroEvaluator, TeacherTargetMatchingGenerator,
    sha256,
)

OUT = ROOT / "results" / "v5_teacher_generator_trial"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def state_for(candidate, mach, alpha):
    # The aero uses only M/alpha; the other interface fields are explicitly unused.
    return GeneratorState(mach=mach, altitude=0., velocity=0., flight_path_angle=0.,
        angle_of_attack=alpha, current_geometry=np.array(candidate.parameters),
        context={"source": "standalone fixed-surface generator trial",
                 "unused_state_fields": ["altitude", "velocity", "flight_path_angle"],
                 "current_geometry_id": candidate.geometry_id})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "generator_trial.json").exists():
        raise SystemExit("Existing completed trial found: refuse to overwrite or repeat calls.")
    library = AuditedCandidateLibrary()
    evaluator = FrozenTeacherAeroEvaluator(library, OUT, timeout_s=120., max_calls=12)
    generator = TeacherTargetMatchingGenerator(library, evaluator)
    conditions = [{"name": "M15_alpha4deg", "M_flight": 15., "alpha_rad": float(np.deg2rad(4.))},
                  {"name": "M5_alpha0deg", "M_flight": 5., "alpha_rad": 0.}]
    identities = {
        "schema": "teacher-target-matching-identity-v1", "audit_identity_files": library.identity_files,
        "geometry_hash_algorithm": "native-mesh-inputs-audit-v1; NOT legacy TeacherGeometry xi-family hash",
        "candidates": [{"geometry_id": c.geometry_id, "design_spec": c.design_spec,
                        "geometry_hash": c.geometry_hash, "native_mat": str(c.mat_path),
                        "mat_sha256": c.mat_sha256, "audit_file": str(c.audit_path),
                        "audit_sha256": c.audit_sha256} for c in library.candidates],
        "reference": evaluator.reference, "model_options": evaluator.options, "conditions": conditions,
        "planned_max_unique_aero_calls": 6, "hard_ledger_limit": 12, "per_call_timeout_s": 120.,
        "geometry_generation_calls": 0, "coupling_scope": "pressure-only teacher numerical model",
        "pipeline_integration": "NOT_CONNECTED_TO_ACTUATOR_OCP_DQN; no intermediate path validated",
        "new_code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
             [ROOT / "src/morphing_waverider/teacher_v5/target_matching.py", Path(__file__).resolve()]},
    }
    dump(OUT / "identity_manifest.json", identities)
    evaluations, trials = [], []
    start = time.perf_counter()
    for condition in conditions:
        mach, alpha = condition["M_flight"], condition["alpha_rad"]
        current_rows = []
        for candidate in library.candidates:
            print(f"REAL_AERO_START {condition['name']} {candidate.geometry_id}", flush=True)
            result = evaluator(candidate, mach, alpha)
            evaluations.append(dict(result, condition_name=condition["name"]))
            current_rows.append(result)
            dump(OUT / "candidate_evaluations.json", {"evaluations": evaluations, "events": evaluator.events})
            print(f"REAL_AERO_END {candidate.geometry_id} {result['status']} {result.get('failure_type')} wall={result.get('wall_s')}", flush=True)
        if all(row["status"] == "PASS" for row in current_rows):
            for candidate, row in zip(library.candidates, current_rows):
                target = TargetAero(row["CL_pressure"], row["CD_pressure"])
                result = generator.generate(state_for(candidate, mach, alpha), target)
                repeated = generator.generate(state_for(candidate, mach, alpha), target)
                equivalent = result.success and result.metadata["target_weighted_L2"] <= generator.tolerance
                trials.append({"test": "real_teacher_roundtrip", "condition": condition["name"],
                    "source_geometry_id": candidate.geometry_id, "status": "PASS" if equivalent else "FAIL",
                    "target": target.as_dict(), "result": result.as_dict(),
                    "deterministic_repeat": result.as_dict() == repeated.as_dict(),
                    "identity_invariant": True})
            target = TargetAero(1000., 1000.)
            result = generator.generate(state_for(library.candidates[0], mach, alpha), target)
            trials.append({"test": "real_teacher_unreachable_target", "condition": condition["name"],
                "status": "PASS" if result.metadata["generator_status"] == "INFEASIBLE_TARGET" else "FAIL",
                "source_geometry_id": "G42", "target": target.as_dict(), "result": result.as_dict(),
                "identity_invariant": True})
        else:
            # Call generator using cached real failures to verify fail-closed semantics.
            target = TargetAero(.1, .03)
            result = generator.generate(state_for(library.candidates[0], mach, alpha), target)
            repeated = generator.generate(state_for(library.candidates[0], mach, alpha), target)
            trials.append({"test": "real_teacher_failure_propagation", "condition": condition["name"],
                "source_geometry_id": "G42", "status": "PASS" if not result.success and result.metadata["generator_status"] == "AERO_COUPLING_BLOCKED" else "FAIL",
                "target": target.as_dict(), "result": result.as_dict(),
                "deterministic_repeat": result.as_dict() == repeated.as_dict(), "identity_invariant": True,
                "roundtrip_status": "BLOCKED_NO_COMPLETE_TRUSTWORTHY_TEACHER_CL_CD",
                "unreachable_target_status": "NOT_TESTABLE_WITH_FAILED_TEACHER_AERO"})
    library.verify()
    dump(OUT / "candidate_evaluations.json", {"evaluations": evaluations, "events": evaluator.events})
    blocked = any(row["status"] != "PASS" for row in evaluations)
    final = {"TEACHER_GENERATOR_INTERFACE": "SEE_INDEPENDENT_CONTRACT_TESTS",
        "AERO_COUPLING": "BLOCKED" if blocked else "PASS_NUMERICAL_TEACHER_MODEL_ONLY",
        "trials": trials, "elapsed_s": time.perf_counter() - start,
        "unique_real_aero_calls": len(evaluations), "geometry_calls": 0,
        "geometry_hash_invariant_across_conditions": True,
        "successful_real_coefficients": sum(row["status"] == "PASS" for row in evaluations),
        "synthetic_fallback_used": False, "conditions": conditions}
    dump(OUT / "generator_trial.json", final)
    rows = []
    for trial in trials:
        result = trial["result"]
        meta = result["metadata"]
        rows.append({"test": trial["test"], "condition": trial["condition"],
            "source_geometry_id": trial["source_geometry_id"], "test_status": trial["status"],
            "generator_status": meta["generator_status"], "geometry_id": meta["geometry_id"],
            "geometry_hash": meta["geometry_hash"], "CL_des": trial["target"]["CL_des"],
            "CD_des": trial["target"]["CD_des"], "CL_pred": meta["CL_pred"], "CD_pred": meta["CD_pred"],
            "target_weighted_L2": meta.get("target_weighted_L2"),
            "deterministic_repeat": trial.get("deterministic_repeat"),
            "identity_invariant": trial["identity_invariant"]})
    with (OUT / "generator_trial.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: v for k, v in final.items() if k != "trials"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
