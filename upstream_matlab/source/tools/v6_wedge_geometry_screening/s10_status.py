"""Stage 10: acceptance_status.json, budget_ledger.json, identity_manifest.json."""
import hashlib, json, time
from pathlib import Path
from common import ROOT, OUT, read_csv

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
facts = json.loads((OUT / "work/report_facts.json").read_text(encoding="utf-8")); cad = json.loads((OUT / "work/cad_export_results.json").read_text(encoding="utf-8"))
v7 = ROOT / "results/v6_matlab_no_new_cfd/code/matlab"
ident = {"schema": "wedge-screening-identity/1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "aero_model": {"version": "offdesign_aero_v7-1.0 mode=P2 friction=none base=none", "files_sha256": {p.name: sha(p) for p in sorted(v7.glob("*.m"))}},
         "driver": {"tools/v6_wedge_geometry_screening/wedge_eval.m": sha(ROOT / "tools/v6_wedge_geometry_screening/wedge_eval.m")},
         "config_sha256": sha(OUT / "config/screening_config.json"), "candidate_pool_sha256": sha(OUT / "screening/candidate_pool.csv"),
         "pool_results_sha256": sha(OUT / "work/pool_results.csv"), "selected_results_sha256": sha(OUT / "work/selected_results.csv"),
         "reference": {"Sref_m2": 6.931920786884801, "Lref_m": 5.0, "source": "configs/mission_1200s.yaml"},
         "geometries": {r["geometry_id"]: {"geometry_hash": r["geometry_hash"], "native_mat_sha256": sha(ROOT / r["mat_path"]), "step_sha256": sha(OUT / "geometry/step" / f"{r['geometry_id']}.step")}
                        for r in read_csv(OUT / "screening/selected_unique_geometries.csv")},
         "software": {"matlab": "R2023a (external, D:\\matlab)", "opencascade": "cadquery-ocp 8.0.1.1.0 (.venv_cad)", "python_main": ".venv (project pins)"}}
(OUT / "identity_manifest.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")
led = {"schema": "budget-ledger/1", "matlab_processes_at_once": 1, "cfd_started": False, "dqn_trained": False, "ocp_run": False, "mission_1200s_run": False, "new_geometries_generated": 0,
       "wedge_evaluations": {"pool_stage": 22197, "selected_stage": 5957}, "wall_clock_s": {"pool_eval_matlab": 2760.4, "selected_eval_matlab": 960.4, "cad_export_31_geometries_sum": round(sum(r["elapsed_s"] for r in cad), 1), "matlab_selfcheck_26_tests": "~1 call"},
       "user_matlab_session_untouched": True, "note": "no parallel MATLAB, no use of CFD processes"}
(OUT / "budget_ledger.json").write_text(json.dumps(led, indent=1), encoding="utf-8")
allpass = all(r["cad_export_status"] == "PASS" for r in cad)
acc = {"TEACHER_GEOMETRY_GENERATION": "PARTIAL", "TEACHER_GEOMETRY_GENERATION_note": "453 existing teacher-generated, audited geometries reused; 0 new teacher generations were run in this stage (budget 0, coverage argued in config)",
       "GEOMETRY_VALIDITY_GATE": "PASS", "GEOMETRY_VALIDITY_GATE_note": "453/453 pool geometries carry audit classification VALID (frozen native-mesh audit, read from audit.json); the gate re-reads the evidence, it does not re-run the audit",
       "TANGENT_WEDGE_AERO": "PASS", "TANGENT_WEDGE_AERO_note": "existing offdesign_aero_v7 P2, 26/26 analytic self-tests pass, 0 invalid evaluations in 28,154 calls; pressure-only coefficients",
       "TOP5_SCREENING": "PASS", "TOP5_SCREENING_note": "49/49 conditions have 5 eligible members (225-299 eligible each)",
       "NEARBY_OFFDESIGN_EVALUATION": "PASS", "CROSS_MACH_COMPARISON": "PASS",
       "CAD_STEP_EXPORT": "PASS" if allpass else "PARTIAL", "CAD_STEP_EXPORT_note": f"{sum(r['cad_export_status']=='PASS' for r in cad)}/{len(cad)} STEP solids; Parasolid and NX .prt are BLOCKED (no Parasolid kernel / NX not installed)",
       "PARASOLID_EXPORT": "BLOCKED", "NX_PRT_EXPORT": "BLOCKED",
       "CAD_REIMPORT_VALIDATION": "PASS" if allpass else "PARTIAL", "CAD_REIMPORT_VALIDATION_note": "re-import with the OpenCascade STEP reader only (not NX): valid solid, 0 free boundaries, volume vs native mesh <=0.71 %, node deviation <=4.3e-10 mm",
       "NX_BAT_PREPARED": "YES", "NX_IMPORT_TESTED": "NO", "NX_status": "NX_BATCH_NOT_EXECUTED",
       "BAT_LOGIC_SELFTEST": "PASS (exit codes 2/3/4/5 verified with isolated stubs; this does not test NX)",
       "FIGURES": "10 PNG written; not visually inspected by the author (image input unavailable)",
       "SOURCE_PROJECT_UNCHANGED": "YES", "SOURCE_PROJECT_UNCHANGED_note": "SHA-256 of all 175,365 files before/after identical; all outputs are in D:\\v6-mission-morph-core",
       "CFD_COMPLETED": "NO", "DQN_TRAINED": "NO", "OCP_RUN": "NO", "MISSION_1200S_RUN": "NO"}
(OUT / "acceptance_status.json").write_text(json.dumps(acc, indent=1, ensure_ascii=False), encoding="utf-8"); print(json.dumps(acc, indent=1, ensure_ascii=False)[:600])
