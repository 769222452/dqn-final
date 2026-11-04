"""Stage 0: read the existing candidate pool, check validity evidence, freeze the screening configuration."""
from __future__ import annotations
import collections, json, math, time
from pathlib import Path
from common import ROOT, OUT, SREF, LREF, MODEL_VERSION, read_csv, write_csv

INV = ROOT / "results/v6_48h_multifidelity_morph_rl/GEOMETRY_INVENTORY.csv"
IDX = ROOT / "data/geometries/candidate_identity_index.csv"

inv = read_csv(INV); idx = {r["geometry_hash"]: r for r in read_csv(IDX)}
by_hash = collections.defaultdict(list)
for r in inv: by_hash[r["geometry_hash"]].append(r)
canon = {}
for h, rows in by_hash.items():
    cid = idx[h]["canonical_geometry_id"]
    canon[h] = next(r for r in rows if r["geometry_id"] == cid)

pool = []; problems = []
for h, r in canon.items():
    mat = ROOT / r["mat_path"]; aud = mat.with_name("audit.json")
    rec = {"geometry_id": r["geometry_id"], "geometry_hash": h, "source": r["source"], "Ma_design": float(r["Ma_design"]),
           "a1": r["a1"], "a2": r["a2"], "b1": r["b1"], "b2": r["b2"], "d1": r["d1"], "a_x": r["a_x"],
           "mat_path": r["mat_path"], "alias_ids": idx[h]["alias_ids"], "alias_count": idx[h]["alias_count"],
           "volume_ratio_G42": r["closed_volume_ratio_G42"], "thickness_ratio_G42": r["thickness_ratio_G42"],
           "closed_volume_full": r["closed_volume_full"]}
    if not mat.is_file(): problems.append((r["geometry_id"], "mat_missing")); continue
    if not aud.is_file(): problems.append((r["geometry_id"], "audit_missing")); rec["geometry_validity"] = "NO_AUDIT"; pool.append(rec); continue
    d = json.loads(aud.read_text(encoding="utf-8-sig"))
    x = d["inputs10"]   # [Ma, a1, a2, b1, b2, c1, c2, d1, AOA_prog, a_x]
    rec.update({"AOA_design_prog": x[8], "c1": x[5], "c2": x[6], "geometry_validity": d["classification"],
                "audit_failure_reason": d.get("failure_reason", ""), "length_scale": d.get("length_scale"),
                "audit_surface_intersections": d.get("surface_intersection_count"), "audit_negative_thickness": d.get("negative_thickness_count")})
    pool.append(rec)
pool.sort(key=lambda r: (r["Ma_design"], r["geometry_id"]))
write_csv(OUT / "screening/candidate_pool.csv", pool)
valid = [r for r in pool if r["geometry_validity"] == "VALID"]
fam = collections.Counter(round(r["Ma_design"], 3) for r in valid)
summary = {"inventory_rows": len(inv), "unique_native_hashes": len(canon), "audit_VALID": len(valid), "problems": problems,
           "n_design_mach_values": len(fam), "design_mach_counts_main": {k: v for k, v in sorted(fam.items()) if v >= 5},
           "design_mach_all": sorted(fam.items()), "G42_in_pool": any(r["geometry_id"] == "G42" for r in valid),
           "exploratory_volume_screen_pass": sum(float(r["volume_ratio_G42"]) >= 0.85 and float(r["thickness_ratio_G42"]) >= 0.5 for r in valid)}
(OUT / "work/pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))

# ---- frozen configuration (written once, before any ranking) ----
config = {
    "schema": "wedge-screening-config/1", "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "aero_backend": {"name": "TANGENT_WEDGE_SHOCK_EXPANSION", "implementation": "results/v6_matlab_no_new_cfd/code/matlab/offdesign_aero_v7.m (existing, unmodified)",
                     "options": {"mode": "P2", "friction": "none", "base_pressure_ratio": "NaN (no base term)"}, "model_version": MODEL_VERSION,
                     "coefficients": "pressure-only CL and CD_pressure on Sref; NOT total drag (no friction, no base drag)",
                     "invalid_states": "no attached weak shock / subsonic / Prandtl-Meyer vacuum / strip fold-back -> model_valid=false, no silent fallback",
                     "Sref_m2": SREF, "Lref_m": LREF, "reference_source": "configs/mission_1200s.yaml model.reference_area_m2; v7 database builder reference"},
    "flight_grid": {"M_flight": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], "alpha_deg": [4, 5, 6, 7, 8, 9, 10], "n_conditions": 49,
                    "note": "primary-mission screening only; the full support domain M[3.5,9.25] alpha[-2,12] is NOT covered by this run"},
    "ranking": {"metric": "L_over_D = CL / CD_pressure at fixed alpha", "top_n": 5,
                "gates": {"geometry_validity": "audit classification == VALID (frozen native-mesh audit)",
                          "aero": "model_valid and finite CL, CD_pressure>0",
                          "lift_floor_CL_min": 0.026150265744858465,
                          "lift_floor_source": "results/fixed_geometry_baselines/comparison.json operating_condition_envelope.CL_actual.p5 (all feasible fixed baselines, 0.5 s samples)",
                          "volume_thickness_screen": "EXPLORATORY: volume_full >= 0.85*G42 and grid_thickness_max >= 0.5*G42 (src/mission_v6 stage2 common.py scenario definition; research assumption, not an engineering requirement)"},
                "low_lift_note": "candidates below the floor are kept in all_candidate_scores.csv with flag LOW_LIFT; they are never ranked"},
    "controls": {"baseline": "G42 (if it already is in Top5 it is not duplicated)",
                 "diversity": "valid gate-passing candidate maximizing min normalized distance to the Top5 over [Ma_design, volume ratio, thickness ratio, a1,a2,b1,b2,d1,a_x]"},
    "nearby": {"dM": 0.25, "dalpha_deg": 1.0, "diagonals": False},
    "cross_mach": {"M_grid": [5.0 + 0.25 * i for i in range(13)], "alpha_deg_grid": [4, 5, 6, 7, 8, 9, 10], "geometry_set": "G_selected_unique"},
    "equal_lift": {"alpha_dense_deg": [2.0 + 0.5 * i for i in range(21)], "M_grid": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                   "CL_targets": [0.026150265744858465, 0.0398, 0.05350025264833205], "targets_source": "mission envelope CL_actual p5 / mid / p95 (comparison.json)",
                   "interpolation": "PCHIP in alpha inside the bracket only, no extrapolation"},
    "geometry_generation": {"new_geometries_generated": 0, "reason": "existing audited pool already spans many design Mach families; new teacher generation deferred (budget 0 in this run)",
                            "budget_note": "no parallel MATLAB, no use of CFD processes"},
    "budget": {"matlab_processes_at_once": 1, "cfd": False, "dqn": False, "ocp": False, "synthetic_data": False},
    "cad": {"unit_policy": "program length unit = 1 m (audit: 'm under existing V5 1 metre/program-unit assumption; physical scale unconfirmed'); STEP written in mm (x1000)"},
}
(OUT / "config/screening_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("config written")
