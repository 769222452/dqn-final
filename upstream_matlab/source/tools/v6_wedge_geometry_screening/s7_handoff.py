"""Stage 7: geometry manifest, UG import manifest and the UG_CFD_HANDOFF package (one geometry copy, many flight cases)."""
import csv, json, shutil
from pathlib import Path
from common import OUT, SREF, LREF, MODEL_VERSION, read_csv, write_csv

sel = read_csv(OUT / "screening/selected_unique_geometries.csv"); pool = {r["geometry_id"]: r for r in read_csv(OUT / "screening/candidate_pool.csv")}
cad = {r["geometry_id"]: r for r in json.loads((OUT / "work/cad_export_results.json").read_text(encoding="utf-8"))}
top = read_csv(OUT / "screening/top5_by_condition.csv"); ctrl = read_csv(OUT / "screening/baseline_diversity.csv")
UNIT = "mm (STEP); program unit 1 m x 1000"; FRAME = "body frame: x downstream (nose x=0, base x=5000 mm), y span (symmetric), z up; alpha>0 nose-up"
cases = {}
for r in top: cases.setdefault(r["geometry_id"], []).append({"condition_id": r["condition_id"], "M_flight": r["M_flight"], "alpha_flight_deg": r["alpha_flight_deg"], "role": f"TOP{r['rank']}", "wedge_CL": r["CL"], "wedge_CD_pressure": r["CD_pressure"], "wedge_L_over_D": r["L_over_D"]})
for r in ctrl: cases.setdefault(r["geometry_id"], []).append({"condition_id": r["condition_id"], "M_flight": r["M_flight"], "alpha_flight_deg": r["alpha_flight_deg"], "role": r["role"], "wedge_CL": r["CL"], "wedge_CD_pressure": r["CD_pressure"], "wedge_L_over_D": r["L_over_D"]})
gm, um = [], []
H = OUT / "UG_CFD_HANDOFF"
for g in sel:
    gid = g["geometry_id"]; p = pool[gid]; k = cad.get(gid, {}); st = k.get("cad_export_status", "CAD_EXPORT_BLOCKED")
    step = f"geometry/step/{gid}.step"; has = (OUT / step).is_file() and st in ("PASS", "PARTIAL")
    row = {"geometry_id": gid, "geometry_hash": g["geometry_hash"], "Ma_design": g["Ma_design"], "AOA_design_prog": p["AOA_design_prog"],
           "a1": p["a1"], "a2": p["a2"], "b1": p["b1"], "b2": p["b2"], "c1": p["c1"], "c2": p["c2"], "d1": p["d1"], "a_x": p["a_x"],
           "native_mat_path": f"geometry/native_mat/{gid}.mat", "step_path": step if has else "", "parasolid_path": "", "prt_path": "",
           "cad_export_status": st, "parasolid_status": "BLOCKED_NO_PARASOLID_KERNEL_OR_NX", "prt_status": "BLOCKED_NX_NOT_INSTALLED", "nx_import_status": "NX_BATCH_NOT_EXECUTED",
           "length_unit": UNIT, "coordinate_frame": FRAME, "associated_case_count": len(cases.get(gid, [])),
           "cad_volume_m3": k.get("cad_volume_m3", ""), "mesh_volume_m3": k.get("mesh_volume_m3", ""), "reimport_valid": k.get("reimport", {}).get("brep_valid", ""),
           "reimport_volume_rel_diff_vs_mesh": k.get("reimport", {}).get("volume_rel_diff_vs_mesh", ""), "reimport_max_node_deviation_mm": k.get("reimport", {}).get("max_node_deviation_mm", ""),
           "free_boundary_wires": k.get("reimport", {}).get("topology", {}).get("free_boundary_wires", "")}
    gm.append(row)
    um.append({"geometry_id": gid, "step_path": step, "prt_path": f"geometry/nx_prt/{gid}.prt", "geometry_hash": g["geometry_hash"], "Ma_design": g["Ma_design"],
               "native_mat_path": row["native_mat_path"], "parasolid_path": "", "cad_export_status": st, "nx_import_status": "NX_BATCH_NOT_EXECUTED",
               "length_unit": UNIT, "coordinate_frame": FRAME, "associated_case_count": row["associated_case_count"]})
    if has:
        d = H / gid; d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT / step, d / "geometry.step"); shutil.copyfile(ROOT_MAT := OUT / row["native_mat_path"], d / "geometry_native.mat")
        info = {k_: row[k_] for k_ in ("geometry_id", "geometry_hash", "Ma_design", "AOA_design_prog", "a1", "a2", "b1", "b2", "c1", "c2", "d1", "a_x", "length_unit", "coordinate_frame", "cad_export_status")}
        info.update({"reference_area_m2": SREF, "reference_length_m": LREF, "reference_area_note": "Sref is the common mission reference area, not rescaled with shape",
                     "aero_prediction_model": MODEL_VERSION, "aero_prediction_note": "wedge-model values in related_flight_cases.csv are low-order pressure-only predictions, NOT CFD results",
                     "cad_checks": {"brep_valid": row["reimport_valid"], "volume_rel_diff_vs_native_mesh": row["reimport_volume_rel_diff_vs_mesh"], "max_node_deviation_mm": row["reimport_max_node_deviation_mm"],
                                    "free_boundary_wires": row["free_boundary_wires"], "base": "CAD_ONLY ruled base cap closes the open trailing edge; the aero model treats the base separately", "symmetry": "full body = native half model + y mirror"},
                     "native_open_base": True})
        (d / "geometry_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        write_csv(d / "related_flight_cases.csv", sorted(cases.get(gid, []), key=lambda r: (float(r["M_flight"]), float(r["alpha_flight_deg"]))))
write_csv(OUT / "geometry/geometry_manifest.csv", gm); write_csv(OUT / "ug_batch/import_manifest.csv", um)
print("geometries", len(gm), "with step", sum(bool(r["step_path"]) for r in gm), "handoff dirs", len([p for p in H.iterdir() if p.is_dir()]))
