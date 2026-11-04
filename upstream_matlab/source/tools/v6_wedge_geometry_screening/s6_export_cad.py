"""Stage 6: real OpenCascade solid + STEP (mm) + re-import check for every selected unique geometry (run with .venv_cad)."""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv
import cad_build as cb
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "results/v6_wedge_geometry_screening"
sel = list(csv.DictReader(open(OUT / "screening/selected_unique_geometries.csv", encoding="utf-8-sig")))
pool = {r["geometry_id"]: r for r in csv.DictReader(open(OUT / "screening/candidate_pool.csv", encoding="utf-8-sig"))}
recs = []
for g in sel:
    gid = g["geometry_id"]; d = OUT / "geometry/step" / gid
    inv = float(pool[gid]["closed_volume_full"]) if pool[gid]["closed_volume_full"] else None
    rec = cb.process(gid, ROOT / g["mat_path"], d, inventory_volume=inv)
    rec["geometry_hash"] = g["geometry_hash"]
    gates = rec.get("reimport", {})
    if rec["cad_export_status"] == "PASS":
        ok = (gates.get("topology", {}).get("free_boundary_wires", 1) == 0 and gates.get("volume_rel_diff_vs_mesh", 1) < 0.01
              and gates.get("max_node_deviation_mm", 1) < 1e-3 and gates.get("shape_type") == "SOLID")
        rec["cad_export_status"] = "PASS" if ok else "PARTIAL"
    # one STEP file per geometry at geometry/step/<id>.step (flat) plus the per-geometry folder copy
    if (d / "geometry.step").is_file(): shutil.copyfile(d / "geometry.step", OUT / "geometry/step" / f"{gid}.step")
    nat = OUT / "geometry/native_mat"; nat.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / g["mat_path"], nat / f"{gid}.mat")
    (d / "cad_check.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    recs.append(rec); print(gid, rec["cad_export_status"], rec.get("elapsed_s"), flush=True)
(OUT / "work/cad_export_results.json").write_text(json.dumps(recs, indent=1, default=float), encoding="utf-8")
print("PASS", sum(r["cad_export_status"] == "PASS" for r in recs), "of", len(recs))
