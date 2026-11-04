"""One-off statistics collection for the final reports."""
import csv
import glob
import json
from collections import Counter
from pathlib import Path

OUT = Path(str((Path(__file__).resolve().parents[3] / 'results/v5_multimach_geometry_db_v2')))

man = json.load(open(OUT / "geometry_manifest.json", encoding="utf-8"))
print("manifest valid:", man["valid_count"], "design:", man["design_geometry_count"],
      "morph:", man["morph_intermediate_geometry_count"])
reg = json.load(open(OUT / "g42_regression.json", encoding="utf-8"))
print("regression:", reg["GEOMETRY_REGRESSION"], reg["max_abs_coordinate_diff"],
      reg["mat_container_sha256_identical"])
for p in sorted(glob.glob(str(OUT / "generation" / "*" / "generation_budget.json"))):
    b = json.load(open(p, encoding="utf-8"))
    print(Path(p).parent.name, "calls=", len(b["calls"]), "sessions=", len(b["sessions"]),
          "elapsed=%.1f" % b.get("geometry_elapsed_s_total", 0))
for f in ("feasibility_scan.json", "feasibility_scan_b.json", "feasibility_scan_c.json"):
    s = json.load(open(OUT / "stage0_feasibility" / f, encoding="utf-8"))
    print(f, s["probe_count"], "completed", s["completed"], "wall %.0fs" % s["wall_s"])

inv = list(csv.DictReader(open(OUT / "geometry_inventory.csv", encoding="utf-8-sig")))
print("inv", len(inv), Counter(r["classification"] for r in inv))
v = [r for r in inv if r["classification"] == "VALID" and r["geometry_id"].startswith("V2_")]
for md in sorted({r["Ma_design"] for r in v}, key=float):
    s = [r for r in v if r["Ma_design"] == md]
    print("design Md", md, len(s), Counter(r["design_space"] for r in s))

morph = json.load(open(OUT / "morph_path_validation.json", encoding="utf-8"))
print("morph edges connected/tested:", morph["edges_connected"], "/", morph["edges_tested"])
met = list(csv.DictReader(open(OUT / "geometry_metrics.csv", encoding="utf-8-sig")))
vm = [r for r in met if r["classification"] == "VALID" and r["geometry_id"].startswith("V2_")]
for md in sorted({r["Ma_design"] for r in vm}, key=float):
    s = [r for r in vm if r["Ma_design"] == md]
    vols = sorted(float(r["volume_full"]) for r in s)
    eta = sorted(float(r["volumetric_efficiency"]) for r in s)
    tmax = sorted(float(r["grid_thickness_max"]) for r in s)
    print("Md", md, "vol %.3f..%.3f" % (vols[0], vols[-1]),
          "eta %.3f..%.3f" % (eta[0], eta[-1]),
          "tmax %.3f..%.3f" % (tmax[0], tmax[-1]))
failed = list(csv.DictReader(open(OUT / "failed_candidates.csv", encoding="utf-8-sig")))
print("failed candidates:", len(failed), Counter(r["classification"] for r in failed))
