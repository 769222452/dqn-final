"""Shared helpers for the wedge-model screening (project-relative, no hardcoded roots)."""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v6_wedge_geometry_screening"
MATLAB = Path(r"D:\matlab\bin\matlab.exe")          # external install (R2023a), not part of the project
SREF = 6.931920786884801                              # configs/mission_1200s.yaml model.reference_area_m2
LREF = 5.0                                            # program length scale of every native geometry (audit length_scale)
MODEL_VERSION = "TANGENT_WEDGE_SHOCK_EXPANSION | offdesign_aero_v7-1.0 mode=P2 friction=none base=none (pressure only)"


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, fields=None):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def cfg():
    return json.loads((OUT / "config" / "screening_config.json").read_text(encoding="utf-8"))


def run_matlab_eval(query_csv, out_csv, log_file):
    """One MATLAB -batch process (single process, no parallel pool)."""
    tool = (ROOT / "tools" / "v6_wedge_geometry_screening").as_posix()
    cmd = f"addpath('{tool}'); wedge_eval('{Path(query_csv).as_posix()}','{Path(out_csv).as_posix()}','{Path(log_file).as_posix()}');"
    r = subprocess.run([str(MATLAB), "-batch", cmd, "-nosplash", "-nodesktop"], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"MATLAB failed rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout
