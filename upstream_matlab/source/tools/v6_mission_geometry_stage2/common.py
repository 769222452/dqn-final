"""Shared paths, guards and helpers for V6 stage 2 (mission-driven geometry + morph paths).

Reads (never writes) V6 copies of the V2 geometry/aero databases and stage-1 results.
Every write goes under results/v6_mission_geometry_stage2 or _runtime.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from morphing_waverider.utils.runtime_paths import runtime_view, resolve_project_path

if ROOT.resolve() == Path("D:/v5-final").resolve():
    raise RuntimeError("Stage 2 must never run inside the V5 project")
OUT = ROOT / "results/v6_mission_geometry_stage2"
CASES = OUT / "cases"
TEACHER_DIR = OUT / "teacher"
STAGE1 = ROOT / "results/fixed_geometry_baselines"
GEO_DB = ROOT / "results/v5_multimach_geometry_db_v2"
AERO_DB = ROOT / "results/v5_multimach_aero_database_v2"
ACTIVE = ROOT / "matlab/active"
AUDIT_TOOLS = ROOT / "tools/geometry_audit"
MATLAB = Path(os.environ.get("V6_MATLAB_EXE", "D:/matlab/bin/matlab.exe"))
AERO_WORKER = AERO_DB / "code/aero_worker.m"
AERO_WORKER_SHA = "9f47935ffa4d6424f3d5caaee44afe1527bfa9396a15c2b8b8cc5dd0cb895641"
OFFDESIGN_SHA = "3678ec2d146157a562fa8860b0464b75358ada6037ab6b981d8af564bef6e68e"
MODEL_VERSION = "teacher-offdesign-v4-repaired-v5.1"
REFERENCE = {"area_m2": 6.931920786884801, "length_m": 5.0, "moment_origin_m": [0.0, 0.0, 0.0]}
MODEL_OPTIONS = {"mesh_stride": 1, "cone_lookup_nodes": 33}

PARAMS = ("a1", "a2", "b1", "b2", "d1", "a_x")
FIXED_INPUTS = {"c1": 1.0331, "c2": 1.0331, "AOA": -4.0}
FROZEN_LOWER = np.array([3, 3, 3, 3, .02, .9])
FROZEN_UPPER = np.array([8, 8, 8, 8, .18, 1.1])
EXT_UPPER = np.array([20, 20, 20, 20, .18, 1.1])
G42_ID = "G42"
G42_INPUTS10 = [15.0, 5.633749605922896, 7.796675937828777, 3.698465133623518, 3.0116105313696075,
                1.0331, 1.0331, 0.11606537709080691, -4.0, 0.9944782795122264]
G42_MAT = ROOT / "results/v5_geometry_audit/cases/design_42/native_geometry.mat"

# Teacher grids (subsets of the V2 grid so labels are directly comparable and reusable).
SCREEN_MACH = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
SCREEN_ALPHA = [float(a) for a in range(0, 13)]
MISSION_MACH = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
MISSION_ALPHA = [float(a) for a in range(-2, 13)]
BANDS = {"M7-8": (7.0, 8.0), "M6-7": (6.0, 7.0), "M5-6": (5.0, 6.0)}
ROLE_OF_BAND = {"M7-8": "A_high_Mach", "M6-7": "B_mid_Mach", "M5-6": "C_low_Mach"}

# Research-only volume/thickness scenarios relative to G42 (not engineering minima).
SCENARIOS = {
    "CONSERVATIVE": {"volume_ratio_min": 1.0, "thickness_ratio_min": 1.0,
                     "definition": "volume_full >= G42 and grid_thickness_max >= G42"},
    "EXPLORATORY": {"volume_ratio_min": 0.85, "thickness_ratio_min": 0.5,
                    "definition": "volume_full >= 0.85*G42 and grid_thickness_max >= 0.5*G42"},
}
ACTUATOR_RATE_PATH_PER_S = 0.1
ACTUATOR_GAIN_PER_S = 0.5


def utc():
    return datetime.now(timezone.utc).isoformat()


def out(rel) -> Path:
    p = (OUT / rel).resolve()
    allowed = (OUT.resolve(), (ROOT / "_runtime").resolve())
    if not any(p == a or a in p.parents for a in allowed):
        raise ValueError(f"Stage-2 write outside allowed roots: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    p = Path(p)
    if not p.resolve().is_relative_to(OUT.resolve()) and not p.resolve().is_relative_to((ROOT / "_runtime").resolve()):
        raise ValueError(f"Stage-2 write outside allowed roots: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(v, indent=2, ensure_ascii=False, default=_default, allow_nan=True), encoding="utf-8")
    os.replace(tmp, p)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load(p):
    return runtime_view(json.loads(resolve_project_path(p, ROOT).read_text(encoding="utf-8-sig")), ROOT)


def read_csv(p):
    with resolve_project_path(p, ROOT).open(encoding="utf-8-sig", newline="") as f:
        return runtime_view(list(csv.DictReader(f)), ROOT)


def write_csv(p, rows, fieldnames=None):
    p = out(Path(p).resolve().relative_to(OUT.resolve())) if Path(p).is_absolute() else out(p)
    fieldnames = fieldnames or (list(dict.fromkeys(k for r in rows for k in r)) if rows else [])
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)
    return p


def append_jsonl(p, rec):
    p = Path(p)
    if not p.resolve().is_relative_to(OUT.resolve()):
        raise ValueError(p)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def native_hash(lower, upper, inputs) -> str:
    """Same identity as tools/geometry_audit/scan.py::native_hash."""
    h = hashlib.sha256(b"native-mesh-inputs-audit-v1\0")
    for x in [np.asarray(inputs), lower, upper]:
        x = np.asarray(x, dtype="<f8", order="C")
        h.update(json.dumps(list(x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def design_space_label(z):
    z = np.asarray(z, float)
    if np.all(z >= FROZEN_LOWER) and np.all(z <= FROZEN_UPPER):
        return "FROZEN"
    if np.all(z >= FROZEN_LOWER) and np.all(z <= EXT_UPPER):
        return "EXTENDED"
    return "OUT_OF_RANGE"


def inputs10(ma_design, z):
    a1, a2, b1, b2, d1, ax = (float(v) for v in z)
    return [float(ma_design), a1, a2, b1, b2, FIXED_INPUTS["c1"], FIXED_INPUTS["c2"], d1,
            FIXED_INPUTS["AOA"], ax]


def candidate_id(ma_design, z, prefix="S2"):
    d = hashlib.sha256(json.dumps({"ma": round(float(ma_design), 10),
                                   "z": [round(float(v), 10) for v in z]}).encode()).hexdigest()[:10]
    return f"{prefix}_MD{float(ma_design):g}_{d}"


def normalise(z):
    return (np.asarray(z, float) - FROZEN_LOWER) / (EXT_UPPER - FROZEN_LOWER)


def add_paths():
    for p in (str(ROOT / "src"), str(AUDIT_TOOLS), str(GEO_DB / "code")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
