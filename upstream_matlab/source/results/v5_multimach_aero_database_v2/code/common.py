"""Shared constants for V5-MULTIMACH-AERO-DB-V2 (full Teacher aero database).

Reads (never writes) the V2 geometry DB, the 30-minute pilot, V1 and matlab/active.
Every write goes under results/v5_multimach_aero_database_v2.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(str(Path(__file__).resolve().parents[3]))
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from morphing_waverider.utils.runtime_paths import runtime_view, resolve_project_path

OUT = ROOT / "results/v5_multimach_aero_database_v2"
STATE = OUT / "state"
CODE = OUT / "code"
GEO_DB = ROOT / "results/v5_multimach_geometry_db_v2"
PILOT = ROOT / "results/v5_multimach_aero_pilot_30min"
V1_DB = ROOT / "results/v5_teacher_aero_database_v1"
ACTIVE = ROOT / "matlab/active"
MATLAB = Path("D:/matlab/bin/matlab.exe")
G42_MAT = ROOT / "results/v5_geometry_audit/cases/design_42/native_geometry.mat"
G42_HASH = "d35074a0032e37b69278fb81afbb44250363255a6f8c880e8b89208e30401f7a"
G42_INPUTS10 = [15.0, 5.633749605922896, 7.796675937828777, 3.698465133623518, 3.0116105313696075,
                1.0331, 1.0331, 0.11606537709080691, -4.0, 0.9944782795122264]

FLIGHT_MACH = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.25]
ALPHA_DEG = [float(a) for a in range(-2, 13)]

# identical to V1 and the 30-minute pilot (fixed G42 projected area for every geometry)
REFERENCE = {"area_m2": 6.931920786884801, "length_m": 5.0, "moment_origin_m": [0.0, 0.0, 0.0]}
MODEL_OPTIONS = {"mesh_stride": 1, "cone_lookup_nodes": 33}
MODEL_VERSION = "teacher-offdesign-v4-repaired-v5.1"
OFFDESIGN_SHA = "3678ec2d146157a562fa8860b0464b75358ada6037ab6b981d8af564bef6e68e"
PILOT_WORKER_SHA = "9f47935ffa4d6424f3d5caaee44afe1527bfa9396a15c2b8b8cc5dd0cb895641"

BUDGET_S = 14 * 3600.0
STOP_DISPATCH_S = BUDGET_S - 25 * 60.0   # leaves >= 10 min for assembly, checks and report
HARD_STOP_S = BUDGET_S - 17 * 60.0
REQUEST_TIMEOUT_S = 180.0                # 15 alphas ~ 9 s typical
MAX_WORKERS = 3
MAX_REQUEST_ATTEMPTS = 3                 # only for infrastructure/transient faults
TRANSIENT_ERROR_IDS = ("MATLAB:nomem", "MATLAB:array:SizeLimitExceeded", "MATLAB:load:couldNotReadFile",
                       "MATLAB:load:unableToReadMatFile")
MIN_FREE_RAM_GB = 1.2
RESUME_FREE_RAM_GB = 1.8                 # 3 idle MATLAB workers leave ~2.2 GB free on this 16 GB host
MIN_FREE_DISK_GB = 2.0


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def native_hash(lower, upper, inputs) -> str:
    """Exact copy of tools/geometry_audit/scan.py::native_hash (frozen geometry identity)."""
    h = hashlib.sha256(b"native-mesh-inputs-audit-v1\0")
    for x in [np.asarray(inputs), lower, upper]:
        x = np.asarray(x, dtype="<f8", order="C")
        h.update(json.dumps(list(x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def dump(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(v, indent=2, ensure_ascii=False, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load(p):
    return runtime_view(json.loads(resolve_project_path(p, ROOT).read_text(encoding="utf-8")), ROOT)


def read_csv(p):
    with resolve_project_path(p, ROOT).open(encoding="utf-8-sig", newline="") as f:
        return runtime_view(list(csv.DictReader(f)), ROOT)


def write_csv(p, rows, fieldnames=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def request_id(gid, M, kind="MAIN"):
    return f"{kind}|{gid}|M{M:g}"


def read_jsonl(p, bad_log=None):
    """Committed JSONL records; an unparsable (e.g. torn last) line is reported, never dropped silently."""
    out, bad = [], []
    p = Path(p)
    if not p.exists():
        return out, bad
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(i)
    if bad and bad_log:
        with open(bad_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": str(p), "bad_line_numbers": bad}) + "\n")
    return out, bad


def append_jsonl(p, rec):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, allow_nan=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
