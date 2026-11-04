"""Shared paths, geometry registry and helpers for the V6 aero physics validation round (v1).

Read-only with respect to every existing V5/V6 artefact; all outputs go to
results/v6_aero_physics_validation_v1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/v6_aero_physics_validation_v1"
WORK = OUT / "work"
FIG = OUT / "figures"
ACTIVE = ROOT / "matlab/active"
MATLAB = Path(r"D:\matlab\bin\matlab.exe")
for d in (OUT, WORK, FIG):
    d.mkdir(parents=True, exist_ok=True)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import teacher_replica as R  # noqa: E402  (verbatim copy of results/v6_matlab_deep_audit_v1/analysis/replica.py)

G = 1.4
SREF = R.SREF
LREF = R.LREF
AOA_NATIVE = -4.0
ALPHA_DESIGN_BODY = 4.0  # native freestream (cos -4, 0, sin -4) -> body diag(1,-1,-1) -> alpha_body = +4 deg
P0_DESIGN = 79.779       # build_geometry_core free-stream pressure (only ratios are used)
R0_DESIGN = 0.001027

V2_DB = ROOT / "results/v5_multimach_aero_database_v2/aero_database.csv"
G42_DB = ROOT / "results/v5_multimach_aero_database_v2/g42_baseline.csv"
S2_DB = ROOT / "results/v6_mission_geometry_stage2/aero_pilot_labels.csv"

GEOMS = {
    "G42": {"case": "design_42", "Ma_design": 15.0, "labels": G42_DB, "label_id": "G42",
            "mat": ROOT / "results/v5_geometry_audit/cases/design_42/native_geometry.mat",
            "inputs10": [15.0, 5.633749605922896, 7.796675937828777, 3.698465133623518, 3.0116105313696075,
                         1.0331, 1.0331, 0.11606537709080691, -4.0, 0.9944782795122264],
            "role": "baseline (Md15)"},
    "S2_MD9.873_4cfddf7fac": {"case": "S2_MD9.873_4cfddf7fac", "Ma_design": 9.873, "labels": S2_DB,
                              "label_id": "S2_MD9.873_4cfddf7fac",
                              "mat": ROOT / "results/v6_mission_geometry_stage2/cases/S2_MD9.873_4cfddf7fac/native_geometry.mat",
                              "inputs10": None, "role": "stage-2 best (Md9.873)"},
    "V2_MD8_020": {"case": "W2_MD8_efe555886b", "Ma_design": 8.0, "labels": V2_DB, "label_id": "V2_MD8_020",
                   "mat": ROOT / "results/v5_multimach_geometry_db_v2/cases/W2_MD8_efe555886b/native_geometry.mat",
                   "inputs10": None, "role": "V2 Md8 (EXTENDED)"},
    "V2_MD8_009": {"case": "ANCH_MD8_f2eb7d8eb3", "Ma_design": 8.0, "labels": V2_DB, "label_id": "V2_MD8_009",
                   "mat": ROOT / "results/v5_multimach_geometry_db_v2/cases/ANCH_MD8_f2eb7d8eb3/native_geometry.mat",
                   "inputs10": None, "role": "V2 Md8 (FROZEN anchor)"},
}
ORDER = list(GEOMS)
SHORT = {"G42": "G42", "S2_MD9.873_4cfddf7fac": "MD9.873", "V2_MD8_020": "MD8_020", "V2_MD8_009": "MD8_009"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def native_hash(lower, upper, inputs) -> str:
    """Identity of tools/geometry_audit/scan.py::native_hash."""
    h = hashlib.sha256(b"native-mesh-inputs-audit-v1\0")
    for x in [np.asarray(inputs), lower, upper]:
        x = np.asarray(x, dtype="<f8", order="C")
        h.update(json.dumps(list(x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def load_mat(p):
    m = loadmat(p)
    S = {k: np.asarray(m[k], float) for k in ("wav_x", "wav_y", "wav_z", "wav_up_x", "wav_up_y", "wav_up_z")}
    S["inputs10"] = np.asarray(m["inputs10"], float).ravel() if "inputs10" in m else None
    return S


def lower_upper(S):
    lo = np.stack([S["wav_x"], S["wav_y"], S["wav_z"]], -1)
    up = np.stack([S["wav_up_x"], S["wav_up_y"], S["wav_up_z"]], -1)
    return lo, up


def inputs10(gid):
    g = GEOMS[gid]
    if g["inputs10"] is not None:
        return list(g["inputs10"])
    S = load_mat(g["mat"])
    return [float(v) for v in S["inputs10"]]


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else repr(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else v


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def labels(gid):
    """Stored Teacher labels {(M, alpha): (CL, CD, Cm, hash)} for one geometry (PASS rows only)."""
    g = GEOMS[gid]
    out = {}
    for r in read_csv(g["labels"]):
        if r["geometry_id"] != g["label_id"] or r["status"] != "PASS":
            continue
        out[(float(r["flight_mach"]), float(r["alpha_deg"]))] = (float(r["CL"]), float(r["CD"]), float(r["Cm"]),
                                                                 r["geometry_hash"])
    return out


def teacher(P, M, alpha, detail=False):
    return R.aero(P, M, alpha, detail=detail)


def alpha_for_cl(fun_cl, cl_target, lo=-2.0, hi=12.0, n=29):
    """Bracket + brentq for alpha with CL(alpha)=cl_target on [lo, hi]; None if not reachable."""
    from scipy.optimize import brentq
    a = np.linspace(lo, hi, n)
    c = np.array([fun_cl(x) for x in a])
    d = c - cl_target
    idx = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)
    if idx.size == 0:
        return None
    k = idx[0]
    if d[k] == 0:
        return float(a[k])
    return float(brentq(lambda x: fun_cl(x) - cl_target, a[k], a[k + 1], xtol=1e-6))
