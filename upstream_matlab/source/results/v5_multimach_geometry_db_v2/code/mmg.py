"""Shared layer for the V5 multimach waverider geometry database V2.

Independent call layer. It reuses, without modifying, the confirmed active
MATLAB geometry source (``matlab/active``) and the frozen native-mesh audit
code/rules (``tools/geometry_audit``, ``results/v5_geometry_audit``).

The only correction applied relative to the historical driver
(``tools/geometry_audit/generate.py``) is the MATLAB search path: the old
driver passed ``teacher_matlab/active``, which does not exist in this project
root, so this layer passes ``matlab/active`` instead. The 16 geometry source
files of both trees were verified byte-identical before this task started; only
the aerodynamic file ``comptue_conicalfield.m`` differs and it is never loaded
by the geometry core.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(str(Path(__file__).resolve().parents[3]))
OUT = ROOT / "results/v5_multimach_geometry_db_v2"
OLD_AUDIT = ROOT / "results/v5_geometry_audit"
AUDIT_TOOLS = ROOT / "tools/geometry_audit"
ACTIVE = ROOT / "matlab/active"
MATLAB = Path("D:/matlab/bin/matlab.exe")
OPTIMIZER = Path("D:/mat/.code/surrogate_assisted_ga_wave_rider_6.m")

PARAMS = ("a1", "a2", "b1", "b2", "d1", "a_x")
MA_DESIGN_GRID = (4.0, 5.0, 6.0, 7.0, 8.0)
MA_DESIGN_REFERENCE = 15.0
MA_DESIGN_PILOT = (4.0, 6.0, 8.0)

# Extended shock-angle upper bound for the V2 database. The historical
# optimizer bounds (3..8 deg, parsed from the real source) admit no geometry at
# all at Ma_design 4 and 5: stage 0 recorded 0 successes in 116 MATLAB calls
# there, while uniform angles of 12 deg (Ma 4) and 10 deg (Ma 5) succeed. The
# upper bound is therefore raised for this database only, and every record
# carries design_space = FROZEN or EXTENDED so the two populations never mix.
EXTENDED_ANGLE_UPPER = 20.0
G42_Z = (5.633749605922896, 7.796675937828777, 3.698465133623518,
         3.0116105313696075, 0.11606537709080691, 0.9944782795122264)
G42_GEOMETRY_HASH = "d35074a0032e37b69278fb81afbb44250363255a6f8c880e8b89208e30401f7a"
G42_MAT_SHA256 = "3077f1748f1beb73bb77e9aa0061bb11dfe4a971462de8bf27873c83ff382dbe"
G42_DESIGN_INDEX = 42

PER_CANDIDATE_TIMEOUT_S = 120
STARTUP_TIMEOUT_S = 240
MAX_CONSECUTIVE_FAILURES = 6


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(p)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write_csv(p, rows, fieldnames=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or (rows[0].keys() if rows else []))
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audit_module():
    """Import the frozen audit code in place (no copy, no modification)."""
    for d in (str(AUDIT_TOOLS),):
        if d not in sys.path:
            sys.path.insert(0, d)
    import scan  # noqa: E402  (frozen project module)
    return scan


# --------------------------------------------------------------------------
# design space
# --------------------------------------------------------------------------

class DesignSpace:
    """Design-variable bounds and fixed inputs parsed from the real optimizer
    source, with the design Mach promoted from a fixed 15 to a task variable."""

    def __init__(self):
        code = OPTIMIZER.read_text(encoding="utf-8-sig")
        b = {k: [float(v) for v in re.search(r"\b" + k + r"\s*=\s*\[([^]]+)\]", code)[1]
                 .replace(",", " ").split()] for k in ("lb", "ub")}
        ev = code.split("function [success, L_D, vol_frac] = evaluatePoint(x)", 1)[1].split("function hv", 1)[0]
        self.fixed = {k: float(re.search(r"\b" + k + r"\s*=\s*([-+\d.]+)\s*;", ev)[1])
                      for k in ("Ma", "c1", "c2", "AOA")}
        self.lower = np.asarray(b["lb"], float)
        self.upper = np.asarray(b["ub"], float)
        if self.lower.shape != (6,) or self.upper.shape != (6,):
            raise ValueError("UNEXPECTED_BOUND_DIMENSION")
        self.extended_lower = self.lower.copy()
        self.extended_upper = self.upper.copy()
        self.extended_upper[:4] = EXTENDED_ANGLE_UPPER
        old = load(ROOT / "results/v5_teacher/design_spec.json")
        if (self.fixed["Ma"] != old["Ma_design"] or self.fixed["c1"] != old["c1"]
                or self.fixed["c2"] != old["c2"] or self.fixed["AOA"] != old["AOA_design_prog_deg"]):
            raise ValueError("FIXED_DESIGN_INPUTS_CHANGED")
        self.historical_ma_design = self.fixed["Ma"]
        self.identity = {
            str(OPTIMIZER): sha(OPTIMIZER),
            str(OLD_AUDIT / "classification_rules.json"): sha(OLD_AUDIT / "classification_rules.json"),
            str(OLD_AUDIT / "pareto_geometry_audit.csv"): sha(OLD_AUDIT / "pareto_geometry_audit.csv"),
            str(OLD_AUDIT / "source_manifest.json"): sha(OLD_AUDIT / "source_manifest.json"),
            **{str(p): sha(p) for p in sorted(AUDIT_TOOLS.glob("*.py"))},
            str(AUDIT_TOOLS / "run_geometry_audit_batch.m"): sha(AUDIT_TOOLS / "run_geometry_audit_batch.m"),
            **{str(p): sha(p) for p in sorted(ACTIVE.glob("*.m"))},
        }

    def in_bounds(self, z) -> bool:
        z = np.asarray(z, float)
        return bool(np.all(z >= self.lower) and np.all(z <= self.upper))

    def in_extended_bounds(self, z) -> bool:
        z = np.asarray(z, float)
        return bool(np.all(z >= self.extended_lower) and np.all(z <= self.extended_upper))

    def label(self, z) -> str:
        if self.in_bounds(z):
            return "FROZEN"
        if self.in_extended_bounds(z):
            return "EXTENDED"
        return "OUT_OF_RANGE"

    def inputs(self, ma_design, z):
        a1, a2, b1, b2, d1, ax = (float(v) for v in z)
        return [float(ma_design), a1, a2, b1, b2, self.fixed["c1"], self.fixed["c2"], d1,
                self.fixed["AOA"], ax]

    def spec(self):
        return {
            "parameter_names": list(PARAMS),
            "bounds_from_optimizer": {"lower": self.lower.tolist(), "upper": self.upper.tolist()},
            "bounds_source": str(OPTIMIZER),
            "extended_bounds_this_database": {"lower": self.extended_lower.tolist(),
                                              "upper": self.extended_upper.tolist()},
            "extended_bounds_reason": ("The frozen shock-angle upper bound of 8 deg admits no "
                                       "geometry at Ma_design 4 or 5 (0 successes in 116 MATLAB "
                                       "calls, stage 0). Raised to "
                                       f"{EXTENDED_ANGLE_UPPER:g} deg for this database only; every "
                                       "record carries design_space = FROZEN or EXTENDED."),
            "fixed_design_inputs": {k: v for k, v in self.fixed.items() if k != "Ma"},
            "Ma_design": {"status": "TASK_VARIABLE", "historical_fixed_value": self.historical_ma_design,
                          "grid": list(MA_DESIGN_GRID), "reference_node": MA_DESIGN_REFERENCE},
            "geometry_source": str(ACTIVE / "build_geometry_core.m"),
            "audit_rules": "native-triangle-audit-1 (frozen 2026-10-01, reused unmodified)",
            "mesh_version": "native50x50-stream40-v5",
            "scale_status": "ASSUMED_1_METRE_PER_PROGRAM_UNIT",
            "boundary": "OPEN_BASE",
            "source_identity": self.identity,
        }

    def verify(self):
        for p, expected in self.identity.items():
            if sha(p) != expected:
                raise RuntimeError("FROZEN_SOURCE_CHANGED:" + p)


def valid_seeds():
    """The 26 VALID Pareto designs of the frozen audit, as (design_index, z)."""
    rows = read_csv(OLD_AUDIT / "pareto_geometry_audit.csv")
    if len(rows) != 52 or sum(r["classification"] == "VALID" for r in rows) != 26:
        raise RuntimeError("AUDIT_CLASSIFICATION_COUNTS_CHANGED")
    seeds = []
    for r in rows:
        if r["classification"] != "VALID":
            continue
        seeds.append({"design_index": int(r["design_index"]),
                      "original_row_index": int(r["original_row_index"]),
                      "z": tuple(float(r[p]) for p in PARAMS),
                      "geometry_hash_md15": r["geometry_hash"],
                      "stored_L_D": float(r["stored_L_D"]),
                      "stored_Vol_eff": float(r["stored_Vol_eff"])})
    return seeds


def candidate_id(ma_design, z, prefix="C"):
    d = hashlib.sha256(json.dumps({"ma": float(ma_design), "z": [float(v) for v in z]}).encode()).hexdigest()[:10]
    return f"{prefix}_MD{ma_design:g}_{d}"


# --------------------------------------------------------------------------
# generation (MATLAB supervisor)
# --------------------------------------------------------------------------

def is_domain_outcome(status) -> bool:
    """True when MATLAB returned a geometry or a typed generator domain error."""
    if status.get("status") == "GENERATED":
        return True
    return str(status.get("error_identifier") or "").startswith("V5:Geometry")


def make_job(space, candidate, job_index):
    case = OUT / "cases" / candidate["candidate_id"]
    candidate["case_dir"] = str(case)
    candidate["job_index"] = job_index
    candidate["inputs10"] = space.inputs(candidate["Ma_design"], candidate["z"])
    return {"design_index": job_index, "case_dir": str(case).replace("\\", "/"),
            "inputs10": candidate["inputs10"]}


def generate(space, candidates, stage, log=print):
    """Serial MATLAB sessions, batched per session, resumable.

    A candidate is skipped when ``generation_status.json`` already exists.
    """
    space.verify()
    if not MATLAB.exists():
        raise RuntimeError("MATLAB_NOT_FOUND")
    stage_dir = OUT / "generation" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    budget_path = stage_dir / "generation_budget.json"
    budget = load(budget_path) if budget_path.exists() else {
        "stage": stage, "matlab": str(MATLAB), "active_dir": str(ACTIVE),
        "per_candidate_timeout_s": PER_CANDIDATE_TIMEOUT_S, "aero_calls": 0, "ocp_calls": 0,
        "training_calls": 0, "sessions": [], "calls": []}
    done = {c["candidate_id"] for c in candidates
            if (Path(c["case_dir"]) / "generation_status.json").exists()}
    consecutive = 0
    while True:
        pending = [c for c in candidates if c["candidate_id"] not in done]
        if not pending:
            break
        session = len(budget["sessions"]) + 1
        batch = stage_dir / f"batch_{session:02d}.json"
        jobs = [make_job(space, c, c["job_index"]) for c in pending]
        dump(batch, jobs)
        for c in pending:
            dump(Path(c["case_dir"]) / "candidate.json", c)
        code = (f"addpath('{AUDIT_TOOLS.as_posix()}'); "
                f"run_geometry_audit_batch('{batch.as_posix()}','{ACTIVE.as_posix()}');")
        logfile = stage_dir / f"matlab_session_{session:02d}.log"
        fh = logfile.open("w", encoding="utf-8")
        start = time.monotonic()
        proc = subprocess.Popen([str(MATLAB), "-batch", code], cwd=str(stage_dir),
                                stdout=fh, stderr=subprocess.STDOUT)
        record = {"session": session, "pid": proc.pid, "utc": now(),
                  "candidates": [c["candidate_id"] for c in pending], "log": logfile.name}
        budget["sessions"].append(record)
        dump(budget_path, budget)
        by_index = {c["job_index"]: c for c in pending}
        watch, seen, timed_out = {}, set(), False
        while proc.poll() is None:
            for c in pending:
                d = Path(c["case_dir"])
                idx = c["job_index"]
                if (d / "started.json").exists() and idx not in watch:
                    watch[idx] = time.monotonic()
                if (d / "generation_status.json").exists() and idx not in seen:
                    s = load(d / "generation_status.json")
                    seen.add(idx)
                    done.add(c["candidate_id"])
                    budget["calls"].append({"candidate_id": c["candidate_id"], "job_index": idx,
                                            "Ma_design": c["Ma_design"], "status": s["status"],
                                            "elapsed_s": s.get("elapsed_s"),
                                            "error_identifier": s.get("error_identifier"),
                                            "error_message": s.get("error_message")})
                    budget["geometry_calls"] = len(budget["calls"])
                    dump(budget_path, budget)
                    log(f"[{stage}] {c['candidate_id']} Md={c['Ma_design']:g} {s['status']} "
                        f"{s.get('elapsed_s', 0):.2f}s")
                    # A V5:Geometry* identifier is the generator reporting that
                    # this design is outside its domain; that is an expected
                    # search outcome. Only infrastructure failures (no typed
                    # identifier, or a supervisor timeout) are counted here.
                    consecutive = 0 if is_domain_outcome(s) else consecutive + 1
                elif idx in watch and idx not in seen and time.monotonic() - watch[idx] > PER_CANDIDATE_TIMEOUT_S:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                    s = {"design_index": idx, "status": "GENERATION_FAILED",
                         "elapsed_s": time.monotonic() - watch[idx], "timeout": True,
                         "error_identifier": "SUPERVISOR:HardTimeout",
                         "error_message": f"Supervisor hard per-candidate timeout {PER_CANDIDATE_TIMEOUT_S} s"}
                    dump(Path(by_index[idx]["case_dir"]) / "generation_status.json", s)
                    budget["calls"].append({"candidate_id": by_index[idx]["candidate_id"], "job_index": idx,
                                            "Ma_design": by_index[idx]["Ma_design"],
                                            "status": "GENERATION_FAILED", "elapsed_s": s["elapsed_s"],
                                            "error_identifier": s["error_identifier"],
                                            "error_message": s["error_message"]})
                    done.add(by_index[idx]["candidate_id"])
                    consecutive += 1
                    timed_out = True
                    break
            if timed_out:
                break
            if not watch and time.monotonic() - start > STARTUP_TIMEOUT_S:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                fh.close()
                raise RuntimeError("MATLAB_STARTUP_TIMEOUT")
            time.sleep(.2)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        fh.close()
        for c in pending:
            d = Path(c["case_dir"])
            if (d / "generation_status.json").exists() and c["candidate_id"] not in done:
                s = load(d / "generation_status.json")
                budget["calls"].append({"candidate_id": c["candidate_id"], "job_index": c["job_index"],
                                        "Ma_design": c["Ma_design"], "status": s["status"],
                                        "elapsed_s": s.get("elapsed_s"),
                                        "error_identifier": s.get("error_identifier"),
                                        "error_message": s.get("error_message")})
                done.add(c["candidate_id"])
        record["wall_s"] = time.monotonic() - start
        record["returncode"] = proc.returncode
        budget["geometry_calls"] = len(budget["calls"])
        budget["geometry_elapsed_s_total"] = sum(c.get("elapsed_s") or 0 for c in budget["calls"])
        dump(budget_path, budget)
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(f"{consecutive} consecutive geometry failures; stopped")
        if proc.returncode not in (0, None) and not timed_out and not any(
                c["candidate_id"] not in done for c in pending):
            log(f"[{stage}] session {session} exited {proc.returncode}; all candidates resolved")
    return budget


# --------------------------------------------------------------------------
# validation (frozen audit, reused unmodified)
# --------------------------------------------------------------------------

AUDIT_COLUMNS = ["candidate_id", "Ma_design", "design_space", "source", "seed_design_index",
                 "path_id", "path_k",
                 "generation_status", "generation_elapsed_s", "geometry_hash", "mat_sha256",
                 "classification", "failure_reason", "min_section_thickness",
                 "negative_thickness_fraction", "surface_intersection_count",
                 "self_intersection_count", "degenerate_face_count", "allowed_tip_degenerate_count",
                 "normal_issue_count", "single_valued_graph", "body_transform_invariant",
                 "length_scale", "audit_elapsed_s", *PARAMS, "inputs10"]


def audit_candidates(candidates, log=print, resume=True):
    scan = audit_module()
    results = []
    for c in candidates:
        case = Path(c["case_dir"])
        target = case / "audit.json"
        if resume and target.exists():
            r = load(target)
        else:
            job = {"design_index": c["job_index"], "original_row_index": c.get("seed_design_index"),
                   "parameters": dict(zip(PARAMS, [float(v) for v in c["z"]])),
                   "inputs10": c["inputs10"], "case_dir": str(case),
                   "stored_L_D": None, "stored_Vol_eff": None}
            r = scan.audit(job)
            r["candidate_id"] = c["candidate_id"]
            r["Ma_design"] = c["Ma_design"]
            r["design_space"] = c.get("design_space")
            r["source"] = c.get("source")
            r["seed_design_index"] = c.get("seed_design_index")
            r["path_id"] = c.get("path_id")
            r["path_k"] = c.get("path_k")
            dump(target, r)
        results.append(r)
        log(f"[audit] {c['candidate_id']} Md={c['Ma_design']:g} {r['classification']} "
            f"xsec={r.get('surface_intersection_count')} self={r.get('self_intersection_count')} "
            f"tmin={r.get('min_section_thickness')}")
    return results


def _audit_one(c):
    """Worker entry point for the parallel audit pool."""
    scan = audit_module()
    case = Path(c["case_dir"])
    target = case / "audit.json"
    if target.exists():
        return load(target)
    job = {"design_index": c["job_index"], "original_row_index": c.get("seed_design_index"),
           "parameters": dict(zip(PARAMS, [float(v) for v in c["z"]])),
           "inputs10": c["inputs10"], "case_dir": str(case),
           "stored_L_D": None, "stored_Vol_eff": None}
    r = scan.audit(job)
    for k in ("candidate_id", "Ma_design", "design_space", "source", "seed_design_index",
              "path_id", "path_k"):
        r[k] = c.get(k)
    dump(target, r)
    return r


def audit_parallel(candidates, workers=6, log=print):
    """Same frozen audit, run over independent case directories."""
    from concurrent.futures import ProcessPoolExecutor
    todo = [c for c in candidates
            if (Path(c["case_dir"]) / "generation_status.json").exists()]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_audit_one, todo, chunksize=1):
            results.append(r)
            log(f"[audit] {r['candidate_id']} Md={r['Ma_design']:g} {r['classification']} "
                f"xsec={r.get('surface_intersection_count')} tmin={r.get('min_section_thickness')}")
    return results


def audit_row(r):
    row = {k: r.get(k) for k in AUDIT_COLUMNS}
    row.update({p: r["parameters"][p] for p in PARAMS})
    row["inputs10"] = json.dumps(r["inputs10"])
    return row
