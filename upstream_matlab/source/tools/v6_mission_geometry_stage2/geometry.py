"""Native MATLAB generation + frozen audit + mesh metrics for stage-2 candidates.

Uses the unmodified V5 geometry wrapper (tools/geometry_audit/run_geometry_audit_batch.m)
and build_geometry_core in matlab/active. Each candidate gets its own case directory;
a candidate with generation_status.json / audit.json / metrics.json is never redone.
"""
from __future__ import annotations

import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import (ACTIVE, AUDIT_TOOLS, CASES, G42_INPUTS10, MATLAB, OUT, PARAMS, add_paths,
                    candidate_id, design_space_label, dump, inputs10, load, sha, utc)

PER_CANDIDATE_TIMEOUT_S = 120
STARTUP_TIMEOUT_S = 300


def make_candidate(ma_design, z, source, **extra):
    z = [float(v) for v in z]
    cid = candidate_id(ma_design, z, prefix=extra.pop("prefix", "S2"))
    return {"candidate_id": cid, "Ma_design": float(ma_design), "z": z, "source": source,
            "design_space": design_space_label(z), "inputs10": inputs10(ma_design, z),
            "case_dir": str(CASES / cid), **extra}


def _launch(batch, stage_dir, tag):
    code = (f"addpath('{AUDIT_TOOLS.as_posix()}'); "
            f"run_geometry_audit_batch('{batch.as_posix()}','{ACTIVE.as_posix()}');")
    logf = open(stage_dir / f"matlab_{tag}.log", "w", encoding="utf-8")
    proc = subprocess.Popen([str(MATLAB), "-batch", code], cwd=str(stage_dir), stdout=logf,
                            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    return proc, logf


def _kill(proc):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


def generate(cands, stage, sessions=3, log=print):
    """Parallel independent MATLAB sessions; resumable; per-candidate hard timeout."""
    stage_dir = OUT / "generation" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = stage_dir / "generation_ledger.json"
    ledger = load(ledger_path) if ledger_path.exists() else {"stage": stage, "sessions": [], "aero_calls": 0}
    for c in cands:
        Path(c["case_dir"]).mkdir(parents=True, exist_ok=True)
        if not (Path(c["case_dir"]) / "candidate.json").exists():
            dump(Path(c["case_dir"]) / "candidate.json", c)
    rounds = 0
    while True:
        pending = [c for c in cands if not (Path(c["case_dir"]) / "generation_status.json").exists()]
        if not pending or rounds > 6:
            break
        rounds += 1
        chunks = [pending[i::sessions] for i in range(min(sessions, len(pending)))]
        procs = []
        for k, chunk in enumerate(chunks):
            tag = f"r{len(ledger['sessions']) + 1:03d}"
            jobs = [{"design_index": j + 1, "case_dir": c["case_dir"].replace("\\", "/"),
                     "inputs10": c["inputs10"]} for j, c in enumerate(chunk)]
            batch = stage_dir / f"batch_{tag}.json"
            dump(batch, jobs)
            proc, logf = _launch(batch, stage_dir, tag)
            rec = {"tag": tag, "pid": proc.pid, "utc": utc(), "n": len(chunk)}
            ledger["sessions"].append(rec)
            procs.append({"proc": proc, "log": logf, "chunk": chunk, "rec": rec, "t0": time.monotonic(),
                          "watch": {}})
        dump(ledger_path, ledger)
        while any(p["proc"].poll() is None for p in procs):
            for p in procs:
                if p["proc"].poll() is not None:
                    continue
                started_any = False
                for c in p["chunk"]:
                    d = Path(c["case_dir"])
                    if (d / "started.json").exists():
                        started_any = True
                        if (d / "generation_status.json").exists():
                            continue
                        p["watch"].setdefault(c["candidate_id"], time.monotonic())
                        if time.monotonic() - p["watch"][c["candidate_id"]] > PER_CANDIDATE_TIMEOUT_S:
                            _kill(p["proc"])
                            dump(d / "generation_status.json",
                                 {"status": "GENERATION_FAILED", "timeout": True,
                                  "error_identifier": "SUPERVISOR:HardTimeout",
                                  "error_message": f"hard per-candidate timeout {PER_CANDIDATE_TIMEOUT_S} s",
                                  "elapsed_s": PER_CANDIDATE_TIMEOUT_S})
                            log(f"[{stage}] timeout {c['candidate_id']}")
                if not started_any and time.monotonic() - p["t0"] > STARTUP_TIMEOUT_S:
                    _kill(p["proc"])
                    raise RuntimeError("MATLAB_STARTUP_TIMEOUT")
            time.sleep(0.5)
        for p in procs:
            p["log"].close()
            p["rec"]["wall_s"] = time.monotonic() - p["t0"]
            p["rec"]["returncode"] = p["proc"].returncode
        dump(ledger_path, ledger)
        n_ok = sum((Path(c["case_dir"]) / "generation_status.json").exists() and
                   load(Path(c["case_dir"]) / "generation_status.json")["status"] == "GENERATED" for c in pending)
        log(f"[{stage}] round {rounds}: {len(pending)} pending, {n_ok} generated")
    return ledger


def _audit_one(c):
    add_paths()
    import scan  # frozen audit (tools/geometry_audit), imported unmodified
    from geometry_metrics import metrics  # V2 metrics code, imported unmodified
    case = Path(c["case_dir"])
    target = case / "audit.json"
    if target.exists():
        r = load(target)
    else:
        job = {"design_index": 0, "original_row_index": None,
               "parameters": dict(zip(PARAMS, c["z"])), "inputs10": c["inputs10"],
               "case_dir": str(case), "stored_L_D": None, "stored_Vol_eff": None}
        r = scan.audit(job)
        r["candidate_id"] = c["candidate_id"]
        dump(target, r)
    mpath = case / "metrics.json"
    if r.get("generation_status") == "GENERATED" and not mpath.exists():
        dump(mpath, metrics(case / "native_geometry.mat"))
    return c["candidate_id"]


def audit(cands, workers=6, log=print):
    todo = [c for c in cands if (Path(c["case_dir"]) / "generation_status.json").exists()
            and not ((Path(c["case_dir"]) / "audit.json").exists()
                     and ((Path(c["case_dir"]) / "metrics.json").exists()
                          or load(Path(c["case_dir"]) / "audit.json").get("generation_status") != "GENERATED"))]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for cid in pool.map(_audit_one, todo, chunksize=1):
                pass
        log(f"[audit] {len(todo)} audited")
    return [summary(c) for c in cands]


def summary(c):
    """One flat record per candidate (generation, audit, metrics, identity)."""
    case = Path(c["case_dir"])
    rec = {k: c.get(k) for k in ("candidate_id", "Ma_design", "source", "design_space", "case_dir")}
    rec.update(dict(zip(PARAMS, c["z"])))
    gs = case / "generation_status.json"
    st = load(gs) if gs.exists() else {"status": "NOT_RUN"}
    rec["generation_status"] = st.get("status")
    rec["generation_error"] = st.get("error_identifier") or ""
    rec["generation_message"] = (st.get("error_message") or "")[:200]
    au = case / "audit.json"
    if au.exists():
        a = load(au)
        rec["classification"] = a.get("classification")
        rec["failure_reason"] = a.get("failure_reason")
        rec["geometry_hash"] = a.get("geometry_hash")
        rec["mat_sha256"] = a.get("mat_sha256")
        rec["min_section_thickness"] = a.get("min_section_thickness")
        rec["surface_intersection_count"] = a.get("surface_intersection_count")
        rec["self_intersection_count"] = a.get("self_intersection_count")
    else:
        rec["classification"] = "GENERATION_FAILED" if rec["generation_status"] == "GENERATION_FAILED" else "NOT_AUDITED"
    mp = case / "metrics.json"
    if mp.exists():
        m = load(mp)
        for k in ("volume_full", "grid_thickness_max", "span", "length", "planform_area_full",
                  "volumetric_efficiency", "wetted_area_lower_half", "wetted_area_upper_half"):
            rec[k] = m.get(k)
    rec["native_geometry_mat"] = str(case / "native_geometry.mat") if rec["generation_status"] == "GENERATED" else ""
    return rec


def surfaces(mat_path):
    from scipy.io import loadmat
    m = loadmat(mat_path)
    lower = np.stack([m["wav_" + a] for a in ("x", "y", "z")], -1).astype(float)
    upper = np.stack([m["wav_up_" + a] for a in ("x", "y", "z")], -1).astype(float)
    return lower, upper


def shape_distance(mat_a, mat_b):
    """Point-wise displacement between two native 50x50 skins with identical grid topology.

    Native index correspondence is a parametric (not physical-material) correspondence.
    """
    la, ua = surfaces(mat_a)
    lb, ub = surfaces(mat_b)
    if la.shape != lb.shape or ua.shape != ub.shape:
        return {"compatible_topology": False}
    d = np.concatenate([np.linalg.norm(la - lb, axis=-1).ravel(), np.linalg.norm(ua - ub, axis=-1).ravel()])
    return {"compatible_topology": True, "max_displacement_m": float(d.max()),
            "rms_displacement_m": float(np.sqrt(np.mean(d ** 2))), "grid_shape": list(la.shape[:2])}


def g42_regression(log=print):
    """Regenerate G42 inside stage 2 and compare its geometry hash with the frozen record."""
    c = make_candidate(15.0, G42_INPUTS10[1:5] + [G42_INPUTS10[7], G42_INPUTS10[9]], "g42_regression",
                       prefix="REG")
    generate([c], "regression", sessions=1, log=log)
    audit([c], workers=1, log=log)
    s = summary(c)
    from common import load as _l
    ok = s.get("geometry_hash") == "d35074a0032e37b69278fb81afbb44250363255a6f8c880e8b89208e30401f7a"
    s["regression"] = "PASS" if ok and s.get("classification") == "VALID" else "FAIL"
    dump(OUT / "g42_regression.json", s)
    return s
