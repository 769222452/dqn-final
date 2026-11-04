"""Persistent Teacher aero workers for stage 2 (resumable, append-only ledger).

* Guard: morphing_waverider.teacher_v5.bridge.guard_v6_teacher (V6 root, whitelisted
  output under results/v6_mission_geometry_stage2/teacher, matlab/active hashes equal to
  the frozen V5 manifest) + offdesign_aero_v4 and aero_worker.m hash checks.
* One request = one geometry MAT x one flight Mach x a list of alpha (deg).
* Ledger is the only source of truth. Deterministic Teacher point failures are final and
  stored as FAIL; only infrastructure faults are retried (max 2 attempts).
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import deque

import psutil

from common import (ACTIVE, AERO_WORKER, AERO_WORKER_SHA, MATLAB, MODEL_OPTIONS, MODEL_VERSION,
                    OFFDESIGN_SHA, REFERENCE, TEACHER_DIR, add_paths, append_jsonl, dump, read_jsonl, sha, utc)

LEDGER = TEACHER_DIR / "ledger.jsonl"
REQUEST_TIMEOUT_S = 240.0
STARTUP_TIMEOUT_S = 300.0
MAX_ATTEMPTS = 2
# Workers hold their memory whether busy or idle, so this only guards against
# near-exhaustion (measured: 3 idle-or-busy workers leave ~0.9 GB free on this machine).
MIN_FREE_RAM_GB = 0.4


def request_id(geometry_hash, mach, alphas):
    return f"{geometry_hash[:16]}|M{mach:g}|A{','.join(f'{a:g}' for a in alphas)}"


def guard():
    add_paths()
    from morphing_waverider.teacher_v5.bridge import guard_v6_teacher
    guard_v6_teacher(TEACHER_DIR)
    if sha(ACTIVE / "offdesign_aero_v4.m") != OFFDESIGN_SHA:
        raise RuntimeError("offdesign_aero_v4 changed")
    if sha(AERO_WORKER) != AERO_WORKER_SHA:
        raise RuntimeError("aero_worker.m differs from the V2 worker")


def committed():
    """(geometry_hash, M, alpha) -> label dict, from committed ledger records."""
    labels = {}
    for r in read_jsonl(LEDGER):
        if r.get("outcome") != "COMMITTED":
            continue
        for p in r["points"]:
            labels[(r["geometry_hash"], float(r["M_flight"]), float(p["alpha_deg"]))] = {
                **p, "request_id": r["request_id"], "geometry_id": r["geometry_id"],
                "geometry_hash": r["geometry_hash"], "M_flight": float(r["M_flight"]),
                "completed_utc": r["completed_utc"], "worker": r["worker"]}
    return labels


class Worker:
    def __init__(self, idx):
        self.idx = idx
        self.dir = TEACHER_DIR / "workers" / f"w{idx}"
        self.proc = None
        self.busy = None
        self.alive = False
        self.seq = 0
        self.restarts = 0

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in list(self.dir.glob("request_*.json")) + list(self.dir.glob("response_*.json")) + \
                [self.dir / "ready.json", self.dir / "stop"]:
            if p.exists():
                p.unlink()
        code = f"addpath('{AERO_WORKER.parent.as_posix()}'); aero_worker('{self.dir.as_posix()}','{ACTIVE.as_posix()}');"
        self.log = open(self.dir / f"matlab_{int(time.time())}.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen([str(MATLAB), "-batch", code], cwd=str(self.dir), stdout=self.log,
                                     stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.t0 = time.time()
        self.alive = False
        self.pid = None

    def ready(self):
        if not self.alive and (self.dir / "ready.json").exists():
            try:
                self.pid = json.loads((self.dir / "ready.json").read_text(encoding="utf-8")).get("pid")
                self.alive = True
            except json.JSONDecodeError:
                pass
        return self.alive

    def send(self, task):
        self.seq += 1
        name = f"request_{self.restarts:02d}_{self.seq:06d}.json"
        payload = {"request_id": task["request_id"], "geometry_mat": task["geometry_mat"],
                   "M_flight": task["M_flight"], "alpha_deg": task["alpha_deg"],
                   "reference": REFERENCE, "model_options": MODEL_OPTIONS}
        tmp = self.dir / ("tmp_" + name)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.dir / name)
        self.busy = (task, self.dir / name, self.dir / name.replace("request_", "response_"), time.time())

    def poll(self):
        if not self.busy:
            return None
        task, req, resp, t0 = self.busy
        if not resp.exists():
            return None
        try:
            data = json.loads(resp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        self.busy = None
        for p in (req, resp):
            if p.exists():
                p.unlink()
        return task, data, time.time() - t0

    def kill(self):
        for pid in (self.pid, self.proc.pid if self.proc else None):
            if pid:
                try:
                    if "matlab" in psutil.Process(int(pid)).name().lower():
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
                except psutil.NoSuchProcess:
                    pass
        self.alive = False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            (self.dir / "stop").write_text("stop", encoding="utf-8")


def run(tasks, n_workers=3, deadline=None, log=print):
    """tasks: dicts with geometry_id, geometry_hash, geometry_mat, M_flight, alpha_deg."""
    guard()
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    done = committed()
    attempts = {}
    for r in read_jsonl(LEDGER):
        attempts[r["request_id"]] = attempts.get(r["request_id"], 0) + 1
    queue = deque()
    for t in tasks:
        alphas = [a for a in t["alpha_deg"] if (t["geometry_hash"], float(t["M_flight"]), float(a)) not in done]
        if alphas:
            t = dict(t, alpha_deg=alphas, request_id=request_id(t["geometry_hash"], t["M_flight"], alphas))
            queue.append(t)
    if not queue:
        return {"requests": 0}
    log(f"[teacher] {len(queue)} requests, {sum(len(t['alpha_deg']) for t in queue)} labels")
    workers = [Worker(i) for i in range(n_workers)]
    for w in workers:
        w.start()
    stats = {"requests": 0, "points_pass": 0, "points_fail": 0, "retries": 0, "timeouts": 0}
    t_start = time.time()
    last = 0
    try:
        while queue or any(w.busy for w in workers):
            if deadline and time.time() > deadline and not any(w.busy for w in workers):
                log("[teacher] deadline reached; remaining requests left for resume")
                break
            ram = psutil.virtual_memory().available / 2 ** 30
            cap = 1 if ram < MIN_FREE_RAM_GB else n_workers
            busy_n = sum(1 for w in workers if w.busy)
            for w in workers:
                if not w.alive:
                    if w.ready():
                        log(f"[teacher] w{w.idx} ready in {time.time() - w.t0:.0f}s")
                    elif w.proc.poll() is not None or time.time() - w.t0 > STARTUP_TIMEOUT_S:
                        w.kill()
                        w.restarts += 1
                        if w.restarts > 5:
                            raise RuntimeError("Teacher worker cannot start")
                        w.start()
                    continue
                got = w.poll()
                if got:
                    task, data, wall = got
                    _commit(task, data, wall, w, attempts, stats, queue)
                    busy_n -= 1
                elif w.busy and time.time() - w.busy[3] > REQUEST_TIMEOUT_S:
                    task = w.busy[0]
                    w.busy = None
                    stats["timeouts"] += 1
                    _commit(task, {"status": "TIMEOUT", "points": []}, REQUEST_TIMEOUT_S, w, attempts, stats, queue)
                    w.kill()
                    w.restarts += 1
                    w.start()
                    busy_n -= 1
                    continue
                elif w.pid and not psutil.pid_exists(int(w.pid)):
                    if w.busy:
                        task = w.busy[0]
                        w.busy = None
                        _commit(task, {"status": "WORKER_DIED", "points": []}, 0, w, attempts, stats, queue)
                    w.restarts += 1
                    w.start()
                    continue
                if not w.busy and queue and busy_n < cap and not (deadline and time.time() > deadline):
                    w.send(queue.popleft())
                    busy_n += 1
            if time.time() - last > 60:
                last = time.time()
                log(f"[teacher] {stats} remaining={len(queue)} elapsed={time.time() - t_start:.0f}s ram={ram:.1f}GB")
            time.sleep(0.05)
    finally:
        for w in workers:
            w.stop()
        t_end = time.time() + 20
        while time.time() < t_end and any(w.proc and w.proc.poll() is None for w in workers):
            time.sleep(0.2)
        for w in workers:
            if w.proc and w.proc.poll() is None:
                w.kill()
    stats["wall_s"] = time.time() - t_start
    stats["remaining"] = len(queue)
    log(f"[teacher] done {stats}")
    return stats


def _commit(task, data, wall, w, attempts, stats, queue):
    rid = task["request_id"]
    attempts[rid] = attempts.get(rid, 0) + 1
    pts = data.get("points") or []
    pts = [pts] if isinstance(pts, dict) else list(pts)
    ok = data.get("status") == "PASS" and sorted(float(p["alpha_deg"]) for p in pts) == sorted(task["alpha_deg"])
    final = ok or attempts[rid] >= MAX_ATTEMPTS
    rec = {"request_id": rid, "geometry_id": task["geometry_id"], "geometry_hash": task["geometry_hash"],
           "geometry_mat": task["geometry_mat"], "M_flight": task["M_flight"], "alpha_deg": task["alpha_deg"],
           "attempt": attempts[rid], "outcome": "COMMITTED" if final else "RETRY",
           "request_status": data.get("status"), "message": str(data.get("message", ""))[:300],
           "worker": f"w{w.idx}", "wall_s": wall, "completed_utc": utc(), "model_version_expected": MODEL_VERSION,
           "points": [{k: p.get(k) for k in ("alpha_deg", "status", "CL", "CD", "Cm", "error_id", "error_message",
                                              "model_version", "kernel_s")} for p in pts]}
    if final and not ok:
        have = {float(p["alpha_deg"]) for p in pts}
        rec["points"] += [{"alpha_deg": a, "status": "FAIL", "CL": None, "CD": None, "Cm": None,
                           "error_id": "INFRA:" + str(data.get("status")), "error_message": rec["message"],
                           "model_version": None, "kernel_s": None} for a in task["alpha_deg"] if a not in have]
    append_jsonl(LEDGER, rec)
    stats["requests"] += 1
    if final:
        for p in rec["points"]:
            stats["points_pass" if p["status"] == "PASS" else "points_fail"] += 1
    else:
        stats["retries"] += 1
        queue.appendleft(task)
