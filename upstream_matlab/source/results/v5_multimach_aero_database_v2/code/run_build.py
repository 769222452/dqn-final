"""Full V2 Teacher aero database supervisor: <=3 persistent MATLAB workers, resumable.

* The 14 h clock starts at the first MATLAB worker launch and is persisted in
  state/clock.json; restarts never reset it.
* state/ledger.jsonl is the only source of truth (fsync'd append). A request is
  done once a COMMITTED record exists; RETRY records count attempts.
* On start: kill workers left by a dead supervisor, harvest responses that were
  written but never committed, delete un-answered request files, then dispatch.
* Retries only for infrastructure faults (timeout, worker death, request-level
  load failure, malformed response, transient MATLAB error ids). Deterministic
  Teacher FAILs are committed as final.
* Dispatch is throttled on low free RAM and paused on low disk.
"""
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone

import psutil

from common import (ACTIVE, BUDGET_S, CODE, HARD_STOP_S, MATLAB, MAX_REQUEST_ATTEMPTS, MAX_WORKERS, MIN_FREE_DISK_GB,
                    MIN_FREE_RAM_GB, MODEL_OPTIONS, OFFDESIGN_SHA, OUT, PILOT_WORKER_SHA, REFERENCE, REQUEST_TIMEOUT_S,
                    RESUME_FREE_RAM_GB, STATE, STOP_DISPATCH_S, TRANSIENT_ERROR_IDS, append_jsonl, dump, load,
                    read_jsonl, sha)

WORKERS_DIR = OUT / "workers"
LEDGER = STATE / "ledger.jsonl"
CLOCK = STATE / "clock.json"
LOCK = STATE / "supervisor.lock"
DONE_MARK = STATE / "build_done.json"
STARTUP_TIMEOUT_S = 240.0
MAX_WORKER_RESTARTS = 25
PROGRESS_EVERY_S = 60.0
CRITICAL_FREE_RAM_GB = 0.6


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    line = "[%s] %s" % (utc(), msg)
    print(line, flush=True)
    with open(STATE / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_ram_gb():
    return psutil.virtual_memory().available / 2 ** 30


def free_disk_gb():
    return shutil.disk_usage(str(OUT)).free / 2 ** 30


def kill_tree(pid):
    if not pid:
        return
    try:
        p = psutil.Process(int(pid))
    except (psutil.NoSuchProcess, ValueError):
        return
    if "matlab" not in p.name().lower():
        log("refusing to kill pid %s (%s): not MATLAB" % (pid, p.name()))
        return
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def as_points(data):
    pts = data.get("points") or []
    return [pts] if isinstance(pts, dict) else list(pts)


def classify(task, data):
    """-> (final: bool, reason). final=False means an infrastructure fault worth a retry."""
    if data.get("request_id") != task["request_id"]:
        return False, "RESPONSE_ID_MISMATCH"
    if data.get("status") != "PASS":
        return False, "REQUEST_FAIL:" + str(data.get("message", ""))[:300]
    pts = as_points(data)
    got = sorted(float(p.get("alpha_deg")) for p in pts)
    if got != sorted(float(a) for a in task["alpha_deg"]):
        return False, "ALPHA_SET_MISMATCH"
    for p in pts:
        if p.get("status") != "PASS" and str(p.get("error_id", "")) in TRANSIENT_ERROR_IDS:
            return False, "TRANSIENT_POINT_ERROR:" + str(p.get("error_id"))
    return True, ""


class Build:
    def __init__(self):
        self.tasks = load(STATE / "tasks.json")["tasks"]
        self.by_id = {t["request_id"]: t for t in self.tasks}
        recs, bad = read_jsonl(LEDGER, STATE / "ledger_bad_lines.jsonl")
        if bad:
            log("ledger has %d unparsable line(s) %s; they are ignored (requests will be redone)" % (len(bad), bad))
        self.done = {}
        self.attempts = Counter()
        for r in recs:
            self.attempts[r["request_id"]] += 1
            if r.get("outcome") == "COMMITTED" and r["request_id"] not in self.done:
                self.done[r["request_id"]] = r
        self.queue = deque(t for t in self.tasks if t["request_id"] not in self.done)
        self.events = deque(maxlen=30)
        self.stats = Counter()

    # -------------------------------------------------- ledger
    def record(self, task, data, worker, wall, final, reason, pid=None, source="live"):
        rid = task["request_id"]
        if rid in self.done:
            self.stats["duplicate_responses_ignored"] += 1
            return
        self.attempts[rid] += 1
        n = self.attempts[rid]
        if not final and n >= MAX_REQUEST_ATTEMPTS:
            final = True
            reason = "INFRA_FAILED_AFTER_%d_ATTEMPTS:%s" % (n, reason)
        rec = {"request_id": rid, "kind": task["kind"], "geometry_id": task["geometry_id"],
               "geometry_hash": task["geometry_hash"], "M_flight": task["M_flight"], "alpha_deg": task["alpha_deg"],
               "attempt": n, "outcome": "COMMITTED" if final else "RETRY", "reason": reason, "worker": worker,
               "matlab_pid": pid, "wall_s": wall, "elapsed_at_done_s": self.el(), "completed_utc": utc(),
               "source": source, "response": data}
        append_jsonl(LEDGER, rec)
        if final:
            self.done[rid] = rec
            pts = as_points(data or {})
            npass = sum(1 for p in pts if p.get("status") == "PASS")
            self.stats["points_pass"] += npass
            self.stats["points_fail"] += len(task["alpha_deg"]) - npass
            self.stats["requests_committed"] += 1
            if reason:
                self.stats["requests_infra_failed"] += 1
                self.events.append("%s INFRA_FAILED %s" % (utc(), rid))
        else:
            self.stats["retries"] += 1
            self.events.append("%s RETRY %s %s" % (utc(), rid, reason[:120]))
            self.queue.appendleft(task)

    # -------------------------------------------------- clock
    def start_clock_if_needed(self):
        if self.clock.get("t0_unix") is None:
            self.clock["t0_unix"] = time.time()
            self.clock["t0_utc"] = utc()
            self.clock["t0_definition"] = "first MATLAB worker launch"
            dump(CLOCK, self.clock)

    def el(self):
        t0 = self.clock.get("t0_unix")
        return 0.0 if t0 is None else time.time() - t0


class Worker:
    def __init__(self, b, idx, runtag):
        self.b, self.idx, self.runtag = b, idx, runtag
        self.dir = WORKERS_DIR / ("w%d" % idx)
        self.proc = None
        self.pid = None
        self.busy = None
        self.alive = False
        self.restarts = 0
        self.seq = 0
        self.dead = False

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in ("ready.json", "stop"):
            if (self.dir / p).exists():
                (self.dir / p).unlink()
        for p in self.dir.glob("request_*.json"):
            if not (self.dir / p.name.replace("request_", "response_")).exists():
                p.unlink()
        self.b.start_clock_if_needed()
        code = "addpath('%s'); aero_worker('%s','%s');" % (CODE.as_posix(), self.dir.as_posix(), ACTIVE.as_posix())
        logf = open(self.dir / ("matlab_%s_%02d.log" % (self.runtag, self.restarts)), "w", encoding="utf-8")
        self.proc = subprocess.Popen([str(MATLAB), "-batch", code], cwd=str(self.dir), stdout=logf,
                                     stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.started = time.time()
        self.alive = False
        self.pid = None
        log("w%d: launched launcher pid=%d (restart %d)" % (self.idx, self.proc.pid, self.restarts))
        self.save_pids()

    def save_pids(self):
        p = STATE / "worker_pids.json"
        cur = load(p) if p.exists() else {}
        cur["w%d" % self.idx] = {"launcher": self.proc.pid if self.proc else None, "matlab": self.pid}
        dump(p, cur)

    def check_ready(self):
        r = self.dir / "ready.json"
        if not self.alive and r.exists():
            try:
                info = json.loads(r.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return False
            self.pid = info.get("pid")
            self.alive = True
            self.save_pids()
            log("w%d: READY in %.1fs matlab pid=%s" % (self.idx, time.time() - self.started, self.pid))
        return self.alive

    def send(self, task):
        self.seq += 1
        name = "request_%s_%02d_%06d.json" % (self.runtag, self.restarts, self.seq)
        payload = {"request_id": task["request_id"], "geometry_mat": task["geometry_mat"],
                   "M_flight": task["M_flight"], "alpha_deg": task["alpha_deg"],
                   "reference": REFERENCE, "model_options": MODEL_OPTIONS}
        tmp = self.dir / ("tmp_" + name)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
        tmp.replace(self.dir / name)
        self.busy = (task, self.dir / name, self.dir / name.replace("request_", "response_"), time.time())

    def poll(self):
        if not self.busy:
            return None
        task, reqp, resp, t_sent = self.busy
        if not resp.exists():
            return None
        try:
            data = json.loads(resp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        self.busy = None
        return task, reqp, data, time.time() - t_sent

    def kill(self):
        for pid in (self.pid, self.proc.pid if self.proc else None):
            kill_tree(pid)
        self.alive = False

    def drop_busy_request(self):
        if self.busy:
            reqp = self.busy[1]
            if reqp.exists():
                reqp.unlink()

    def fault(self, reason):
        """Worker died / timed out: return the in-flight task as RETRY and restart."""
        now = self.b.el()
        if self.busy:
            task = self.busy[0]
            self.drop_busy_request()
            self.b.record(task, {"status": reason, "points": []}, "w%d" % self.idx,
                          time.time() - self.busy[3], False, reason, self.pid)
            self.busy = None
        self.kill()
        self.b.events.append("%s w%d %s" % (utc(), self.idx, reason))
        if self.restarts < MAX_WORKER_RESTARTS and now < STOP_DISPATCH_S - 300:
            self.restarts += 1
            self.b.stats["worker_restarts"] += 1
            self.start()
        else:
            self.proc = None
            self.dead = True
            log("w%d: retired (restarts=%d)" % (self.idx, self.restarts))


def cleanup_previous(b):
    """Kill MATLAB left by a dead supervisor and harvest its uncommitted responses."""
    p = STATE / "worker_pids.json"
    if p.exists():
        for w, pids in load(p).items():
            for pid in (pids.get("matlab"), pids.get("launcher")):
                if pid and psutil.pid_exists(int(pid)):
                    log("killing leftover %s pid=%s" % (w, pid))
                    kill_tree(pid)
        time.sleep(2)
    harvested = removed = 0
    for d in sorted(WORKERS_DIR.glob("w*")):
        for f in list(d.glob("tmp_request_*")):
            f.unlink()
        for reqp in sorted(d.glob("request_*.json")):
            resp = d / reqp.name.replace("request_", "response_")
            if resp.exists():
                try:
                    data = json.loads(resp.read_text(encoding="utf-8"))
                    rid = json.loads(reqp.read_text(encoding="utf-8"))["request_id"]
                except (json.JSONDecodeError, OSError, KeyError):
                    reqp.unlink()
                    removed += 1
                    continue
                task = b.by_id.get(rid)
                if task and rid not in b.done:
                    final, reason = classify(task, data)
                    b.record(task, data, d.name, None, final, reason, None, source="orphan_harvest")
                    harvested += 1
            reqp.unlink()
            removed += 1
    b.queue = deque(t for t in b.tasks if t["request_id"] not in b.done)
    log("cleanup: harvested %d orphan response(s), removed %d request file(s)" % (harvested, removed))


def write_progress(b, workers, final=False, note=""):
    el = b.el()
    tasks_total = len(b.tasks)
    pts_total = sum(len(t["alpha_deg"]) for t in b.tasks)
    pts_done = sum(len(b.by_id[r]["alpha_deg"]) for r in b.done)
    rate_hist = b.rate_hist
    rate_hist.append((time.time(), pts_done))
    while len(rate_hist) > 2 and rate_hist[-1][0] - rate_hist[0][0] > 1800:
        rate_hist.popleft()
    dt = rate_hist[-1][0] - rate_hist[0][0]
    rate = (rate_hist[-1][1] - rate_hist[0][1]) / dt if dt > 60 else None
    remaining = pts_total - pts_done
    kinds = Counter(b.by_id[r]["kind"] for r in b.done)
    prog = {
        "updated_utc": utc(), "final": final, "note": note,
        "clock_t0_utc": b.clock.get("t0_utc"), "elapsed_s": round(el, 1), "elapsed_h": round(el / 3600, 3),
        "budget_s": BUDGET_S, "stop_dispatch_at_s": STOP_DISPATCH_S, "hard_stop_at_s": HARD_STOP_S,
        "tasks_total": tasks_total, "tasks_committed": len(b.done), "tasks_remaining": tasks_total - len(b.done),
        "committed_by_kind": dict(kinds),
        "new_points_total": pts_total, "new_points_committed": pts_done, "new_points_remaining": remaining,
        "fraction_done": round(pts_done / pts_total, 5) if pts_total else 1.0,
        "this_run": dict(b.stats), "points_per_s_recent": rate,
        "eta_s": (remaining / rate) if rate else None,
        "eta_within_budget": (el + remaining / rate < STOP_DISPATCH_S) if rate else None,
        "workers": [{"w": w.idx, "alive": w.alive, "dead": w.dead, "matlab_pid": w.pid, "restarts": w.restarts,
                     "busy": w.busy[0]["request_id"] if w.busy else None} for w in workers],
        "throttle": b.throttle, "free_ram_gb": round(free_ram_gb(), 2), "free_disk_gb": round(free_disk_gb(), 2),
        "supervisor_resumes": b.clock.get("resumes", 0), "recent_events": list(b.events)}
    dump(OUT / "progress.json", prog)
    dump(OUT / "resume_manifest.json", {
        "updated_utc": utc(), "how_to_resume": "python code/guardian.py (or code/run_build.py); "
                                               "state/clock.json keeps the original t0; done requests are skipped",
        "tasks_file": "state/tasks.json", "tasks_sha256": b.tasks_sha, "ledger": "state/ledger.jsonl",
        "reused_labels": "state/reused_labels.jsonl", "clock": "state/clock.json",
        "committed_request_ids": len(b.done), "remaining_request_ids": [t["request_id"] for t in b.queue][:50],
        "remaining_count": len(b.queue) + sum(1 for w in workers if w.busy),
        "dedup_rule": "first COMMITTED record per request_id wins; later responses ignored",
        "retry_rule": "RETRY only for infrastructure faults, max %d attempts per request" % MAX_REQUEST_ATTEMPTS})


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            other = int(LOCK.read_text().strip())
        except ValueError:
            other = None
        if other and psutil.pid_exists(other) and other != psutil.Process().pid:
            try:
                name = psutil.Process(other).name().lower()
            except psutil.NoSuchProcess:
                name = ""
            if "python" in name:
                print("another supervisor (pid %d) is running; exiting" % other)
                return 3
    LOCK.write_text(str(psutil.Process().pid))
    if DONE_MARK.exists():
        log("build already marked done; nothing to do")
        return 0

    assert sha(ACTIVE / "offdesign_aero_v4.m") == OFFDESIGN_SHA, "offdesign_aero_v4 changed"
    assert sha(CODE / "aero_worker.m") == PILOT_WORKER_SHA, "aero_worker.m differs from the pilot worker"
    b = Build()
    b.tasks_sha = sha(STATE / "tasks.json")
    b.clock = load(CLOCK) if CLOCK.exists() else {"t0_unix": None, "resumes": -1}
    b.clock["resumes"] = b.clock.get("resumes", -1) + 1
    b.clock.setdefault("supervisor_starts", []).append(utc())
    dump(CLOCK, b.clock)
    b.rate_hist = deque()
    b.throttle = "none"
    resume = b.clock["resumes"]
    runtag = "r%02d" % resume
    log("supervisor start pid=%d resume#%d elapsed=%.1fs committed=%d remaining=%d"
        % (psutil.Process().pid, resume, b.el(), len(b.done), len(b.queue)))
    cleanup_previous(b)
    dump(STATE / "identity.json", {
        "offdesign_aero_v4_sha256": OFFDESIGN_SHA, "aero_worker_sha256": PILOT_WORKER_SHA,
        "reference": REFERENCE, "model_options": MODEL_OPTIONS, "max_workers": MAX_WORKERS,
        "budget_s": BUDGET_S, "stop_dispatch_s": STOP_DISPATCH_S, "hard_stop_s": HARD_STOP_S,
        "request_timeout_s": REQUEST_TIMEOUT_S, "max_request_attempts": MAX_REQUEST_ATTEMPTS})

    workers = []
    reason = "queue_empty"
    if b.queue and b.el() < STOP_DISPATCH_S:
        workers = [Worker(b, i, runtag) for i in range(MAX_WORKERS)]
        for w in workers:
            w.start()
        last_progress = 0.0
        last_res = 0.0
        low_disk = False
        while True:
            now = b.el()
            if time.time() - last_res > 10:
                last_res = time.time()
                ram, disk = free_ram_gb(), free_disk_gb()
                if ram < MIN_FREE_RAM_GB and b.throttle != "ram_1_worker":
                    b.throttle = "ram_1_worker"
                    log("free RAM %.2f GB < %.2f: throttling dispatch to 1 worker" % (ram, MIN_FREE_RAM_GB))
                elif b.throttle == "ram_1_worker" and ram >= RESUME_FREE_RAM_GB:
                    b.throttle = "none"
                    log("free RAM %.2f GB recovered: full dispatch" % ram)
                if ram < CRITICAL_FREE_RAM_GB:
                    idle = [w for w in workers if w.alive and not w.busy and not w.dead]
                    if len([w for w in workers if w.alive]) > 1 and idle:
                        w = idle[-1]
                        log("free RAM %.2f GB critical: stopping idle w%d" % (ram, w.idx))
                        w.kill()
                        w.proc, w.dead = None, True
                if disk < MIN_FREE_DISK_GB and not low_disk:
                    low_disk = True
                    log("free disk %.2f GB < %.2f: dispatch paused" % (disk, MIN_FREE_DISK_GB))
                elif low_disk and disk >= MIN_FREE_DISK_GB + 1:
                    low_disk = False
                    log("free disk recovered: dispatch resumed")
            busy_n = sum(1 for w in workers if w.busy)
            cap = 0 if low_disk else (1 if b.throttle == "ram_1_worker" else MAX_WORKERS)
            for w in workers:
                if w.proc is None:
                    continue
                if not w.alive:
                    if w.check_ready():
                        pass
                    elif w.proc.poll() is not None:
                        w.fault("STARTUP_EXIT_%s" % w.proc.poll())
                    elif time.time() - w.started > STARTUP_TIMEOUT_S:
                        w.fault("STARTUP_TIMEOUT")
                    continue
                got = w.poll()
                if got:
                    task, reqp, data, wall = got
                    final, why = classify(task, data)
                    b.record(task, data, "w%d" % w.idx, wall, final, why, w.pid)
                    if reqp.exists():
                        reqp.unlink()
                    busy_n -= 1
                elif w.busy and time.time() - w.busy[3] > REQUEST_TIMEOUT_S:
                    log("w%d: TIMEOUT %s" % (w.idx, w.busy[0]["request_id"]))
                    b.stats["timeouts"] += 1
                    w.fault("TIMEOUT")
                    busy_n -= 1
                    continue
                elif (not psutil.pid_exists(int(w.pid))) if w.pid else (w.proc.poll() is not None):
                    log("w%d: MATLAB exited unexpectedly" % w.idx)
                    had = bool(w.busy)
                    w.fault("WORKER_DIED")
                    busy_n -= int(had)
                    continue
                if w.alive and not w.busy and b.queue and now < STOP_DISPATCH_S and busy_n < cap:
                    w.send(b.queue.popleft())
                    busy_n += 1
            busy = [w for w in workers if w.busy]
            if time.time() - last_progress > PROGRESS_EVERY_S:
                last_progress = time.time()
                write_progress(b, workers)
            if not b.queue and not busy:
                reason = "queue_empty"
                break
            if now >= STOP_DISPATCH_S and not busy:
                reason = "budget_stop_dispatch"
                break
            if now >= HARD_STOP_S:
                reason = "budget_hard_stop"
                log("HARD STOP; abandoning %d in-flight request(s)" % len(busy))
                break
            if all(w.proc is None for w in workers):
                reason = "all_workers_dead"
                break
            time.sleep(0.05)
        for w in workers:
            if w.proc is not None:
                (w.dir / "stop").write_text("stop", encoding="utf-8")
        deadline = time.time() + 20
        while time.time() < deadline and any(w.proc and w.proc.poll() is None for w in workers):
            time.sleep(0.2)
        for w in workers:
            if w.proc and w.proc.poll() is None:
                w.kill()
        log("workers stopped: reason=%s elapsed=%.1fs stats=%s" % (reason, b.el(), dict(b.stats)))
    elif b.queue:
        reason = "budget_stop_dispatch"
    write_progress(b, workers, final=True, note=reason)
    b.clock.setdefault("runs", []).append({"stopped_utc": utc(), "elapsed_s": b.el(), "reason": reason,
                                           "stats": dict(b.stats), "remaining": len(b.queue)})
    dump(CLOCK, b.clock)
    if reason in ("queue_empty", "budget_stop_dispatch", "budget_hard_stop"):
        dump(DONE_MARK, {"utc": utc(), "reason": reason, "elapsed_s": b.el(), "committed": len(b.done),
                         "remaining": len(b.queue)})
        return 0
    return 4


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        STATE.mkdir(parents=True, exist_ok=True)
        with open(STATE / "supervisor.log", "a", encoding="utf-8") as f:
            f.write("[%s] CRASH\n%s\n" % (utc(), traceback.format_exc()))
        rc = 1
    finally:
        try:
            if LOCK.exists() and LOCK.read_text().strip() == str(psutil.Process().pid):
                LOCK.unlink()
        except OSError:
            pass
    sys.exit(rc)
