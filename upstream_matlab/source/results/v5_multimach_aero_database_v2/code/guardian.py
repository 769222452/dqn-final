"""Keeps run_build.py alive until it marks the build done or the budget is spent.

Launch detached:  Start-Process python guardian.py -WindowStyle Hidden
A second guardian exits immediately (state/guardian.lock with a live python pid).
"""
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from common import CODE, HARD_STOP_S, STATE, load

LOCK = STATE / "guardian.lock"
MAX_RESTARTS = 60


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    with open(STATE / "guardian.log", "a", encoding="utf-8") as f:
        f.write("[%s] %s\n" % (utc(), msg))


def elapsed():
    p = STATE / "clock.json"
    if not p.exists():
        return 0.0
    t0 = load(p).get("t0_unix")
    return 0.0 if t0 is None else time.time() - t0


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    me = psutil.Process().pid
    if LOCK.exists():
        try:
            other = int(LOCK.read_text().strip())
            if other != me and psutil.pid_exists(other) and "python" in psutil.Process(other).name().lower():
                return 3
        except (ValueError, psutil.NoSuchProcess):
            pass
    LOCK.write_text(str(me))
    log("guardian start pid=%d elapsed=%.1fs" % (me, elapsed()))
    starts = 0
    try:
        while True:
            if (STATE / "build_done.json").exists():
                log("build_done present; guardian exits")
                return 0
            if elapsed() >= HARD_STOP_S:
                log("budget hard stop reached; guardian exits")
                return 0
            if starts > MAX_RESTARTS:
                log("too many supervisor restarts; guardian gives up")
                return 5
            starts += 1
            out = open(STATE / "supervisor_stdout.log", "a", encoding="utf-8")
            r = subprocess.run([sys.executable, str(CODE / "run_build.py")], cwd=str(CODE), stdout=out,
                               stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            out.close()
            log("supervisor run #%d exit=%s elapsed=%.1fs" % (starts, r.returncode, elapsed()))
            if r.returncode == 3:
                time.sleep(60)
            elif r.returncode != 0:
                time.sleep(20)
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(me):
            LOCK.unlink()


if __name__ == "__main__":
    sys.exit(main())
