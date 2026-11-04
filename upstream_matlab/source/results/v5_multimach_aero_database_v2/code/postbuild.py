"""Post-build pipeline: assemble -> hold-out validation -> analysis -> after-snapshot -> report.

Refuses to run while a MATLAB worker or the supervisor is alive, so the after-snapshot
and the database reflect a finished build.
"""
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from common import CODE, STATE, dump

STEPS = [["assemble.py"], ["validate_holdout.py"], ["analyze.py"], ["snapshot.py", "after"], ["report.py"]]


def main():
    matlab = [p.pid for p in psutil.process_iter(["name"]) if "matlab" in (p.info["name"] or "").lower()]
    if matlab and "--force" not in sys.argv:
        raise SystemExit("MATLAB still running: %s" % matlab)
    if not (STATE / "build_done.json").exists() and "--force" not in sys.argv:
        raise SystemExit("build not marked done")
    log = {"started_utc": datetime.now(timezone.utc).isoformat(), "matlab_running_at_start": matlab, "steps": []}
    for s in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable] + [str(CODE / s[0])] + s[1:], cwd=str(CODE))
        log["steps"].append({"step": " ".join(s), "exit": r.returncode, "seconds": round(time.time() - t, 1)})
        print(log["steps"][-1], flush=True)
        if r.returncode != 0:
            break
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["matlab_running_at_end"] = [p.pid for p in psutil.process_iter(["name"])
                                    if "matlab" in (p.info["name"] or "").lower()]
    dump(STATE / "postbuild_log.json", log)


if __name__ == "__main__":
    main()
