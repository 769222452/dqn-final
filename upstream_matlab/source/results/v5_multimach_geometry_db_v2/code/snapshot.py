"""Read-only identity snapshot of D:/v5-final for the multimach geometry DB V2 task.

    python snapshot.py before|after

Hashes all source/code trees, stats the bulk data trees, and ignores only this
task's own output directory. Any difference between before and after therefore
proves that the old project was or was not modified.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(str(Path(__file__).resolve().parents[3]))
TASK_OUT = ROOT / "results/v5_multimach_geometry_db_v2"
HASHED = ("src", "configs", "matlab", "tools", "scripts", "tests", "provenance", "docs")
STATTED = ("results", "data", "runs", "audit_artifacts")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def skip(p):
    return "__pycache__" in p.parts or p.suffix == ".pyc" or ".pytest_cache" in p.parts


def snap():
    hashed, stat = {}, {}
    for t in HASHED:
        d = ROOT / t
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and not skip(p):
                hashed[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in sorted(ROOT.glob("*")):
        if p.is_file():
            s = p.stat()
            hashed[p.name] = sha(p) if s.st_size < 50_000_000 else f"size={s.st_size};mtime={s.st_mtime_ns}"
    for t in STATTED:
        d = ROOT / t
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and TASK_OUT not in p.parents and not skip(p):
                s = p.stat()
                stat[p.relative_to(ROOT).as_posix()] = [s.st_size, s.st_mtime_ns]
    pyc = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                 if p.is_file() and skip(p) and TASK_OUT not in p.parents)
    return {"hashed": hashed, "stat": stat, "cache_files": pyc}


def main():
    mode = sys.argv[1]
    out_dir = TASK_OUT / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = out_dir / "baseline_before.json"
    s = snap() | {"utc": datetime.now(timezone.utc).isoformat()}
    if mode == "before":
        if baseline.exists():
            raise SystemExit("baseline exists; refusing to overwrite the before snapshot")
        baseline.write_text(json.dumps(s), encoding="utf-8")
        print(json.dumps({"hashed": len(s["hashed"]), "stat": len(s["stat"]),
                          "cache_files": len(s["cache_files"]), "utc": s["utc"]}))
        return
    b = json.loads(baseline.read_text(encoding="utf-8"))
    rep = {"before_utc": b["utc"], "after_utc": s["utc"],
           "task_output_dir_excluded": TASK_OUT.relative_to(ROOT).as_posix()}
    for k in ("hashed", "stat"):
        rep[k] = {"n_before": len(b[k]), "n_after": len(s[k]),
                  "changed": sorted(x for x in b[k] if x in s[k] and s[k][x] != b[k][x]),
                  "removed": sorted(x for x in b[k] if x not in s[k]),
                  "added": sorted(x for x in s[k] if x not in b[k])}
    rep["cache_files_added"] = sorted(set(s["cache_files"]) - set(b["cache_files"]))
    rep["PROJECT_UNCHANGED"] = "YES" if not any(rep[k][j] for k in ("hashed", "stat")
                                                for j in ("changed", "removed", "added")) else "NO"
    (out_dir / "verify_after.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {j: (len(x) if isinstance(x, list) else x) for j, x in v.items()})
                      for k, v in rep.items()}))


if __name__ == "__main__":
    main()
