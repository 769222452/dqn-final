"""Stage-1 preservation check: hash every stage-1 artefact, compare with the stage-2 start snapshot.

Files whose mtime is later than the stage-1 closure (V6_LABELS.json mtime + 120 s) at snapshot time
are listed separately (they were already touched by stage 2 and are documented changes).
"""
from __future__ import annotations

import json
import sys

from common import OUT, ROOT, dump, sha, utc

STAGE1 = ["results/fixed_geometry_baselines", "src/mission_v6", "tests_v6", "configs/mission_1200s.yaml",
          "V6_REPORT_ZH.md", "README.md", "docs/MISSION_DEFINITION_ZH.md", "docs/ARCHITECTURE_ZH.md",
          "docs/CHANGELOG_FROM_V5_ZH.md", "docs/NEXT_STAGES_ZH.md", "docs/legacy/V5_README_COPY.md",
          "src/morphing_waverider/teacher_v5/bridge.py", "src/morphing_waverider/teacher_v5/teacher.py"]
SNAP = ROOT / "_runtime/stage1_snapshot_at_stage2.json"


def files():
    for rel in STAGE1:
        p = ROOT / rel
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts)


def snapshot():
    closure = (ROOT / "results/fixed_geometry_baselines/V6_LABELS.json").stat().st_mtime + 120
    rec = {str(p.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha(p), "mtime": p.stat().st_mtime,
                                                        "after_stage1_closure": p.stat().st_mtime > closure}
           for p in files()}
    SNAP.write_text(json.dumps({"utc": utc(), "stage1_closure_mtime": closure, "files": rec}, indent=1),
                    encoding="utf-8")
    return rec


def verify():
    snap = json.loads(SNAP.read_text(encoding="utf-8"))
    now = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files()}
    old = snap["files"]
    changed = sorted(k for k in old if k in now and now[k] != old[k]["sha256"])
    removed = sorted(k for k in old if k not in now)
    added = sorted(k for k in now if k not in old)
    pre = sorted(k for k, v in old.items() if v["after_stage1_closure"])
    res = {"utc": utc(), "snapshot_utc": snap["utc"], "n_files": len(now), "changed_since_snapshot": changed,
           "removed": removed, "added": added, "modified_by_stage2_before_snapshot": pre}
    dump(OUT / "stage1_preservation.json", res)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        r = snapshot()
        print(len(r), [k for k, v in r.items() if v["after_stage1_closure"]])
    else:
        print(json.dumps(verify(), indent=1))
