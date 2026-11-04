"""V5 preservation check for stage 2 (read-only on D:/v5-final; writes only the stage-2 result).

Reuses provenance/freeze.py::inventory and compares with provenance/v5_before.json, like
`freeze.py end`, without overwriting the stage-1 evidence file v5_after.json.
"""
from __future__ import annotations

import importlib.util
import json

from common import OUT, ROOT, dump, utc

spec = importlib.util.spec_from_file_location("freeze", ROOT / "provenance/freeze.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)

before = json.loads((ROOT / "provenance/v5_before.json").read_text(encoding="utf-8"))["files"]
after = freeze.inventory(freeze.SOURCE)
added = sorted(after.keys() - before.keys())
removed = sorted(before.keys() - after.keys())
changed_hash = sorted(k for k in after.keys() & before.keys()
                      if before[k]["sha256"] is not None and before[k] != after[k])
changed_stat = sorted(k for k in after.keys() & before.keys()
                      if before[k]["sha256"] is None and before[k] != after[k])
res = {"utc": utc(), "source": str(freeze.SOURCE), "files": len(after), "added": added, "removed": removed,
       "changed_hash": changed_hash, "changed_stat": changed_stat,
       "V5_PRESERVED": not (added or removed or changed_hash or changed_stat)}
dump(OUT / "v5_preservation.json", res)
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}))
