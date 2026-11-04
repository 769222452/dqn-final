"""Isolate the unchanged previous audit in a killable, deadline-bound process."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tools/geometry_audit'))
from scan import audit  # exact frozen rules and native triangle algorithms

job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
result = audit(job)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
print(result['classification'], flush=True)
