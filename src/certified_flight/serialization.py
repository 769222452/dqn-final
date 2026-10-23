"""JSON conversion of evidence records (NumPy values, non-finite floats) and atomic writes."""
import json
import os
import time
from pathlib import Path

import numpy as np


def clean(value):
    """Plain JSON types; non-finite floats become None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    """Write cleaned JSON through a hidden temporary file and an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name('.' + path.name + '.writing')
    with pending.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(40):  # tolerate transient Windows sharing locks
        try:
            os.replace(pending, path)
            break
        except PermissionError:
            if os.name != 'nt' or attempt == 39:
                raise
            time.sleep(.25)


def append_jsonl(path, value):
    """Append one cleaned JSON record as a line."""
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(clean(value), allow_nan=False) + '\n')
