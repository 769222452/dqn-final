"""JSON helpers and the bootstrap confidence interval."""
import json
import os
from pathlib import Path

import numpy as np


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def save_json(path, value):
    """Atomic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(_jsonable(value), f, indent=1, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cluster_bootstrap_ci(values_by_cluster, level=.95, resamples=10000, seed=0):
    """Percentile CI of the mean over clusters (each cluster: list of paired differences), resampling clusters."""
    cl = [np.asarray(v, float) for v in values_by_cluster if len(v)]
    sums = np.array([v.sum() for v in cl])
    ns = np.array([len(v) for v in cl])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cl), (resamples, len(cl)))
    means = sums[idx].sum(1)/ns[idx].sum(1)
    return [float(np.quantile(means, (1-level)/2)), float(np.quantile(means, (1+level)/2))]
