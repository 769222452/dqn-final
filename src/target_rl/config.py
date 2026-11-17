"""Task configuration (configs/task.yaml) and repository paths."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load(path=None):
    p = Path(path) if path else ROOT/'configs'/'task.yaml'
    cfg = yaml.safe_load(p.read_text(encoding='utf-8'))
    if cfg.get('schema') != 'target_task/1':
        raise ValueError('unexpected configuration schema')
    return cfg


def seed_range(pair):
    lo, hi = pair
    return list(range(int(lo), int(hi)+1))
