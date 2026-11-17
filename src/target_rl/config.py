"""Task configuration (configs/task.yaml) and repository paths."""
import os
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


def library_name(cfg=None):
    """Active geometry library: environment variable TARGET_RL_LIBRARY, else configs/task.yaml aero.active."""
    cfg = cfg or load()
    name = os.environ.get('TARGET_RL_LIBRARY') or cfg['aero']['active']
    if name not in cfg['aero']['libraries']:
        raise ValueError(f'unknown library {name}; choose from {sorted(cfg["aero"]["libraries"])}')
    return name


def library(cfg=None):
    cfg = cfg or load()
    return dict(cfg['aero']['libraries'][library_name(cfg)], name=library_name(cfg))


def runs_dir(cfg=None):
    """Results of the active library: runs/<library>/ (the aerodynamic cache runs/cache/ is shared)."""
    return ROOT/'runs'/library_name(cfg)


def seed_range(pair):
    lo, hi = pair
    return list(range(int(lo), int(hi)+1))
