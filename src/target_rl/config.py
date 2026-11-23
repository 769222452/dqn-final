"""Task configuration (configs/task.yaml, or the file named by TARGET_RL_CONFIG) and repository paths."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load(path=None):
    """Configuration file: `path`, else the environment variable TARGET_RL_CONFIG (relative to the repository root
    or absolute; used by study A: configs/study_a.yaml), else configs/task.yaml."""
    path = path or os.environ.get('TARGET_RL_CONFIG')
    p = Path(path) if path else ROOT/'configs'/'task.yaml'
    if not p.is_absolute() and not p.exists():
        p = ROOT/p
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


def runs_dir(cfg=None, library=None):
    """Results of the current round: runs/<tag>/<library>/ (round 1, without a tag: runs/<library>/).

    The preregistered target regions (region_path) and the aerodynamic cache (runs/cache/) are shared by all
    rounds."""
    cfg = cfg or load()
    lib = library or library_name(cfg)
    tag = cfg.get('experiment', {}).get('tag')
    return ROOT/'runs'/tag/lib if tag else ROOT/'runs'/lib


def round1_dir(library=None, cfg=None):
    """Round-1 results (runs/<library>/), used to re-evaluate the round-1 policies on the new test set."""
    return ROOT/'runs'/(library or library_name(cfg))


def region_path(library=None, cfg=None):
    cfg = cfg or load()
    ex = cfg.get('experiment', {})
    if ex.get('region_tag'):                                  # v3b: the region of another tag (same task)
        return ROOT/'runs'/ex['region_tag']/(library or library_name(cfg))/'region.json'
    if ex.get('own_region'):                                  # v3: a region of its own (other task, plant with mismatch)
        return runs_dir(cfg, library)/'region.json'
    return ROOT/'runs'/(library or library_name(cfg))/'region.json'


def seed_range(pair):
    lo, hi = pair
    return list(range(int(lo), int(hi)+1))
