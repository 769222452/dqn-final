"""Logging and per-run result directory management.

Every pipeline / RL run writes to  results/<run_name>/  containing:
    config_snapshot.yaml, trajectory.csv, constraint_margins.csv,
    summary.json, solver_status.json, figures/
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import project_root

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


class RunDirectory:
    """Creates results/<name>_<timestamp>/ and provides typed writers."""

    def __init__(self, run_name: str, base: Path | None = None, timestamp: bool = True):
        base = Path(base) if base else project_root() / "results"
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
        name = f"{run_name}_{stamp}" if stamp else run_name
        self.path = base / name
        self.figures = self.path / "figures"
        self.figures.mkdir(parents=True, exist_ok=True)

    def save_config(self, cfg: dict[str, Any], seed: int | None = None) -> Path:
        p = self.path / "config_snapshot.yaml"
        payload = {"seed": seed, **cfg} if seed is not None else cfg
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        return p

    def save_json(self, name: str, data: Any) -> Path:
        p = self.path / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        return p

    def save_csv(self, name: str, columns: dict[str, Any]) -> Path:
        import pandas as pd

        p = self.path / name
        pd.DataFrame(columns).to_csv(p, index=False)
        return p

    def figure_path(self, name: str) -> Path:
        return self.figures / name

    def __str__(self) -> str:
        return str(self.path)
