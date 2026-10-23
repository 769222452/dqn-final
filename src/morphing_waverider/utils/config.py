"""YAML configuration loading. All physical numbers live in configs/*.yaml, never in code."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def project_root() -> Path:
    """Repository root (directory containing configs/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by short name ("vehicle") or explicit path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = project_root() / "configs" / f"{p.name}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class ProjectConfig:
    vehicle: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    trajectory: dict[str, Any] = field(default_factory=dict)
    rl: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vehicle": copy.deepcopy(self.vehicle),
            "environment": copy.deepcopy(self.environment),
            "trajectory": copy.deepcopy(self.trajectory),
            "rl": copy.deepcopy(self.rl),
        }


def load_all_configs(overrides: Mapping[str, Mapping[str, Any]] | None = None) -> ProjectConfig:
    overrides = overrides or {}
    return ProjectConfig(
        vehicle=deep_update(load_config("vehicle"), overrides.get("vehicle")),
        environment=deep_update(load_config("environment"), overrides.get("environment")),
        trajectory=deep_update(load_config("trajectory"), overrides.get("trajectory")),
        rl=deep_update(load_config("rl"), overrides.get("rl")),
    )
