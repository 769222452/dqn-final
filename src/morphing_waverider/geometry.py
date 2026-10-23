"""Morphing geometry vector  eta = [Lambda, b, kappa_1, ..., kappa_n].

V0 uses eta = [sweep_angle, span]. The spec is data-driven from configs/vehicle.yaml,
so adding parameters later is a config change, not a code change. Every module that
touches geometry (aero, generator, validator, OCP, env) goes through GeometrySpec and
never hard-codes the parameter count or ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class GeometryParameter:
    name: str
    symbol: str
    unit: str
    min: float
    max: float
    nominal: float
    rate_max: float


class GeometrySpec:
    def __init__(self, parameters: Sequence[GeometryParameter]):
        if not parameters:
            raise ValueError("GeometrySpec needs at least one parameter")
        self.parameters = tuple(parameters)
        self.names = tuple(p.name for p in self.parameters)
        self.lower = np.array([p.min for p in self.parameters], dtype=float)
        self.upper = np.array([p.max for p in self.parameters], dtype=float)
        self.nominal = np.array([p.nominal for p in self.parameters], dtype=float)
        self.rate_max = np.array([p.rate_max for p in self.parameters], dtype=float)
        self.range = self.upper - self.lower
        if np.any(self.upper <= self.lower):
            raise ValueError("geometry bounds must satisfy max > min")
        if np.any(self.nominal < self.lower) or np.any(self.nominal > self.upper):
            raise ValueError("nominal geometry must lie inside bounds")

    # -------------------------------------------------------------- factory
    @classmethod
    def from_config(cls, vehicle_cfg: dict[str, Any]) -> "GeometrySpec":
        params = [
            GeometryParameter(
                name=p["name"], symbol=p.get("symbol", p["name"]), unit=p.get("unit", "-"),
                min=float(p["min"]), max=float(p["max"]), nominal=float(p["nominal"]),
                rate_max=float(p.get("rate_max", np.inf)),
            )
            for p in vehicle_cfg["geometry"]["parameters"]
        ]
        return cls(params)

    # ---------------------------------------------------------------- utils
    @property
    def dim(self) -> int:
        return len(self.parameters)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def as_array(self, eta) -> np.ndarray:
        arr = np.asarray(eta, dtype=float).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"geometry must have {self.dim} entries, got {arr.shape[0]}")
        return arr

    def clip(self, eta) -> np.ndarray:
        return np.clip(self.as_array(eta), self.lower, self.upper)

    def is_within_bounds(self, eta, tol: float = 1e-9) -> bool:
        arr = self.as_array(eta)
        return bool(np.all(arr >= self.lower - tol) and np.all(arr <= self.upper + tol)
                    and np.all(np.isfinite(arr)))

    def normalize(self, eta) -> np.ndarray:
        """Map physical eta -> [0, 1]^dim."""
        return (self.as_array(eta) - self.lower) / (self.upper - self.lower)

    def denormalize(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        return self.lower + np.clip(z, 0.0, 1.0) * (self.upper - self.lower)

    def normalized_distance(self, eta_a, eta_b) -> float:
        """L2 distance in normalized coordinates (dimensionless morphing cost)."""
        return float(np.linalg.norm(self.normalize(eta_a) - self.normalize(eta_b)))

    def to_dict(self, eta) -> dict[str, float]:
        arr = self.as_array(eta)
        return {n: float(v) for n, v in zip(self.names, arr)}

    def describe(self) -> list[dict[str, Any]]:
        return [p.__dict__ for p in self.parameters]
