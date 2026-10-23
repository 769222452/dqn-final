"""Simplified ISA-like standard atmosphere (documented in docs/MODEL_ASSUMPTIONS.md).

Model
-----
Piecewise-linear temperature profile with the seven US-Standard-Atmosphere-1976 layers
up to 86 km (geopotential-altitude layers applied directly to geometric altitude, a
documented simplification). Pressure follows the hydrostatic solution in each layer,
density from the ideal gas law. Above 86 km temperature is held constant and density
decays exponentially (extrapolation, flagged as low fidelity). Below 0 m values are
clamped to sea level.

All functions accept floats, numpy arrays, or CasADi symbols (via dynamics.backend).
Units: SI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import backend as B

# (base altitude [m], lapse rate [K/m]) for the seven standard layers
_LAYERS: tuple[tuple[float, float], ...] = (
    (0.0, -0.0065),
    (11000.0, 0.0),
    (20000.0, 0.001),
    (32000.0, 0.0028),
    (47000.0, 0.0),
    (51000.0, -0.0028),
    (71000.0, -0.002),
)
_H_TOP = 86000.0


@dataclass(frozen=True)
class AtmosphereConfig:
    rho0: float = 1.225
    p0: float = 101325.0
    T0: float = 288.15
    R_gas: float = 287.05287
    gamma_gas: float = 1.4
    g0: float = 9.80665
    R_earth: float = 6371000.0
    h_min_valid: float = 0.0
    h_max_valid: float = 100000.0
    rho_floor: float = 1e-9

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AtmosphereConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})


class Atmosphere:
    """Callable standard-atmosphere object.

    >>> atm = Atmosphere()
    >>> rho = atm.density(30000.0)
    """

    def __init__(self, cfg: AtmosphereConfig | dict | None = None):
        if cfg is None:
            cfg = AtmosphereConfig()
        elif isinstance(cfg, dict):
            cfg = AtmosphereConfig.from_dict(cfg)
        self.cfg = cfg
        self._layer_T, self._layer_p = self._precompute_layer_bases()

    # ------------------------------------------------------------------ setup
    def _precompute_layer_bases(self):
        c = self.cfg
        T_b = [c.T0]
        p_b = [c.p0]
        for i in range(1, len(_LAYERS)):
            h_prev, L_prev = _LAYERS[i - 1]
            h_i = _LAYERS[i][0]
            T_prev, p_prev = T_b[-1], p_b[-1]
            T_i = T_prev + L_prev * (h_i - h_prev)
            if abs(L_prev) < 1e-12:
                p_i = p_prev * np.exp(-c.g0 * (h_i - h_prev) / (c.R_gas * T_prev))
            else:
                p_i = p_prev * (T_i / T_prev) ** (-c.g0 / (c.R_gas * L_prev))
            T_b.append(T_i)
            p_b.append(p_i)
        return T_b, p_b

    # --------------------------------------------------------------- helpers
    def _clamped_h(self, h):
        return B.clip(h, self.cfg.h_min_valid, self.cfg.h_max_valid)

    def _layer_values(self, h):
        """Return (T, p) at clamped altitude h using nested if_else over layers."""
        c = self.cfg
        h = self._clamped_h(h)
        # start from top layer and fold backwards so the innermost condition is layer 0
        n = len(_LAYERS)
        T_expr = None
        p_expr = None
        for i in reversed(range(n)):
            h_b, L = _LAYERS[i]
            T_b, p_b = self._layer_T[i], self._layer_p[i]
            dh = h - h_b
            if i == n - 1:
                # last layer up to 86 km, then isothermal extrapolation above
                h_eff = B.fmin(h, _H_TOP)
                dh_eff = h_eff - h_b
                T_top = T_b + L * (_H_TOP - h_b)
                T_i = T_b + L * dh_eff
                p_i = p_b * B.power(B.fmax(T_i / T_b, 1e-6), -c.g0 / (c.R_gas * L))
                # isothermal above 86 km
                p_above = p_i * B.exp(-c.g0 * (h - h_eff) / (c.R_gas * T_top))
                T_expr = T_i
                p_expr = p_above
            else:
                if abs(L) < 1e-12:
                    T_i = T_b + 0.0 * dh
                    p_i = p_b * B.exp(-c.g0 * dh / (c.R_gas * T_b))
                else:
                    T_i = T_b + L * dh
                    p_i = p_b * B.power(B.fmax(T_i / T_b, 1e-6), -c.g0 / (c.R_gas * L))
                cond = h < _LAYERS[i + 1][0]
                T_expr = B.if_else(cond, T_i, T_expr)
                p_expr = B.if_else(cond, p_i, p_expr)
        return T_expr, p_expr

    # ------------------------------------------------------------------ API
    def temperature(self, h):
        T, _ = self._layer_values(h)
        return T

    def pressure(self, h):
        _, p = self._layer_values(h)
        return p

    def density(self, h):
        T, p = self._layer_values(h)
        rho = p / (self.cfg.R_gas * T)
        return B.fmax(rho, self.cfg.rho_floor)

    def speed_of_sound(self, h):
        T = self.temperature(h)
        return B.sqrt(self.cfg.gamma_gas * self.cfg.R_gas * T)

    def gravity(self, h, spherical: bool = True):
        c = self.cfg
        if not spherical:
            return c.g0 + 0.0 * h
        r = c.R_earth + h
        return c.g0 * (c.R_earth / r) ** 2

    def mach(self, V, h):
        return V / self.speed_of_sound(h)

    def dynamic_pressure(self, V, h):
        return 0.5 * self.density(h) * V**2

    def evaluate(self, h) -> dict[str, float]:
        """Convenience: all scalar properties at altitude h (numpy only)."""
        T = float(self.temperature(h))
        p = float(self.pressure(h))
        return {
            "h": float(h),
            "temperature": T,
            "pressure": p,
            "density": float(self.density(h)),
            "speed_of_sound": float(self.speed_of_sound(h)),
            "gravity": float(self.gravity(h)),
        }
