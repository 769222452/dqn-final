"""Morphing actuator dynamics (V0.5).

The generator commands a geometry eta_cmd; the vehicle cannot jump there. The actuator
turns the command into a continuous, bounded, rate-limited trajectory:

    eta_dot = sat( K * (eta_cmd - eta), -eta_rate_max, +eta_rate_max )
    eta_min <= eta(t) <= eta_max                               (component-wise)

`K` (1/s) and the minimum substep count come from configs/vehicle.yaml -> morphing_actuator;
eta_rate_max and the bounds come from the GeometrySpec. The integration is explicit Euler on
sub-steps; between sub-steps eta moves along a straight line, so
|eta(t2)-eta(t1)|/(t2-t1) <= eta_rate_max for any t1 < t2.

Substep rule (V0.6): for an interval of length dt

    stability_required = ceil(dt / min(1/max(K), 1 s))     # K*h <= 1: first-order lag never overshoots
    n_substeps         = max(min_substeps_per_interval, stability_required)
    h                  = dt / n_substeps

so the configured value is a real lower bound on the resolution (config 5 -> 5 substeps of
0.6 s on a 3 s interval; config 10 -> 10 substeps of 0.3 s). `substep_plan(dt)` reports
configured / stability-required / actual counts and the actual actuator timestep. The
pre-V0.6 key `substeps_per_interval` is accepted with the same (minimum) meaning.

`predict()` returns the geometry at the N+1 shooting nodes of an OCP horizon; the OCP reads
this trajectory as a *parameter* (prescribed eta), never as a decision variable. It also
keeps the full substep trajectory (`t_sub`, `eta_sub`) so that a dense audit can evaluate
eta(t) between the nodes with exactly the actuator's own integration (`eta_at(t)`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry import GeometrySpec


@dataclass
class ActuatorTrajectory:
    t: np.ndarray          # (N+1,) node times relative to the start
    eta: np.ndarray        # (N+1, n_eta) geometry at the nodes
    eta_cmd: np.ndarray    # (n_eta,) command that produced the trajectory
    converged_at: float | None   # first node time at which |eta - eta_cmd| < tol on every axis
    t_sub: np.ndarray | None = None      # (N*n_sub+1,) actuator substep times (dense record)
    eta_sub: np.ndarray | None = None    # (N*n_sub+1, n_eta) geometry at every actuator substep
    diagnostics: dict[str, Any] = field(default_factory=dict)   # substep_plan() of the interval

    @property
    def final(self) -> np.ndarray:
        return self.eta[-1].copy()

    def rates(self) -> np.ndarray:
        dt = np.diff(self.t)[:, None]
        return np.diff(self.eta, axis=0) / dt

    @property
    def n_substeps_per_interval(self) -> int | None:
        return None if not self.diagnostics else int(self.diagnostics["actual_substeps"])

    def eta_at(self, t) -> np.ndarray:
        """eta at arbitrary time(s) t (relative to the start): linear interpolation on the
        actuator substep grid -- exact for the explicit-Euler actuator, whose eta is piecewise
        linear between substeps. Falls back to the node grid if no substep record exists.
        Times outside the horizon are clamped (eta is held)."""
        tg, eg = (self.t, self.eta) if self.eta_sub is None else (self.t_sub, self.eta_sub)
        tq = np.clip(np.atleast_1d(np.asarray(t, float)), tg[0], tg[-1])
        out = np.stack([np.interp(tq, tg, eg[:, j]) for j in range(eg.shape[1])], axis=-1)
        return out[0] if np.ndim(t) == 0 else out


class MorphingActuator:
    def __init__(self, spec: GeometrySpec, cfg: dict[str, Any] | None = None):
        self.spec = spec
        cfg = cfg or {}
        gain_cfg = cfg.get("gain", {})
        if isinstance(gain_cfg, dict):
            self.gain = np.array([float(gain_cfg.get(n, 0.5)) for n in spec.names], dtype=float)
        else:
            self.gain = np.full(spec.dim, float(gain_cfg), dtype=float)
        self.rate_max = np.asarray(spec.rate_max, dtype=float)
        if "min_substeps_per_interval" in cfg:
            n_min = cfg["min_substeps_per_interval"]
        else:
            n_min = cfg.get("substeps_per_interval", 5)        # pre-V0.6 key, same (minimum) meaning
        self.min_substeps = max(1, int(n_min))
        if np.any(self.gain <= 0) or np.any(self.rate_max <= 0):
            raise ValueError("morphing actuator needs positive gains and positive rate limits")
        self.last_step_diagnostics: dict[str, Any] = {}

    @property
    def substeps(self) -> int:
        """Configured minimum number of substeps per interval (legacy attribute name)."""
        return self.min_substeps

    # ------------------------------------------------------------------ core
    def rate(self, eta: np.ndarray, eta_cmd: np.ndarray) -> np.ndarray:
        raw = self.gain * (np.asarray(eta_cmd, float) - np.asarray(eta, float))
        return np.clip(raw, -self.rate_max, self.rate_max)

    def stability_substeps(self, dt: float) -> int:
        """Substeps needed so that gain*h <= 1 (and h <= 1 s) on an interval of length dt."""
        return max(1, int(np.ceil(float(dt) / self.substep_dt() - 1e-12)))

    def substep_plan(self, dt: float) -> dict[str, Any]:
        """configured-minimum / stability-required / actual substeps and the actual timestep."""
        req = self.stability_substeps(dt)
        n = max(self.min_substeps, req)
        return {"interval_dt_s": float(dt), "configured_min_substeps": int(self.min_substeps),
                "stability_required_substeps": int(req), "actual_substeps": int(n),
                "dt_actuator_s": float(dt) / n, "stability_dt_max_s": self.substep_dt()}

    def step(self, eta: np.ndarray, eta_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Advance eta by dt toward eta_cmd (rate- and bound-limited) with
        max(min_substeps, stability_required) explicit-Euler substeps."""
        return self.step_dense(eta, eta_cmd, dt)[-1]

    def step_dense(self, eta: np.ndarray, eta_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Like `step` but returns every substep state, shape (n_substeps + 1, n_eta)."""
        eta = self.spec.as_array(eta)
        eta_cmd = self.spec.as_array(eta_cmd)
        plan = self.substep_plan(dt)
        n_sub = plan["actual_substeps"]
        h = float(dt) / n_sub
        out = np.empty((n_sub + 1, eta.shape[0]))
        out[0] = eta
        for i in range(n_sub):
            eta_next = eta + h * self.rate(eta, eta_cmd)
            # explicit Euler with |rate| <= rate_max cannot overshoot the command when
            # gain*h <= 1; clip guards the rare case of a large dt and keeps bounds hard
            over = np.sign(eta_cmd - eta) != np.sign(eta_cmd - eta_next)
            eta_next = np.where(over, eta_cmd, eta_next)
            eta = self.spec.clip(eta_next)
            out[i + 1] = eta
        self.last_step_diagnostics = plan
        return out

    def substep_dt(self) -> float:
        # ensure gain*h <= 1 so the first-order lag never overshoots
        return float(min(1.0 / np.max(self.gain), 1.0))

    def predict(self, eta_current: np.ndarray, eta_cmd: np.ndarray, horizon: float, n_intervals: int,
                *, tol: float = 1e-6) -> ActuatorTrajectory:
        """Geometry at the N+1 nodes of a horizon split into `n_intervals` equal intervals."""
        eta_current = self.spec.as_array(eta_current)
        eta_cmd = self.spec.clip(eta_cmd)
        dt = float(horizon) / int(n_intervals)
        plan = self.substep_plan(dt)
        n_sub = plan["actual_substeps"]
        etas = np.empty((int(n_intervals) * n_sub + 1, self.spec.dim))
        etas[0] = eta_current
        for k in range(int(n_intervals)):
            dense = self.step_dense(etas[k * n_sub], eta_cmd, dt)
            etas[k * n_sub + 1:(k + 1) * n_sub + 1] = dense[1:]
        eta_arr = etas[::n_sub].copy()
        t = np.arange(n_intervals + 1) * dt
        t_sub = np.arange(etas.shape[0]) * (dt / n_sub)
        conv = np.all(np.abs(eta_arr - eta_cmd) < tol, axis=1)
        conv_at = float(t[np.argmax(conv)]) if conv.any() else None
        return ActuatorTrajectory(t=t, eta=eta_arr, eta_cmd=eta_cmd.copy(), converged_at=conv_at,
                                  t_sub=t_sub, eta_sub=etas, diagnostics=plan)

    def describe(self) -> dict[str, Any]:
        return {"gain": self.gain.tolist(), "rate_max": self.rate_max.tolist(),
                "lower": self.spec.lower.tolist(), "upper": self.spec.upper.tolist(),
                "min_substeps_per_interval": int(self.min_substeps), "stability_dt_max_s": self.substep_dt(),
                "substep_rule": "n = max(min_substeps_per_interval, ceil(dt / min(1/max(K), 1 s)))",
                "law": "eta_dot = sat(K (eta_cmd - eta), -rate_max, rate_max)"}
