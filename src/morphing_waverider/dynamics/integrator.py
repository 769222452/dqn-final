"""Fixed-step RK4 integration for the 3-DOF model (numpy). A symbolic RK4 for CasADi is
built by the OCP layer from `PointMass3DOF.rhs` (same math, see trajectory/ocp.py).

Time-varying geometry (V0.6.1). The geometry eta is NOT constant inside an OCP interval: the
morphing actuator produces a continuous, rate-limited eta(t). `rk4_interval` is the single
propagation kernel shared by the executed rollout (`Integrator.rollout` / `propagate`) and
the Dense Constraint Audit (trajectory/dense_audit.py):

    x_dot = f(x, alpha_k, eta(t)),   alpha zero-order hold on the interval, eta from a provider

and every RK4 stage evaluates eta at its own stage time

    k1 = f(x,            u, eta(t))
    k2 = f(x + h/2 k1,   u, eta(t + h/2))
    k3 = f(x + h/2 k2,   u, eta(t + h/2))
    k4 = f(x + h   k3,   u, eta(t + h))

Rollout and audit differ only in the step size and in what they store; the physics and the
eta(t) definition are the same code path. An `eta` argument may be
  * a fixed vector (n_eta,)                 -> constant geometry
  * an `ActuatorTrajectory`                 -> eta(t) = plan.eta_at(t - t0)   (authoritative)
  * a callable eta(t) of absolute time
  * an (N+1, n_eta) node array              -> LEGACY piecewise-constant hold eta_k on [t_k, t_{k+1});
                                               regression-test / constant-geometry callers only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .morphing_actuator import ActuatorTrajectory
from .point_mass_3dof import NX, STATE_NAMES, DynamicsError, PointMass3DOF

EtaProvider = Callable[[float], np.ndarray]


def rk4_step(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def constant_eta(eta) -> EtaProvider:
    e = np.asarray(eta, dtype=float).reshape(-1)
    return lambda t: e


def node_hold_eta(t_nodes, eta_nodes) -> EtaProvider:
    """LEGACY (pre-V0.6.1) execution semantics: eta_k held constant on [t_k, t_{k+1})."""
    tn = np.asarray(t_nodes, float).ravel()
    en = np.asarray(eta_nodes, float)

    def hold(t: float) -> np.ndarray:
        k = int(np.clip(np.searchsorted(tn, t + 1e-9, side="right") - 1, 0, en.shape[0] - 1))
        return en[k]
    return hold


def make_eta_provider(eta, t0: float, t_nodes=None) -> tuple[EtaProvider, str]:
    """Normalise the accepted `eta` forms to a callable of absolute time; returns (fn, mode)."""
    if isinstance(eta, ActuatorTrajectory):
        plan = eta
        return (lambda t: plan.eta_at(t - t0)), "actuator"
    if callable(eta):
        return eta, "callable"
    arr = np.asarray(eta, dtype=float)
    if arr.ndim == 1:
        return constant_eta(arr), "constant"
    if t_nodes is None:
        raise ValueError("a node eta array needs the node times")
    if arr.shape[0] != np.asarray(t_nodes).shape[0]:
        raise ValueError("eta node trajectory must have N+1 rows")
    return node_hold_eta(t_nodes, arr), "node_hold"


def rk4_interval(model: PointMass3DOF, x0, u, eta_fn: EtaProvider, t0: float, dt: float, n_sub: int, *,
                 store: bool = False, quantities: bool = False):
    """Shared propagation kernel: RK4 over [t0, t0+dt] in `n_sub` equal substeps, control u held,
    eta(t) evaluated at every RK4 stage time.

    Returns (x_end, t_sub, x_sub, eta_sub, qty_sub):
      store=False     -> t_sub / x_sub / eta_sub are None
      store=True      -> (n_sub+1,) substep times, (n_sub+1, NX) states, (n_sub+1, n_eta) eta(t)
      quantities=True -> list of path-quantity dicts at samples 0..n_sub-1 (from the k1 evaluation;
                         the end sample is left to the caller), else None
    Raises DynamicsError on a non-finite state."""
    n_sub = max(1, int(n_sub))
    h = float(dt) / n_sub
    x = np.asarray(x0, float).copy()
    ts = xs = es = None
    if store:
        ts = t0 + h * np.arange(n_sub + 1)
        xs = np.empty((n_sub + 1, NX))
        e0 = np.asarray(eta_fn(t0), float)
        es = np.empty((n_sub + 1, e0.shape[0]))
        xs[0], es[0] = x, e0
    qtys: list | None = [] if quantities else None
    for i in range(n_sub):
        t = t0 + i * h
        eta_a = np.asarray(eta_fn(t), float)
        eta_m = np.asarray(eta_fn(t + 0.5 * h), float)
        eta_b = np.asarray(eta_fn(t + h), float)
        if quantities:
            k1, qty = model.rhs_with_quantities(x, u, eta_a)
            qtys.append(qty)
        else:
            k1 = model.rhs(x, u, eta_a)
        k2 = model.rhs(x + 0.5 * h * k1, u, eta_m)
        k3 = model.rhs(x + 0.5 * h * k2, u, eta_m)
        k4 = model.rhs(x + h * k3, u, eta_b)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(x)):
            raise DynamicsError("integration produced non-finite state")
        if store:
            xs[i + 1], es[i + 1] = x, eta_b
    return x, ts, xs, es, qtys


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray           # (N+1, NX)
    u: np.ndarray           # (N, NU)  (control held on each interval)
    eta: np.ndarray         # (N+1, n_eta)  geometry actually used at the nodes
    outputs: dict[str, np.ndarray] = field(default_factory=dict)  # path quantities per node
    eta_mode: str = "constant"                 # constant | actuator | callable | node_hold
    sub_t: np.ndarray | None = None            # integration substep grid actually executed
    sub_x: np.ndarray | None = None            # state at every substep
    sub_eta: np.ndarray | None = None          # eta(t) used at every substep

    def column(self, name: str) -> np.ndarray:
        if name in STATE_NAMES:
            return self.x[:, STATE_NAMES.index(name)]
        if name == "alpha":
            return np.append(self.u[:, 0], self.u[-1, 0]) if len(self.u) else np.array([])
        return self.outputs[name]

    def as_columns(self) -> dict[str, np.ndarray]:
        cols = {"t": self.t}
        for i, n in enumerate(STATE_NAMES):
            cols[n] = self.x[:, i]
        cols["alpha"] = self.column("alpha")
        for k, v in self.outputs.items():
            cols[k] = v
        for j in range(self.eta.shape[1]):
            cols[f"eta_{j}"] = self.eta[:, j]
        return cols

    def sampled(self, model: PointMass3DOF) -> "Trajectory":
        """View of the actual executed RK samples, without a second integration.

        Control rate is a decision-grid constraint and must be checked on the original
        nodes, not by dividing a ZOH jump by the finer sampling interval.
        """
        if self.sub_t is None:
            return self
        indices = np.clip(np.searchsorted(self.t, self.sub_t + 1e-9, side="right") - 1,
                          0, len(self.u) - 1)
        controls = self.u[indices]
        outputs = [model.numeric_outputs(x, u, eta)
                   for x, u, eta in zip(self.sub_x, controls, self.sub_eta)]
        return Trajectory(self.sub_t, self.sub_x, controls, self.sub_eta,
                          {k: np.array([q[k] for q in outputs]) for k in outputs[0]}, self.eta_mode)


class Integrator:
    def __init__(self, model: PointMass3DOF, max_substep_dt: float = 0.5):
        self.model = model
        self.max_substep_dt = float(max_substep_dt)

    def n_substeps(self, dt: float) -> int:
        return max(1, int(np.ceil(float(dt) / self.max_substep_dt - 1e-9)))

    def propagate(self, x, u, eta, dt: float, t0: float = 0.0) -> np.ndarray:
        """Advance state by dt with control u held. `eta` is a fixed vector (constant geometry), an
        ActuatorTrajectory (eta(t) = eta_at(t - t0)) or a callable of absolute time; every RK4 stage
        uses eta at its own stage time."""
        x = self.model.validate_state(x)
        u = self.model.validate_control(u)
        if not (isinstance(eta, ActuatorTrajectory) or callable(eta)):
            eta = self.model.spec.as_array(eta)
        eta_fn, _ = make_eta_provider(eta, t0)
        x_end, *_ = rk4_interval(self.model, x, u, eta_fn, float(t0), float(dt), self.n_substeps(dt))
        return x_end

    def rollout(self, x0, controls: Sequence | np.ndarray | Callable, eta, dt: float,
                n_steps: int | None = None, t0: float = 0.0, *, store_substeps: bool = True) -> Trajectory:
        """Roll out for n_steps. `controls` is an (N, NU) array, a sequence, or a callable
        u = controls(t, x) -- alpha is zero-order held on each interval. `eta`: fixed vector,
        ActuatorTrajectory (authoritative eta(t), relative to t0), callable eta(t) of absolute
        time, or the LEGACY (N+1, n_eta) node array (piecewise-constant hold)."""
        x0 = self.model.validate_state(x0)
        if callable(controls):
            if n_steps is None:
                raise ValueError("n_steps required when controls is callable")
            N = int(n_steps)
        else:
            controls = np.atleast_2d(np.asarray(controls, dtype=float))
            N = controls.shape[0] if n_steps is None else int(n_steps)
        t = t0 + np.arange(N + 1) * dt
        eta_fn, mode = make_eta_provider(eta, t0, t)
        eta_arr = np.stack([np.asarray(eta_fn(tk), float) for tk in t], axis=0)
        xs = np.zeros((N + 1, NX))
        us = np.zeros((N, 1))
        xs[0] = x0
        outs: dict[str, list[float]] = {}
        n_sub = self.n_substeps(dt)
        sub_t, sub_x, sub_eta = [np.array([t0])], [xs[:1].copy()], [eta_arr[:1].copy()]
        for k in range(N):
            u_k = np.asarray(controls(t[k], xs[k]) if callable(controls) else controls[k], float).reshape(-1)
            us[k] = u_k
            u_k = self.model.validate_control(u_k)
            q = self.model.numeric_outputs(xs[k], u_k, eta_arr[k])
            for key, v in q.items():
                outs.setdefault(key, []).append(v)
            xs[k + 1], ts, xsub, esub, _ = rk4_interval(self.model, xs[k], u_k, eta_fn, float(t[k]), float(dt),
                                                        n_sub, store=store_substeps)
            if store_substeps:
                sub_t.append(ts[1:])
                sub_x.append(xsub[1:])
                sub_eta.append(esub[1:])
        # last node outputs with last control
        q = self.model.numeric_outputs(xs[N], us[-1] if N else np.zeros(1), eta_arr[N])
        for key, v in q.items():
            outs.setdefault(key, []).append(v)
        traj = Trajectory(t=t, x=xs, u=us, eta=eta_arr, outputs={k: np.array(v) for k, v in outs.items()},
                          eta_mode=mode)
        if store_substeps:
            traj.sub_t = np.concatenate(sub_t)
            traj.sub_x = np.concatenate(sub_x, axis=0)
            traj.sub_eta = np.concatenate(sub_eta, axis=0)
        return traj
