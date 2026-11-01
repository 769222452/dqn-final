"""P4: preliminary 1200 s comparison (fixed new geometry / ideal selection / executable morph).

Same mission file, V5 3DOF plant (PointMass3DOF + rk4_interval) and limits as stage 1
(mission_v6.simulate.limits_at). Differences, all explicit:
  * aero comes from stage-2 Teacher tables on the MISSION grid (M 4.5-8.5, alpha -2..12);
    leaving that grid is AERO_DOMAIN_EXIT (stricter than the V2 3.5-9.25 grid);
  * eta[0] carries the morph path progress s(t) in [0, 1]. Aero at s between two natively
    generated, Teacher-labelled path nodes is linearly blended between the node tables
    (the path is refined until the Teacher midpoint check passes; see morph.py);
  * the ideal-selection case B switches tables instantaneously (no actuator, no transition aero).
Nothing here writes to stage-1 results; every output goes under results/v6_mission_geometry_stage2.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from aerotab import DomainExit, Table
from common import MISSION_ALPHA, MISSION_MACH, ROOT, STAGE1, add_paths

add_paths()
from mission_v6.mission import Mission  # noqa: E402
from mission_v6.simulate import limits_at  # noqa: E402
from morphing_waverider.dynamics.integrator import rk4_interval  # noqa: E402

RATE = 0.1  # path/s


def mission():
    return Mission(ROOT)


def mission_table(name, labels):
    return Table(name, labels, MISSION_MACH, MISSION_ALPHA)


class PathAero:
    """Aero along a native morph path: nodes [(s, Table)], s ascending (s in [0,1], or [0,2] for a
    concatenated G42->A->C path)."""

    def __init__(self, nodes, mode="blend"):
        self.s = np.array([s for s, _ in nodes], float)
        self.tables = [t for _, t in nodes]
        self.mode = mode  # "blend" (executable path) or "step" (ideal instantaneous switch)

    def coefficients(self, mach, alpha, eta=None):
        s = float(eta[0]) if eta is not None and len(self.tables) > 1 else 0.0
        if len(self.tables) == 1:
            return self.tables[0].coefficients(mach, alpha)
        if self.mode == "step":
            t = self.tables[-1] if s >= 0.5 else self.tables[0]
            return t.coefficients(mach, alpha)
        s = min(max(s, self.s[0]), self.s[-1])
        i = min(int(np.searchsorted(self.s, s, side="right") - 1), len(self.s) - 2)
        w = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        if w <= 1e-12:
            return self.tables[i].coefficients(mach, alpha)
        if w >= 1 - 1e-12:
            return self.tables[i + 1].coefficients(mach, alpha)
        a = self.tables[i].coefficients(mach, alpha)
        b = self.tables[i + 1].coefficients(mach, alpha)
        return (1 - w) * a[0] + w * b[0], (1 - w) * a[1] + w * b[1], None


def schedule_fn(kind, t_switch, rate=RATE):
    """Path progress s(t) packed into eta[0].

    fixed      s = 0
    step       ideal selection: s jumps 0 -> 1 at t_switch (no actuator, no transition aero)
    morph      family start: s ramps 0 -> 1 at `rate` path/s from t_switch
    morph_g42  G42 start on a concatenated path G42 -> A (s 0..1) -> C (s 1..2): the first
               segment is flown from t=0 at `rate`, the second from max(t_switch, 1/rate)."""
    z = [0, 0, 0, 0, 0.0]
    if kind == "fixed":
        return lambda t: np.array([0.0, *z])
    if kind == "step":
        return lambda t: np.array([1.0 if t >= t_switch else 0.0, *z])
    if kind == "morph":
        return lambda t: np.array([min(1.0, rate * max(0.0, t - t_switch)), *z])
    if kind == "morph_g42":
        ts = max(float(t_switch), 1.0 / rate)
        return lambda t: np.array([min(1.0, rate * t) + min(1.0, rate * max(0.0, t - ts)), *z])
    raise ValueError(kind)


def simulate(m, aero, controls, eta, dt=None, collect=False):
    dt = float(dt or m.data["plant_dt_s"])
    controls = np.asarray(controls, float).reshape(-1)
    model = m.model(aero)
    state = np.array(m.data["initial_state"]["x_h_V_gamma_s"], float)
    x0 = state.copy()
    prev = float(m.data["initial_state"]["alpha_rad"])
    rows, mins, work, reason, step = [], {}, 0.0, "", 0
    for step, alpha in enumerate(controls):
        t = step * dt
        if not np.isfinite(alpha) or abs(alpha - prev) > m.data["limits"]["alpha_rate_radps"] * dt + 1e-10:
            reason = "alpha_rate"
            break
        try:
            q = model.path_quantities(state, [alpha], eta(t))
            mg = limits_at(m, state, alpha, q)
            mg["alpha_rate_rad"] = m.data["limits"]["alpha_rate_radps"] * dt - abs(alpha - prev)
            for k, v in mg.items():
                mins[k] = min(mins.get(k, np.inf), v)
            if min(mg.values()) < -1e-7:
                reason = "path_constraint:" + ",".join(k for k, v in mg.items() if v < -1e-7)
                break
            if collect:
                rows.append({"time_s": t, "h_m": state[0], "V_mps": state[1], "gamma_rad": state[2],
                             "s_m": state[3], "Mach": q["mach"], "q_pa": q["q_dyn"], "alpha_rad": alpha,
                             "CL": q["CL"], "CD": q["CD"], "path_s": float(eta(t)[0]),
                             "drag_power_w": q["D"] * state[1], "cumulative_drag_work_j": work})
            work += float(q["D"]) * state[1] * dt
            state, *_ = rk4_interval(model, state, np.array([alpha]), eta, t, dt, 1)
            prev = alpha
            if not np.isfinite(state).all() or state[1] <= 0:
                reason = "nonfinite_state"
                break
        except DomainExit as e:
            reason = "aero_domain_exit:" + str(e)
            break
        except (FloatingPointError, OverflowError, ValueError) as e:
            reason = "numerical_failure:" + str(e)
            break
    else:
        step = len(controls)
    t_end = float(step * dt)
    if not reason:
        try:
            q = model.path_quantities(state, [prev], eta(t_end))
            mg = limits_at(m, state, prev, q)
            for k, v in mg.items():
                mins[k] = min(mins.get(k, np.inf), v)
            if min(mg.values()) < -1e-7:
                reason = "terminal_path_constraint"
        except DomainExit as e:
            reason = "aero_domain_exit_at_end:" + str(e)
    kind = None
    if reason:
        kind = ("AERO_DOMAIN_EXIT" if reason.startswith("aero_domain") else
                "NUMERICAL_FAILURE" if reason.startswith(("numerical", "nonfinite")) else "CONSTRAINT_VIOLATION")
    outcome = m.outcome(t_end, state, kind)
    return {"outcome": outcome, "reason": reason or ("terminal_condition" if outcome == "CONSTRAINT_VIOLATION" else ""),
            "time_s": t_end, "range_m": float(state[3] - x0[3]), "min_margins": mins, "final_state": state.tolist(),
            "drag_work_j": work, "rows": rows}


def fine_replay(m, aero, controls, eta):
    dt = m.data["plant_dt_s"]
    model = m.model(aero)
    x = np.asarray(m.data["initial_state"]["x_h_V_gamma_s"], float)
    for i, alpha in enumerate(controls):
        for half in (0, 1):
            t = (i + half / 2) * dt
            q = model.path_quantities(x, [alpha], eta(t))
            if min(limits_at(m, x, alpha, q).values()) < -1e-7:
                raise ValueError(f"fine replay constraint at {t}s")
            x, *_ = rk4_interval(model, x, np.array([alpha]), eta, t, dt / 2, 1)
    return x


SCALE = {"altitude_min_m": 20000, "altitude_max_m": 80000, "velocity_min_mps": 800, "velocity_max_mps": 6000,
         "gamma_min_rad": .35, "gamma_max_rad": .35, "mach_min": 3.5, "mach_max": 9.25,
         "alpha_min_rad": .2, "alpha_max_rad": .2, "alpha_rate_rad": .087, "q_max_pa": 80000,
         "heat_max_wm2": 2000000, "load_max_g": 3}


def g42_stage1_alpha():
    return np.load(STAGE1 / "G42" / "alpha_0p5s.npy")  # read-only warm start


_G42_TRAJ = None


def cl_matched_alpha(m, aero, eta, scales=(1.0,)):
    """Warm start that makes the geometry produce the CL of the stage-1 G42 best flight at the same
    time and Mach (rate-limited, clipped to the alpha limits). Only a seed; the plant decides."""
    global _G42_TRAJ
    if _G42_TRAJ is None:
        import csv
        with open(STAGE1 / "G42" / "trajectory_0p5s.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        _G42_TRAJ = np.array([[float(r["Mach"]), float(r["CL_actual"])] for r in rows])
    dt = float(m.data["plant_dt_s"])
    lo = max(m.data["limits"]["alpha_rad"][0], np.deg2rad(MISSION_ALPHA[0]))
    hi = min(m.data["limits"]["alpha_rad"][1], np.deg2rad(MISSION_ALPHA[-1]))
    grid = np.linspace(lo, hi, 57)
    outs = np.empty((len(scales), len(_G42_TRAJ)))
    for i, (mach, cl) in enumerate(_G42_TRAJ):
        e = eta(i * dt)
        mach = min(max(mach, MISSION_MACH[0]), MISSION_MACH[-1])
        curve = np.array([aero.coefficients(mach, a, e)[0] for a in grid])
        for n, sc in enumerate(scales):
            c = curve - cl * sc
            k = np.nonzero(np.diff(np.sign(c)) != 0)[0]
            if c[0] >= 0:
                outs[n, i] = grid[0]
            elif not len(k):
                outs[n, i] = grid[-1]
            else:
                j = k[0]
                outs[n, i] = grid[j] - c[j] * (grid[j + 1] - grid[j]) / (c[j + 1] - c[j])
    rate = m.data["limits"]["alpha_rate_radps"] * dt * 0.999
    for out in outs:
        prev = float(m.data["initial_state"]["alpha_rad"])
        for i in range(len(out)):
            out[i] = prev = float(np.clip(out[i], prev - rate, prev + rate))
    return list(outs)


def optimise(m, aero, kind, deadline, *, t_switch_grid=(None,), knots=(6,), maxiter=6, seeds=None, log=print):
    """Best-found range for one strategy. Decision variables: alpha knot perturbations around a
    warm start (+/-0.05 rad, as stage 1) and, for morph/step, the switch time.
    Acceptance: 0.5 s SUCCESS + independent 0.25 s replay agreement (same rule as stage 1)."""
    dt = float(m.data["plant_dt_s"])
    n = int(m.data["duration_s"] / dt)
    tgrid = np.arange(n) * dt
    seeds = seeds or [("G42_stage1_alpha", g42_stage1_alpha())]
    best = {"range_m": -np.inf}
    evals = [0]
    t0 = time.monotonic()

    def run(ctrl, ts):
        evals[0] += 1
        return simulate(m, aero, ctrl, schedule_fn(kind, ts if ts is not None else 0.0))

    def accept(ctrl, ts, f):
        if f["outcome"] != "SUCCESS" or f["range_m"] <= best["range_m"]:
            return
        eta = schedule_fn(kind, ts if ts is not None else 0.0)
        try:
            xf = fine_replay(m, aero, ctrl, eta)
        except (ValueError, DomainExit) as e:
            log(f"   fine replay reject: {e}")
            return
        diff = np.abs(np.asarray(f["final_state"]) - xf)
        if np.max(diff) > 20.0 or diff[0] > 2 or diff[1] > 0.2:
            return
        best.update({"range_m": f["range_m"], "controls": np.asarray(ctrl).copy(), "t_switch": ts,
                     "flight": {k: v for k, v in f.items() if k != "rows"},
                     "fine_disagreement_max": float(np.max(diff)), "evaluations": evals[0]})

    starts = []
    seed_cache = {}

    def seed_ctrl(label, base, ts):
        if not callable(base):
            return base
        if (label, ts) not in seed_cache:
            seed_cache[(label, ts)] = base(ts)
        return seed_cache[(label, ts)]

    for ts in t_switch_grid:
        for label, base in seeds:
            if time.monotonic() > deadline:
                break
            ctrl = seed_ctrl(label, base, ts)
            f = run(ctrl, ts)
            starts.append({"seed": label, "t_switch": ts, "outcome": f["outcome"], "range_m": f["range_m"],
                           "reason": f["reason"][:120], "time_s": f["time_s"]})
            accept(ctrl, ts, f)
    # local polish from the best feasible start (or the best partial start)
    if np.isfinite(best["range_m"]):
        base, ts0 = best["controls"].copy(), best["t_switch"]
    else:
        st = max(starts, key=lambda r: (r["time_s"], r["range_m"])) if starts else None
        if st is None:
            return {"status": "NO_EVALUATION", "starts": starts}
        ts0 = st["t_switch"]
        base = seed_ctrl(st["seed"], dict(seeds)[st["seed"]], ts0)
    solver = []
    for nk in knots:
        if time.monotonic() > deadline:
            break
        has_ts = ts0 is not None
        cache = {}

        def unpack(z):
            ctrl = base + np.interp(tgrid, np.linspace(0, n * dt, nk), z[:nk])
            ts = float(np.clip(ts0 + 100.0 * z[nk], 0, 1150)) if has_ts else None
            return ctrl, ts

        def ev(z):
            k = tuple(np.round(z, 12))
            if k not in cache:
                c, ts = unpack(z)
                cache[k] = run(c, ts)
            return cache[k]

        def obj(z):
            f = ev(z)
            miss = (m.data["duration_s"] - f["time_s"]) / m.data["duration_s"]
            return (-f["range_m"] / 1e6 + 10 * miss + 100 * sum(min(0, f["min_margins"].get(k, -s) / s) ** 2
                                                               for k, s in SCALE.items()) + .01 * np.dot(z[:nk], z[:nk]))

        def ineq(z):
            f = ev(z)
            if f["time_s"] < m.data["duration_s"]:
                return np.full(len(SCALE), -1.0)
            return np.array([f["min_margins"].get(k, -1.0) / s for k, s in SCALE.items()])

        def cb(z):
            c, ts = unpack(z)
            accept(c, ts, ev(z))
            if time.monotonic() > deadline:
                raise TimeoutError

        nz = nk + (1 if has_ts else 0)
        bounds = [(-.05, .05)] * nk + ([(-3.0, 3.0)] if has_ts else [])
        t1 = time.monotonic()
        try:
            sol = minimize(obj, np.zeros(nz), method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": ineq}], callback=cb,
                           options={"maxiter": maxiter, "ftol": 1e-7})
            cb(sol.x)
            st = {"message": sol.message, "iterations": int(sol.nit), "success": bool(sol.success)}
        except TimeoutError:
            st = {"message": "budget reached; best feasible retained", "success": False}
        solver.append({"n_knots": nk, "switch_time_variable": has_ts, "wall_s": time.monotonic() - t1, **st})
        if np.isfinite(best["range_m"]):
            base, ts0 = best["controls"].copy(), best["t_switch"]
    out = {"status": "BEST_FOUND" if np.isfinite(best["range_m"]) else "NO_FEASIBLE_RESULT_FOUND",
           "kind": kind, "starts": starts, "solver": solver, "evaluations": evals[0],
           "wall_s": time.monotonic() - t0,
           "note": "local search from warm starts; not a global optimum; SLSQP iteration-limited"}
    if np.isfinite(best["range_m"]):
        out.update({"range_m": best["range_m"], "t_switch_s": best["t_switch"], "flight": best["flight"],
                    "fine_disagreement_max": best["fine_disagreement_max"]})
        out["_controls"] = best["controls"]
    return out
