"""Fast disturbed flight for the disturbance study (experiments/perturb, fast-model stage).

Truth: `headroom.Truth` (the point-mass dynamics and path limits of the physical bound, lower-hull polar of the
298 geometries, no friction) with a multiplicative density field along the track and constant lift/drag biases.
Every 0.5 s RK4 stage is checked against all path limits; any violation ends the flight as a failure.

Action: every 12 s a dynamic-pressure set point ratio r = q_ref/q_max (R_LEVELS, 19 levels). In an equilibrium
glide r maps one-to-one to a lift coefficient; the set point form lets the 0.5 s inner loop

    CL = CL_level (1 + kp (q/q_ref - 1)) - kg gamma,   CL_level = m (g - V^2/r) / (q S),   clipped to [0, CL_MAX]

provide the fast feedback that damps the phugoid and keeps q below its limit. All policies share this inner loop.
Mode 'const' (baseline B1) holds a constant CL and lets the same loop take over when it asks for more lift.

Observations: dynamic pressure and density ratio are measured exactly (flush air data); lift/drag biases are
estimated online with `headroom.LoadEstimator`. Return: downrange / 2400 km + 0.10 on success, -0.25 on failure
(the certified mission's reward interface).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import headroom as H

ROOT = Path(__file__).resolve().parents[2]
R_LEVELS = (.50, .60, .70, .75, .80, .83, .85, .87, .88, .89, .90, .91, .92, .93, .94, .95, .96, .97, .98)
DECISION_S = 12.
DT = .5
MISSION_S = 1200.
CL_MAX = .10
DOWNRANGE_SCALE_M = 2.4e6
SUCCESS_BONUS, FAILURE_PENALTY = .10, -.25
#: (density sigma, correlation length km, aero bias sigma); see PREREGISTRATION.md section 3
LEVELS = {'L0': (0., 200., 0.), 'L1': (.03, 200., .05), 'L2': (.06, 200., .05), 'L3': (.12, 200., .05),
          'L3b': (.06, 50., .05)}
TRAIN_SEEDS = tuple(range(1000, 1016))
TEST_SEEDS = tuple(range(2000, 2040))

_WORLD = {}


def world():
    """(mission, polar surface, Mach support, nominal initial state), loaded once per process."""
    if not _WORLD:
        from .aero import AeroDatabase
        from .build import runtime_config
        from .mission import Mission
        physical = runtime_config(ROOT)['physical']
        mission = Mission(ROOT, physical['mission'])
        db = AeroDatabase(ROOT/physical['snapshot'])
        z = np.load(ROOT/'experiments/headroom/polar_table.npz')
        _WORLD.update(mission=mission, surface=H.PolarSurface(z['knots'], z['cd']),
                      support=(float(db.M[0]), float(db.M[-1])),
                      x0=np.array(mission.data['initial_state']['x_h_V_gamma_s'], float))
    return _WORLD


@dataclass(frozen=True)
class Scenario:
    level: str
    seed: int
    dh: float
    dv: float
    k_lift: float
    k_drag: float

    @property
    def name(self):
        return f'{self.level}_s{self.seed}'


def scenario(level, seed):
    """Deterministic scenario: the same seed gives the same initial offset and standard-normal draws at every level."""
    sigma, corr_km, aero = LEVELS[level]
    rng = np.random.default_rng([int(seed), 7])
    dh = float(np.round(rng.uniform(-2000., 2000.), 0))
    dv = float(np.round(rng.uniform(-100., 100.), 1))
    zl, zd = rng.standard_normal(2)
    k = lambda z: float(1+aero*np.clip(z, -2., 2.))
    return Scenario(level, int(seed), dh, dv, k(zl), k(zd))


def density_field(sc: Scenario):
    sigma, corr_km, _ = LEVELS[sc.level]
    if sigma <= 0:
        s = np.arange(0, 3.2e6+5e3, 5e3)
        return s, np.ones_like(s)
    return H.density_field(sigma, corr_km*1e3, sc.seed)


def truth(sc: Scenario):
    w = world()
    return H.Truth(w['mission'], w['surface'], w['support'], density_field(sc), sc.k_lift, sc.k_drag, dt=DT)


class FastFlight:
    """One disturbed flight; `step(action)` flies one 12 s decision with the 0.5 s inner loop."""

    def __init__(self, sc: Scenario, kp=2., kg=1., mode='law', cl_const=None, _truth=None):
        w = world()
        self.sc, self.kp, self.kg, self.mode, self.cl_const = sc, float(kp), float(kg), mode, cl_const
        if mode not in ('law', 'const') or (mode == 'const' and cl_const is None):
            raise ValueError('mode law, or const with cl_const')
        self.truth = _truth if _truth is not None else truth(sc)
        self.qmax = w['mission'].data['limits']['dynamic_pressure_pa_max']
        self.m, self.S = self.truth.m, self.truth.s_ref
        self.x = w['x0']+np.array([sc.dh, sc.dv, 0., 0.])
        self.t, self.cl, self.alpha = 0., .03, 0.
        self.previous_cl_raw, self.history_valid = None, False
        self.est = H.LoadEstimator(self.truth)
        self.rho_prev = self.truth.k_rho(self.x[3])
        self.done, self.failed, self.reason = False, False, None
        self.min_margin = np.inf
        self.G = 0.

    # -------------------------------------------------------------------------- observations
    def _q(self, x):
        return float(self.truth.eval(x, self.cl)[1][0])

    def extra_features(self):
        rho = self.truth.k_rho(self.x[3])
        kl, kd = self.est.values
        return [(rho-1)/.1, (self._q(self.x)/self.qmax-.9)/.1, (kl-1)/.05, (kd-1)/.05, (self.rho_prev-1)/.1]

    # -------------------------------------------------------------------------- inner loop
    def _cl(self, q_ref):
        x = self.x
        q = self._q(x)
        h, v, g = x[0], x[1], x[2]
        atm = self.truth.model.atm
        grav = float(atm.gravity(h, spherical=True))
        r = atm.cfg.R_earth+h
        lvl = self.m*(grav-v*v/r)/(q*self.S)
        cl = lvl*(1+self.kp*(q/q_ref-1))-self.kg*g
        if self.mode == 'const':
            cl = max(self.cl_const, cl)
        return float(np.clip(cl, 0., CL_MAX))

    def step(self, action):
        """Fly one decision (12 s or to the end); returns (reward, done)."""
        if self.done:
            raise RuntimeError('flight finished')
        q_ref = R_LEVELS[int(action)]*self.qmax
        self.rho_prev = self.truth.k_rho(self.x[3])
        s0 = self.x[3]
        end = min(self.t+DECISION_S, MISSION_S)
        while self.t < end-1e-9:
            self.cl = self._cl(q_ref)
            k_meas = self.truth.k_rho(self.x[3])
            xn, margin = self.truth.step(self.x, self.cl)
            m = float(np.min(margin))
            self.min_margin = min(self.min_margin, m)
            if not np.all(np.isfinite(xn)) or m < -1e-6:
                self.failed, self.done = True, True
                names = H.STAGE_MARGINS
                self.reason = 'VIOLATION:'+names[int(np.argmin(margin))] if np.all(np.isfinite(margin)) else 'NONFINITE'
                break
            self.est.add(self.x, xn, self.cl, k_meas)
            self.est.decay()
            self.x, self.t = xn, self.t+DT
        r = (self.x[3]-s0)/DOWNRANGE_SCALE_M
        if self.failed:
            r += FAILURE_PENALTY
        elif self.t >= MISSION_S-1e-9:
            self.done = True
            self.reason = 'DONE'
            r += SUCCESS_BONUS
        self.G += r
        return r, self.done

    def fork(self):
        """Independent copy sharing the (stateless) truth model."""
        c = object.__new__(FastFlight)
        c.__dict__.update(self.__dict__)
        c.x = self.x.copy()
        est = object.__new__(H.LoadEstimator)
        est.__dict__.update(self.est.__dict__)
        est.num, est.den = self.est.num.copy(), self.est.den.copy()
        c.est = est
        return c

    @property
    def range_km(self):
        return float((self.x[3]-world()['x0'][3])/1000)

    def summary(self):
        return dict(scenario=asdict(self.sc), range_km=self.range_km, G=self.G, success=not self.failed and self.done,
                    reason=self.reason, min_margin=float(self.min_margin), t=self.t)
