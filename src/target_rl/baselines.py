"""Baselines (PREREGISTRATION.md section 6).

pc_fair    predictor-corrector, every cycle_s: choose (r, w), held to the end, so that the predicted range and
           speed at 1200 s hit the target. Prediction: same model, RK4 predict_dt_s, density ratio frozen at the
           currently measured value, lift/drag biases = the ONLINE ESTIMATES (the same information the policy
           has); the shared protection loop is simulated. Correction: damped Gauss-Newton with finite
           differences (minimum-norm steps), `newton` iterations per cycle (`newton_first` at t = 0), warm
           start; a step is accepted only if the predicted trajectory keeps `margin` on every path limit and
           the predicted miss does not grow (backtracking 1, 1/2, 1/4).
pc_strong  as pc_fair with one (r, w) per phase (0-400, 400-800, 800-1200 s); every cycle re-solves the
           current and later phases.
pc_fixed   as pc_fair without morphing (fixed geometry, r only, range target only).
pc_oracle  as pc_strong, but the prediction uses the TRUE density field ahead and the TRUE lift/drag biases
           (full information): a strong reference for what perfect knowledge of the disturbance buys.
constant   the best constant action on the training cases (the initial policy pi0, "learning removed").
reference  clairvoyant three-phase OPEN-LOOP plan, Nelder-Mead from the nearest nominal region-grid plans
           (150 evaluations). Kept for completeness; not run by default (amendment 4: weaker than pc_strong).
"""
import dataclasses
import os

import numpy as np
from scipy.optimize import minimize

from . import config, flight as F


class _Const:
    def __init__(self, k):
        self.k = k

    def __call__(self, s):
        return self.k


def _phase(t, phases):
    return 0 if t < phases[0] else (1 if t < phases[1] else 2)


def _command(params, phases, morph):
    """params: r[3] + w[3] per phase."""
    def command(env):
        k = _phase(env.t, phases)
        w = env.w
        p = w['prot']
        cl = F.law_cl(w['model'], env.x, env.q(), params[k], w['qmax'], p['kp'], p['kg'])
        return cl, (params[3+k] if morph else 0.)
    return command


def predict(env, params, phases, pc, oracle=False):
    """Prediction from the current state: ((range, speed), keeps the margin). Measured density ratio frozen and
    estimated biases; with oracle=True the true density field and true biases."""
    p = env.fork()
    p.trace = None
    if oracle:
        p.field, p.sc = env.field, env.sc
    else:
        kl, kd = env.est.values
        kl, kd = float(np.clip(kl, .7, 1.3)), float(np.clip(kd, .7, 1.3))
        p.field = _Const(env.k_rho())
        p.sc = dataclasses.replace(env.sc, k_lift=kl, k_drag=kd)
    p.w = dict(env.w, dt=pc['predict_dt_s'])
    p.advance(env.w['T']-env.t, _command(params, phases, env.morph))
    return np.array([p.range_km(), p.x[1]]), (not p.failed) and p.min_margin >= pc['margin']


class PredictorCorrector:
    def __init__(self, kind, cfg=None):
        cfg = cfg or config.load()
        self.kind, self.pc = kind, cfg['predictor_corrector']
        if kind not in ('pc_fair', 'pc_strong', 'pc_fixed', 'pc_oracle'):
            raise ValueError(kind)
        self.strong = kind in ('pc_strong', 'pc_oracle')
        self.oracle = kind == 'pc_oracle'
        self.morph = kind != 'pc_fixed'
        self.n_predictions = 0
        self.phases = self.pc['strong_phases_s'] if self.strong else [1e9, 1e9]
        self.u = np.array([.8, .8, .8, 0., 0., 0.])

    def _free(self, t):
        """Indices of the parameters the corrector may change at time t."""
        if not self.strong:
            idx = [0, 1, 2, 3, 4, 5] if self.morph else [0, 1, 2]
            return idx
        k = _phase(t, self.phases)
        return [j for j in range(k, 3)]+([3+j for j in range(k, 3)] if self.morph else [])

    def _tie(self, u):
        """pc_fair / pc_fixed hold one (r, w) for the whole rest of the flight."""
        if not self.strong:
            u[1] = u[2] = u[0]
            u[4] = u[5] = u[3]
        return u

    def _predict(self, env, u):
        self.n_predictions += 1
        return predict(env, u, self.phases, self.pc, self.oracle)

    def correct(self, env, iterations):
        pc = self.pc
        rb = pc['r_bounds']
        lo = np.array([rb[0]]*3+[0.]*3)
        hi = np.array([rb[1]]*3+[1. if self.morph else 0.]*3)
        free = self._free(env.t)
        if not self.strong:
            free = [0, 3] if self.morph else [0]
        tgt = np.array([env.target.range_km, env.target.speed_mps])
        rows = [0, 1] if self.morph else [0]
        scale = np.array([1., 10.])[rows]
        u = self._tie(np.clip(self.u.copy(), lo, hi))
        for _ in range(iterations):
            f0, ok0 = self._predict(env, u)
            e = (f0[rows]-tgt[rows])/scale
            J = np.zeros((len(rows), len(free)))
            for c, j in enumerate(free):
                d = .02 if j < 3 else .05
                uu = u.copy()
                uu[j] = min(u[j]+d, hi[j]) if u[j]+d <= hi[j] else u[j]-d
                uu = self._tie(uu)
                fj, _ = self._predict(env, uu)
                J[:, c] = ((fj[rows]-tgt[rows])/scale-e)/(uu[j]-u[j])
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(e))):
                break
            step = np.linalg.lstsq(J, -e, rcond=None)[0]
            step = np.clip(step, -np.where(np.array(free) < 3, .15, .3), np.where(np.array(free) < 3, .15, .3))
            for frac in (1., .5, .25):
                un = u.copy()
                un[free] = np.clip(u[free]+frac*step, lo[free], hi[free])
                un = self._tie(un)
                fn, okn = self._predict(env, un)
                if okn and (not ok0 or np.sum(((fn[rows]-tgt[rows])/scale)**2) <= np.sum(e**2)):
                    u = un
                    break
        self.u = u
        return u

    def fly(self, sc, target, trace=False):
        env = F.TargetFlight(sc, target, self.morph, trace=trace)
        pc = self.pc
        nxt = 0.
        while not env.done:
            if env.t >= nxt-1e-9:
                self.correct(env, pc['newton_first'] if env.t == 0 else pc['newton'])
                nxt += pc['cycle_s']
            env.advance(nxt-env.t, _command(self.u, self.phases, self.morph))
        env.info['n_predictions'] = self.n_predictions
        return env


def fly_constant(sc, target, action, morph=True, trace=False, kg=None):
    env = F.TargetFlight(sc, target, morph, trace=trace, kg=kg)
    while not env.done:
        env.step(action)
    return env


def constant_table(cases, workers, morph=True, kg=None):
    """Return G[action, case] and violation[action, case] of every constant action on every case."""
    from .parallel import pool_map
    n = F.n_actions(morph)
    jobs = [('constant', dict(case=c, action=a, morph=morph, kg=kg)) for a in range(n) for c in cases]
    res = pool_map(jobs, workers, chunksize=4)
    G = np.array([r['G'] for r in res]).reshape(n, len(cases))
    V = np.array([not r['feasible'] for r in res]).reshape(n, len(cases))
    return G, V


def tune_constant(cases, workers, morph=True):
    """Best constant action on the training cases (mean return); returns (action, table)."""
    from .parallel import pool_map
    n = F.n_actions(morph)
    jobs = [('constant', dict(case=c, action=a, morph=morph)) for a in range(n) for c in cases]
    res = pool_map(jobs, workers)
    table = []
    for a in range(n):
        part = res[a*len(cases):(a+1)*len(cases)]
        table.append(dict(action=a, decoded=F.decode(a, morph), mean_G=float(np.mean([r['G'] for r in part])),
                          hits=int(sum(r['hit'] for r in part)), violations=int(sum(not r['feasible'] for r in part))))
    best = max(table, key=lambda r: (r['mean_G'], -r['violations']))
    return best['action'], table


def reference(sc, target, region=None, cfg=None):
    """Clairvoyant three-phase plan (true disturbances); best of Nelder-Mead runs from the 3 nearest grid plans."""
    from .region import fly_plan, load_region
    cfg = cfg or config.load()
    region = region or load_region()
    maxfev = cfg['reference']['maxfev']
    rows = [r for r in region['rows'] if r['morph'] and r['feasible']]
    starts = sorted(rows, key=lambda r: (r['range_km']-target.range_km)**2+((r['V_end']-target.speed_mps)/10)**2)[:3]
    best = dict(cost=np.inf)

    def cost(p):
        plan = dict(r=list(np.clip(p[:3], .3, .95)), w=list(np.clip(p[3:], 0., 1.)))
        env = fly_plan(sc, plan, True, target)
        dD, dV = env.errors()
        c = dD**2+(dV/10)**2 if not env.failed else 1e6+(env.w['T']-env.t)*1e3
        if c < best['cost']:
            best.update(cost=c, plan=plan, summary=env.summary())
        return c
    for s in starts:
        minimize(cost, np.r_[s['plan']['r'], s['plan']['w']], method='Nelder-Mead',
                 options=dict(maxfev=maxfev//len(starts), xatol=1e-4, fatol=1e-3))
    out = dict(best['summary'])
    out['plan'] = best['plan']
    return out
