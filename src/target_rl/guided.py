"""Study A (PREREGISTRATION.md amendment 6): flight guided by the cheap strong predictor-corrector.

The strong predictor-corrector (3 phases, (r, w) per phase) flies with cheap settings (configs/study_a.yaml
guidance: re-plan every 48 s with 1 Gauss-Newton iteration, 4 at t = 0, prediction step 4 s). Every decision (12 s)
the policy chooses an OFFSET to the planned morph level w of the current phase (configs/study_a.yaml
actions.morph_levels; offset 0 = the plan, so the policy that always picks offset 0 IS the predictor-corrector).
Shield: a non-zero offset is reduced to half of it, then to 0, unless the prediction (offset for one decision, then
the plan; measured density frozen, estimated biases, as the predictor-corrector predicts) keeps the margin on every
path limit.

Extra features (appended to TargetFlight.features(extra=True), names policy.GUIDE): the predicted range and speed
errors of the last prediction of the corrector (/100 km, /100 m/s), the planned w and r of the current phase.
"""
import copy
import dataclasses

import numpy as np

from . import baselines as B, config, flight as F


def enabled(cfg=None):
    return (cfg or config.load()).get('guidance', {}).get('kind') in ('pc_offset', 'rel_pred')


class PCOffsetFlight(F.TargetFlight):
    def __init__(self, sc, target, morph=True, trace=False, kg=None):
        super().__init__(sc, target, True, trace, kg)
        g = config.load()['guidance']
        self.shield = bool(g.get('shield', True))
        self.pc = B.PredictorCorrector('pc_strong')
        self.pc.pc = dict(self.pc.pc, cycle_s=g['cycle_s'], newton=g['newton'], newton_first=g['newton_first'],
                          predict_dt_s=g['predict_dt_s'])
        self.last_f = None
        self.offsets_applied = 0
        self.shield_cuts = 0
        self.next_cycle = 0.
        self._hook()
        self._plan()

    # -------------------------------------------------------------- predictor-corrector
    def _hook(self):
        """Record the latest prediction of the corrector (a feature of the policy)."""
        pc, env = self.pc, self
        orig = B.PredictorCorrector._predict.__get__(pc)

        def recorded(e, u):
            out = orig(e, u)
            env.last_f = out[0]
            return out
        pc._predict = recorded

    def _plan(self):
        if self.t >= self.next_cycle-1e-9:
            p = self.pc.pc
            self.pc.correct(self, p['newton_first'] if self.t == 0 else p['newton'])
            self.next_cycle += p['cycle_s']

    def _phase(self):
        return B._phase(self.t, self.pc.phases)

    def planned_w(self):
        return float(self.pc.u[3+self._phase()])

    def features(self, extra=False):
        base = super().features(extra)
        if not extra:
            return base
        e = np.asarray(self.last_f)-np.array([self.target.range_km, self.target.speed_mps])
        return base+[float(e[0])/100., float(e[1])/100., self.planned_w(), float(self.pc.u[self._phase()])]

    # -------------------------------------------------------------- decision
    def _keeps_margin(self, w):
        """Prediction: w for one decision, then the plan; True if every path limit keeps the margin."""
        self.pc.n_predictions += 1
        p = super().fork()
        p.trace = None
        kl, kd = self.est.values
        p.field = B._Const(self.k_rho())
        p.sc = dataclasses.replace(self.sc, k_lift=float(np.clip(kl, .7, 1.3)), k_drag=float(np.clip(kd, .7, 1.3)))
        p.w = dict(self.w, dt=self.pc.pc['predict_dt_s'])
        u = self.pc.u.copy()
        uw = u.copy()
        uw[3:] = w
        F.TargetFlight.advance(p, self.w['decision'], B._command(uw, self.pc.phases, True))
        if not p.done:
            F.TargetFlight.advance(p, self.w['T']-p.t, B._command(u, self.pc.phases, True))
        return (not p.failed) and p.min_margin >= self.pc.pc['margin']

    def step(self, action):
        if self.done:
            raise RuntimeError('flight finished')
        _, i_w = F.decode(action, True)
        wp = self.planned_w()
        w = float(np.clip(wp+self.w['w_levels'][i_w], 0., 1.))
        if abs(w-wp) > 1e-9:
            self.offsets_applied += 1
            if self.shield:
                for x in (w, .5*(w+wp), wp):
                    if x == wp or self._keeps_margin(x):
                        break
                    self.shield_cuts += 1
                w = x
        self.rho_prev = self.k_rho()
        self.last_action = int(action)
        u = self.pc.u.copy()
        u[3:] = w
        self.advance(self.w['decision'], B._command(u, self.pc.phases, True))
        if not self.done:
            self._plan()
        else:
            self.info.update(n_predictions=self.pc.n_predictions, offsets_applied=self.offsets_applied,
                             shield_cuts=self.shield_cuts)
        return self

    def fork(self):
        c = super().fork()
        pc = copy.copy(self.pc)
        pc.u = self.pc.u.copy()
        c.pc = pc
        c.last_f = None if self.last_f is None else np.array(self.last_f)
        c._hook()
        return c


ABS_CL = [-0.01, 0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09,
          0.10, 0.11, 0.12]
_NOMMAP = {}


def _hold(cl, wl):
    """Execution of round 2: CL target - kg gamma, refreshed every update_s; w held."""
    held = dict(next=-1., cl=cl)

    def command(env):
        if env.t >= held['next']-1e-9:
            held['cl'] = cl-env.kg*env.x[2]
            held['next'] = env.t+env.w['update_s']
        return held['cl'], wl
    return command


def nominal_map():
    """Nominal (no disturbance) end range of a constant CL target, w = 0, for the absolute CL levels."""
    if 'm' not in _NOMMAP:
        from .scenario import Scenario, Target
        nom = Scenario('L0', 0, 0., 0., 1., 1.)
        rows = []
        for cl in ABS_CL:
            e = F.TargetFlight(nom, Target('x', 0., 0.), True)
            e.advance(e.w['T'], _hold(cl, 0.))
            if not e.failed:
                rows.append((cl, e.range_km()))
        _NOMMAP['m'] = rows
    return _NOMMAP['m']


class RelPredFlight(F.TargetFlight):
    """Study R (PREREGISTRATION.md amendment 7). The policy decides every decision (12 s): an increment of the CL
    target (actions.cl_levels) and an absolute morph level w (actions.morph_levels; fixed geometry: w = 0). Input
    features: the last coarse prediction (every guidance.predict_every_s; CL target held, no more braking, density
    frozen at the measured value, estimated biases, step guidance.predict_dt_s) of the range and speed errors at
    1200 s, the current w and CL target. Initial CL target: the nominal constant CL (w = 0) whose range equals the
    target range. Execution as round 2 (CL target - kg gamma every 3 s, shared protection loop)."""

    def __init__(self, sc, target, morph=True, trace=False, kg=None):
        super().__init__(sc, target, morph, trace, kg)
        rows = sorted(nominal_map(), key=lambda r: r[1])
        cl = [r[0] for r in rows]
        rg = [r[1] for r in rows]
        self.cl_t = float(np.interp(target.range_km, rg, cl))
        self.w_cur = 0.
        self.npred = 0
        self.pdt = config.load()['guidance'].get('predict_dt_s', 4.)
        self._predict()

    def _predict(self):
        self.npred += 1
        self.last_pred_t = self.t
        p = F.TargetFlight.fork(self)
        kl, kd = self.est.values
        p.field = B._Const(self.k_rho())
        p.sc = dataclasses.replace(self.sc, k_lift=float(np.clip(kl, .7, 1.3)), k_drag=float(np.clip(kd, .7, 1.3)))
        p.w = dict(self.w, dt=self.pdt)
        F.TargetFlight.advance(p, self.w['T']-p.t, _hold(self.cl_t, 0.))
        self.pred = np.array([p.range_km(), p.x[1]])

    def features(self, extra=False):
        base = super().features(extra)
        if not extra:
            return base
        e = self.pred-np.array([self.target.range_km, self.target.speed_mps])
        return base+[float(e[0])/100., float(e[1])/100., self.w_cur, self.cl_t*10.]

    def step(self, action):
        if self.done:
            raise RuntimeError('flight finished')
        i_cl, i_w = F.decode(action, self.morph)
        lo, hi = self.w['cl_clip']
        self.cl_t = float(np.clip(self.cl_t+self.w['cl_levels'][i_cl], lo, hi))
        self.w_cur = self.w['w_levels'][i_w]
        self.rho_prev = self.k_rho()
        self.last_action = int(action)
        self.advance(self.w['decision'], _hold(self.cl_t, self.w_cur))
        every = config.load()['guidance'].get('predict_every_s', self.w['decision'])
        if not self.done and self.t >= self.last_pred_t+every-1e-6:
            self._predict()
        if self.done:
            self.info.update(n_predictions=self.npred)
        return self

    def fork(self):
        c = super().fork()
        c.pred = self.pred.copy()
        return c


def make(sc, target, morph=True, trace=False, kg=None):
    """Flight of the configured kind: the guided flight under study A, else the round-2 TargetFlight."""
    kind = config.load().get('guidance', {}).get('kind')
    if kind == 'pc_offset':
        return PCOffsetFlight(sc, target, morph, trace, kg)
    if kind == 'rel_pred':
        return RelPredFlight(sc, target, morph, trace, kg)
    return F.TargetFlight(sc, target, morph, trace=trace, kg=kg)


def plan_action(morph=True):
    """Action index of offset 0 (= follow the predictor-corrector plan); rel_pred: CL increment 0, w = 0."""
    w = F.world(morph)
    i_cl = int(np.argmin(np.abs(np.asarray(w['cl_levels']))))
    i_w = int(np.argmin(np.abs(np.asarray(w['w_levels']))))
    return F.encode(i_cl, i_w, morph)
