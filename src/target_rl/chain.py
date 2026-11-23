"""v4c: the target-arrival flight on the CERTIFIED CHAIN (certified_flight), with the interface of guided.RelPredFlight.

What flies (every 3 s decision of the chain):
  CL target of the policy (changed every 12 s by its CL increment) - kg gamma, raised by the protection loop (dynamic-pressure
  law with the online lift estimate, altitude-floor law; configs/v4c.yaml protection) = desired CL;
  requested morph level w of the policy -> the chain's geometry generator picks the F8D2000B shape whose equal-lift drag
  level is nearest to w (8 s transition, a new shape only after the current transition is complete);
  the certifier probes the chain's CL actions in order of closeness to the desired CL and executes the closest one with a
  VERIFIED_PASS certificate: surrogate of the two-stage OCP (pre: planned Mach, post: six 0.5 s alpha commands) or the
  exact-physics baseline on an out-of-distribution refusal, post corrector, 30 s look-ahead guard, 0.1 s audit, all on the
  NOMINAL model. The true-OCP takeover is switched off (configs/runtime_v4c.json; cost), so such an action is uncertified.
  The motion itself is the TRUTH plant (certified_flight.truth): the scenario's density field and lift/drag biases and the
  configured model mismatch; the certifier never sees them.
Failures: no certified action (NOT_CERTIFIED), an execution refusal, or a truth path-limit violation (TRUTH_VIOLATION).

The policy's own model (prediction every 48 s, initial CL map) is the NOMINAL point-mass model of target_rl with the same
F8D2000B tables (what the designer knows): guided.RelPredFlight._predict on a nominal copy of the current state.
"""
import copy
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np

from . import config, flight as F, guided as G
from .scenario import Field

_CORE = {}


def _quiet_certification():
    """No proof files, pool logs or event logs (thousands of training flights); certificates are still cached in memory."""
    from certified_flight import certification as C
    C.write_json = lambda *a, **k: None
    C.append_jsonl = lambda *a, **k: None


def runtime_path():
    return config.load()['chain']['runtime']


def core(physics_controller=False):
    """Per-process chain: (env, certifier). physics_controller=True: the exact-physics baseline flies instead of the surrogate
    and the look-ahead guard is off (used only to record contexts for the surrogate refit, before a surrogate exists)."""
    key = (runtime_path(), bool(physics_controller))
    if key in _CORE:
        return _CORE[key]
    os.environ['CF_RUNTIME'] = runtime_path()
    from certified_flight.build import ROOT, build_flight, runtime_config
    from certified_flight.certification import Certifier
    from certified_flight.environment import CertifiedFlightEnvironment
    from certified_flight.mission import Mission
    from certified_flight.aero import AeroDatabase
    from certified_flight.objectives import contract
    from certified_flight.ocp import TwoStageOCP
    from certified_flight.plant import CompiledPlant
    from certified_flight.surrogate import PhysicalBaseline
    from certified_flight.truth import TruthPlant
    import yaml
    _quiet_certification()
    rt = runtime_config(ROOT)
    folder = config.runs_dir()/'chain_scratch'/str(os.getpid())
    if physics_controller:
        physical = rt['physical']
        mission = Mission(ROOT, physical['mission'])
        db = AeroDatabase(ROOT/physical['snapshot'])
        plant = CompiledPlant(mission, db)
        traj = yaml.safe_load(mission.path(physical['trajectory']).read_text(encoding='utf-8'))
        oracle = TwoStageOCP(mission, db, traj, physical['ocp'], plant)
        safety = copy.deepcopy(rt['safety_options'])
        safety['lookahead_enabled'] = False
        env = CertifiedFlightEnvironment(plant, PhysicalBaseline(db, mission, contract(physical)), physical, oracle, safety)
        certifier = Certifier(env, folder/'proofs', dict(label='v4c-physics', physical=physical), memo=True,
                              retry_options=rt['dead_end_retry'])
    else:
        env, certifier = build_flight(folder, 'v4c', config=rt)
    env.oracle_enabled = bool(config.load()['chain'].get('true_ocp_fallback', False))
    env.event_sink = env.artifact_sink = None
    env.truth = TruthPlant(env.ocp.plant, rt['physical']['start_geometry'])
    from certified_flight.state import STATE_FIELDS
    _CORE[key] = dict(env=env, certifier=certifier, fields=STATE_FIELDS, x0=env.x.copy(), alpha0=float(env.alpha))
    return _CORE[key]


class ChainFlight:
    """One flight of the v4c task (scenario sc, target) on the certified chain; RelPredFlight interface."""

    def __init__(self, sc, target, morph=True, trace=False, kg=None, physics_controller=False, record=None):
        cfg = config.load()
        self.w = F.world(morph)
        self.kg = self.w['kg'] if kg is None else float(kg)
        self.morph = bool(morph)
        self.sc, self.target = sc, target
        self.phys = bool(physics_controller)
        self.record = record                  # list: committed pre contexts are appended (surrogate refit)
        c = core(self.phys)
        env = c['env']
        env.reset()
        env.x = np.array(c['x0'], float)
        env.x[0] += sc.dh
        env.x[1] += sc.dv
        self.chain_state = {k: copy.deepcopy(getattr(env, k)) for k in c['fields']}
        self.field = Field(sc, cfg)
        self.plant_cfg = dict(cfg.get('plant', {}) or {})
        self.x = list(map(float, env.x))
        self.s0 = self.x[3]
        self.t = 0.
        self.done = self.failed = False
        self.reason = None
        self.G = None
        self.est = F.LoadEstimator()
        self.rho_prev = self.field(self.x[3])
        self.min_margin = 1e9
        self.protect_steps = self.steps = 0
        self.n_switches = 0
        self.channels = {}
        self.last_action = None
        self.info = {}
        self.trace = [] if trace else None
        self.cl_t = _initial_cl(target)
        self.w_cur = 0.
        self.npred = 0
        self.pdt = cfg['guidance'].get('predict_dt_s', 4.)
        self._predict()

    # ------------------------------------------------------------------ measurements (as TargetFlight)
    def k_rho(self):
        return self.field(self.x[3])

    def q(self):
        return .5*self.w['model'].atm.density(self.x[0])*self.k_rho()*self.x[1]**2

    def range_km(self):
        return (self.x[3]-self.s0)/1e3

    def features(self, extra=False):
        base = F.TargetFlight.features(self, extra)
        return base+self.guide_features() if extra else base

    errors = F.TargetFlight.errors
    hit = F.TargetFlight.hit
    _finish = F.TargetFlight._finish

    def guide_features(self):
        e = self.pred-np.array([self.target.range_km, self.target.speed_mps])
        return [float(e[0])/100., float(e[1])/100., self.w_cur, self.cl_t*10.]

    # ------------------------------------------------------------------ the policy's nominal prediction
    def _nominal_copy(self):
        """A target_rl nominal-model flight at the current (measured) state: no plant mismatch, density frozen at the measured
        value, estimated biases."""
        from .scenario import Scenario
        kl, kd = self.est.values
        sc = dataclasses.replace(self.sc, k_lift=float(np.clip(kl, .7, 1.3)), k_drag=float(np.clip(kd, .7, 1.3)))
        p = F.TargetFlight(sc, self.target, True)
        p.plant = False
        p.realistic = False
        p.field = _Const(self.k_rho())
        p.x = list(self.x)
        p.s0 = self.s0
        p.t = self.t
        p.est = copy.copy(self.est)
        p.est.num, p.est.den = list(self.est.num), list(self.est.den)
        p.w = dict(self.w, dt=self.pdt)
        return p

    def _predict(self):
        self.npred += 1
        self.last_pred_t = self.t
        p = self._nominal_copy()
        F.TargetFlight.advance(p, self.w['T']-p.t, G._hold(self.cl_t, 0.))
        self.pred = np.array([p.range_km(), p.x[1]])

    # ------------------------------------------------------------------ protection loop (desired CL)
    def desired_cl(self):
        w, model, p = self.w, self.w['model'], self.w['prot']
        x = self.x
        q = self.q()
        cl = self.cl_t-self.kg*x[2]
        kl = min(max(self.est.values[0], .7), 1.3) if p.get('use_lift_estimate') else 1.
        guard = model.level_cl(x, q)/kl*(1+p['kp']*(q/(p['q_ratio']*w['qmax'])-1))-p['kg']*x[2]
        protected = guard > cl
        cl = max(cl, guard)
        buf = p.get('floor_buffer_m', 0.)
        if buf > 0:
            h0 = w['limits'].h[0]+buf
            if x[0] < h0:
                need = model.level_cl(x, q)/kl*(1+p.get('floor_gain', 2.)*(h0-x[0])/buf)-p['kg']*x[2]
                if need > cl:
                    cl, protected = need, True
        lo, hi = w['cl_clip']
        return min(max(cl, lo), hi), protected

    # ------------------------------------------------------------------ one chain decision (3 s)
    def _chain_step(self, level):
        c = core(self.phys)
        env, cert = c['env'], c['certifier']
        for k, v in self.chain_state.items():
            setattr(env, k, copy.deepcopy(v))
        env.truth.set_scenario(self.field, self.sc, self.plant_cfg)
        env.db.level = level
        cert.tag['morph_level'] = level
        cert.records.clear()
        cert.pool_records.clear()
        env.physics_events.clear()
        cl_des, protected = self.desired_cl()
        acts = np.asarray(env.cl_actions)
        pref = -np.abs(acts-cl_des)-1e-6*np.arange(len(acts))
        from certified_flight.aero import DomainExit
        try:
            pool = cert.pool(pref)
        except (DomainExit, ValueError, FloatingPointError) as exc:      # state outside the tables / model domain
            return self._fail('OUT_OF_DOMAIN:'+type(exc).__name__)
        if not pool['mask'].any():
            reasons = {}
            for r_ in (cert.pool_records[-1]['reasons'] if cert.pool_records else []):
                reasons[str(r_)] = reasons.get(str(r_), 0)+1
            self.info['not_certified'] = dict(t=self.t, h=self.x[0], V=self.x[1], gamma=self.x[2], cl_des=cl_des,
                                              level=level, source=env.db.ids[env.transition.source],
                                              target=env.db.ids[env.transition.target], reasons=reasons)
            return self._fail('NOT_CERTIFIED')
        a = int(np.argmax(np.where(pool['mask'], pref, -np.inf)))
        from certified_flight.surrogate import OracleRequired
        x0 = np.array(self.x)
        try:
            _, _, info = cert.commit(pool, a)
        except (OracleRequired, RuntimeError, ValueError) as exc:
            return self._fail('EXECUTION_REFUSED:'+str(getattr(exc, 'reason', exc))[:60])
        if len(cert.cache) > 2000:
            cert.cache.clear()
        if self.record is not None:
            self.record.append(dict(t=self.t, CL_raw=env.cl_actions[a], channel=info.get('channel'),
                                    pre_context=info['pre_context']))
        self.channels[info.get('channel')] = self.channels.get(info.get('channel'), 0)+1
        self.n_switches += int(bool(info.get('transition_started')))
        self.protect_steps += int(protected)
        self.steps += 1
        self.chain_state = {k: copy.deepcopy(getattr(env, k)) for k in c['fields']}
        margin = float(info['plant']['min_normalized_margin'])
        self.min_margin = min(self.min_margin, margin)
        k_meas = self.k_rho()
        self.x = list(map(float, env.x))
        self.t = float(env.t)
        if self.trace is not None:
            self.trace.append([self.t, *self.x[:3], self.range_km(), env.cl_actions[a], level or 0., margin, int(protected)])
        if margin < -1e-7:                    # a truth path limit (incl. the Mach range of the tables) was crossed
            return self._fail('TRUTH_VIOLATION')
        # online lift/drag estimate: truth motion vs the nominal coefficients at the executed alpha (0.5 s steps)
        plant = env.ocp.plant
        pc = info['post_context']
        xs = np.vstack([x0, np.asarray(info['plant']['x'])])
        try:
            for j, al in enumerate(info['plant']['alpha']):
                frac = min(1., pc['fraction']+pc['rate']*.5*j)
                qq = plant.quantities(xs[j], float(al), pc['source'], pc['target'], frac)
                self.est.add(self.w, list(xs[j]), list(xs[j+1]), float(qq[4]), float(qq[5]), k_meas, .5)
                self.est.decay()
        except DomainExit:
            return self._fail('OUT_OF_DOMAIN:DomainExit')
        if self.t >= self.w['T']-1e-9:
            self.done = True
            self.reason = 'END'
            self._finish()

    def _fail(self, reason):
        self.failed = self.done = True
        self.reason = reason
        self._finish()

    def step(self, action):
        if self.done:
            raise RuntimeError('flight finished')
        i_cl, i_w = F.decode(action, self.morph)
        lo, hi = self.w['cl_clip']
        self.cl_t = float(np.clip(self.cl_t+self.w['cl_levels'][i_cl], lo, hi))
        self.w_cur = self.w['w_levels'][i_w]
        self.rho_prev = self.k_rho()
        self.last_action = int(action)
        level = (self.w_cur if self.w_cur > 0 else None) if self.morph else None
        end = self.t+self.w['decision']
        while not self.done and self.t < end-1e-9:
            self._chain_step(level)
        every = config.load()['guidance'].get('predict_every_s', self.w['decision'])
        if not self.done and self.t >= self.last_pred_t+every-1e-6:
            self._predict()
        if self.done:
            self.info.update(n_predictions=self.npred)
        return self

    def fork(self):
        c = copy.copy(self)
        c.x = list(self.x)
        c.est = copy.copy(self.est)
        c.est.num, c.est.den = list(self.est.num), list(self.est.den)
        c.chain_state = {k: copy.deepcopy(v) for k, v in self.chain_state.items()}
        c.channels = dict(self.channels)
        c.info = dict(self.info)
        c.pred = self.pred.copy()
        c.trace = None
        c.record = None
        return c

    def summary(self):
        dD, dV = self.errors()
        return dict(level=self.sc.level, seed=self.sc.seed, target=self.target.name, target_range_km=self.target.range_km,
                    target_speed_mps=self.target.speed_mps, G=self.G, hit=bool(self.hit()), feasible=not self.failed,
                    reason=self.reason, t_end=self.t, range_km=self.range_km(), V_end=self.x[1], h_end=self.x[0],
                    dD_km=dD, dV_mps=dV, protect_fraction=self.protect_steps/max(self.steps, 1),
                    min_margin=self.min_margin, n_switches=self.n_switches, channels=dict(self.channels), **self.info)


class _Const:
    def __init__(self, k):
        self.k = float(k)

    def __call__(self, s):
        return self.k


def _initial_cl(target):
    rows = sorted(G.nominal_map(), key=lambda r: r[1])
    return float(np.interp(target.range_km, [r[1] for r in rows], [r[0] for r in rows]))
