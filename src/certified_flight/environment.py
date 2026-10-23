"""Flight environment: CL action -> pre controller -> geometry generator -> post controller -> plant.

One decision lasts 3 s (six 0.5 s alpha commands). A CL action is screened for
reachability, the pre head predicts the planning Mach, the generator selects the
minimum-drag geometry for that CL, the post head proposes six alpha commands,
the post corrector and the lookahead guard may adjust them, and dense (.1 s) and
plant-grid (.5 s) re-integrations must satisfy every path limit. Any refusal
raises OracleRequired without changing the environment state.
"""
from __future__ import annotations

from copy import copy, deepcopy
import time
from types import SimpleNamespace

import numpy as np

from .corrector import PostCorrector, audit_controls, compact_audit
from .contracts import matching_contract
from .diagnostics import MARGIN_NAMES, constraint_evidence, interval_moments
from .lookahead import LookaheadGuard
from .ocp import OracleController
from .reachability import CLReachability
from .state import STATE_FIELDS
from .surrogate import (CheckedSurrogate, OOD_REASONS, OracleRequired, PhysicalBaseline,
                        PhysicsOnRefusal, SurrogateUnavailable)
from .transition import Transition


class FlightEnvironment:
    """Surrogate-driven decision step with exact plant, generator and audits."""

    def __init__(self, plant, controller, config, safety_options=None, post_corrector=None):
        # env.ocp groups the shared numeric objects (plant, database, mission).
        self.ocp = SimpleNamespace(plant=plant, db=plant.db, m=plant.mission)
        self.db, self.m, self.cfg = plant.db, plant.mission, config
        self.cl_actions = [float(cl) for cl in config['cl_actions']]
        self.start = self.db.index['G42']
        self.control = config.get('control', {})
        if self.control.get('pre_transition_model', 'frozen') not in ('actual', 'frozen'):
            raise ValueError('Unknown preliminary transition model')
        if not np.isfinite(self.control.get('switch_cost', 0.)) or self.control.get('switch_cost', 0.) < 0:
            raise ValueError('switch_cost must be finite and nonnegative')
        self.control_contract = matching_contract(controller.metadata.get('control_contract'), config)
        self.controller = controller
        self.reachability = CLReachability(plant)
        self.safety_options = dict(safety_options or {})
        self.post_corrector = post_corrector
        self.lookahead_guard = None
        self.lookahead_qualification = None
        self.reset()

    @property
    def n_actions(self):
        return len(self.cl_actions)

    def reset(self):
        self.x = np.array(self.m.data['initial_state']['x_h_V_gamma_s'], dtype=float)
        self.alpha = float(self.m.data['initial_state']['alpha_rad'])
        self.t, self.done = 0., False
        self.transition = Transition(self.start, self.start, duration=self.cfg['transition']['duration_s'])
        self.total_reward, self.switches, self.last_outcome = 0., 0, 'INCOMPLETE'
        self.previous_cl_raw = self.previous_cl_cmd = self.previous_cl_actual = 0.
        self.history_valid = False
        self.last_transition_completed = None
        self.decision_index = 0

    def state_summary(self):
        """Scaled plant quantities and history at the current state (float32).

        Evaluating the plant raises DomainExit outside the aerodynamic table, and
        the pre-commit check requires every entry to be finite in single precision.
        """
        z = self.transition
        f = z.fraction(self.t)
        q = self.ocp.plant.quantities(self.x, self.alpha, z.source, z.target, f)
        flight = [self.t/self.m.data['duration_s'], self.x[0]/40000, self.x[1]/2400,
                  self.x[2]/.1, self.x[3]/2.4e6, self.alpha/.1, q[3]/8,
                  q[4]/.1, q[5]/.03, q[0]/80000, q[1]/2e6, f,
                  max(0, z.started+z.duration-self.t)/z.duration if z.active(self.t) else 0]
        history = [float(z.active(self.t)), self.previous_cl_raw/.1, self.previous_cl_cmd/.1,
                   self.previous_cl_actual/.1, float(self.history_valid),
                   (self.m.data['duration_s']-self.t)/self.m.data['duration_s']]
        return np.r_[flight, history].astype(np.float32)

    def context(self, z, cl, phase, raw=None):
        """Controller input; 'post' includes the transition the generator just commanded."""
        rate = (1/z.duration if z.active(self.t) else 0.)
        if phase == 'pre' and self.control.get('pre_transition_model', 'frozen') == 'frozen':
            rate = 0.
        return {'x': self.x.tolist(), 'alpha_prev': self.alpha, 't': self.t, 'source': z.source, 'target': z.target,
                'fraction': z.fraction(self.t), 'rate': rate, 'CL_des': cl, 'CD_des': None,
                'CL_raw': cl if raw is None else raw, 'control_contract': self.control_contract}

    def step(self, action):
        """Execute one 3 s decision; returns (state_finite, reward, done, info)."""
        if self.done:
            raise RuntimeError('reset() required after termination')
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(action)
        start = time.perf_counter()
        raw = self.cl_actions[int(action)]
        old_x = self.x.copy()
        old_alpha = self.alpha
        old_cl = self.previous_cl_cmd if self.history_valid else None
        z = copy(self.transition)
        z.settle(self.t)
        locked = z.active(self.t)
        current_q = self.ocp.plant.quantities(self.x, self.alpha, z.source, z.target, z.fraction(self.t))
        reachable = self.reachability.check(self.x, self.alpha, z, self.t, raw,
                                            enabled=self.control.get('reachability', False))
        cl = reachable['CL_cmd']
        stage = time.perf_counter()
        pre_c = self.context(z, cl, 'pre', raw)
        mach, pre_diag = self.controller.predict(pre_c, 'pre')
        pre_wall = time.perf_counter()-stage
        stage = time.perf_counter()
        choice = self.db.generate(float(mach[0]), cl, current=z.target)
        generator_wall = time.perf_counter()-stage
        switched = False
        if choice['geometry_index'] is not None:
            switched = z.command(choice['geometry_index'], self.t)
        post_c = self.context(z, cl, 'post', raw)
        stage = time.perf_counter()
        try:
            alpha, post_diag = self.controller.predict(post_c, 'post')
        except OracleRequired as exc:
            exc.diagnostics['pre_context'] = pre_c
            raise
        post_wall = time.perf_counter()-stage
        supported = bool(pre_diag.get('supported', True) and post_diag.get('supported', True))
        correction = None
        network_alpha = np.asarray(alpha).copy()
        if self.post_corrector is not None:
            corrected = self.post_corrector.correct(post_c, alpha)
            correction = corrected.diagnostics
            correction.update(empirical_support=supported)
            if not corrected.accepted:
                raise OracleRequired(corrected.status, post_c, 'post', dict(
                    correction=correction, pre_context=pre_c, CL_reachability=reachable, generator=choice,
                    network_alpha_rad=network_alpha.tolist()))
            alpha = corrected.alpha
        future = None
        if self.lookahead_guard is not None:
            checked = self.lookahead_guard.assess(post_c, alpha)
            future = checked.diagnostics
            if not checked.accepted:
                raise OracleRequired(checked.status, post_c, 'post', dict(
                    lookahead=future, pre_context=pre_c, CL_reachability=reachable, generator=choice,
                    current_alpha_rad=np.asarray(alpha).tolist()))
            alpha = checked.alpha
        alpha = np.asarray(alpha, float)
        limits = self.m.data['limits']
        lo = max(float(self.db.A[0]), float(limits['alpha_rad'][0]))
        hi = min(float(self.db.A[-1]), float(limits['alpha_rad'][1]))
        control_ok = bool(alpha.shape == (6,) and np.isfinite(alpha).all() and np.all(alpha >= lo-1e-9)
                          and np.all(alpha <= hi+1e-9)
                          and np.all(abs(np.diff(np.r_[old_alpha, alpha])) <= limits['alpha_rate_radps']*.5+1e-7))
        if not control_ok:
            raise OracleRequired('DENSE_AUDIT_REJECTED', post_c, 'post', dict(
                pre_context=pre_c, reason='INDEPENDENT_ALPHA_BOUND_OR_RATE_CHECK', alpha_rad=alpha.tolist(),
                CL_reachability=reachable, generator=choice))
        times = np.arange(7)*self.m.data['plant_dt_s']
        audit = self.ocp.plant.integrate(self.x, alpha, times, z.source, z.target, post_c['fraction'],
                                         post_c['rate'], self.cfg['ocp']['audit_dt_s'])
        if not audit['valid']:
            raise OracleRequired('FAST_EXECUTION_AUDIT_FAILURE', post_c, 'post',
                                 self.rejection_evidence(audit, alpha, pre_c, reachable, choice))
        actual = self.ocp.plant.integrate(self.x, alpha, times, z.source, z.target, post_c['fraction'],
                                          post_c['rate'], self.m.data['plant_dt_s'])
        if not actual['valid']:
            raise OracleRequired('FAST_PLANT_RK_STAGE_FAILURE', post_c, 'post',
                                 self.rejection_evidence(actual, alpha, pre_c, reachable, choice))
        ds = float(actual['x'][-1, 3]-self.x[3])
        t0 = self.t
        info = {'channel': 'FAST_APPROXIMATE', 'control_contract': self.control_contract,
                'empirical_support': supported, 'action': int(action), 'CL_raw': raw, 'CL_cmd': cl, 'CL_des': cl,
                't_start_s': t0, 'planned_mach': float(mach[0]), 'generator': choice, 'pre_surrogate': pre_diag,
                'post_surrogate': post_diag, 'pre_context': pre_c, 'post_context': post_c,
                'source_geometry': self.db.ids[z.source], 'target_geometry': self.db.ids[z.target],
                'transition_start_s': z.started, 'transition_locked': locked, 'transition_started': switched,
                'fraction_start': post_c['fraction'], 'fraction_end': z.fraction(t0+3),
                'executed_plan': 'APPROX_CONTROLLER', 'execution_audit_horizon_s': 3.,
                'execution_audit_dt_s': audit['dt'], 'execution_audit_min_margin': audit['min_margin'],
                'execution_audit_margins': dict(zip(MARGIN_NAMES, audit['margins'])),
                'plant': {'dt_s': actual['dt'], 'x': actual['x'].tolist(), 'alpha': alpha.tolist(),
                          'q_start': actual['samples'][4:, ::5].T.tolist(), 'q_end': actual['samples'][4:, 4::5].T.tolist(),
                          'min_normalized_margin': actual['min_margin']}}
        if correction is not None:
            info.update(post_correction=correction, network_alpha_rad=network_alpha.tolist(),
                        safety_options=self.safety_options)
        if future is not None:
            info.update(lookahead=future, lookahead_qualification=self.lookahead_qualification)
        event = None
        if switched:
            _, drag, feasible = self.db.equal_lift(float(mach[0]), cl)
            ranked = np.sort(drag[feasible])
            gap = float(ranked[1]-ranked[0]) if len(ranked) > 1 else None
            source_cd = float(drag[z.source]) if feasible[z.source] else None
            event = {'decision_index': self.decision_index, 'start_time_s': self.t,
                     'source_geometry': self.db.ids[z.source], 'target_geometry': self.db.ids[z.target],
                     'previous_completion_time_s': self.last_transition_completed,
                     'scheduled_completion_time_s': self.t+z.duration,
                     'CL_raw': raw, 'CL_cmd': cl, 'CL_previous_cmd': old_cl,
                     'CL_target_changed': old_cl is not None and abs(cl-old_cl) > 1e-12,
                     'Mach_preview': float(mach[0]), 'target_CD_equal_CL': float(drag[z.target]),
                     'source_CD_equal_CL': source_cd,
                     'CD_gain_equal_CL': None if source_cd is None else source_cd-float(drag[z.target]),
                     'runner_up_CD_gap': gap, 'ranking_near_tie': gap is not None and gap <= 1e-6,
                     'near_tie_diagnostic_tolerance': 1e-6, 'actual_new_pair': z.source != z.target}
        self.x = actual['x'][-1]
        self.alpha = float(alpha[-1])
        self.t += 3
        if z.source != z.target and not z.active(self.t):
            self.last_transition_completed = z.started+z.duration
        self.transition = z
        z.settle(self.t)
        self.switches += int(switched)
        self.previous_cl_raw = raw
        self.previous_cl_cmd = cl
        moments = interval_moments(actual, horizon=3., target=cl)
        self.previous_cl_actual = moments['mean']
        self.history_valid = True
        outcome = self.m.outcome(self.t, self.x)
        rew = self.m.reward(ds, outcome)
        rew['original_reward'] = rew['reward']
        rew['switch_cost'] = float(self.control.get('switch_cost', 0.)) if event else 0.
        rew['reward'] -= rew['switch_cost']
        self.done = rew['terminated'] or rew['truncated']
        self.last_outcome = outcome
        self.total_reward += rew['reward']
        info.update(t_end_s=self.t, outcome=outcome, reward=rew, range_m=float(self.x[3]), switches=self.switches,
                    wall_s=time.perf_counter()-start)
        final = interval_moments(audit, target=cl)
        violations, active = constraint_evidence(audit)
        info.update(transition_event=event, executed_CL_mean=moments['mean'], executed_CL_target_rmse=moments['rmse'])
        info['diagnostics'] = dict(
            reachable, decision_index=self.decision_index, mission_time_s=t0,
            CL_pre=None, CL_pre_unavailable_reason='PRE_SURROGATE_PREDICTS_MACH_ONLY', CL_gen=choice.get('CL'),
            CL_final=final['mean'], CL_actual=moments['mean'],
            CL_final_time_basis='predicted_3s_dense_audit_of_surrogate_controls',
            CL_time_basis='RK4_quadrature_first_3s_mean',
            CL_gen_time_basis='standalone_candidate_pointwise_at_own_matched_alpha',
            CL_decision_instantaneous=float(current_q[4]), Mach_actual=float(current_q[3]), alpha_actual=old_alpha,
            height=float(old_x[0]), velocity=float(old_x[1]), gamma=float(old_x[2]),
            dynamic_pressure=float(current_q[0]),
            Mach_preview=float(mach[0]), alpha_preview=None, alpha_generator_matched=choice.get('alpha_rad'),
            CL_target=cl, generator_geometry=choice.get('geometry_id'), geometry_start_id=info['source_geometry'],
            geometry_target_id=info['target_geometry'], morph_fraction=post_c['fraction'],
            morph_remaining_s=(1-post_c['fraction'])*z.duration,
            pre_ocp_status='SURROGATE_NO_OCP', final_ocp_status='SURROGATE_NO_OCP', generator_status=choice['status'],
            fallback_type='NONE', constraint_violations=violations, near_active_constraints=active,
            CL_tracking_error=moments['mean']-cl, raw_to_cmd=cl-raw, cmd_to_pre=None,
            cmd_to_gen_pointwise=None if choice.get('CL') is None else choice['CL']-cl,
            cmd_to_final=final['mean']-cl, final_to_actual=moments['mean']-final['mean'],
            pre_surrogate_wall_time=pre_wall, generator_wall_time=generator_wall, final_surrogate_wall_time=post_wall,
            pre_transition_rate=pre_c['rate'], final_transition_rate=post_c['rate'], transition_started=bool(event),
            decision_wall_time=time.perf_counter()-start)
        self.decision_index += 1
        state_finite = bool(np.isfinite(self.state_summary()).all())
        return state_finite, rew['reward'], self.done, info

    def rejection_evidence(self, audit, alpha, pre_c, reachable, choice):
        samples = audit['samples']
        finite = bool(np.isfinite(samples).all())
        return {'min_margin': audit['min_margin'], 'pre_context': pre_c,
                'CL_reachability': reachable, 'generator': choice, 'alpha_rad': np.asarray(alpha).tolist(),
                'audit_dt_s': audit['dt'], 'finite': finite,
                'constraint_margins': dict(zip(MARGIN_NAMES, audit['margins'])),
                'constraint_violations': constraint_evidence(audit)[0],
                'RK_stage_ranges': {key: [float(np.min(samples[i])), float(np.max(samples[i]))]
                                    for key, i in (('h_m', 0), ('V_mps', 1), ('gamma_rad', 2), ('q_pa', 4),
                                                   ('heat', 5), ('n_load', 6), ('Mach', 7), ('CL', 8))}}


class CertifiedFlightEnvironment(FlightEnvironment):
    """Transactional decision step with the fallback chain.

    Every attempt runs on a candidate copy. A surrogate refusal at the prediction
    boundary triggers, in order: the exact-physics baseline (out-of-distribution
    refusals only) and the true dual OCP. Corrector, lookahead and audit refusals
    remain refusals. The candidate state is copied back only after an independent
    pre-commit audit.
    """

    def __init__(self, plant, controller, config, oracle, safety_options, event_sink=None, artifact_sink=None,
                 ood_physics_fallback=False):
        super().__init__(plant, controller, config, safety_options)
        if self.safety_options.get('constraint_aware_post_enabled', False):
            self.post_corrector = PostCorrector(plant, self.safety_options)
        if self.safety_options.get('lookahead_enabled', False):
            qualification = LookaheadGuard.qualify(plant, controller, self.safety_options)
            if self.post_corrector is None:
                raise ValueError('Lookahead requires the qualified post corrector')
            self.lookahead_guard = LookaheadGuard(plant, controller, config, self.safety_options, self.post_corrector)
            self.lookahead_qualification = qualification
        if oracle.db is not self.db or oracle.m is not self.m:
            raise ValueError('Oracle must use the identical database and mission instances')
        if oracle.cfg != config['ocp']:
            raise ValueError('Oracle configuration mismatch')
        self.oracle = oracle
        self.event_sink, self.artifact_sink = event_sink, artifact_sink
        self.oracle_events = []
        self.attempt_counter = 0
        self.ood_physics_fallback = bool(ood_physics_fallback)
        self.physics_events = []
        self.physics_baseline = None
        if self.ood_physics_fallback:
            self.physics_baseline = PhysicalBaseline(self.db, self.m, controller.control_contract)

    def _physics_attempt(self, action, exc, started):
        if not self.ood_physics_fallback or exc.reason not in OOD_REASONS:
            return None, None, None
        physics_started = time.perf_counter()
        record = dict(attempt=self.attempt_counter, t=self.t, action=int(action),
                      trigger_head=exc.head, trigger_reason=exc.reason, accepted=False)
        controller = PhysicsOnRefusal(self.controller, self.physics_baseline)
        candidate = self._candidate(CheckedSurrogate(controller))
        try:
            result = FlightEnvironment.step(candidate, action)
        except (OracleRequired, ValueError, FloatingPointError, RuntimeError) as failure:
            result = None
            record.update(reason=getattr(failure, 'reason', None) or repr(failure),
                          diagnostics=deepcopy(getattr(failure, 'diagnostics', {})))
        else:
            record.update(accepted=True, reason=None, baseline_heads=[u['head'] for u in controller.used])
        record['wall_s'] = time.perf_counter()-physics_started
        self.physics_events.append(record)
        return result, candidate, record

    def _candidate(self, controller):
        """Shallow copy with private state fields and the given controller."""
        candidate = copy(self)
        for name in STATE_FIELDS:
            setattr(candidate, name, deepcopy(getattr(self, name)))
        candidate.controller = controller
        return candidate

    def step(self, action):
        """Execute one decision transactionally; returns (reward, done, info) or raises OracleRequired."""
        started = time.perf_counter()
        self.attempt_counter += 1
        candidate = self._candidate(CheckedSurrogate(self.controller))
        event = None
        physics = None
        exc = None
        try:
            result = FlightEnvironment.step(candidate, action)
        except SurrogateUnavailable as refusal:
            exc = refusal
            result, physics_candidate, physics = self._physics_attempt(action, exc, started)
            if result is not None:
                candidate = physics_candidate
        if result is None:
            event = dict(attempt=self.attempt_counter, t=self.t, action=int(action),
                         CL_raw=self.cl_actions[int(action)], trigger_head=exc.head, trigger_reason=exc.reason,
                         trigger_context=deepcopy(exc.context), trigger_diagnostics=deepcopy(exc.diagnostics),
                         solver_calls={'pre': 0, 'post': 0}, phases={}, accepted=False, committed=False)
            if physics is not None:
                event['physics_fallback_attempt'] = deepcopy(physics)
            self.oracle_events.append(event)
            oracle_controller = OracleController(self.oracle, event, self.artifact_sink)
            candidate = self._candidate(oracle_controller)
            try:
                result = FlightEnvironment.step(candidate, action)
            except OracleRequired as failure:
                event.update(reason=failure.reason, failure_head=failure.head,
                             failure_diagnostics=deepcopy(failure.diagnostics), wall_s=time.perf_counter()-started)
                if self.event_sink:
                    self.event_sink(event)
                raise
            except Exception as failure:
                event.update(reason='TRUE_OCP_OR_AUDIT_EXCEPTION', error=repr(failure),
                             wall_s=time.perf_counter()-started)
                if self.event_sink:
                    self.event_sink(event)
                raise OracleRequired(event['reason'], exc.context, exc.head, deepcopy(event)) from failure
        state_finite, reward, done, info = result
        # Validate independently before copying ANY physical/history/reward state.
        audit = audit_controls(self.ocp.plant, info['post_context'], info['plant']['alpha'])
        if not audit['valid'] or not state_finite or not np.isfinite(reward):
            if event is not None:
                event.update(reason='PRECOMMIT_AUDIT_REJECTED', audit=compact_audit(audit),
                             wall_s=time.perf_counter()-started)
                if self.event_sink:
                    self.event_sink(event)
            raise OracleRequired('PRECOMMIT_AUDIT_REJECTED', info['post_context'], 'post', compact_audit(audit))
        if event is not None:
            event.update(accepted=True, committed=True, reason=None, wall_s=time.perf_counter()-started,
                         executed_alpha_rad=info['plant']['alpha'], execution_audit=compact_audit(audit))
            info.update(channel='TRUE_DUAL_OCP_ON_DEMAND', executed_plan='TRUE_OCP_THEN_CORRECTOR_AND_LOOKAHEAD',
                        oracle_event=deepcopy(event), fallback_trigger_head=event['trigger_head'])
            if self.event_sink:
                self.event_sink(event)
        elif physics is not None:
            info.update(channel='PHYSICS_BASELINE_ON_OOD', executed_plan='PHYSICS_BASELINE_THEN_CORRECTOR_AND_LOOKAHEAD',
                        ood_physics_fallback=deepcopy(physics), fallback_trigger_head=physics['trigger_head'])
        for name in STATE_FIELDS:
            setattr(self, name, deepcopy(getattr(candidate, name)))
        info['measured_decision_wall_s'] = time.perf_counter()-started
        info['precommit_audit'] = compact_audit(audit)
        return reward, done, info
