"""Lookahead guard: audited future witnesses with the true plant, not a recursive-safety proof.

A post plan is accepted only if, after its 3 s segment, one candidate future
policy keeps every RK stage feasible over the lookahead horizon while holding the
current CL action. If the unchanged plan has no witness, bounded alpha offsets
corrected by the post corrector are tried, within an evaluation count and a
wall-clock cap.
"""
from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from .corrector import CorrectionResult, audit_controls
from .diagnostics import MARGIN_NAMES, interval_moments
from .surrogate import OracleRequired, PhysicalBaseline
from .transition import Transition

POLICIES = ('equal_lift', 'rolling_corrector', 'load_recovery')
# Offsets are bounded candidate searches, never changes to CL actions or the generator.
ALPHA_OFFSETS_DEG = (0., .5, 1., 2., -.5, -1.)
# Settings of the qualification report that must equal the runtime corrector settings
# (boundary_smoothing is absent from V16 reports and runtime alike, so V16 still qualifies).
QUALIFIED_KEYS = ('constraint_aware_post_enabled', 'post_correction_method', 'post_correction_max_wall_s',
                  'max_iterations', 'ftol', 'alpha_scale_rad', 'CL_scale', 'weights', 'internal_margin',
                  'dense_refinement', 'boundary_smoothing')


class LoadRecoveryBaseline(PhysicalBaseline):
    """State-feedback seed for audited future segments: alpha for a bounded target load factor."""

    def __init__(self, plant, contract):
        super().__init__(plant.db, plant.mission, contract)
        self.plant = plant

    def predict(self, c, head):
        if head == 'pre':
            return super().predict(c, head)
        if c.get('control_contract') != self.control_contract:
            raise ValueError('Recovery control contract mismatch')
        self.db.check(self.features.mach(c), c['alpha_prev'])
        # Arrest descent with bounded load; ease toward unit load as gamma rises.
        target = float(np.clip(1.-c['x'][1]*c['x'][2]/(9.80665*12.), .5, 2.7))
        lo = float(self.db.A[0])
        hi = float(self.db.A[-1])
        for _ in range(24):
            a = (lo+hi)/2
            n = float(self.plant.quantities(c['x'], a, c['source'], c['target'], c['fraction'])[2])
            if n < target:
                lo = a
            else:
                hi = a
        return (self.features.project(np.full(6, (lo+hi)/2), c['alpha_prev']),
                dict(supported=True, target_load=target, kind='PHYSICAL_RECOVERY_SEED'))


class LookaheadGuard:
    def __init__(self, plant, controller, base, options, corrector):
        self.plant, self.controller, self.base, self.options, self.corrector = \
            plant, controller, base, dict(options), corrector
        self.horizon = float(options.get('lookahead_horizon_s', 20.))
        self.budget = float(options.get('lookahead_max_wall_s', 3.))
        self.policies = list(options.get('lookahead_candidate_policy', ['equal_lift', 'rolling_corrector']))
        if not options.get('lookahead_policy_search_enabled', True):
            self.policies = self.policies[:1]
        self.search_order = options.get('lookahead_search_order', 'declared')
        if self.search_order not in ('declared', 'pressure_risk', 'risk_first'):
            raise ValueError('Unknown lookahead search order')
        # Opt-in (experiments/improve, Phase 1): after a measured pressure risk, try positive alpha offsets
        # smallest first ('minimal_first') instead of largest first (V16 default 'aggressive_first'). Same
        # candidate set, same audits and witness requirement; only the least deviation that passes is kept.
        self.offset_order = options.get('lookahead_offset_order', 'aggressive_first')
        if self.offset_order not in ('aggressive_first', 'minimal_first'):
            raise ValueError('Unknown lookahead offset order')
        self.max_evaluations = int(options.get('recovery_search_max_evaluations', 12))
        if self.max_evaluations <= 0:
            raise ValueError('Recovery search evaluation limit must be positive')
        if self.horizon not in (12., 20., 30.) or self.budget <= 0 or not np.isfinite(self.budget):
            raise ValueError('Lookahead requires 12, 20 or 30 seconds and a positive budget')
        if not self.policies or set(self.policies)-set(POLICIES):
            raise ValueError('Unknown future candidate policy')
        self.baseline = PhysicalBaseline(plant.db, plant.mission, controller.control_contract)

    @staticmethod
    def qualify(plant, controller, options):
        """Check the offline qualification report: completed, same surrogate, same corrector settings."""
        report_path = options.get('lookahead_qualification_report')
        if not report_path:
            raise ValueError('Lookahead requires the corrector qualification report')
        path = plant.mission.path(report_path)
        report = json.loads(path.read_text(encoding='utf-8'))
        manifest = json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        if not report.get('P1_physical_gate_passed') or report.get('status') != 'COMPLETE':
            raise ValueError('Lookahead requires a completed corrector qualification')
        model_hash = controller.metadata.get('model_sha256')
        if not model_hash or model_hash not in manifest['source_sha256'].values():
            raise ValueError('Corrector qualification belongs to a different model')
        frozen = manifest['config']['fast_safety']
        for key in QUALIFIED_KEYS:
            if options.get(key) != frozen.get(key):
                raise ValueError('Qualified correction settings changed: '+key)
        return dict(report=str(path.relative_to(plant.mission.root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def forecast(self, context, alpha, policy, deadline):
        """Forecast from the current 3 s plan endpoint; never mutate caller state.

        The last horizon segment may be shorter than 3 s. Its own exact prefix
        is checked even if the additional (outside horizon) remainder fails.
        """
        from .environment import FlightEnvironment
        initial = audit_controls(self.plant, context, alpha)
        result = dict(policy=policy, horizon_s=min(self.horizon, 1200-context['t']-3), accepted_witness=False,
                      safe_duration_s=0., risk=[], stop_reason=None, min_margin=None, q_peak_pa=None, load_peak_g=None,
                      height_min_m=None, universal_infeasibility_proven=False)
        if self.options.get('lookahead_save_witness', False):
            result['control_sequence'] = []
        if not initial['valid']:
            result['stop_reason'] = 'CURRENT_3S_UNSAFE'
            return result
        horizon = result['horizon_s']
        if horizon <= 1e-9:
            result['accepted_witness'] = True
            return result
        if policy == 'equal_lift':
            env = FlightEnvironment(self.plant, self.baseline, self.base)
        else:
            ctrl = (LoadRecoveryBaseline(self.plant, self.controller.control_contract) if policy == 'load_recovery'
                    else self.controller)
            env = FlightEnvironment(self.plant, ctrl, self.base, dict(self.options, lookahead_enabled=False),
                                    post_corrector=self.corrector)
        env.x = initial['actual']['x'][-1].copy()
        env.alpha = float(alpha[-1])
        env.t = context['t']+3
        env.transition = Transition(context['source'], context['target'], context['t']-8*context['fraction'], 8.)
        env.transition.settle(env.t)
        env.previous_cl_raw = context.get('CL_raw', context['CL_des'])
        env.previous_cl_cmd = context['CL_des']
        env.previous_cl_actual = interval_moments(initial['actual'])['mean']
        env.history_valid = True
        begin = env.t
        all_samples = []
        for _ in range(int(math.ceil(horizon/3))):
            if time.perf_counter() >= deadline:
                result['stop_reason'] = 'LOOKAHEAD_WALL_BUDGET'
                break
            duration = min(3., horizon-(env.t-begin))
            old_alpha = env.alpha
            # The current absolute CL action is held over the whole horizon.
            raw = float(context.get('CL_raw', context['CL_des']))
            if raw not in env.cl_actions:
                raise ValueError('Lookahead CL is not one of the CL actions')
            rejected = None
            info = None
            old_budget = self.corrector.budget
            self.corrector.budget = min(old_budget, max(1e-9, deadline-time.perf_counter()))
            try:
                _, _, _, info = env.step(env.cl_actions.index(raw))
            except OracleRequired as exc:
                rejected = exc
                result['stop_reason'] = exc.reason
                result.update(failure_context=deepcopy(exc.context), failure_diagnostics=deepcopy(exc.diagnostics))
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                result['stop_reason'] = 'NUMERICAL_OR_DOMAIN_FAILURE'
                result['error'] = repr(exc)
                break
            finally:
                self.corrector.budget = old_budget
            # Candidate identity comes from the controller's explicit post output.
            # Reachability searches also integrate controls, and are never evidence
            # that the future policy produced those controls.
            if info is not None:
                plan_context = info['post_context']
                plan_alpha = info['plant']['alpha']
            elif (rejected is not None and rejected.reason in ('FAST_EXECUTION_AUDIT_FAILURE', 'FAST_PLANT_RK_STAGE_FAILURE')
                  and 'alpha_rad' in rejected.diagnostics):
                plan_context = rejected.context
                plan_alpha = rejected.diagnostics['alpha_rad']
            else:
                break
            valid = True
            segments = []
            for dt in (.1, .5):
                n = int(round(duration/.5))
                u = np.asarray(plan_alpha)[:n]
                r = self.plant.integrate(plan_context['x'], u, np.arange(n+1)*.5, plan_context['source'],
                                         plan_context['target'], plan_context['fraction'], plan_context['rate'], dt)
                lim = self.plant.mission.data['limits']
                lo = max(self.plant.db.A[0], lim['alpha_rad'][0])
                hi = min(self.plant.db.A[-1], lim['alpha_rad'][1])
                control_valid = (np.isfinite(u).all() and np.all(u >= lo-1e-9) and np.all(u <= hi+1e-9)
                                 and np.all(abs(np.diff(np.r_[old_alpha, u])) <= lim['alpha_rate_radps']*.5+1e-7))
                mm = self.plant.normalized_margins(r['samples'])
                valid = bool(valid and control_valid and np.isfinite(r['samples']).all() and np.min(mm) >= -1e-7)
                segments.append(r['samples'])
            if segments:
                all_samples.extend(segments)
            if not valid or (rejected is not None and duration >= 3-1e-9):
                result['stop_reason'] = result['stop_reason'] or 'FUTURE_DENSE_AUDIT_REJECTED'
                break
            if 'control_sequence' in result:
                result['control_sequence'].append(dict(context=deepcopy(plan_context), alpha_rad=u.tolist(),
                                                       duration_s=duration, end_state=env.x.tolist(),
                                                       audit_min_margin=float(np.min(mm))))
            result['safe_duration_s'] += duration
            if result['safe_duration_s'] >= horizon-1e-9:
                result['accepted_witness'] = True
                result['stop_reason'] = None
                break
        if all_samples:
            s = np.concatenate(all_samples, axis=1)
            mm = self.plant.normalized_margins(s)
            result.update(min_margin=float(np.min(mm)), q_peak_pa=float(np.max(s[4])), load_peak_g=float(np.max(s[6])),
                          height_min_m=float(np.min(s[0])), margins=dict(zip(MARGIN_NAMES, np.min(mm, axis=1).tolist())),
                          risk=[k for k, v in zip(MARGIN_NAMES, np.min(mm, axis=1)) if v < -1e-7])
        return result

    def assess(self, context, alpha):
        """Accept alpha (or an audited offset correction) only with a future witness."""
        started = time.perf_counter()
        deadline = started+self.budget
        diag = dict(horizon_s=self.horizon, budget_s=self.budget, oracle_calls=0, trials=[],
                    search_order=self.search_order, pressure_risk_reordered=False,
                    evaluations=0, max_evaluations=self.max_evaluations,
                    current_3s_safe=False, universal_infeasibility_proven=False,
                    scope='Audited future candidate policies, not all controls; current CL held')

        def finish(status, u=None):
            diag.update(status=status, wall_s=time.perf_counter()-started, budget_exceeded=time.perf_counter() > deadline)
            return CorrectionResult(u is not None, status, None if u is None else np.asarray(u).copy(), diag)

        current = audit_controls(self.plant, context, alpha)
        if not current['valid']:
            return finish('DENSE_AUDIT_REJECTED')
        diag['current_3s_safe'] = True
        pending = list(ALPHA_OFFSETS_DEG)
        diag['declared_offsets_deg'] = pending.copy()
        deferred = []
        risk_priority = False

        def evaluate_future(ctx, u, policy, trial):
            tick = time.perf_counter()
            future = self.forecast(ctx, u, policy, deadline)
            future['wall_s'] = time.perf_counter()-tick
            trial['forecasts'].append(future)
            diag['evaluations'] += 1
            return future

        while pending:
            offset = pending.pop(0)
            if time.perf_counter() >= deadline:
                break
            if offset == 0:
                u = np.asarray(alpha).copy()
                correction = None
            else:
                proposal = self.corrector.features.project(np.asarray(alpha)+np.deg2rad(offset), context['alpha_prev'])
                old_budget = self.corrector.budget
                self.corrector.budget = min(old_budget, max(1e-9, deadline-time.perf_counter()))
                try:
                    corrected = self.corrector.correct(context, proposal)
                finally:
                    self.corrector.budget = old_budget
                correction = corrected.diagnostics
                if not corrected.accepted:
                    diag['trials'].append(dict(offset_deg=offset, current_3s_safe=False, correction=correction,
                                               forecasts=[]))
                    continue
                u = corrected.alpha
            trial = dict(offset_deg=offset, current_3s_safe=True, alpha_rad=u.tolist(), correction=correction, forecasts=[])
            diag['trials'].append(trial)
            policies = list(self.policies)
            if risk_priority and self.search_order == 'risk_first':
                policies.sort(key=lambda p: {'load_recovery': 0, 'rolling_corrector': 1, 'equal_lift': 2}[p])
            for index, policy in enumerate(policies):
                if time.perf_counter() >= deadline or diag['evaluations'] >= self.max_evaluations:
                    break
                future = evaluate_future(context, u, policy, trial)
                if future['accepted_witness'] and time.perf_counter() <= deadline:
                    diag.update(selected_policy=policy, selected_offset_deg=offset, selected_alpha_rad=u.tolist(),
                                future_candidate_witness=True)
                    return finish('FUTURE_CANDIDATE_FOUND' if offset == 0 else 'EARLY_RECOVERY_FOUND', u)
                if (self.search_order == 'risk_first' and offset == 0. and context['x'][2] < 0.
                        and 'dynamic_pressure' in future.get('risk', [])):
                    deferred.extend((u.copy(), trial, p) for p in policies[index+1:])
                    risk_priority = True
                    break
            # Positive alpha can help arrest descent. This only prioritizes
            # existing candidates after a measured future pressure risk; every
            # current and future control still requires its own audits.
            if (self.search_order in ('pressure_risk', 'risk_first') and offset == 0. and context['x'][2] < 0.
                    and any('dynamic_pressure' in f.get('risk', []) for f in trial['forecasts'])):
                pending = (sorted((v for v in pending if v > 0.), reverse=self.offset_order == 'aggressive_first')
                           + [v for v in pending if v <= 0.])
                diag.update(pressure_risk_reordered=True, remaining_offsets_deg=pending.copy())
        for u, trial, policy in deferred:
            if time.perf_counter() >= deadline or diag['evaluations'] >= self.max_evaluations:
                break
            future = evaluate_future(context, u, policy, trial)
            if future['accepted_witness'] and time.perf_counter() <= deadline:
                diag.update(selected_policy=policy, selected_offset_deg=trial['offset_deg'], selected_alpha_rad=u.tolist(),
                            future_candidate_witness=True)
                return finish('FUTURE_CANDIDATE_FOUND', u)
        diag['future_candidate_witness'] = False
        completed = sum(not t['current_3s_safe'] or (len(t['forecasts']) == len(self.policies)
                        and not any(f.get('stop_reason') == 'LOOKAHEAD_WALL_BUDGET' for f in t['forecasts']))
                        for t in diag['trials'])
        diag.update(fully_evaluated_candidates=completed, all_candidates_evaluated=completed == len(diag['declared_offsets_deg']))
        if time.perf_counter() >= deadline or diag['evaluations'] >= self.max_evaluations and not diag['all_candidates_evaluated']:
            return finish('SEARCH_BUDGET_EXHAUSTED')
        if diag['all_candidates_evaluated']:
            return finish('ALL_CANDIDATES_EVALUATED_NO_WITNESS')
        return finish('RECOVERY_NOT_FOUND')
