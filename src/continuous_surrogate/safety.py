"""Independent continuous-alpha correction and finite-horizon safety witnesses.

Every plan is seven *nodes*, including the measured alpha at time zero.  The
same linear interpolation is used by P1, both audit grids, and the caller's
commit.  A search failure means UNKNOWN, never universal physical infeasibility.
The H30 witness is a bounded future policy search, not recursive feasibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time

import casadi as ca
import numpy as np
from scipy.optimize import brentq, minimize

from certified_flight import codegen
from certified_flight.diagnostics import MARGIN_NAMES, interval_moments
from certified_flight.transition import Transition
from continuous_tracking.plant import prefix, validate_plan
from .reachability import ContinuousCLReachability

OCP_TAIL_PROTOCOL = 'continuous-ocp-tail-witness/1'


@dataclass
class SafetyResult:
    accepted: bool
    status: str
    alpha_nodes: np.ndarray | None
    times: np.ndarray | None
    audit: dict | None
    arrays: dict | None
    diagnostics: dict

    @property
    def alpha(self):
        return self.alpha_nodes


class _Deadline(Exception):
    pass


class ContinuousSafety:
    """Continuous P1 plus independent current-3s and future-H30 audits."""

    def __init__(self, plant, options=None):
        self.plant, self.base, self.db = plant, plant.base, plant.db
        self.cfg = dict(options or {})
        self.limits = plant.mission.data['limits']
        self.lo = max(float(self.db.A[0]), self.limits['alpha_rad'][0])
        self.hi = min(float(self.db.A[-1]), self.limits['alpha_rad'][1])
        self.rate = float(self.limits['alpha_rate_radps'])
        self.duration = float(plant.mission.data['duration_s'])
        self.horizon = float(self.cfg.get('lookahead_horizon_s', 30.))
        self.budget = float(self.cfg.get('lookahead_max_wall_s', 30.))
        self.correction_budget = float(self.cfg.get('post_correction_max_wall_s', 2.))
        self.max_evaluations = int(self.cfg.get('recovery_search_max_evaluations', 12))
        self.reachability = ContinuousCLReachability(plant)
        quality = self.cfg.get('tracking_quality', {})
        self.quality_tolerance = float(quality.get('rmse_max', self.cfg.get('tracking_rmse_tolerance', .005)))
        self.mean_tolerance = float(quality.get('mean_abs_error_max', .0025))
        self.peak_tolerance = float(quality.get('peak_abs_error_max', .01))
        self.scale = float(self.cfg.get('alpha_scale_rad', .05))
        self.clscale = float(self.cfg.get('CL_scale', .1))
        weights = self.cfg.get('weights', {})
        self.wn = np.broadcast_to(np.asarray(weights.get('network', 1.), float), (6,)).copy()
        self.ws = float(weights.get('smooth', .01))
        self.wcl = float(weights.get('CL', .05))
        vals = np.r_[self.horizon, self.budget, self.correction_budget,
                     self.quality_tolerance, self.scale, self.clscale, self.wn, self.ws, self.wcl,
                     self.mean_tolerance, self.peak_tolerance]
        if not np.isfinite(vals).all() or min(vals[:6]) <= 0 or min(vals[6:]) < 0:
            raise ValueError('Invalid continuous safety configuration')
        if self.horizon != 30. or self.max_evaluations <= 0:
            raise ValueError('This experimental protocol requires H30 and a positive search limit')
        # Compile before any certification deadline.  All flights share this
        # exact RHS; no fast-math or surrogate is involved in the audit.
        self.rk_numeric = codegen.compile_function(plant.rk)
        self.p1_function = self._build_p1()
        nominal = plant.mission.data['initial_state']
        g = self.db.index['G42']
        for dt in (.1, .5):
            plant.integrate(nominal['x_h_V_gamma_s'], np.full(7, nominal['alpha_rad']),
                            np.arange(7)*.5, g, g, 1., 0., dt)

    def project(self, context, nodes):
        """Decode a proposed node plan before auditing, never after certification."""
        a = np.asarray(nodes, float).copy()
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('Continuous post output must have seven finite nodes')
        a[0] = float(context['alpha_prev'])
        if not self.lo <= a[0] <= self.hi:
            raise ValueError('Measured alpha is outside the aerodynamic/physical bounds')
        for i in range(1, 7):
            a[i] = np.clip(a[i], max(self.lo, a[i-1]-.5*self.rate), min(self.hi, a[i-1]+.5*self.rate))
        return a

    def _step(self, x, a0, a1, context, tau):
        xx, _ = self.rk_numeric(x, a0, (a1-a0)/.5, context['fraction'], context['rate'],
                                self.base._dm(context['source']), self.base._dm(context['target']), .5, tau)
        return np.asarray(xx).ravel()

    def _angle(self, x, context, fraction, policy):
        """Invert the actual pair at one predicted state, with no extrapolation."""
        target = float(context['CL_des'])
        row = 4
        if policy == 'load_recovery':
            target = float(np.clip(1.-x[1]*x[2]/(9.80665*12.), .5, 2.7))
            row = 2
        elif policy != 'equal_lift':
            raise ValueError('Unknown continuous physical policy')
        def residual(alpha):
            return float(self.base.quantities(x, alpha, context['source'], context['target'], fraction)[row])-target
        low, high = residual(self.lo), residual(self.hi)
        info = dict(quantity='CL' if policy == 'equal_lift' else 'load_factor_g', request=target,
                    available_range=[low+target, high+target],
                    within_endpoint_range=bool(low <= 1e-12 and high >= -1e-12),
                    physical_infeasibility_proven=False,
                    meaning='Fixed-state alpha inversion seed, not a dynamic reachability certificate')
        # A bound is only a candidate when the request lies outside this pair's
        # CL range. Its safety and tracking error still require explicit audits.
        if low >= 0:
            return self.lo, info
        if high <= 0:
            return self.hi, info
        return float(brentq(residual, self.lo, self.hi, xtol=1e-12, rtol=1e-12)), info

    def candidate(self, context, policy='equal_lift', duration=3., offset_deg=0., return_diagnostics=False):
        """Feedback seed; update Mach/state and morph fraction at every .5s node."""
        n = int(round(duration/.5))
        if n < 1 or abs(n*.5-duration) > 1e-9:
            raise ValueError('Candidate duration must be a positive half-second multiple')
        x = np.asarray(context['x'], float).copy()
        nodes = [float(context['alpha_prev'])]
        details = dict(policy=policy, offset_deg=float(offset_deg), nodes=[],
                       universal_infeasibility_proven=False)
        if not self.lo <= nodes[0] <= self.hi:
            raise ValueError('Initial alpha outside physical/database bounds')
        for i in range(n):
            tau = i*.5
            fraction = min(1., float(context['fraction'])+float(context['rate'])*(tau+.5))
            guess = self._step(x, nodes[-1], nodes[-1], context, tau)
            for _ in range(2):
                desired, inverse = self._angle(guess, context, fraction, policy)
                desired += np.deg2rad(offset_deg)
                target = float(np.clip(desired, max(self.lo, nodes[-1]-.5*self.rate),
                                       min(self.hi, nodes[-1]+.5*self.rate)))
                guess = self._step(x, nodes[-1], target, context, tau)
            details['nodes'].append(dict(time_s=tau+.5, fraction=fraction, inversion=inverse,
                                         alpha_requested_rad=desired, alpha_node_rad=target,
                                         rate_or_angle_limited=bool(abs(target-desired) > 1e-12)))
            nodes.append(target)
            x = guess
        details.update(endpoint_support_saturated=any(not v['inversion']['within_endpoint_range'] for v in details['nodes']),
                       rate_or_angle_limited=any(v['rate_or_angle_limited'] for v in details['nodes']))
        result = np.asarray(nodes)
        return (result, details) if return_diagnostics else result

    def _quality(self, audit):
        if not audit or not audit.get('grids'):
            return dict(passed=False, rmse=None, tolerance=self.quality_tolerance)
        rmse = max(v['CL_rmse_first3'] for v in audit['grids'].values())
        mean = max(abs(v['CL_mean_first3']-audit['CL_target']) for v in audit['grids'].values())
        peak = max(v['CL_peak_error_first3'] for v in audit['grids'].values())
        passed = bool(rmse <= self.quality_tolerance and mean <= self.mean_tolerance and peak <= self.peak_tolerance)
        return dict(passed=passed, status='TRACKING_QUALIFIED' if passed else 'TRACKING_DEVIATION',
                    rmse=float(rmse), mean_abs_error=float(mean), peak_abs_error=float(peak),
                    tolerance=self.quality_tolerance,
                    thresholds=dict(rmse_max=self.quality_tolerance, mean_abs_error_max=self.mean_tolerance,
                                    peak_abs_error_max=self.peak_tolerance),
                    rate_limit_active=bool(audit['maximum_alpha_rate_degps'] >= np.rad2deg(self.rate)-1e-7),
                    meaning='Tracking diagnostic, not a hard safety constraint; rate flag does not establish causality')

    def audit(self, context, nodes, times=None):
        times = np.arange(len(nodes))*.5 if times is None else np.asarray(times, float)
        audit, arrays = self.plant.audit(context, nodes, times, float(context['CL_des']))
        audit['CL_target'] = float(context['CL_des'])
        return audit, arrays

    def _build_p1(self):
        """Six free future nodes, exact continuous RK constraints and derivatives."""
        nc = self.db.ncoef
        z = ca.MX.sym('future_nodes_scaled', 6)
        p = ca.MX.sym('continuous_context', 14+2*nc)
        nodes = ca.vertcat(p[4], z*self.scale)
        x, cl, ref, fraction, rate = p[:4], p[5], p[6:12], p[12], p[13]
        c0, c1 = p[14:14+nc], p[14+nc:]
        constraints = []
        track = 0.
        for i in range(6):
            x, ss = self.plant.rk(x, nodes[i], (nodes[i+1]-nodes[i])/.5,
                                  fraction, rate, c0, c1, .5, i*.5)
            for axis, key, scale in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
                constraints.extend([(ss[axis, :]-self.limits[key][0])/scale,
                                    (self.limits[key][1]-ss[axis, :])/scale])
            constraints.extend([1-ss[4, :]/self.limits['dynamic_pressure_pa_max'],
                                1-ss[5, :]/self.limits['heat_proxy_wm2_max'],
                                1-ss[6, :]/self.limits['load_factor_g_max'],
                                ss[7, :]-self.db.M[0], self.db.M[-1]-ss[7, :]])
            error = (ss[8, :4]-cl)/self.clscale
            track += .5*(error[0]**2+2*error[1]**2+2*error[2]**2+error[3]**2)/6
        slew = (nodes[1:]-nodes[:-1])/(self.rate*.5)
        constraints.extend([1-slew, 1+slew])
        g = ca.vertcat(*[ca.vec(v) for v in constraints])
        objective = (ca.dot(ca.DM(self.wn), ((nodes[1:]-ref)/self.scale)**2)
                     + self.ws*ca.sumsqr((nodes[1:]-nodes[:-1])/self.scale)
                     + self.wcl*track/3.)
        fn = ca.Function('continuous_post_correction', [z, p],
                         [objective, ca.gradient(objective, z), g, ca.jacobian(g, z)])
        return codegen.compile_function(fn)

    def correct(self, context, nodes):
        """P1: explicit tracking gate, bounded constrained optimization, audit."""
        started = time.perf_counter()
        deadline = started+self.correction_budget
        ref = np.asarray(nodes, float)
        diag = dict(method='continuous_six_future_nodes_slsqp', budget_s=self.correction_budget,
                    tracking_gate_is_physical_constraint=False, attempts=[], oracle_calls=0,
                    universal_infeasibility_proven=False)
        best = None

        def finish(status):
            diag.update(status=status, wall_s=time.perf_counter()-started,
                        budget_exceeded=time.perf_counter() > deadline)
            if best is None:
                return SafetyResult(False, status, None, None, None, None, diag)
            _, a, audit, arrays, source = best
            diag.update(selected_source=source, quality=self._quality(audit), final_audit=audit,
                        correction_max_deg=float(np.rad2deg(np.max(abs(a-ref)))))
            return SafetyResult(True, status, a.copy(), np.arange(7)*.5, audit, arrays, diag)

        def consider(a, source):
            nonlocal best
            a = np.asarray(a, float)
            audit, arrays = self.audit(context, a)
            q = self._quality(audit)
            diag['attempts'].append(dict(source=source, audit=audit, quality=q, alpha_nodes_rad=a.tolist()))
            if audit['valid']:
                score = float(np.sum(self.wn*((a[1:]-ref[1:])/self.scale)**2)
                              + self.ws*np.sum((np.diff(a)/self.scale)**2)
                              + self.wcl*(q['rmse']/self.clscale)**2)
                # Prefer a tracking-qualified plan if one exists. If none does,
                # keep the least tracking error among physically safe plans.
                key = (not q['passed'], score if q['passed'] else q['rmse'])
                if best is None or key < best[0]:
                    best = (key, a.copy(), audit, arrays, source)
            return audit, q

        try:
            if ref.shape != (7,) or not np.isfinite(ref).all() or ref[0] != context['alpha_prev']:
                raise ValueError('Malformed continuous proposal or discontinuous first node')
            original, q = consider(ref, context.get('proposal_origin', 'SURROGATE'))
            if original['valid'] and q['passed']:
                return finish('NETWORK_PLAN_ALREADY_SAFE_AND_TRACKING_QUALIFIED')
            if time.perf_counter() >= deadline:
                return finish('SAFE_WITH_TRACKING_DEVIATION' if best else 'CORRECTION_BUDGET_EXHAUSTED')
            baseline, baseline_details = self.candidate(context, return_diagnostics=True)
            diag['baseline_candidate'] = baseline_details
            consider(baseline, 'CONTINUOUS_EQUAL_LIFT')
            p = np.r_[context['x'], context['alpha_prev'], context['CL_des'], ref[1:],
                      context['fraction'], context['rate'], self.db.parameters[context['source']],
                      self.db.parameters[context['target']]]
            cache = {}

            def evaluate(z):
                if time.perf_counter() >= deadline:
                    raise _Deadline()
                if 'z' not in cache or not np.array_equal(cache['z'], z):
                    out = self.p1_function(z, p)
                    cache.update(z=np.asarray(z).copy(), f=float(out[0]), df=np.asarray(out[1]).ravel(),
                                 g=np.asarray(out[2]).ravel(), dg=np.asarray(out[3]))
                return cache

            initial = (best[1] if best is not None else self.project(context, ref))[1:]/self.scale
            res = minimize(lambda z: evaluate(z)['f'], initial, jac=lambda z: evaluate(z)['df'],
                           method='SLSQP', bounds=[(self.lo/self.scale, self.hi/self.scale)]*6,
                           constraints=[dict(type='ineq', fun=lambda z: evaluate(z)['g'],
                                             jac=lambda z: evaluate(z)['dg'])],
                           options=dict(maxiter=int(self.cfg.get('max_iterations', 60)),
                                        ftol=float(self.cfg.get('ftol', 1e-10)), disp=False))
            diag['optimizer'] = dict(success=bool(res.success), status=int(res.status),
                                     message=str(res.message), iterations=int(res.nit))
            # Solver success is not a certificate; the candidate is independently
            # checked even after an optimizer iteration limit.
            consider(np.r_[context['alpha_prev'], res.x*self.scale], 'CONTINUOUS_SLSQP')
        except _Deadline:
            diag['optimizer_deadline'] = True
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, KeyError) as exc:
            diag['error'] = repr(exc)
        if best is None:
            return finish('CORRECTION_NO_CERTIFIED_WITNESS')
        return finish('SAFE_CORRECTION_FOUND' if self._quality(best[2])['passed'] else 'SAFE_WITH_TRACKING_DEVIATION')

    def forecast(self, context, nodes, arrays, policy, deadline):
        """Independent actual and dense paths, with the raw CL held in future."""
        horizon = max(0., min(self.horizon, self.duration-float(context['t'])-3.))
        result = dict(policy=policy, horizon_s=horizon, accepted_witness=False,
                      safe_duration_s=0., stop_reason=None, risk=[], control_sequence=[],
                      universal_infeasibility_proven=False)
        if horizon <= 1e-9:
            result['accepted_witness'] = True
            return result
        states = {dt: arrays[dt]['x'][-1].copy() for dt in ('0.1', '0.5')}
        current_t = float(context['t'])+3.
        alpha = float(nodes[-1])
        z = Transition(context['source'], context['target'],
                       float(context['t'])-8.*float(context['fraction']), 8.)
        z.settle(current_t)
        samples = []
        while result['safe_duration_s'] < horizon-1e-9:
            if time.perf_counter() >= deadline:
                result['stop_reason'] = 'LOOKAHEAD_WALL_BUDGET'
                break
            duration = min(3., horizon-result['safe_duration_s'])
            raw = float(context.get('CL_raw', context['CL_des']))
            reachable = self.reachability.check(states['0.5'], alpha, z, current_t, raw,
                                                enabled=self.cfg.get('reachability_enabled', True))
            cl = float(reachable['CL_cmd'])
            c = dict(context, x=states['0.5'].tolist(), alpha_prev=alpha, t=current_t,
                     source=z.source, target=z.target, fraction=z.fraction(current_t),
                     rate=.125 if z.active(current_t) else 0., CL_des=cl, CL_raw=raw)
            try:
                # Original Pre definition: Mach at the midpoint of the current
                # and predicted 3s endpoint states, never a design Mach.
                pre = self.candidate(c, 'equal_lift', duration)
                pre_times = np.arange(len(pre))*.5
                preview = self.plant.integrate(c['x'], pre, pre_times, c['source'], c['target'],
                                               c['fraction'], c['rate'], .5)
                midpoint = (np.asarray(c['x'])+preview['x'][-1])/2
                mach = float(midpoint[1]/self.plant.atmosphere.speed_of_sound(midpoint[0]))
                choice = self.db.generate(mach, cl, current=z.target)
                if choice['geometry_index'] is not None:
                    z.command(choice['geometry_index'], current_t)
                c.update(source=z.source, target=z.target, fraction=z.fraction(current_t),
                         rate=.125 if z.active(current_t) else 0.)
                candidate, candidate_details = self.candidate(c, policy, duration, return_diagnostics=True)
                times = np.arange(len(candidate))*.5
                proof, future = self.audit(c, candidate, times)
                # The .1s trajectory is continued from its own previous state,
                # rather than being reset to the .5s path at every decision.
                dense = self.plant.integrate(states['0.1'], candidate, times, c['source'], c['target'],
                                             c['fraction'], c['rate'], .1)
                future['0.1'] = dense
                good = proof['control_valid'] and future['0.5']['valid'] and dense['valid']
                for dt in ('0.1', '0.5'):
                    samples.append(future[dt]['samples'])
                if not good:
                    result['stop_reason'] = 'FUTURE_DENSE_AUDIT_REJECTED'
                    result['rejected_context'] = c
                    break
                result['control_sequence'].append(dict(context=deepcopy(c), alpha_nodes_rad=candidate.tolist(),
                    times_s=times.tolist(), planned_mach=mach, generator=choice,
                    candidate_diagnostics=candidate_details,
                    CL_reachability=reachable,
                    end_states={dt: future[dt]['x'][-1].tolist() for dt in future},
                    min_margins={dt: future[dt]['min_margin'] for dt in future}))
                states = {dt: future[dt]['x'][-1].copy() for dt in future}
                alpha = float(candidate[-1])
                current_t += duration
                z.settle(current_t)
                result['safe_duration_s'] += duration
            except (ValueError, RuntimeError, FloatingPointError, OverflowError, KeyError) as exc:
                result.update(stop_reason='NUMERICAL_OR_DOMAIN_FAILURE', error=repr(exc))
                break
        result['accepted_witness'] = bool(result['safe_duration_s'] >= horizon-1e-9
                                          and time.perf_counter() <= deadline)
        if samples:
            s = np.concatenate(samples, axis=1)
            mm = self.plant.normalized_margins(s)
            result.update(min_margin=float(np.min(mm)), q_peak_pa=float(np.max(s[4])),
                          load_peak_g=float(np.max(s[6])), heat_peak=float(np.max(s[5])),
                          margins=dict(zip(MARGIN_NAMES, np.min(mm, axis=1).tolist())),
                          risk=[k for k, v in zip(MARGIN_NAMES, np.min(mm, axis=1)) if v < -1e-7])
        return result

    def assess(self, context, nodes, times=None):
        """P1 and bounded H30 candidate search; only audited nodes can escape."""
        started = time.perf_counter()
        times = np.arange(7)*.5 if times is None else np.asarray(times, float)
        if not np.array_equal(times, np.arange(7)*.5):
            raise ValueError('Current execution must be seven nodes over exactly three seconds')
        corrected = self.correct(context, nodes)
        diag = dict(correction=corrected.diagnostics, trials=[], horizon_s=self.horizon,
                    budget_s=self.budget, max_evaluations=self.max_evaluations, evaluations=0,
                    oracle_calls=0, universal_infeasibility_proven=False,
                    scope='Audited current 3s plus one future H30 policy witness; no recursive guarantee')
        if not corrected.accepted:
            diag.update(status=corrected.status, wall_s=time.perf_counter()-started, proof_status='UNKNOWN_NOT_CERTIFIED')
            return SafetyResult(False, corrected.status, None, None, None, None, diag)
        deadline = time.perf_counter()+self.budget
        candidates = [('CORRECTED_PROPOSAL', None, 0.), ('EQUAL_LIFT', 'equal_lift', 0.),
                      ('LOAD_RECOVERY', 'load_recovery', 0.)]
        candidates.extend((f'EQUAL_LIFT_OFFSET_{v:+g}', 'equal_lift', v) for v in (.5, 1., 2., -.5, -1.))
        seen = set()
        for name, policy, offset in candidates:
            if time.perf_counter() >= deadline or diag['evaluations'] >= self.max_evaluations:
                break
            trial = dict(source=name, forecasts=[])
            diag['trials'].append(trial)
            try:
                if policy is None:
                    a = corrected.alpha_nodes.copy()
                else:
                    a, trial['candidate_diagnostics'] = self.candidate(context, policy, 3., offset, return_diagnostics=True)
                identity = a.tobytes()
                if identity in seen:
                    trial['duplicate'] = True
                    continue
                seen.add(identity)
                proof, arrays = self.audit(context, a)
                trial.update(current_audit=proof, alpha_nodes_rad=a.tolist(), quality=self._quality(proof))
                if not proof['valid']:
                    trial['status'] = 'CURRENT_CANDIDATE_REJECTED'
                    continue
                for future_policy in ('equal_lift', 'load_recovery'):
                    if time.perf_counter() >= deadline or diag['evaluations'] >= self.max_evaluations:
                        break
                    tick = time.perf_counter()
                    witness = self.forecast(context, a, arrays, future_policy, deadline)
                    witness['wall_s'] = time.perf_counter()-tick
                    trial['forecasts'].append(witness)
                    diag['evaluations'] += 1
                    if witness['accepted_witness'] and time.perf_counter() <= deadline:
                        diag.update(status='FUTURE_CANDIDATE_FOUND', selected_source=name,
                                    selected_policy=future_policy, quality=self._quality(proof),
                                    selected_alpha_nodes_rad=a.tolist(), proof_status='VERIFIED_PASS',
                                    wall_s=time.perf_counter()-started,
                                    safety_intervention=bool(not np.array_equal(a, np.asarray(nodes))))
                        return SafetyResult(True, 'FUTURE_CANDIDATE_FOUND', a.copy(), times.copy(), proof, arrays, diag)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError, KeyError) as exc:
                trial.update(status='NUMERICAL_OR_DOMAIN_FAILURE', error=repr(exc))
        status = ('SEARCH_BUDGET_EXHAUSTED' if time.perf_counter() >= deadline
                  or diag['evaluations'] >= self.max_evaluations else 'ALL_CANDIDATES_EVALUATED_NO_WITNESS')
        diag.update(status=status, proof_status='UNKNOWN_NOT_CERTIFIED',
                    wall_s=time.perf_counter()-started, future_candidate_witness=False)
        return SafetyResult(False, status, None, None, None, None, diag)

    def assess_ocp_plan(self, context, plan):
        """Certify an unchanged real-OCP prefix using its independently audited tail.

        P1 is deliberately not called: changing the prefix would invalidate the
        supplied tail.  At each future decision, the static OCP geometry pair
        must still agree with the original Generator and transition lock.  A
        mismatch is a witness-policy incompatibility, not physical infeasibility.
        """
        started = time.perf_counter()
        deadline = started+self.budget
        diag = dict(witness_protocol=OCP_TAIL_PROTOCOL, budget_s=self.budget, horizon_s=self.horizon,
                    oracle_calls=0, source='FRESH_CONTINUOUS_REAL_OCP', p1_applied=False,
                    universal_infeasibility_proven=False, generator_checks=[],
                    scope='Unmodified audited OCP prefix and Generator-compatible finite H30 tail; no recursive guarantee')

        def finish(status, current_nodes=None, audit=None, arrays=None):
            accepted = current_nodes is not None
            diag.update(status=status, proof_status='VERIFIED_PASS' if accepted else 'UNKNOWN_NOT_CERTIFIED',
                        wall_s=time.perf_counter()-started, budget_exceeded=time.perf_counter() > deadline,
                        future_candidate_witness=accepted)
            if accepted:
                diag.update(quality=self._quality(audit), safety_intervention=False,
                            selected_source='UNCHANGED_REAL_OCP_PREFIX', selected_policy='EXACT_OCP_TAIL',
                            selected_alpha_nodes_rad=current_nodes.tolist())
            return SafetyResult(accepted, status, None if current_nodes is None else current_nodes.copy(),
                                None if current_nodes is None else np.arange(7)*.5, audit, arrays, diag)

        def expired():
            return time.perf_counter() > deadline

        try:
            if not bool(plan.success):
                diag['solver_status'] = str(getattr(plan, 'status', 'UNSUCCESSFUL'))
                return finish('OCP_SOLVE_NOT_SUCCESSFUL')
            nodes, times = validate_plan(plan.alpha_nodes, plan.times)
            remaining = self.duration-float(context['t'])
            expected = min(60., remaining)
            if remaining < 3.-1e-9 or abs(times[-1]-expected) > 1e-9:
                diag.update(remaining_s=remaining, expected_plan_horizon_s=expected,
                            supplied_plan_horizon_s=float(times[-1]))
                return finish('OCP_PLAN_HORIZON_MISMATCH')
            current_nodes, current_times = prefix(nodes, times, 3.)
            if not np.array_equal(current_times, np.arange(7)*.5):
                return finish('OCP_PLAN_UNSUPPORTED_EXECUTION_GRID')
            if current_nodes[0] != context['alpha_prev']:
                return finish('OCP_PLAN_DISCONTINUOUS_ENTRY')
            # Store and hash the actual controls and physical context, not an
            # optimizer's stored state trajectory or a success flag.
            bound_context = {k: deepcopy(context[k]) for k in
                             ('x', 'alpha_prev', 't', 'source', 'target', 'fraction', 'rate', 'CL_des')}
            bound_context['CL_raw'] = float(context.get('CL_raw', context['CL_des']))
            bound_context['x'] = np.asarray(bound_context['x'], float).tolist()
            for key in ('source', 'target'):
                bound_context[key] = int(bound_context[key])
            for key in ('alpha_prev', 't', 'fraction', 'rate', 'CL_des'):
                bound_context[key] = float(bound_context[key])
            binding = dict(protocol=OCP_TAIL_PROTOCOL, context=bound_context,
                           alpha_nodes_rad=nodes.tolist(), times_s=times.tolist())
            encoded = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
            diag.update(plan_binding=binding, plan_sha256=hashlib.sha256(encoded).hexdigest(),
                        solver_status=str(getattr(plan, 'status', 'SUCCESS')))

            full_audit, _ = self.audit(context, nodes, times)
            diag['full_plan_audit'] = full_audit
            if not full_audit['valid']:
                return finish('OCP_FULL_PLAN_AUDIT_REJECTED')
            if expired():
                return finish('SEARCH_BUDGET_EXHAUSTED')
            current_audit, current_arrays = self.audit(context, current_nodes, current_times)
            diag['current_3s_audit'] = current_audit
            if not current_audit['valid']:
                return finish('OCP_PREFIX_AUDIT_REJECTED')
            if expired():
                return finish('SEARCH_BUDGET_EXHAUSTED')
            witness_duration = min(3.+self.horizon, remaining)
            witness_nodes, witness_times = prefix(nodes, times, witness_duration)
            witness_audit, witness_arrays = self.audit(context, witness_nodes, witness_times)
            diag.update(witness_duration_s=witness_duration,
                        future_horizon_s=max(0., witness_duration-3.),
                        full_witness_audit=witness_audit,
                        witness_alpha_nodes_rad=witness_nodes.tolist(), witness_times_s=witness_times.tolist())
            if not witness_audit['valid']:
                return finish('OCP_TAIL_AUDIT_REJECTED')
            if expired():
                return finish('SEARCH_BUDGET_EXHAUSTED')

            # Compatibility is evaluated along the actual .5s trajectory, as
            # future decisions would be. The independent .1s path has already
            # audited the same entire witness without resets at decision times.
            actual = witness_arrays['0.5']
            z = Transition(int(context['source']), int(context['target']),
                           float(context['t'])-8.*float(context['fraction']), 8.)
            raw = float(context.get('CL_raw', context['CL_des']))
            for tau in np.arange(3., witness_duration-1e-9, 3.):
                if expired():
                    return finish('SEARCH_BUDGET_EXHAUSTED')
                t = float(context['t'])+float(tau)
                z.settle(t)
                idx = int(round(tau/.5))-1
                x = np.asarray(actual['x'][idx], float)
                alpha = float(np.interp(tau, times, nodes))
                reachable = self.reachability.check(x, alpha, z, t, raw,
                                                    enabled=self.cfg.get('reachability_enabled', True))
                cl = float(reachable['CL_cmd'])
                stop = min(tau+3., witness_duration)
                endpoint = np.asarray(actual['x'][int(round(stop/.5))-1], float)
                midpoint = (x+endpoint)/2
                mach = float(midpoint[1]/self.plant.atmosphere.speed_of_sound(midpoint[0]))
                choice = self.db.generate(mach, cl, current=z.target)
                geometry = choice['geometry_index']
                locked = z.active(t)
                compatible = bool(locked or geometry is None or int(geometry) == z.target)
                diag['generator_checks'].append(dict(time_s=t, relative_time_s=float(tau),
                    context=dict(x=x.tolist(), alpha_prev=alpha, source=z.source, target=z.target,
                                 fraction=z.fraction(t), rate=.125 if locked else 0., CL_raw=raw, CL_des=cl),
                    planned_mach=mach, pre_mach_definition='mach_of_mean_current_and_exact_plan_3s_state',
                    generator=choice, CL_reachability=reachable, transition_locked=locked,
                    compatible=compatible))
                if not compatible:
                    diag.update(physical_plan_audit_passed=True, generator_compatible=False,
                                rejection_meaning='Audited static-pair tail differs from future greedy Generator; not physical infeasibility')
                    return finish('OCP_TAIL_GENERATOR_INCOMPATIBLE')
            if expired():
                return finish('SEARCH_BUDGET_EXHAUSTED')
            diag.update(generator_compatible=True, physical_plan_audit_passed=True,
                        accepted_witness=True, safe_duration_s=max(0., witness_duration-3.))
            return finish('OCP_TAIL_WITNESS_FOUND', current_nodes, current_audit, current_arrays)
        except (AttributeError, ValueError, RuntimeError, FloatingPointError, OverflowError, KeyError, IndexError) as exc:
            diag['error'] = repr(exc)
            return finish('OCP_PLAN_NUMERICAL_OR_DOMAIN_FAILURE')
