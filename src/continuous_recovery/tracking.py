"""Tracking-priority continuous P1, isolated from the previous experiment.

The current alpha is fixed.  CL tracking uses pointwise RK quadrature and is
the primary objective; the old nominal/smoothing cost is secondary.  Tracking
quality thresholds are diagnostics only and never choose an execution mode.
All candidates require the unchanged independent .1/.5 s physical audits.
This module supplies a current-segment witness, not an H30 safety certificate.
"""
from __future__ import annotations

import time

import casadi as ca
import numpy as np
from scipy.optimize import minimize

from certified_flight import codegen
from continuous_surrogate.safety import ContinuousSafety, SafetyResult


class _Deadline(Exception):
    pass


class TrackingPrioritySafety(ContinuousSafety):
    """Wrap an initialized guard, preserving its physical constraints/budgets."""

    relative_allowance = .01
    absolute_allowance = 1e-8
    near_zero_primary = 1e-8

    def __init__(self, existing_safety):
        self.__dict__.update(existing_safety.__dict__)
        # Compilation occurs before a certification call and is separately
        # timed by the experiment. It must not consume hidden runtime budget.
        self.priority_function = self._build_priority_function()

    def _build_priority_function(self):
        nc = self.db.ncoef
        z = ca.MX.sym('recovery_future_nodes_scaled', 6)
        p = ca.MX.sym('recovery_context', 14+2*nc)
        a = ca.vertcat(p[4], z*self.scale)
        x, cl, ref = p[:4], p[5], p[6:12]
        fraction, rate = p[12], p[13]
        c0, c1 = p[14:14+nc], p[14+nc:]
        constraints, primary = [], 0.
        for i in range(6):
            x, ss = self.plant.rk(x, a[i], (a[i+1]-a[i])/.5,
                                  fraction, rate, c0, c1, .5, i*.5)
            for axis, key, scale in ((0, 'altitude_m', 1e4),
                                     (1, 'velocity_mps', 1e3),
                                     (2, 'gamma_rad', .1)):
                constraints.extend([(ss[axis, :]-self.limits[key][0])/scale,
                                    (self.limits[key][1]-ss[axis, :])/scale])
            constraints.extend([1-ss[4, :]/self.limits['dynamic_pressure_pa_max'],
                                1-ss[5, :]/self.limits['heat_proxy_wm2_max'],
                                1-ss[6, :]/self.limits['load_factor_g_max'],
                                ss[7, :]-self.db.M[0], self.db.M[-1]-ss[7, :]])
            error = (ss[8, :4]-cl)/self.clscale
            primary += .5*(error[0]**2+2*error[1]**2+2*error[2]**2+error[3]**2)/18.
        slew = (a[1:]-a[:-1])/(self.rate*.5)
        constraints.extend([1-slew, 1+slew])
        g = ca.vertcat(*[ca.vec(v) for v in constraints])
        # Preserve the previous cost as the secondary preference. The tracking
        # term here cannot trade away the separately enforced primary budget.
        secondary = (ca.dot(ca.DM(self.wn), ((a[1:]-ref)/self.scale)**2)
                     + self.ws*ca.sumsqr((a[1:]-a[:-1])/self.scale)
                     + self.wcl*primary)
        fn = ca.Function('recovery_tracking_priority', [z, p],
                         [primary, ca.gradient(primary, z), secondary,
                          ca.gradient(secondary, z), g, ca.jacobian(g, z)])
        return codegen.compile_function(fn)

    def correct(self, context, nodes):
        started = time.perf_counter()
        deadline = started+self.correction_budget
        diag = dict(method='continuous_tracking_priority',
                    budget_s=self.correction_budget, attempts=[], optimizer=[],
                    relative_allowance=self.relative_allowance,
                    absolute_allowance=self.absolute_allowance,
                    near_zero_primary=self.near_zero_primary,
                    quality_gate_used_for_selection=False,
                    universal_infeasibility_proven=False,
                    global_optimality_claimed=False, oracle_calls=0,
                    function_evaluations=0, cache_hits=0)

        def check_time():
            if time.perf_counter() >= deadline:
                raise _Deadline()

        def unknown(status):
            diag.update(status=status, proof_status='UNKNOWN_NOT_CERTIFIED',
                        wall_s=time.perf_counter()-started,
                        budget_exceeded=time.perf_counter() >= deadline)
            return SafetyResult(False, status, None, None, None, None, diag)

        try:
            check_time()
            ref = np.asarray(nodes, dtype=float).copy()
            if (ref.shape != (7,) or not np.isfinite(ref).all()
                    or ref[0] != float(context['alpha_prev'])):
                raise ValueError('Proposal must have seven finite nodes and exact measured alpha at node zero')
            if not self.lo <= ref[0] <= self.hi:
                raise ValueError('Measured alpha outside physical/database bounds')
            p = np.r_[context['x'], context['alpha_prev'], context['CL_des'], ref[1:],
                      context['fraction'], context['rate'],
                      self.db.parameters[context['source']], self.db.parameters[context['target']]]
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite physical context')
            cache = {}
            candidates = []

            def evaluate(z):
                check_time()
                z = np.asarray(z, float)
                if z.shape != (6,) or not np.isfinite(z).all():
                    raise ValueError('Invalid optimization variables')
                if 'z' in cache and np.array_equal(cache['z'], z):
                    diag['cache_hits'] += 1
                    return cache
                out = self.priority_function(z, p)
                check_time()
                values = dict(j=float(out[0]), dj=np.asarray(out[1]).ravel(),
                              f=float(out[2]), df=np.asarray(out[3]).ravel(),
                              g=np.asarray(out[4]).ravel(), dg=np.asarray(out[5]))
                if (not all(np.isfinite(v).all() for v in values.values())
                        or values['j'] < -1e-14):
                    raise FloatingPointError('Nonfinite objective/derivative or negative squared tracking error')
                values['j'] = max(0., values['j'])
                cache.clear()
                cache.update(z=z.copy(), **values)
                diag['function_evaluations'] += 1
                return cache

            def consider(a, source):
                check_time()
                a = np.asarray(a, float)
                if a.shape != (7,) or not np.isfinite(a).all() or a[0] != ref[0]:
                    raise ValueError('Candidate changed measured alpha or contains nonfinite nodes')
                value = evaluate(a[1:]/self.scale)
                primary, secondary = value['j'], value['f']
                audit, arrays = self.audit(context, a)
                check_time()  # A late audit cannot turn into a PASS.
                diag['attempts'].append(dict(source=source, tracking_objective=primary,
                    secondary_objective=secondary, audit=audit,
                    quality=self._quality(audit), alpha_nodes_rad=a.tolist()))
                if audit['valid']:
                    candidate = dict(j=primary, f=secondary, nodes=a.copy(), audit=audit,
                                     arrays=arrays, source=source)
                    candidates.append(candidate)
                    return candidate
                return None

            baseline, details = self.candidate(context, return_diagnostics=True)
            check_time()
            diag['baseline_candidate'] = details
            base = consider(baseline, 'CONTINUOUS_EQUAL_LIFT')
            proposal = consider(ref, context.get('proposal_origin', 'SURROGATE'))

            if base is not None and base['j'] <= self.near_zero_primary:
                # Nonnegativity supplies a genuine absolute lower bound. A
                # small primary value therefore permits an inexpensive,
                # explicitly limited line-candidate search, not a global solve.
                diag.update(path='NEAR_ZERO_LINE_CANDIDATE_PROJECTION',
                            primary_lower_bound=0.,
                            projection_scope='Audited candidates on equal-lift-to-proposal line only; not a global secondary optimum')
                best_primary = min(v['j'] for v in candidates)
                limit = best_primary*(1.+self.relative_allowance)+self.absolute_allowance
                diag.update(primary_best=best_primary, primary_limit=limit)
                feasible_line = []
                fractions = np.linspace(0., 1., 17)
                previous = None
                for fraction in fractions:
                    a = baseline+fraction*(ref-baseline)
                    a[0] = ref[0]
                    value = evaluate(a[1:]/self.scale)
                    good = bool(value['j'] <= limit and np.min(value['g']) >= 0.)
                    if good:
                        feasible_line.append((float(value['f']), float(fraction), a.copy()))
                    # Refine every sampled feasible-to-infeasible boundary.
                    # No assumption that the whole line is convex/monotone is
                    # made; disconnected unsampled feasible sets are not claimed.
                    if previous is not None and previous[1] != good:
                        left, right = previous[0], float(fraction)
                        left_good = previous[1]
                        for _ in range(20):
                            mid = (left+right)/2
                            ma = baseline+mid*(ref-baseline)
                            ma[0] = ref[0]
                            mv = evaluate(ma[1:]/self.scale)
                            mgood = bool(mv['j'] <= limit and np.min(mv['g']) >= 0.)
                            if mgood:
                                feasible_line.append((float(mv['f']), mid, ma.copy()))
                            if mgood == left_good:
                                left = mid
                            else:
                                right = mid
                    previous = (float(fraction), good)
                # Independent audit is authoritative; a symbolic constraint
                # pass alone never permits execution. Try the best few line
                # candidates; the already audited baseline remains an option.
                for _, fraction, a in sorted(feasible_line, key=lambda v: v[0])[:3]:
                    found = consider(a, 'BOUNDED_LINE_PROJECTION')
                    if found is not None:
                        diag['line_fraction'] = fraction
                        break
                diag['line_candidates_evaluated'] = len(feasible_line)
            else:
                diag['path'] = 'TWO_STAGE_CONSTRAINED_OPTIMIZATION'
                initial = (min(candidates, key=lambda v: v['j'])['nodes']
                           if candidates else self.project(context, ref))
                bounds = [(self.lo/self.scale, self.hi/self.scale)]*6
                physics = dict(type='ineq', fun=lambda z: evaluate(z)['g'],
                               jac=lambda z: evaluate(z)['dg'])
                options = dict(maxiter=int(self.cfg.get('max_iterations', 60)),
                               ftol=float(self.cfg.get('ftol', 1e-10)), disp=False)
                diag['optimizer_iteration_budget'] = options['maxiter']
                first = minimize(lambda z: evaluate(z)['j'], initial[1:]/self.scale,
                    jac=lambda z: evaluate(z)['dj'], method='SLSQP', bounds=bounds,
                    constraints=[physics], options=options)
                check_time()
                diag['optimizer'].append(dict(stage=1, success=bool(first.success),
                    status=int(first.status), message=str(first.message), iterations=int(first.nit)))
                if not bool(first.success):
                    return unknown('TRACKING_PRIMARY_OPTIMIZER_NOT_CONVERGED')
                remaining_iterations = options['maxiter']-int(first.nit)
                if remaining_iterations <= 0:
                    return unknown('TRACKING_CORRECTION_ITERATION_BUDGET_EXHAUSTED')
                consider(np.r_[ref[0], first.x*self.scale], 'TRACKING_PRIMARY')
                if not candidates:
                    return unknown('TRACKING_NO_CERTIFIED_CURRENT_CANDIDATE')
                best_primary_candidate = min(candidates, key=lambda v: v['j'])
                best_primary = best_primary_candidate['j']
                limit = best_primary*(1.+self.relative_allowance)+self.absolute_allowance
                diag.update(primary_best=best_primary, primary_limit=limit)
                primary_constraint = dict(type='ineq', fun=lambda z: limit-evaluate(z)['j'],
                                          jac=lambda z: -evaluate(z)['dj'])
                second = minimize(lambda z: evaluate(z)['f'],
                    best_primary_candidate['nodes'][1:]/self.scale,
                    jac=lambda z: evaluate(z)['df'], method='SLSQP', bounds=bounds,
                    constraints=[physics, primary_constraint],
                    options=dict(options, maxiter=remaining_iterations))
                check_time()
                diag['optimizer'].append(dict(stage=2, success=bool(second.success),
                    status=int(second.status), message=str(second.message), iterations=int(second.nit)))
                diag['optimizer_total_iterations'] = int(first.nit)+int(second.nit)
                if not bool(second.success):
                    return unknown('TRACKING_SECONDARY_OPTIMIZER_NOT_CONVERGED')
                consider(np.r_[ref[0], second.x*self.scale], 'TRACKING_BOUNDED_SECONDARY')

            check_time()
            allowed = [v for v in candidates if v['j'] <= limit+1e-14]
            if not allowed:
                return unknown('TRACKING_PRIMARY_BUDGET_NO_AUDITED_CANDIDATE')
            selected = min(allowed, key=lambda v: v['f'])
            a, audit, arrays = selected['nodes'], selected['audit'], selected['arrays']
            diag.update(status='TRACKING_PRIORITY_CURRENT_WITNESS', selected_source=selected['source'],
                        primary_selected=selected['j'], secondary_selected=selected['f'],
                        quality=self._quality(audit), final_audit=audit,
                        correction_max_deg=float(np.rad2deg(np.max(np.abs(a-ref)))),
                        wall_s=time.perf_counter()-started, budget_exceeded=False,
                        proof_status='CURRENT_SEGMENT_AUDITED_ONLY')
            check_time()
            return SafetyResult(True, diag['status'], a.copy(), np.arange(7)*.5, audit, arrays, diag)
        except _Deadline:
            diag['optimizer_deadline'] = True
            return unknown('TRACKING_CORRECTION_BUDGET_EXHAUSTED')
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, KeyError, IndexError) as exc:
            diag['error'] = repr(exc)
            return unknown('TRACKING_NUMERICAL_OR_DOMAIN_FAILURE')
