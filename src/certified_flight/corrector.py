"""Post corrector: six-command correction with exact plant constraints and independent audits.

This is a three-second candidate search, not a feasibility certificate. A failed
search never exposes an executable alpha sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

import casadi as ca
import numpy as np
from scipy.optimize import minimize

from . import codegen
from .aero import DomainExit
from .diagnostics import MARGIN_NAMES, interval_moments
from .surrogate import Features

_SHARED = {}


@dataclass
class CorrectionResult:
    accepted: bool
    status: str
    alpha: np.ndarray | None
    diagnostics: dict


def audit_controls(plant, context, alpha):
    """Re-integrate six half-second commands on the .1 s (dense) and .5 s (actual) grids.

    Adds the alpha bound and rate checks that integrate() does not perform.
    """
    u = np.asarray(alpha, float)
    limits = plant.mission.data['limits']
    lo = max(plant.db.A[0], limits['alpha_rad'][0])
    hi = min(plant.db.A[-1], limits['alpha_rad'][1])
    finite = u.shape == (6,) and np.isfinite(u).all()
    control_valid = bool(finite and np.all(u >= lo-1e-9) and np.all(u <= hi+1e-9)
                         and np.all(abs(np.diff(np.r_[context['alpha_prev'], u])) <= limits['alpha_rate_radps']*.5+1e-7))
    result = dict(valid=False, control_valid=control_valid, dense=None, actual=None,
                  violation_names=[] if control_valid else ['alpha_bounds_or_rate'])
    if not finite:
        return result
    for name, dt in (('dense', .1), ('actual', .5)):
        r = plant.integrate(context['x'], u, np.arange(7)*.5, context['source'], context['target'],
                            context['fraction'], context['rate'], dt)
        # Recompute rather than trusting an optimizer or a caller-supplied valid flag.
        margins = plant.normalized_margins(r['samples'])
        good = np.isfinite(r['x']).all() and np.isfinite(r['samples']).all() and np.isfinite(margins).all()
        r['valid'] = bool(good and np.min(margins) >= -1e-7)
        r['min_margin'] = float(np.min(margins)) if good else -np.inf
        r['margins'] = np.min(margins, axis=1).tolist()
        result[name] = r
        result['violation_names'].extend(n for n, v in zip(MARGIN_NAMES, r['margins']) if v < -1e-7)
        if not good:
            result['violation_names'].append('nonfinite_dynamics')
    result['violation_names'] = sorted(set(result['violation_names']))
    result['valid'] = bool(control_valid and result['dense']['valid'] and result['actual']['valid'])
    return result


def compact_audit(audit):
    """JSON-sized summary of an audit_controls result."""
    result = {k: audit[k] for k in ('valid', 'control_valid', 'violation_names')}
    for name in ('dense', 'actual'):
        r = audit[name]
        if r is None:
            result[name] = None
            continue
        s = r['samples']
        result[name] = dict(valid=r['valid'], min_margin=r['min_margin'], margins=dict(zip(MARGIN_NAMES, r['margins'])),
                            q_peak_pa=float(np.max(s[4])), load_peak_g=float(np.max(s[6])), heat_peak=float(np.max(s[5])),
                            height_min_m=float(np.min(s[0])), Mach_range=[float(np.min(s[7])), float(np.max(s[7]))],
                            CL_mean=interval_moments(r)['mean'], end_state=r['x'][-1].tolist())
    return result


class _Deadline(Exception):
    pass


class PostCorrector:
    """Bounded SLSQP in six scaled alpha variables, with CasADi derivatives.

    Objective: weighted distance to the proposed commands, smoothness and the
    3 s mean-CL error; constraints: every RK stage on the planning grid plus slew.
    Candidates are accepted only after audit_controls.
    """

    def __init__(self, plant, options=None):
        self.plant = plant
        self.cfg = dict(options or {})
        if self.cfg.get('post_correction_method', 'six_alpha_slsqp') != 'six_alpha_slsqp':
            raise ValueError('Unknown Post correction method')
        self.budget = float(self.cfg.get('post_correction_max_wall_s', 2.))
        self.scale = float(self.cfg.get('alpha_scale_rad', .05))
        self.clscale = float(self.cfg.get('CL_scale', .1))
        self.margin = float(self.cfg.get('internal_margin', 0.))
        weights = self.cfg.get('weights', {})
        self.wn = np.broadcast_to(np.asarray(weights.get('network', 1.), float), (6,)).copy()
        self.ws = float(weights.get('smooth', .01))
        self.wcl = float(weights.get('CL', .05))
        vals = np.r_[self.budget, self.scale, self.clscale, self.margin, self.wn, self.ws, self.wcl]
        if not np.isfinite(vals).all() or min(self.budget, self.scale, self.clscale) <= 0 or \
           min(self.margin, self.ws, self.wcl, np.min(self.wn)) < 0:
            raise ValueError('Correction scales/budgets must be positive and weights/margin nonnegative')
        self.features = Features(plant.db, plant.mission)
        self.functions = {}
        self.build_times = {}
        if codegen.enabled():
            # Build/compile now, never inside a correction wall-clock budget.
            for dt in self._planning_grids():
                self._function(dt)

    def _planning_grids(self):
        return [.5, .1] if self.cfg.get('dense_refinement', True) else [.5]

    def _shared_key(self, dt):
        limits = json.dumps(self.plant.mission.data['limits'], sort_keys=True)
        return (float(dt), self.scale, self.clscale, self.margin, tuple(self.wn.tolist()), self.ws, self.wcl,
                str(self.plant.db.directory), self.plant.db.ncoef, limits, codegen.enabled())

    def _function(self, dt):
        if dt in self.functions:
            return self.functions[dt]
        started = time.perf_counter()
        # The expression graph depends only on these settings, so the compiled
        # function is shared within a process. Without code generation each
        # instance builds its own MX function lazily.
        shared = _SHARED.get(self._shared_key(dt)) if codegen.enabled() else None
        if shared is not None:
            self.functions[dt] = shared
            self.build_times[dt] = time.perf_counter()-started
            return shared
        nc = self.plant.db.ncoef
        z = ca.MX.sym('alpha_scaled', 6)
        u = z*self.scale
        p = ca.MX.sym('context', 14+2*nc)
        state = p[:4]
        prev = p[4]
        cl = p[5]
        ref = p[6:12]
        fraction = p[12]
        rate = p[13]
        c0 = p[14:14+nc]
        c1 = p[14+nc:]
        limits = self.plant.mission.data['limits']
        constraints = []
        cl_integral = 0.
        for j in range(int(round(3/dt))):
            t = j*dt
            slot = min(int(round(t*10))//5, 5)
            state, ss = self.plant.rk(state, u[slot], fraction, rate, c0, c1, dt, t)
            for axis, key, scale in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
                constraints.extend([(ss[axis, :]-limits[key][0])/scale-self.margin,
                                    (limits[key][1]-ss[axis, :])/scale-self.margin])
            constraints.extend([1-ss[4, :]/limits['dynamic_pressure_pa_max']-self.margin,
                                1-ss[5, :]/limits['heat_proxy_wm2_max']-self.margin,
                                1-ss[6, :]/limits['load_factor_g_max']-self.margin,
                                ss[7, :]-self.plant.db.M[0]-self.margin, self.plant.db.M[-1]-ss[7, :]-self.margin])
            cl_integral += dt*(ss[8, 0]+2*ss[8, 1]+2*ss[8, 2]+ss[8, 3])/6
        slew = (u-ca.vertcat(prev, u[:-1]))/(limits['alpha_rate_radps']*.5)
        constraints.extend([1-slew, 1+slew])
        g = ca.vertcat(*[ca.vec(v) for v in constraints])
        objective = (ca.dot(ca.DM(self.wn), ((u-ref)/self.scale)**2)+self.ws*ca.sumsqr((u[1:]-u[:-1])/self.scale)
                     + self.wcl*((cl_integral/3-cl)/self.clscale)**2)
        fn = ca.Function(f'post_correction_{int(round(1/dt))}', [z, p],
                         [objective, ca.gradient(objective, z), g, ca.jacobian(g, z)])
        if codegen.enabled():
            fn = codegen.compile_function(fn)
            _SHARED[self._shared_key(dt)] = fn
        self.functions[dt] = fn
        self.build_times[dt] = time.perf_counter()-started
        return fn

    def correct(self, context, network_alpha):
        started = time.perf_counter()
        deadline = started+self.budget
        diag = dict(method='six_alpha_slsqp', horizon_s=3., control_slots=6, control_dt_s=.5,
                    independent_audit_dt_s=.1, attempts=[], optimizer_success=False, oracle_calls=0,
                    internal_margin=self.margin, universal_infeasibility_proven=False,
                    budget_s=self.budget, scope='Finite candidate witness only; no long-term feasibility certificate')

        def finish(status, alpha=None):
            diag.update(wall_s=time.perf_counter()-started, status=status,
                        build_times_s={str(k): v for k, v in self.build_times.items()})
            return CorrectionResult(alpha is not None, status, None if alpha is None else np.asarray(alpha).copy(), diag)

        ref = np.asarray(network_alpha, float)
        if ref.shape != (6,) or not np.isfinite(ref).all():
            return finish('NUMERICAL_FAILURE')
        try:
            state = np.asarray(context['x'], float)
            scalars = [context[k] for k in ('alpha_prev', 't', 'fraction', 'rate', 'CL_des')]
            if state.shape != (4,) or not np.isfinite(np.r_[state, scalars]).all():
                raise ValueError('Nonfinite or malformed current physical context')
            if not 0 <= context['fraction'] <= 1 or context['rate'] not in (0., .125):
                raise ValueError('Invalid eight-second morph context')
            if any(not isinstance(context[k], (int, np.integer)) or not 0 <= context[k] < len(self.plant.db.ids)
                   for k in ('source', 'target')):
                raise ValueError('Unknown geometry endpoint index')
            self.plant.quantities(context['x'], context['alpha_prev'], context['source'], context['target'],
                                  context['fraction'])
            original = audit_controls(self.plant, context, ref)
            diag['network_audit'] = compact_audit(original)
        except DomainExit as exc:
            diag['error'] = str(exc)
            return finish('AERO_DOMAIN_EXIT')
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, KeyError) as exc:
            diag['error'] = repr(exc)
            return finish('NUMERICAL_FAILURE')
        if original['valid']:
            diag.update(selected_source='NETWORK', final_audit=compact_audit(original), correction_max_deg=0.)
            return finish('NETWORK_PLAN_ALREADY_SAFE', ref)
        if time.perf_counter() >= deadline:
            return finish('CORRECTION_NO_FEASIBLE_WITNESS')
        p = np.r_[context['x'], context['alpha_prev'], context['CL_des'], ref, context['fraction'], context['rate'],
                  self.plant.db.parameters[context['source']], self.plant.db.parameters[context['target']]]
        lo = max(self.plant.db.A[0], self.plant.mission.data['limits']['alpha_rad'][0])/self.scale
        hi = min(self.plant.db.A[-1], self.plant.mission.data['limits']['alpha_rad'][1])/self.scale
        best = None
        best_objective = np.inf
        last = ref.copy()
        timed_out = False

        def score(alpha, audit):
            cl = interval_moments(audit['dense'])['mean']
            return float(np.sum(self.wn*((alpha-ref)/self.scale)**2)+self.ws*np.sum((np.diff(alpha)/self.scale)**2)
                         + self.wcl*((cl-context['CL_des'])/self.clscale)**2)

        def consider(alpha, source):
            nonlocal best, best_objective
            a = audit_controls(self.plant, context, alpha)
            # An optional buffer is a planning preference; report if a witness only meets hard limits.
            val = score(alpha, a) if a['valid'] else np.inf
            if a['valid'] and val < best_objective:
                best = (np.asarray(alpha).copy(), source, a)
                best_objective = val
            return a

        try:
            baseline = self.features.baseline(context, 'post')
            base_audit = consider(baseline, 'EQUAL_LIFT_AUDITED_WITNESS')
            diag['baseline_audit'] = compact_audit(base_audit)
            for dt in self._planning_grids():
                if time.perf_counter() >= deadline:
                    timed_out = True
                    break
                fn = self._function(dt)
                cache = {}

                def evaluate(z):
                    if time.perf_counter() >= deadline:
                        raise _Deadline()
                    if 'z' not in cache or not np.array_equal(z, cache['z']):
                        out = fn(z, p)
                        cache.update(z=np.asarray(z).copy(), f=float(out[0]), df=np.asarray(out[1]).ravel(),
                                     g=np.asarray(out[2]).ravel(), dg=np.asarray(out[3]))
                    return cache

                initial = self.features.project(last, context['alpha_prev'])/self.scale
                trial = minimize(lambda z: evaluate(z)['f'], initial, jac=lambda z: evaluate(z)['df'], method='SLSQP',
                                 bounds=[(lo, hi)]*6,
                                 constraints=[{'type': 'ineq', 'fun': lambda z: evaluate(z)['g'],
                                               'jac': lambda z: evaluate(z)['dg']}],
                                 options=dict(maxiter=int(self.cfg.get('max_iterations', 60)),
                                              ftol=float(self.cfg.get('ftol', 1e-10)), disp=False))
                last = trial.x*self.scale
                a = consider(last, f'SLSQP_GRID_{dt:g}')
                diag['attempts'].append(dict(planning_dt_s=dt, success=bool(trial.success), status=int(trial.status),
                                             message=str(trial.message), iterations=int(trial.nit),
                                             alpha_rad=last.tolist(), audit=compact_audit(a)))
                diag['optimizer_success'] = diag['optimizer_success'] or bool(trial.success)
                if a['valid']:
                    break
        except _Deadline:
            timed_out = True
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            diag['optimizer_error'] = repr(exc)
        diag['timed_out'] = bool(timed_out or time.perf_counter() >= deadline)
        if best is None:
            status = 'DENSE_AUDIT_REJECTED' if diag['optimizer_success'] else 'CORRECTION_NO_FEASIBLE_WITNESS'
            return finish(status)
        alpha, source, a = best
        diag.update(selected_source=source, final_audit=compact_audit(a), objective=best_objective,
                    correction_max_deg=float(np.rad2deg(np.max(abs(alpha-ref)))),
                    internal_margin_met=bool(min(a['dense']['margins']) >= self.margin-1e-7 and
                                             min(a['actual']['margins']) >= self.margin-1e-7))
        return finish('SAFE_CORRECTION_FOUND', alpha)
