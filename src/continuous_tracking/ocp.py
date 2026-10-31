"""Full-horizon pointwise tracking with continuous alpha and boundary rate cost."""
from dataclasses import dataclass
import time

import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator

from certified_flight.ocp import ACCEPTED_STATUS, TwoStageOCP
from .compiled_solver import compile_solver


@dataclass
class TrackingPlan:
    success: bool
    status: str
    times: np.ndarray
    alpha_nodes: np.ndarray
    states: np.ndarray
    solve_s: float
    iterations: int
    violation: float
    objective: float | None
    terms: dict
    stages: dict

    def summary(self):
        return dict(success=self.success, status=self.status, solve_s=self.solve_s,
                    iterations=self.iterations, violation=self.violation, objective=self.objective, terms=self.terms,
                    stages=self.stages)


class TrackingOCP(TwoStageOCP):
    def __init__(self, mission, db, trajectory, physical, plant, experiment):
        super().__init__(mission, db, trajectory, physical, plant)
        self.experiment = experiment
        if (experiment['tracking'] != 'rk4_pointwise_full_horizon'
                or experiment['reference_prediction'] != 'hold_current_CL'
                or experiment['objective_priority'] != 'current_tracking_then_full_horizon_economics'):
            raise ValueError('Unsupported tracking contract')
        for k in ('normalization_time_s', 'rate_cost_reference_dt_s', 'planning_integration_max_dt_s'):
            if not np.isfinite(experiment[k]) or experiment[k] <= 0:
                raise ValueError('Invalid tracking normalization')
        for k in ('tracking_optimality_relative_allowance','tracking_optimality_absolute_allowance','planning_normalized_margin'):
            if not np.isfinite(experiment[k]) or experiment[k] < 0:
                raise ValueError('Invalid tracking allowance or planning margin')

    def build(self, dts):
        key = tuple(dts)
        if key in self.cache:
            return self.cache[key]
        n, nc = len(dts), self.db.ncoef
        times = np.r_[0., np.cumsum(dts)]
        p = ca.MX.sym('p', 9+2*nc)
        x0, previous, cl, fraction, rate = p[:4], p[4], p[5], p[6], p[7]
        c0, c1 = p[8:8+nc], p[8+nc:8+2*nc]
        tracking_budget = p[-1]
        X, U = ca.MX.sym('X', 4, n+1), ca.MX.sym('alpha_end', n)
        x = ca.times(X, ca.repmat(ca.DM(self.xscale), 1, n+1))
        nodes = ca.vertcat(previous, U*self.uscale)
        gs, lb, ub = [], [], []

        def add(v, lower, upper):
            gs.append(ca.vec(v)); lb.extend([lower]*v.numel()); ub.extend([upper]*v.numel())

        add(X[:, 0]-x0/self.xscale, 0., 0.)
        l, o = self.m.data['limits'], self.traj['objective']
        tracking = self.traj['aero_target_tracking']
        terms = {k: 0 for k in ('drag', 'heat', 'target', 'control_rate')}
        margin = self.experiment['planning_normalized_margin']
        first_cl = 0
        current_tracking = 0
        for i, dt in enumerate(dts):
            slope = (nodes[i+1]-nodes[i])/dt
            add(slope/l['alpha_rate_radps'], -1., 1.)
            state = x[:, i]
            ns = max(1, int(np.ceil(dt/self.experiment['planning_integration_max_dt_s']-1e-9)))
            h = dt/ns
            for j in range(ns):
                tau = float(times[i]+j*h)
                state, ss = self.plant.rk(state, nodes[i]+slope*j*h, slope,
                                         fraction, rate, c0, c1, h, tau)
                for axis, name, scale in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
                    add((ss[axis, :]-l[name][0])/scale, margin, np.inf)
                    add((l[name][1]-ss[axis, :])/scale, margin, np.inf)
                for row, name in ((4, 'dynamic_pressure_pa_max'), (5, 'heat_proxy_wm2_max'), (6, 'load_factor_g_max')):
                    add(1-ss[row, :]/l[name], margin, np.inf)
                add(ss[7, :]-self.db.M[0], margin, np.inf)
                add(self.db.M[-1]-ss[7, :], margin, np.inf)
                # Pointwise squared errors: average errors cannot cancel.
                errors = ((ss[8, :4]-cl)/float(tracking['CL_scale']))**2
                quad = (errors[0]+2*errors[1]+2*errors[2]+errors[3])/6
                terms['target'] += float(tracking['weight_CL'])*h*quad/self.experiment['normalization_time_s']
                if tau < min(3., times[-1])-1e-9:
                    first_cl += h*(ss[8, 0]+2*ss[8, 1]+2*ss[8, 2]+ss[8, 3])/6
                    current_tracking += float(tracking['weight_CL'])*h*quad/min(3., times[-1])
            add(X[:, i+1]-state/self.xscale, 0., 0.)
            _, q = self.plant.base.fn(x[:, i], nodes[i], ca.fmin(1., fraction+rate*times[i]), c0, c1)
            # The original economic terms and their weights are retained.
            terms['drag'] += q[6]*x[1, i]*dt/float(o['reference']['energy_ref'])
            terms['heat'] += q[1]*dt/float(o['reference']['heat_load_ref'])
            # Includes the first interval from the *actual* previous alpha;
            # equivalent to integral(alpha_dot^2) with a fixed reference dt.
            terms['control_rate'] += ((nodes[i+1]-nodes[i])/self.uscale)**2*self.experiment['rate_cost_reference_dt_s']/dt
        terminal = o['terminal']
        terms['terminal'] = 0
        for j, name in enumerate(('h', 'V', 'gamma', 's')):
            if terminal.get(name+'_target') is not None and terminal['weights'][name] > 0:
                terms['terminal'] += terminal['weights'][name]*((x[j, -1]-terminal[name+'_target'])/terminal['scales'][name])**2
        terms['range'] = (x[3, -1]-x0[3])/float(self.traj['solver']['scaling']['s'])
        objective = (o['w_drag']*terms['drag']+o['w_heat']*terms['heat']+o['w_terminal']*terms['terminal']
                     +o['w_control_rate']*terms['control_rate']-o['w_range']*terms['range']+terms['target'])
        terms['CL_first_interval_mean'] = first_cl/min(3., times[-1])
        terms['current_tracking_cost'] = current_tracking
        v, g = ca.vertcat(ca.vec(X), U), ca.vertcat(*gs)
        opts = {'print_time': False, 'calc_lam_p': False, 'ipopt': dict(self.traj['solver']['ipopt'])}
        opts['ipopt']['bound_relax_factor'] = 0.
        primary = ca.nlpsol(f'tracking_priority_{n}', 'ipopt', {'x': v, 'p': p, 'f': current_tracking, 'g': g}, opts)
        solver = ca.nlpsol(f'continuous_tracking_{n}', 'ipopt',
                          {'x': v, 'p': p, 'f': objective, 'g': ca.vertcat(g, current_tracking-tracking_budget)}, opts)
        raw_primary,raw_secondary=primary,solver
        compiled={}
        if self.experiment.get('compile_nlp',False) and times[-1]>=60.:
            primary,compiled['primary']=compile_solver(primary,f'compiled_primary_{n}',opts)
            solver,compiled['secondary']=compile_solver(solver,f'compiled_secondary_{n}',opts)
        lower = np.array([l['altitude_m'][0], l['velocity_mps'][0], l['gamma_rad'][0], -np.inf])/self.xscale
        upper = np.array([l['altitude_m'][1], l['velocity_mps'][1], l['gamma_rad'][1], np.inf])/self.xscale
        alo, ahi = max(self.db.A[0], l['alpha_rad'][0]), min(self.db.A[-1], l['alpha_rad'][1])
        b = dict(solver=solver, primary=primary, raw_primary=raw_primary, raw_secondary=raw_secondary,
                 compiled=compiled, times=times, lbg=np.array(lb), ubg=np.array(ub),
                 lbw=np.r_[np.tile(lower, n+1), np.full(n, alo/self.uscale)],
                 ubw=np.r_[np.tile(upper, n+1), np.full(n, ahi/self.uscale)],
                 terms=ca.Function(f'continuous_terms_{n}', [v, p], [ca.vertcat(*terms.values())]),
                 term_names=list(terms))
        self.cache[key] = b
        return b

    def initial_nodes(self, c, times):
        mach = c['x'][1]/float(self.plant.atmosphere.speed_of_sound(c['x'][0]))
        curves = self.db.curves(mach, [c['source'], c['target']])
        funcs = [PchipInterpolator(self.db.A, y[:, 0], extrapolate=False) for y in curves]
        w = np.minimum(1., c['fraction']+c['rate']*times[1:])
        low, high = np.full(len(w), self.db.A[0]), np.full(len(w), self.db.A[-1])
        for _ in range(32):
            aa = (low+high)/2
            lift = (1-w)*funcs[0](aa)+w*funcs[1](aa)
            low, high = np.where(lift < c['CL_des'], aa, low), np.where(lift >= c['CL_des'], aa, high)
        return np.r_[c['alpha_prev'], (low+high)/2]

    def solve_context(self, c, guess=None):
        dts = self.grid(self.m.data['duration_s']-c['t'])
        b = self.build(dts)
        n = len(dts)
        a = self.initial_nodes(c, b['times']) if guess is None else np.asarray(guess, float).copy()
        if a.shape != (n+1,) or not np.isfinite(a).all():
            raise ValueError('Warm-start nodes must match the new grid')
        a[0] = c['alpha_prev']
        limits = self.m.data['limits']
        lo, hi = max(self.db.A[0], limits['alpha_rad'][0]), min(self.db.A[-1], limits['alpha_rad'][1])
        xs = [np.asarray(c['x'], float)]
        for i, dt in enumerate(dts):
            allowance = limits['alpha_rate_radps']*dt
            a[i+1] = np.clip(a[i+1], max(lo, a[i]-allowance), min(hi, a[i]+allowance))
            state = xs[-1]
            ns = max(1, int(np.ceil(dt/self.experiment['planning_integration_max_dt_s']-1e-9)))
            slope = (a[i+1]-a[i])/dt
            for j in range(ns):
                h = dt/ns
                state, _ = self.plant.rk(state, a[i]+slope*j*h, slope, c['fraction'], c['rate'],
                    self.db.parameters[c['source']], self.db.parameters[c['target']], h, b['times'][i]+j*h)
            xs.append(np.asarray(state).ravel())
        states = np.asarray(xs)
        p = np.r_[c['x'], c['alpha_prev'], c['CL_des'], c['fraction'], c['rate'],
                  self.db.parameters[c['source']], self.db.parameters[c['target']], 0.]
        initial = np.r_[(states/self.xscale).ravel(), a[1:]/self.uscale]
        start = time.perf_counter()
        try:
            primary = b['primary'](x0=np.clip(initial, b['lbw'], b['ubw']), p=p,
                                  lbx=b['lbw'], ubx=b['ubw'], lbg=b['lbg'], ubg=b['ubg'])
            ps = b['primary'].stats()
            best_tracking = float(primary['f'])
            budget = (best_tracking*(1+self.experiment['tracking_optimality_relative_allowance'])
                      + self.experiment['tracking_optimality_absolute_allowance'])
            stages = dict(primary_status=ps['return_status'], primary_iterations=int(ps['iter_count']),
                          primary_tracking_cost=best_tracking, allowed_tracking_cost=budget,
                          primary_wall_s=time.perf_counter()-start,compiled=b['compiled'])
            r, stats = primary, ps
            lows, highs = b['lbg'], b['ubg']
            if ps['return_status'] in ACCEPTED_STATUS and np.isfinite(best_tracking):
                p[-1] = budget
                candidate = b['solver'](x0=primary['x'],p=p,lbx=b['lbw'],ubx=b['ubw'],
                                       lbg=np.r_[b['lbg'],-np.inf],ubg=np.r_[b['ubg'],0.])
                secondary = b['solver'].stats()
                stages.update(secondary_status=secondary['return_status'],secondary_iterations=int(secondary['iter_count']))
                if secondary['return_status'] in ACCEPTED_STATUS:
                    r, stats = candidate, secondary
                    lows, highs = np.r_[b['lbg'],-np.inf],np.r_[b['ubg'],0.]
                    stages['selected']='secondary_economics'
                else:
                    # A successful primary remains only a candidate; the same
                    # independent full-horizon audits still decide acceptance.
                    stages['selected']='primary_tracking_secondary_failed'
            else:
                stages['selected']='primary_failed'
            v, gv = np.asarray(r['x']).ravel(), np.asarray(r['g']).ravel()
            viol = float(max(np.max(np.maximum(lows-gv, 0)+np.maximum(gv-highs, 0)),
                             np.max(np.maximum(b['lbw']-v, 0)+np.maximum(v-b['ubw'], 0))))
            terms = dict(zip(b['term_names'], map(float, np.asarray(b['terms'](v, p)).ravel())))
            ok = bool(stats['return_status'] in ACCEPTED_STATUS and np.isfinite(v).all()
                      and np.isfinite(viol) and viol <= self.cfg['acceptance_violation'])
            return TrackingPlan(ok, stats['return_status'], b['times'].copy(),
                np.r_[c['alpha_prev'], v[4*(n+1):]*self.uscale],
                v[:4*(n+1)].reshape(n+1, 4)*self.xscale, time.perf_counter()-start,
                int(ps['iter_count'])+int(stages.get('secondary_iterations',0)), viol, float(r['f']), terms, stages)
        except RuntimeError as exc:
            return TrackingPlan(False, 'SOLVER_EXCEPTION:'+str(exc)[-300:], b['times'].copy(), a, states,
                                time.perf_counter()-start, 0, float('inf'), None, {}, {})
