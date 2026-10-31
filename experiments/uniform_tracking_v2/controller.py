"""Isolated experiment: uniform pointwise CL band, with explicit safety slacks.

No first-three-second optimality budget. Physical constraints remain hard.
One NLP per Pre/Final. Slope-change cost includes the last executed slope.
The slack penalty is a finite engineering choice, not a proof that every
nonzero slack is physically necessary. This is not an identified actuator.
"""
import time

import casadi as ca
import numpy as np

from certified_flight.ocp import ACCEPTED_STATUS, TwoStageOCP
from continuous_tracking.ocp import TrackingOCP, TrackingPlan
from continuous_tracking.compiled_solver import compile_solver

PROTOCOL = 'continuous-alpha-uniform-pointwise-tracking/2'


class UniformTrackingOCP(TrackingOCP):
    def __init__(self, mission, db, trajectory, physical, plant, experiment):
        TwoStageOCP.__init__(self, mission, db, trajectory, physical, plant)
        self.experiment = experiment
        if (experiment['protocol'] != PROTOCOL or experiment['tracking'] != 'uniform_pointwise_soft_band'
                or experiment['reference_prediction'] != 'hold_current_CL'):
            raise ValueError('Unsupported uniform tracking protocol')
        for name in ('cl_tracking_band', 'slack_weight', 'planning_integration_max_dt_s',
                     'normalization_time_s', 'rate_cost_reference_dt_s'):
            if not np.isfinite(experiment[name]) or experiment[name] <= 0:
                raise ValueError('Invalid '+name)
        for name in ('rate_change_weight_multiplier', 'planning_normalized_margin'):
            if not np.isfinite(experiment[name]) or experiment[name] < 0:
                raise ValueError('Invalid '+name)

    def build(self, dts):
        key = tuple(dts)
        if key in self.cache:
            return self.cache[key]
        n, nc = len(dts), self.db.ncoef
        counts = np.maximum(1, np.ceil(dts/self.experiment['planning_integration_max_dt_s']-1e-9).astype(int))
        ns = int(sum(counts))
        times = np.r_[0., np.cumsum(dts)]
        p = ca.MX.sym('p', 10+2*nc)
        x0, alpha0, cl, fraction, rate = p[:4], p[4], p[5], p[6], p[7]
        c0, c1 = p[8:8+nc], p[8+nc:8+2*nc]
        previous_slope, previous_dt = p[-2], p[-1]
        X, U, E = ca.MX.sym('X', 4, n+1), ca.MX.sym('alpha_end', n), ca.MX.sym('CL_slack_scaled', ns)
        x = ca.times(X, ca.repmat(ca.DM(self.xscale), 1, n+1))
        nodes = ca.vertcat(alpha0, U*self.uscale)
        gs, lb, ub = [], [], []

        def add(value, low, high):
            gs.append(ca.vec(value)); lb.extend([low]*value.numel()); ub.extend([high]*value.numel())

        add(X[:, 0]-x0/self.xscale, 0., 0.)
        l, o = self.m.data['limits'], self.traj['objective']
        tr = self.traj['aero_target_tracking']
        scale = float(tr['CL_scale'])
        band = self.experiment['cl_tracking_band']/scale
        margin = self.experiment['planning_normalized_margin']
        refdt = self.experiment['rate_cost_reference_dt_s']
        norm = self.experiment['normalization_time_s']
        terms = {k: 0 for k in ('drag', 'heat', 'target', 'slack', 'control_rate', 'rate_change')}
        error_rows, slack_times, slack_dts = [], [], []
        z = 0
        for i, dt in enumerate(dts):
            slope = (nodes[i+1]-nodes[i])/dt
            add(slope/l['alpha_rate_radps'], -1., 1.)
            # Approximate integral of squared changes of alpha_dot, including
            # the interval preceding this decision. This is a cost, not a new
            # physical acceleration limit or a claim of C1 continuity.
            center_gap = .5*(dt+(previous_dt if i == 0 else dts[i-1]))
            terms['rate_change'] += ((slope-previous_slope)*refdt/self.uscale)**2*refdt/center_gap
            previous_slope = slope
            terms['control_rate'] += ((nodes[i+1]-nodes[i])/self.uscale)**2*refdt/dt
            state = x[:, i]
            h = dt/counts[i]
            for j in range(counts[i]):
                tau = float(times[i]+j*h)
                state, ss = self.plant.rk(state, nodes[i]+slope*j*h, slope,
                                         fraction, rate, c0, c1, h, tau)
                for axis, name, sc in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
                    add((ss[axis, :]-l[name][0])/sc, margin, np.inf)
                    add((l[name][1]-ss[axis, :])/sc, margin, np.inf)
                for row, name in ((4, 'dynamic_pressure_pa_max'), (5, 'heat_proxy_wm2_max'), (6, 'load_factor_g_max')):
                    add(1-ss[row, :]/l[name], margin, np.inf)
                add(ss[7, :]-self.db.M[0], margin, np.inf)
                add(self.db.M[-1]-ss[7, :], margin, np.inf)
                err = (ss[8, :]-cl)/scale
                # Identical inequalities at EVERY RK stage over the entire
                # horizon. No rule changes at tau=3 s, 6 s, etc.
                add(err-E[z], -np.inf, band)
                add(-err-E[z], -np.inf, band)
                error_rows.append(err)
                sq = err[:, :4]**2
                terms['target'] += float(tr['weight_CL'])*h*(sq[0]+2*sq[1]+2*sq[2]+sq[3])/6/norm
                terms['slack'] += E[z]*h/norm
                slack_times.append(tau); slack_dts.append(h); z += 1
            add(X[:, i+1]-state/self.xscale, 0., 0.)
            _, q = self.plant.base.fn(x[:, i], nodes[i], ca.fmin(1., fraction+rate*times[i]), c0, c1)
            terms['drag'] += q[6]*x[1, i]*dt/float(o['reference']['energy_ref'])
            terms['heat'] += q[1]*dt/float(o['reference']['heat_load_ref'])
        terminal = o['terminal']
        terms['terminal'] = 0
        for j, name in enumerate(('h', 'V', 'gamma', 's')):
            if terminal.get(name+'_target') is not None and terminal['weights'][name] > 0:
                terms['terminal'] += terminal['weights'][name]*((x[j, -1]-terminal[name+'_target'])/terminal['scales'][name])**2
        terms['range'] = (x[3, -1]-x0[3])/float(self.traj['solver']['scaling']['s'])
        objective = (o['w_drag']*terms['drag']+o['w_heat']*terms['heat']+o['w_terminal']*terms['terminal']
                     +o['w_control_rate']*(terms['control_rate']+self.experiment['rate_change_weight_multiplier']*terms['rate_change'])
                     -o['w_range']*terms['range']+terms['target']+self.experiment['slack_weight']*terms['slack'])
        v, g = ca.vertcat(ca.vec(X), U, E), ca.vertcat(*gs)
        opts = {'print_time': False, 'calc_lam_p': False, 'ipopt': dict(self.traj['solver']['ipopt'])}
        opts['ipopt']['bound_relax_factor'] = 0.
        raw = ca.nlpsol(f'uniform_tracking_{n}', 'ipopt', {'x': v, 'p': p, 'f': objective, 'g': g}, opts)
        solver, compiled = raw, {'compiled': False}
        if self.experiment.get('compile_nlp', False) and times[-1] >= 60.:
            solver, compiled = compile_solver(raw, f'uniform_compiled_{n}', opts)
        low = np.array([l['altitude_m'][0], l['velocity_mps'][0], l['gamma_rad'][0], -np.inf])/self.xscale
        high = np.array([l['altitude_m'][1], l['velocity_mps'][1], l['gamma_rad'][1], np.inf])/self.xscale
        alo, ahi = max(self.db.A[0], l['alpha_rad'][0]), min(self.db.A[-1], l['alpha_rad'][1])
        b = dict(solver=solver, raw_solver=raw, compiled=compiled, times=times, counts=counts,
                 lbg=np.array(lb), ubg=np.array(ub), n_slack=ns, slack_start=4*(n+1)+n,
                 slack_times=np.array(slack_times), slack_dts=np.array(slack_dts),
                 lbw=np.r_[np.tile(low, n+1), np.full(n, alo/self.uscale), np.zeros(ns)],
                 ubw=np.r_[np.tile(high, n+1), np.full(n, ahi/self.uscale), np.full(ns, np.inf)],
                 terms=ca.Function(f'uniform_terms_{n}', [v, p], [ca.vertcat(*terms.values())]),
                 errors=ca.Function(f'uniform_errors_{n}', [v, p], [ca.vertcat(*error_rows)]),
                 term_names=list(terms), band_scaled=band)
        self.cache[key] = b
        return b

    def solve_context(self, c, guess=None):
        for key in ('alpha_rate_prev', 'alpha_interval_prev_s'):
            if key not in c or not np.isfinite(c[key]):
                raise ValueError('Missing/invalid executed control memory: '+key)
        if c['alpha_interval_prev_s'] <= 0:
            raise ValueError('Nonpositive previous control interval')
        dts = self.grid(self.m.data['duration_s']-c['t'])
        b = self.build(dts); n = len(dts)
        a = self.initial_nodes(c, b['times']) if guess is None else np.asarray(guess, float).copy()
        if a.shape != (n+1,) or not np.isfinite(a).all():
            raise ValueError('Warm-start grid mismatch')
        a[0] = c['alpha_prev']
        lim = self.m.data['limits']; lo = max(self.db.A[0], lim['alpha_rad'][0]); hi = min(self.db.A[-1], lim['alpha_rad'][1])
        states = [np.asarray(c['x'], float)]; slack = []
        for i, dt in enumerate(dts):
            allowance = lim['alpha_rate_radps']*dt
            a[i+1] = np.clip(a[i+1], max(lo, a[i]-allowance), min(hi, a[i]+allowance))
            slope = (a[i+1]-a[i])/dt; state = states[-1]; h = dt/b['counts'][i]
            for j in range(b['counts'][i]):
                state, ss = self.plant.rk(state, a[i]+slope*j*h, slope, c['fraction'], c['rate'],
                    self.db.parameters[c['source']], self.db.parameters[c['target']], h, b['times'][i]+j*h)
                err = np.max(abs(np.asarray(ss)[8]-c['CL_des']))
                slack.append(max(0., err-self.experiment['cl_tracking_band'])/float(self.traj['aero_target_tracking']['CL_scale']))
            states.append(np.asarray(state).ravel())
        states = np.asarray(states)
        p = np.r_[c['x'], c['alpha_prev'], c['CL_des'], c['fraction'], c['rate'],
                  self.db.parameters[c['source']], self.db.parameters[c['target']], c['alpha_rate_prev'], c['alpha_interval_prev_s']]
        initial = np.r_[(states/self.xscale).ravel(), a[1:]/self.uscale, slack]
        start = time.perf_counter()
        try:
            r = b['solver'](x0=np.clip(initial, b['lbw'], b['ubw']), p=p,
                            lbx=b['lbw'], ubx=b['ubw'], lbg=b['lbg'], ubg=b['ubg'])
            stat = b['solver'].stats(); v = np.asarray(r['x']).ravel(); gv = np.asarray(r['g']).ravel()
            violation = float(max(np.max(np.maximum(b['lbg']-gv, 0)+np.maximum(gv-b['ubg'], 0)),
                                  np.max(np.maximum(b['lbw']-v, 0)+np.maximum(v-b['ubw'], 0))))
            terms = dict(zip(b['term_names'], map(float, np.asarray(b['terms'](v, p)).ravel())))
            e = v[b['slack_start']:]*float(self.traj['aero_target_tracking']['CL_scale'])
            terms['maximum_CL_slack'] = float(np.max(e))
            stages = dict(protocol=PROTOCOL, optimizer_calls=1, compiled=b['compiled'],
                          tracking_band=self.experiment['cl_tracking_band'], slack_times=b['slack_times'].tolist(),
                          slack_CL=e.tolist(), slack_required_for_safety_proven=False)
            ok = bool(stat['return_status'] in ACCEPTED_STATUS and np.isfinite(v).all()
                      and np.isfinite(violation) and violation <= self.cfg['acceptance_violation'])
            return TrackingPlan(ok, stat['return_status'], b['times'].copy(),
                np.r_[c['alpha_prev'], v[4*(n+1):b['slack_start']]*self.uscale],
                v[:4*(n+1)].reshape(n+1, 4)*self.xscale, time.perf_counter()-start,
                int(stat['iter_count']), violation, float(r['f']), terms, stages)
        except RuntimeError as exc:
            return TrackingPlan(False, 'SOLVER_EXCEPTION:'+str(exc)[-300:], b['times'].copy(), a, states,
                                time.perf_counter()-start, 0, float('inf'), None, {}, {'optimizer_calls': 1})
