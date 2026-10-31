"""True dual-OCP fallback: parameterized pre/post alpha OCPs with the trajectory objective.

The NLP is built once per remaining-horizon grid, not once per geometry pair.
Geometry coefficients and the prescribed eight-second blend are parameters.
Both OCPs optimize alpha only. Table values are never changed by the solver.
The oracle controller feeds the planned Mach (pre) and the first six half-second
alpha commands (post) into the same execution chain as the surrogate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time

import casadi as ca
import numpy as np

from .diagnostics import MARGIN_NAMES
from .surrogate import OracleRequired

ACCEPTED_STATUS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


@dataclass
class Plan:
    success: bool
    status: str
    alpha: np.ndarray
    x: np.ndarray
    times: np.ndarray
    solve_s: float
    iterations: int
    violation: float
    objective: float
    terms: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    dual: dict = field(default_factory=dict, repr=False)

    def summary(self):
        return {k: getattr(self, k) for k in
                ('success', 'status', 'solve_s', 'iterations', 'violation', 'objective', 'terms', 'timings')}


class TwoStageOCP:
    def __init__(self, mission, db, traj, config, plant):
        self.m, self.db, self.traj, self.cfg = mission, db, traj, config
        self.plant = plant
        sc = traj['solver']['scaling']
        self.xscale = np.array([float(sc[k]) for k in ('h', 'V', 'gamma', 's')])
        self.uscale = float(sc['alpha'])
        self.cache = {}

    def grid(self, remaining):
        """Fine steps up to fine_until_s, then coarse steps to the (capped) remaining horizon."""
        horizon = min(float(remaining), self.cfg['horizon_s'])
        fine = min(horizon, self.cfg['fine_until_s'])
        dts = [self.cfg['fine_dt_s']]*int(round(fine/self.cfg['fine_dt_s']))
        rest = horizon-sum(dts)
        while rest > 1e-8:
            dt = min(rest, self.cfg['coarse_dt_s'])
            dts.append(dt)
            rest -= dt
        return np.array(dts)

    def build(self, dts):
        backoff = float(self.cfg.get('constraint_backoff', 0.))
        if not np.isfinite(backoff) or not 0 <= backoff < .5:
            raise ValueError('constraint_backoff must be finite and in [0, .5)')
        mode = self.cfg.get('tracking_mode', 'instantaneous')
        boundary = bool(self.cfg.get('boundary_smoothing', False))
        horizon_weight = float(self.cfg.get('horizon_tracking_weight', 1.))
        curvature_weight = float(self.cfg.get('second_difference_weight', 0.))
        if not np.isfinite(horizon_weight) or horizon_weight < 0 or not np.isfinite(curvature_weight) \
                or curvature_weight < 0:
            raise ValueError('horizon_tracking_weight and second_difference_weight must be finite and nonnegative')
        key = (tuple(dts), backoff, mode, boundary, horizon_weight, curvature_weight)
        if key in self.cache:
            return self.cache[key]
        n = len(dts)
        nc = self.db.ncoef
        times = np.r_[0, np.cumsum(dts)]
        p = ca.MX.sym('p', 11+2*nc)
        x0 = p[:4]
        prev = p[4]
        cl = p[5]
        cd = p[6]
        track_cd = p[7]
        w0 = p[8]
        rate = p[9]
        c0 = p[10:10+nc]
        c1 = p[10+nc:10+2*nc]
        weight = p[-1]
        X = ca.MX.sym('X', 4, n+1)
        U = ca.MX.sym('U', n)
        x = ca.times(X, ca.repmat(ca.DM(self.xscale), 1, n+1))
        u = U*self.uscale
        gs = []
        lb = []
        ub = []

        def add(v, lo, hi):
            gs.append(ca.vec(v))
            lb.extend([lo]*v.numel())
            ub.extend([hi]*v.numel())

        add(X[:, 0]-x0/self.xscale, 0, 0)
        l = self.m.data['limits']
        o = self.traj['objective']
        tr = self.traj['aero_target_tracking']
        terms = {k: 0 for k in ('drag', 'heat', 'target', 'control_rate')}
        if curvature_weight > 0:
            terms['control_curvature'] = 0
        tracking_mode = mode
        # 'pointwise_horizon' (opt-in): the commanded CL is the setpoint over the WHOLE
        # prediction horizon, penalised pointwise at every RK stage with one stationary
        # weight. Head and tail of the plan then share one objective, so a re-solve 3 s
        # later has no incentive to undo the executed prefix (no replanning sawtooth).
        if tracking_mode not in ('instantaneous', 'mean_3s', 'pointwise_horizon'):
            raise ValueError(tracking_mode)
        horizon = float(times[-1])
        tracking_duration = min(float(times[-1]), self.m.data['decision_dt_s'])
        cl_integral = 0
        cd_integral = 0
        for i, dt in enumerate(dts):
            state = x[:, i]
            ns = 1 if dt <= .5+1e-9 else int(self.cfg['coarse_rk4_substeps'])
            h = dt/ns
            for j in range(ns):
                state, ss = self.plant.rk(state, u[i], w0, rate, c0, c1, h, float(times[i]+j*h))
                if times[i]+j*h < tracking_duration-1e-9:
                    cl_integral += h*(ss[8, 0]+2*ss[8, 1]+2*ss[8, 2]+ss[8, 3])/6
                    cd_integral += h*(ss[9, 0]+2*ss[9, 1]+2*ss[9, 2]+ss[9, 3])/6
                if tr['enabled'] and tracking_mode == 'pointwise_horizon':
                    e = ((ss[8, :4]-cl)/float(tr['CL_scale']))**2
                    terms['target'] += (horizon_weight*weight*float(tr['weight_CL'])
                                        * h*(e[0]+2*e[1]+2*e[2]+e[3])/6/horizon)
                # Every RK stage, not only shooting nodes, must remain in the supported domain.
                for axis, key2, scale in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
                    margin = backoff*(l[key2][1]-l[key2][0])/scale
                    add((ss[axis, :]-l[key2][0])/scale, margin, np.inf)
                    add((l[key2][1]-ss[axis, :])/scale, margin, np.inf)
                add(1-ss[4, :]/l['dynamic_pressure_pa_max'], backoff, np.inf)
                add(1-ss[5, :]/l['heat_proxy_wm2_max'], backoff, np.inf)
                add(1-ss[6, :]/l['load_factor_g_max'], backoff, np.inf)
                margin = backoff*(self.db.M[-1]-self.db.M[0])
                add(ss[7, :]-self.db.M[0], margin, np.inf)
                add(self.db.M[-1]-ss[7, :], margin, np.inf)
            add(X[:, i+1]-state/self.xscale, 0, 0)
            _, q = self.plant.fn(x[:, i], u[i], ca.fmin(1, w0+rate*float(times[i])), c0, c1)
            terms['drag'] += q[6]*x[1, i]*dt/float(o['reference']['energy_ref'])
            terms['heat'] += q[1]*dt/float(o['reference']['heat_load_ref'])
            if times[i] < self.m.data['decision_dt_s']-1e-9 and tr['enabled'] and tracking_mode == 'instantaneous':
                terms['target'] += (weight*float(tr['weight_CL'])*((q[4]-cl)/float(tr['CL_scale']))**2 +
                                    track_cd*float(tr['weight_CD'])*((q[5]-cd)/float(tr['CD_scale']))**2
                                    )*dt/self.m.data['decision_dt_s']
            # A new command is applied at the start of this interval, so its slew
            # allowance comes from the PREVIOUS interval (0.5 s at block entry).
            rdt = .5 if i == 0 else dts[i-1]
            add((u[i]-(prev if i == 0 else u[i-1]))/(l['alpha_rate_radps']*rdt), -1, 1)
            if i > 0:
                terms['control_rate'] += ((u[i]-u[i-1])/self.uscale)**2
            elif boundary:
                # Opt-in: the jump from the measured (last executed) alpha is costed
                # like every other command change instead of being free.
                terms['control_rate'] += ((u[0]-prev)/self.uscale)**2
            if curvature_weight > 0:
                before = prev if i == 0 else u[i-1]
                if i >= 1:
                    earlier = prev if i == 1 else u[i-2]
                    terms['control_curvature'] += curvature_weight*((u[i]-2*before+earlier)/self.uscale)**2
        cl_mean = cl_integral/tracking_duration
        cd_mean = cd_integral/tracking_duration
        if tr['enabled'] and tracking_mode == 'mean_3s':
            terms['target'] = (weight*float(tr['weight_CL'])*((cl_mean-cl)/float(tr['CL_scale']))**2 +
                               track_cd*float(tr['weight_CD'])*((cd_mean-cd)/float(tr['CD_scale']))**2)
        terminal = o['terminal']
        terms['terminal'] = 0
        for j, name in enumerate(('h', 'V', 'gamma', 's')):
            if terminal.get(name+'_target') is not None and terminal['weights'][name] > 0:
                terms['terminal'] += terminal['weights'][name]*((x[j, -1]-terminal[name+'_target']) /
                                                                terminal['scales'][name])**2
        terms['range'] = (x[3, -1]-x0[3])/float(self.traj['solver']['scaling']['s'])
        obj = (o['w_drag']*terms['drag']+o['w_heat']*terms['heat']+o['w_terminal']*terms['terminal'] +
               o['w_control_rate']*(terms['control_rate']+terms.get('control_curvature', 0))
               - o['w_range']*terms['range']+terms['target'])
        # Prediction-only diagnostics; these entries are not additional costs.
        terms['CL_first_interval_mean'] = cl_mean
        terms['CD_first_interval_mean'] = cd_mean
        v = ca.vertcat(ca.vec(X), U)
        g = ca.vertcat(*gs)
        opts = {'print_time': False, 'calc_lam_p': False, 'ipopt': dict(self.traj['solver']['ipopt'])}
        opts['ipopt']['bound_relax_factor'] = 0.0
        opts['ipopt'].update(warm_start_init_point='yes', warm_start_bound_push=1e-7,
                             warm_start_mult_bound_push=1e-7, warm_start_slack_bound_push=1e-7)
        solver = ca.nlpsol(f'ocp_{n}', 'ipopt', {'x': v, 'p': p, 'f': obj, 'g': g}, opts)
        lo = np.array([l['altitude_m'][0], l['velocity_mps'][0], l['gamma_rad'][0], -np.inf])/self.xscale
        hi = np.array([l['altitude_m'][1], l['velocity_mps'][1], l['gamma_rad'][1], np.inf])/self.xscale
        b = {'solver': solver, 'lbg': np.array(lb), 'ubg': np.array(ub),
             'lbw': np.r_[np.tile(lo, n+1), np.full(n, self.db.A[0]/self.uscale)],
             'ubw': np.r_[np.tile(hi, n+1), np.full(n, self.db.A[-1]/self.uscale)], 'times': times,
             'terms': ca.Function(f'terms_{n}', [v, p], [ca.vertcat(*terms.values())]), 'term_names': list(terms)}
        self.cache[key] = b
        return b

    def solve(self, x, alpha_prev, source, target, w0, rate, cl_des, cd_des=None, guess=None, remaining=1200,
              warm=None, tracking_weight_multiplier=1.):
        if not np.isfinite(tracking_weight_multiplier) or tracking_weight_multiplier < 0:
            raise ValueError('Invalid CL tracking weight')
        dts = self.grid(remaining)
        b = self.build(dts)
        n = len(dts)
        ug = np.full(n, alpha_prev) if guess is None else np.asarray(guess, float).copy()
        if len(ug) != n:
            ug = np.full(n, float(ug[0]))
        prev = alpha_prev
        for i in range(n):
            r = self.m.data['limits']['alpha_rate_radps']*(.5 if i == 0 else dts[i-1])
            ug[i] = prev = np.clip(ug[i], max(self.db.A[0], prev-r), min(self.db.A[-1], prev+r))
        # A dynamically consistent warm start; candidate iterates may be infeasible.
        xs = [np.asarray(x, float)]
        time0 = 0
        for a, dt in zip(ug, dts):
            ns = 1 if dt <= .5+1e-9 else int(self.cfg['coarse_rk4_substeps'])
            y = xs[-1]
            for j in range(ns):
                y, _ = self.plant.rk(y, a, w0, rate, self.db.parameters[source], self.db.parameters[target],
                                     dt/ns, time0+j*dt/ns)
            xs.append(np.asarray(y).ravel())
            time0 += dt
        xg = np.asarray(xs)
        if not np.isfinite(xg).all():
            xg = np.tile(x, (n+1, 1))
        v0 = np.r_[(xg/self.xscale).ravel(), ug/self.uscale]
        p = np.r_[x, alpha_prev, cl_des, 0 if cd_des is None else cd_des, float(cd_des is not None), w0, rate,
                  self.db.parameters[source], self.db.parameters[target], tracking_weight_multiplier]
        start = time.perf_counter()
        try:
            dual = {}
            if warm is not None and warm.success and len(warm.alpha) == n and warm.dual:
                dual = {'lam_x0': warm.dual['x'], 'lam_g0': warm.dual['g']}
            r = b['solver'](x0=np.clip(v0, b['lbw'], b['ubw']), p=p, lbx=b['lbw'], ubx=b['ubw'],
                            lbg=b['lbg'], ubg=b['ubg'], **dual)
            stat = b['solver'].stats()
            v = np.asarray(r['x']).ravel()
            gv = np.asarray(r['g']).ravel()
            viol = float(max(np.max(np.maximum(b['lbg']-gv, 0)+np.maximum(gv-b['ubg'], 0)),
                             np.max(np.maximum(b['lbw']-v, 0)+np.maximum(v-b['ubw'], 0))))
            status = stat['return_status']
            ok = status in ACCEPTED_STATUS and np.isfinite(v).all() and viol <= self.cfg['acceptance_violation']
            tv = np.asarray(b['terms'](v, p)).ravel()
            # Remove only floating-point roundoff at the exact database endpoints.
            alpha = np.clip(v[4*(n+1):]*self.uscale, self.db.A[0], self.db.A[-1])
            timing = {k: float(val) for k, val in stat.items() if k.startswith('t_wall_')}
            return Plan(bool(ok), status, alpha, v[:4*(n+1)].reshape(n+1, 4)*self.xscale,
                        b['times'], time.perf_counter()-start, int(stat['iter_count']), viol, float(r['f']),
                        dict(zip(b['term_names'], map(float, tv))), timing,
                        {'x': np.asarray(r['lam_x']).ravel(), 'g': np.asarray(r['lam_g']).ravel()})
        except RuntimeError as e:
            return Plan(False, 'SOLVER_EXCEPTION:'+str(e)[-160:], ug, xg, b['times'], time.perf_counter()-start,
                        0, np.inf, np.inf)


def audit_full_plan(ocp, plan, context):
    """Mission limits, control slew and independent .1/.5 s re-integration of the full plan."""
    u = np.asarray(plan.alpha, float)
    tt = np.asarray(plan.times, float)
    lim = ocp.m.data['limits']
    lo = max(ocp.db.A[0], lim['alpha_rad'][0])
    hi = min(ocp.db.A[-1], lim['alpha_rad'][1])
    shape_ok = bool(u.ndim == 1 and tt.shape == (len(u)+1,) and len(u) >= 6
                    and np.isfinite(u).all() and np.isfinite(tt).all() and tt[0] == 0 and np.all(np.diff(tt) > 0))
    controls = bool(shape_ok and np.all(u >= lo-1e-9) and np.all(u <= hi+1e-9)
                    and np.all(abs(np.diff(np.r_[context['alpha_prev'], u])) <=
                               lim['alpha_rate_radps']*np.r_[.5, np.diff(tt)[:-1]]+1e-7))
    result = dict(valid=False, control_valid=controls, solver_accepted=bool(plan.success), grids={})
    arrays = {}
    if not shape_ok:
        return result, arrays
    for dt in (.1, .5):
        r = ocp.plant.integrate(context['x'], u, tt, context['source'], context['target'],
                                context['fraction'], context['rate'], dt)
        mm = ocp.plant.normalized_margins(r['samples'])
        finite = np.isfinite(r['x']).all() and np.isfinite(r['samples']).all() and np.isfinite(mm).all()
        result['grids'][str(dt)] = dict(valid=bool(finite and controls and mm.min() >= -1e-7),
                                        min_margin=float(mm.min()) if finite else None,
                                        margins=dict(zip(MARGIN_NAMES, mm.min(axis=1).tolist())),
                                        end_state=r['x'][-1].tolist())
        arrays[str(dt)] = r
    result['valid'] = bool(plan.success and controls and all(r['valid'] for r in result['grids'].values()))
    return result, arrays


class OracleController:
    """Controller interface backed by the true pre/post OCPs (one fresh instance per takeover)."""

    def __init__(self, ocp, record, artifact_sink=None):
        self.ocp, self.record, self.sink = ocp, record, artifact_sink
        self.pre = None

    def predict(self, c, head):
        remaining = self.ocp.m.data['duration_s']-c['t']
        tick = time.perf_counter()
        self.record['solver_calls'][head] += 1
        plan = self.ocp.solve(np.asarray(c['x']), c['alpha_prev'], c['source'], c['target'],
                              c['fraction'], c['rate'], c['CL_des'], c['CD_des'], remaining=remaining,
                              guess=None if head == 'pre' else self.pre.alpha,
                              warm=None if head == 'pre' else self.pre,
                              tracking_weight_multiplier=(1. if head == 'pre' else
                                                          self.ocp.cfg.get('tracking_weight_multiplier', 1.)))
        wall = time.perf_counter()-tick
        summary = dict(context=deepcopy(c), solver=plan.summary(), solve_and_build_wall_s=wall,
                       planned_horizon_s=float(plan.times[-1]), controls_rad=plan.alpha.tolist(),
                       control_times_s=plan.times.tolist(), optimized_states=plan.x.tolist())
        self.record['phases'][head] = summary
        if not plan.success:
            if self.sink:
                self.sink(self.record, head, plan, {})
            raise OracleRequired('TRUE_OCP_SOLVE_FAILED', c, head, summary)
        audit, arrays = audit_full_plan(self.ocp, plan, c)
        summary['full_plan_audit'] = audit
        if self.sink:
            self.sink(self.record, head, plan, arrays)
        if not audit['valid']:
            raise OracleRequired('TRUE_OCP_PLAN_AUDIT_REJECTED', c, head, summary)
        if head == 'pre':
            self.pre = plan
            planned = (np.asarray(c['x'])+plan.x[min(6, len(plan.x)-1)])/2
            output = np.array([planned[1]/float(self.ocp.plant.atmosphere.speed_of_sound(planned[0]))])
        else:
            # Fine OCP grid must exactly match physical half-second commands.
            if not np.allclose(plan.times[:7], np.arange(7)*.5, rtol=0, atol=1e-12):
                raise OracleRequired('TRUE_OCP_EXECUTION_GRID_MISMATCH', c, head, summary)
            output = plan.alpha[:6].copy()
        return output, dict(supported=True, support_basis='TRUE_OCP_DUAL_GRID_AUDIT',
                            solver_status=plan.status, solve_wall_s=wall, raw_output=output.tolist())
