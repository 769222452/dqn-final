"""Fresh continuous-control OCP labels for the isolated surrogate protocol.

The primary optimization tracks the requested CL pointwise over the WHOLE
planning horizon. Economics is optimized only within a declared allowance of
that optimum. This finite-horizon convention assumes the request stays constant;
it does not prove that a nonzero tracking error is physically unavoidable.
Old six-value zero-order-hold labels are deliberately incompatible.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import casadi as ca
import numpy as np

from certified_flight.ocp import TwoStageOCP
from certified_flight.diagnostics import interval_moments
from continuous_tracking.ocp import TrackingOCP
from continuous_tracking.compiled_solver import compile_solver

PROTOCOL = 'continuous_surrogate/1'
DEFAULT_TEACHER_CONFIG = {
    'protocol': PROTOCOL,
    'tracking': 'rk4_pointwise_full_horizon',
    'reference_prediction': 'hold_current_CL',
    'objective_priority': 'whole_horizon_tracking_then_economics',
    'tracking_optimality_relative_allowance': 0.01,
    'tracking_optimality_absolute_allowance': 1e-8,
    'planning_integration_max_dt_s': 0.5,
    'planning_normalized_margin': 1e-4,
    'compile_nlp': True,
    'normalization_time_s': 60.0,
    'rate_cost_reference_dt_s': 0.5,
}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class FullHorizonTeacher(TrackingOCP):
    """The existing continuous solver with a whole-horizon primary objective."""

    def __init__(self, mission, db, trajectory, physical, plant, config=None):
        TwoStageOCP.__init__(self, mission, db, trajectory, physical, plant)
        self.experiment = dict(DEFAULT_TEACHER_CONFIG)
        self.experiment.update(config or {})
        if (self.experiment['protocol'] != PROTOCOL
                or self.experiment['tracking'] != 'rk4_pointwise_full_horizon'
                or self.experiment['reference_prediction'] != 'hold_current_CL'
                or self.experiment['objective_priority'] != 'whole_horizon_tracking_then_economics'):
            raise ValueError('Incompatible continuous teacher protocol')
        for key in ('planning_integration_max_dt_s', 'normalization_time_s', 'rate_cost_reference_dt_s'):
            if not np.isfinite(self.experiment[key]) or self.experiment[key] <= 0:
                raise ValueError('Invalid teacher '+key)
        for key in ('tracking_optimality_relative_allowance', 'tracking_optimality_absolute_allowance',
                    'planning_normalized_margin'):
            if not np.isfinite(self.experiment[key]) or self.experiment[key] < 0:
                raise ValueError('Invalid teacher '+key)
        if self.experiment['planning_integration_max_dt_s'] > .5:
            raise ValueError('Planning integration may not be coarser than 0.5 s')

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
        primary = ca.nlpsol(f'whole_horizon_priority_{n}', 'ipopt', {'x': v, 'p': p, 'f': terms['target'], 'g': g}, opts)
        solver = ca.nlpsol(f'continuous_teacher_{n}', 'ipopt',
                          {'x': v, 'p': p, 'f': objective, 'g': ca.vertcat(g, terms['target']-tracking_budget)}, opts)
        raw_primary,raw_secondary=primary,solver
        compiled={}
        if self.experiment.get('compile_nlp',False) and times[-1]>=60.:
            primary,compiled['primary']=compile_solver(primary,f'teacher_compiled_primary_{n}',opts)
            solver,compiled['secondary']=compile_solver(solver,f'teacher_compiled_secondary_{n}',opts)
        lower = np.array([l['altitude_m'][0], l['velocity_mps'][0], l['gamma_rad'][0], -np.inf])/self.xscale
        upper = np.array([l['altitude_m'][1], l['velocity_mps'][1], l['gamma_rad'][1], np.inf])/self.xscale
        alo, ahi = max(self.db.A[0], l['alpha_rad'][0]), min(self.db.A[-1], l['alpha_rad'][1])
        b = dict(solver=solver, primary=primary, raw_primary=raw_primary, raw_secondary=raw_secondary,
                 compiled=compiled, times=times, lbg=np.array(lb), ubg=np.array(ub),
                 lbw=np.r_[np.tile(lower, n+1), np.full(n, alo/self.uscale)],
                 ubw=np.r_[np.tile(upper, n+1), np.full(n, ahi/self.uscale)],
                 terms=ca.Function(f'teacher_terms_{n}', [v, p], [ca.vertcat(*terms.values())]),
                 term_names=list(terms))
        self.cache[key] = b
        return b


    def solve_audited(self, context, head, output_dir=None, guess=None):
        """Return (serializable label record, plan); every label gets two audits.

        No optimization success flag is treated as a safety certificate. A
        rejected record has no y and cannot be passed to the model fitter.
        The complete trajectory is retained when output_dir is provided.
        """
        if head not in ('pre', 'post'):
            raise ValueError('Unknown teacher head')
        c = deepcopy(context)
        if c.get('protocol', PROTOCOL) != PROTOCOL or c.get('control_contract', PROTOCOL) != PROTOCOL:
            raise ValueError('Teacher input control protocol mismatch')
        if float(c['t']) >= float(self.m.data['duration_s']):
            raise ValueError('No remaining prediction horizon')
        plan = self.solve_context(c, guess)
        record = {
            'protocol': PROTOCOL,
            'teacher_contract': deepcopy(self.experiment),
            'head': head, 'context': json_value(c),
            'accepted': False,
            'label_provenance': 'FRESH_CONTINUOUS_REAL_OCP_AND_FULL_DUAL_GRID_AUDIT',
            'plan_summary': json_value(plan.summary()),
            'tracking_scope': 'whole_prediction_horizon',
            'future_reference_assumption': 'hold_current_CL',
            'tracking_deviation_physically_necessary_proven': False,
        }
        # Audit even a finite rejected optimizer candidate for diagnostics, but
        # do not accept a failed optimizer solely because its rollout is safe.
        audit = None
        arrays = {}
        if np.isfinite(plan.alpha_nodes).all():
            audit, arrays = self.plant.audit(c, plan.alpha_nodes, plan.times, c['CL_des'])
            for key, rollout in arrays.items():
                full = interval_moments(rollout, horizon=float(plan.times[-1]), target=c['CL_des'])
                audit['grids'][key].update(CL_mean_full=full['mean'], CL_rmse_full=full['rmse'])
            record['audit'] = json_value(audit)
        record['accepted'] = bool(plan.success and audit is not None and audit['valid'])
        if record['accepted']:
            if head == 'post':
                if plan.times[-1] < 3.-1e-9:
                    raise ValueError('Post label requires a full three-second decision')
                record['y'] = np.interp(np.arange(1, 7)*.5, plan.times, plan.alpha_nodes).tolist()
                record['output_times_s'] = (np.arange(1, 7)*.5).tolist()
                record['entry_alpha_fixed_rad'] = float(c['alpha_prev'])
            else:
                # Preserve the meaning of Pre: Mach at the arithmetic mean of
                # the actual initial state and the predicted state at 3 s.
                r = arrays['0.5']
                end_index = min(6, len(r['x']))-1
                middle = .5*(np.asarray(c['x'], float)+r['x'][end_index])
                record['y'] = [float(middle[1]/self.plant.atmosphere.speed_of_sound(middle[0]))]
                record['pre_mach_definition'] = 'mach_of_mean_initial_and_3s_state'
        else:
            record['reason'] = 'OCP_SOLVE_FAILED' if not plan.success else 'FULL_PLAN_AUDIT_REJECTED'
        if output_dir is not None:
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(folder/'plan.npz', times=plan.times,
                                alpha_nodes=plan.alpha_nodes, states=plan.states,
                                **{f'{k.replace(".", "p")}_{name}': value
                                   for k, rollout in arrays.items()
                                   for name, value in rollout.items()
                                   if isinstance(value, np.ndarray)})
            (folder/'label.json').write_text(
                json.dumps(json_value(record), ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                encoding='utf-8', newline='\n')
        return record, plan


def create(mission, db, trajectory, physical_ocp_config, continuous_plant, config=None):
    return FullHorizonTeacher(mission, db, trajectory, physical_ocp_config, continuous_plant, config)

