"""Continuous piecewise-linear alpha on the unchanged translational RHS."""
import casadi as ca
import numpy as np

from certified_flight import codegen
from certified_flight.diagnostics import MARGIN_NAMES, interval_moments


def validate_plan(nodes, times):
    a, t = np.asarray(nodes, float), np.asarray(times, float)
    if (a.ndim != 1 or t.shape != a.shape or len(t) < 2 or t[0] != 0.
            or not np.isfinite(np.r_[a, t]).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Expected finite alpha nodes at strictly increasing times starting at zero')
    return a, t


def prefix(nodes, times, duration):
    a, t = validate_plan(nodes, times)
    if not 0 < duration <= t[-1]+1e-9:
        raise ValueError('Prefix duration outside plan')
    tt = np.r_[t[t < duration-1e-9], duration]
    return np.interp(tt, t, a), tt


class ContinuousPlant:
    def __init__(self, base):
        self.base, self.db, self.mission = base, base.db, base.mission
        self.atmosphere = base.atmosphere
        self.normalized_margins = base.normalized_margins
        x = ca.MX.sym('x', 4)
        a = ca.MX.sym('alpha_start')
        v = ca.MX.sym('alpha_rate')
        w, rate, dt, tau = (ca.MX.sym(k) for k in ('fraction', 'morph_rate', 'dt', 'tau'))
        c0, c1 = (ca.MX.sym(k, self.db.ncoef) for k in ('source', 'target'))
        f = lambda xx, aa, tt: base.fn(xx, aa, ca.fmin(1., w+rate*tt), c0, c1)
        k1, q1 = f(x, a, tau)
        x2 = x+dt*k1/2
        k2, q2 = f(x2, a+v*dt/2, tau+dt/2)
        x3 = x+dt*k2/2
        k3, q3 = f(x3, a+v*dt/2, tau+dt/2)
        x4 = x+dt*k3
        k4, q4 = f(x4, a+v*dt, tau+dt)
        end = x+dt*(k1+2*k2+2*k3+k4)/6
        _, qe = f(end, a+v*dt, tau+dt)
        samples = ca.horzcat(*[ca.vertcat(xx, qq) for xx, qq in
                              ((x, q1), (x2, q2), (x3, q3), (x4, q4), (end, qe))])
        self.rk = ca.Function('continuous_alpha_rk', [x, a, v, w, rate, c0, c1, dt, tau], [end, samples])
        self.maps = {}

    def integrate(self, x, nodes, times, source, target, fraction, rate, dt):
        a, t = validate_plan(nodes, times)
        if not np.isfinite(dt) or dt <= 0 or not 0 <= fraction <= 1 or rate not in (0., .125):
            raise ValueError('Invalid integration step or eight-second transition')
        n = int(round(t[-1]/dt))
        # Every alpha-slope breakpoint must be a grid point; never integrate
        # across one using the wrong slope.
        if n < 1 or np.max(abs(t/dt-np.round(t/dt))) > 1e-7:
            raise ValueError('Audit grid must divide every control interval')
        tt = np.arange(n)*dt
        i = np.clip(np.searchsorted(t, tt+1e-9, side='right')-1, 0, len(a)-2)
        slope = np.diff(a)/np.diff(t)
        aa = np.interp(tt, t, a)
        if n not in self.maps:
            self.maps[n] = codegen.compile_function(self.rk.mapaccum(f'continuous_rollout_{n}', n))
        xx, samples = self.maps[n](x, aa[None], slope[i][None], fraction, rate,
                                  self.base._dm(source), self.base._dm(target), dt, tt[None])
        xx, samples = np.asarray(xx).T, np.asarray(samples)
        mm = self.normalized_margins(samples)
        finite = np.isfinite(xx).all() and np.isfinite(samples).all() and np.isfinite(mm).all()
        return dict(x=xx, samples=samples, times=tt, dt=dt, alpha=aa,
                    alpha_end=np.interp(tt+dt, t, a), alpha_rate=slope[i],
                    valid=bool(finite and np.min(mm) >= -1e-7),
                    min_margin=float(np.min(mm)) if finite else None,
                    margins=np.min(mm, axis=1).tolist())

    def audit(self, context, nodes, times, cl_target):
        a, t = validate_plan(nodes, times)
        limits = self.mission.data['limits']
        lo, hi = max(self.db.A[0], limits['alpha_rad'][0]), min(self.db.A[-1], limits['alpha_rad'][1])
        slope = np.diff(a)/np.diff(t)
        continuous = bool(a[0] == context['alpha_prev'])
        control_valid = bool(continuous and np.all(a >= lo-1e-9) and np.all(a <= hi+1e-9)
                             and np.max(abs(slope)) <= limits['alpha_rate_radps']+1e-9)
        result = dict(valid=False, continuous_at_entry=continuous, control_valid=control_valid,
                      maximum_alpha_rate_degps=float(np.rad2deg(np.max(abs(slope)))), grids={})
        arrays = {}
        for dt in (.1, .5):
            r = self.integrate(context['x'], a, t, context['source'], context['target'],
                               context['fraction'], context['rate'], dt)
            stats = interval_moments(r, horizon=min(3., t[-1]), target=cl_target)
            n = int(round(min(3., t[-1])/dt))
            err = r['samples'][8, :5*n]-cl_target
            result['grids'][str(dt)] = dict(valid=bool(control_valid and r['valid']),
                min_margin=r['min_margin'], margins=dict(zip(MARGIN_NAMES, r['margins'])),
                CL_mean_first3=stats['mean'], CL_rmse_first3=stats['rmse'],
                CL_peak_error_first3=float(np.max(abs(err))), end_state=r['x'][-1].tolist())
            arrays[str(dt)] = r
        result['valid'] = bool(control_valid and all(v['valid'] for v in result['grids'].values()))
        return result, arrays
