"""Conservative 3-second CL screen, without an additional optimal-control solve.

Two distinct objects are logged: a necessary outer envelope for every legal
trajectory, and feasible extreme-control witnesses for the fixed current pair.
Finite sampling alone is NEVER used to declare a request unreachable. Projection
requires both an outer-envelope exclusion and an independently audited witness.
When a new generator target is still unknown, projection is deferred.
"""
from time import perf_counter

import numpy as np
from morphing_waverider.dynamics.atmosphere import _LAYERS
from morphing_waverider.utils.units import G0

from .diagnostics import interval_moments


class CLReachability:
    def __init__(self, plant):
        self.plant, self.db, self.m = plant, plant.db, plant.mission

    def _mach_box(self, x, tau):
        """Outer state tube from existing path limits, no guessed speed tolerance.

        ||(L,D)||/m <= n_max*G0; |sin gamma| uses the mission gamma limit.
        These bounds also enclose the positive-weight RK4 stage construction.
        Sound-speed extrema occur at layer endpoints (piecewise affine T).
        """
        lim = self.m.data['limits']
        atm = self.plant.atmosphere
        gmax = float(atm.gravity(lim['altitude_m'][0]))
        sg = max(abs(np.sin(lim['gamma_rad'])))
        av = lim['load_factor_g_max']*G0+gmax*sg
        vlo = max(lim['velocity_mps'][0], x[1]-tau*av)
        vhi = min(lim['velocity_mps'][1], x[1]+tau*gmax*sg)  # CD is positive.
        ag = (lim['load_factor_g_max']*G0+gmax)/max(vlo, 1.)+vhi/(atm.cfg.R_earth+lim['altitude_m'][0])
        glo = max(lim['gamma_rad'][0], x[2]-tau*ag)
        ghi = min(lim['gamma_rad'][1], x[2]+tau*ag)
        vertical = [v*np.sin(g) for v in (vlo, vhi) for g in (glo, ghi)]
        hlo = max(lim['altitude_m'][0], x[0]+tau*min(0., *vertical))
        hhi = min(lim['altitude_m'][1], x[0]+tau*max(0., *vertical))
        hs = [hlo, hhi]+[h for h, _ in _LAYERS if hlo < h < hhi]
        sounds = np.array([float(atm.speed_of_sound(h)) for h in hs])
        return vlo/max(sounds), vhi/min(sounds)

    def _coefficient_box(self, mach_lo, mach_hi, alo, ahi, indices):
        # Shape-preserving PCHIP cannot exceed the endpoint ordinates of each
        # occupied interval, in either interpolation direction. Include boundary
        # Mach evaluations and all interior nodes; bracket alpha outward by nodes.
        ms = [mach_lo, mach_hi]+self.db.M[(self.db.M > mach_lo) & (self.db.M < mach_hi)].tolist()
        jlo = max(0, np.searchsorted(self.db.A, alo, side='right')-1)
        jhi = min(len(self.db.A)-1, np.searchsorted(self.db.A, ahi, side='left'))
        # curves_slice is bit-identical to curves(M,indices)[:,jlo:jhi+1,0].
        values = np.stack([self.db.curves_slice(M, indices, jlo, jhi+1, 0) for M in ms])
        return values.min(axis=(0, 2)), values.max(axis=(0, 2))

    def check(self, x, alpha, z, t, raw, enabled=False):
        """Screen the requested CL over the next decision interval; CL_cmd is the (possibly projected) command."""
        begin = perf_counter()
        lim = self.m.data['limits']
        dt = self.m.data['plant_dt_s']
        duration = min(self.m.data['decision_dt_s'], self.m.data['duration_s']-t)
        result = {'CL_raw': float(raw), 'CL_cmd': float(raw), 'CL_reachable_min': None,
                  'CL_reachable_max': None, 'CL_outer_min': None, 'CL_outer_max': None,
                  'projection_applied': False, 'projection_reason': 'UNCERTAIN',
                  'projection_magnitude': 0., 'unreachable_certified': False,
                  'reachability_method': 'path_limit_outer_box_and_audited_alpha_extremes',
                  'reachable_interval_semantics': 'sampled_witness_extrema_not_complete_reachable_set'}

        def finish(reason):
            result['projection_reason'] = reason
            result['reachability_wall_time'] = perf_counter()-begin
            return result

        if duration <= 0 or not np.isfinite(x).all() or not np.isfinite(alpha):
            return finish('INVALID_STATE')
        try:
            current = self.plant.quantities(x, alpha, z.source, z.target, z.fraction(t))
        except ValueError:
            return finish('CURRENT_AERO_DOMAIN_UNCERTAIN')
        state_sample = np.r_[x, current][:, None]
        if np.min(self.plant.normalized_margins(state_sample)) < -1e-7:
            return finish('CURRENT_STATE_CONSTRAINT_VIOLATION')
        n = int(round(duration/dt))
        alow = max(self.db.A[0], lim['alpha_rad'][0])
        ahigh = min(self.db.A[-1], lim['alpha_rad'][1])
        low = np.maximum(alow, alpha-lim['alpha_rate_radps']*dt*np.arange(1, n+1))
        high = np.minimum(ahigh, alpha+lim['alpha_rate_radps']*dt*np.arange(1, n+1))
        active = z.active(t)
        ids = [z.source, z.target] if active else list(range(len(self.db.ids)))
        bounds = []
        for i in range(n):
            ml, mh = self._mach_box(np.asarray(x), dt*(i+1))
            if not self.db.M[0] <= ml <= mh <= self.db.M[-1]:
                return finish('FUTURE_MACH_DOMAIN_UNCERTAIN_NO_PROJECTION')
            lo, hi = self._coefficient_box(ml, mh, low[i], high[i], ids)
            # Use the plant's positive RK4 quadrature weights, including a
            # completion kink inside a slot. Continuous exact blend integration
            # differs at that kink and must not certify the discrete plant mean.
            f0 = z.fraction(t) if active else 0.
            ts = np.array([i*dt, i*dt+dt/2, (i+1)*dt])
            fractions = np.minimum(1., f0+ts/z.duration)
            fbar = float(fractions@np.array([1., 4., 1.])/6)
            if active:
                bounds.append(((1-fbar)*lo[0]+fbar*lo[1], (1-fbar)*hi[0]+fbar*hi[1]))
            else:
                # Any target may be chosen next, including staying at source.
                bounds.append(((1-fbar)*lo[z.source]+fbar*lo.min(),
                               (1-fbar)*hi[z.source]+fbar*hi.max()))
        outer = np.mean(bounds, axis=0)+np.array([-1e-12, 1e-12])
        result.update(CL_outer_min=float(outer[0]), CL_outer_max=float(outer[1]))
        outside = raw < outer[0] or raw > outer[1]
        result['unreachable_certified'] = bool(outside)
        if not active:
            return finish('GENERATOR_TARGET_PENDING_NO_PROJECTION' if outside else 'WITHIN_OUTER_ENVELOPE_TARGET_PENDING')
        times = np.arange(n+1)*dt
        witnesses = []
        for controls in (low, high):
            a = self.plant.integrate(x, controls, times, z.source, z.target, z.fraction(t),
                                     1/z.duration if active else 0., .1)
            # Independently enforce the same 0.5-second plant constraint audit.
            b = self.plant.integrate(x, controls, times, z.source, z.target, z.fraction(t),
                                     1/z.duration if active else 0., dt)
            if a['valid'] and b['valid']:
                witnesses.append(interval_moments(b, horizon=duration)['mean'])
        if witnesses:
            result.update(CL_reachable_min=float(min(witnesses)), CL_reachable_max=float(max(witnesses)))
        if not outside:
            return finish('WITHIN_OUTER_ENVELOPE_NOT_PROVEN_UNREACHABLE')
        if len(witnesses) != 2:
            return finish('OUTSIDE_OUTER_ENVELOPE_EXTREME_WITNESS_UNSAFE')
        if not enabled:
            return finish('CERTIFIED_UNREACHABLE_PROJECTION_DISABLED')
        command = min(witnesses, key=lambda v: abs(v-raw))
        result.update(CL_cmd=float(command), projection_applied=True, projection_magnitude=float(abs(command-raw)))
        return finish('OUTSIDE_DYNAMIC_OUTER_ENVELOPE_PROJECTED_TO_SAFE_WITNESS')
