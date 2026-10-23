"""Numeric and symbolic plant: PointMass3DOF of morphing_waverider with the aerodynamic database.

Every RK4 stage reports the constraint quantities, so audits and the OCP check the
path limits and the table support at all stages, not only at shooting nodes.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from . import codegen
from .atmosphere import FastAtmosphere


class SymbolicPair:
    """Aerodynamic provider for a geometry pair: linear blend of the endpoint coefficients."""

    def __init__(self, db, c0, c1):
        self.db, self.c0, self.c1 = db, c0, c1

    def coefficients(self, mach, alpha, eta):
        a = self.db.symbolic(mach, alpha, self.c0)
        b = self.db.symbolic(mach, alpha, self.c1)
        q = (1-eta[0])*a+eta[0]*b
        return q[0], q[1], None


class CompiledPlant:
    """Unchanged PointMass3DOF, with explicit table-domain gates on every RK stage.

    State x = (h, V, gamma, s); control alpha; the blend fraction of the active
    transition is an input, the two endpoint coefficient vectors are parameters.
    Quantity rows: q_dyn, heat_flux, n_load, mach, CL, CD, D, L.
    """

    def __init__(self, mission, db):
        self.mission, self.db = mission, db
        self.atmosphere = FastAtmosphere(mission.model(None).atm)
        x = ca.MX.sym('x', 4)
        u = ca.MX.sym('alpha')
        w = ca.MX.sym('fraction')
        c0 = ca.MX.sym('source', db.ncoef)
        c1 = ca.MX.sym('target', db.ncoef)
        model = mission.model(SymbolicPair(db, c0, c1))
        rhs, q = model.rhs_with_quantities(x, [u], [w])
        qv = ca.vertcat(*[q[k] for k in ('q_dyn', 'heat_flux', 'n_load', 'mach', 'CL', 'CD', 'D', 'L')])
        self.fn = ca.Function('plant', [x, u, w, c0, c1], [rhs, qv],
                              {'is_diff_in': [True, True, True, False, False]})
        # One RK4 step with the blend fraction w0 + rate*t (capped at 1); samples hold
        # state and quantities at the four stages and at the end state.
        w0 = ca.MX.sym('fraction0')
        rate = ca.MX.sym('rate')
        dt = ca.MX.sym('dt')
        tau = ca.MX.sym('tau')
        frac = lambda t: ca.fmin(1.0, w0+rate*t)
        k1, q1 = self.fn(x, u, frac(tau), c0, c1)
        x2 = x+dt*k1/2
        k2, q2 = self.fn(x2, u, frac(tau+dt/2), c0, c1)
        x3 = x+dt*k2/2
        k3, q3 = self.fn(x3, u, frac(tau+dt/2), c0, c1)
        x4 = x+dt*k3
        k4, q4 = self.fn(x4, u, frac(tau+dt), c0, c1)
        nxt = x+dt*(k1+2*k2+2*k3+k4)/6
        _, qe = self.fn(nxt, u, frac(tau+dt), c0, c1)
        samples = ca.horzcat(*[ca.vertcat(xx, qq) for xx, qq in ((x, q1), (x2, q2), (x3, q3), (x4, q4), (nxt, qe))])
        self.rk = ca.Function('rk_step', [x, u, w0, rate, c0, c1, dt, tau], [nxt, samples])
        self._maps = {}
        # Numeric point evaluations use a C-compiled copy of the same graph
        # (bit-identical); symbolic users keep the MX self.fn.
        self._quantities_fn = self.fn
        if codegen.enabled():
            # Compile the common 3 s audit grids now, outside any wall-clock budget.
            for n in (30, 6):
                self._maps[n] = codegen.compile_function(self.rk.mapaccum(f'rollout_{n}', n))
            self._quantities_fn = codegen.compile_function(self.fn)

    def _dm(self, index):
        """Cached exact DM copy of one geometry's aero coefficients (identical float64 values)."""
        cache = self.__dict__.setdefault('_param_dm', {})
        dm = cache.get(index)
        if dm is None:
            dm = cache[index] = ca.DM(np.asarray(self.db.parameters[index], dtype=float))
        return dm

    def quantities(self, x, alpha, source, target, fraction):
        """Quantity vector at one state; raises DomainExit outside the table support."""
        M = x[1]/float(self.atmosphere.speed_of_sound(x[0]))
        self.db.check(M, alpha)
        _, q = self._quantities_fn(x, alpha, fraction, self._dm(source), self._dm(target))
        return np.asarray(q).ravel()

    def normalized_margins(self, samples):
        """Scaled constraint margins (rows ordered as diagnostics.MARGIN_NAMES); >= 0 is feasible."""
        s = np.asarray(samples)
        l = self.mission.data['limits']
        cols = []
        for j, key, scale in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
            cols.extend([(s[j]-l[key][0])/scale, (l[key][1]-s[j])/scale])
        cols.extend([1-s[4]/l['dynamic_pressure_pa_max'], 1-s[5]/l['heat_proxy_wm2_max'],
                     1-s[6]/l['load_factor_g_max'], (s[7]-self.db.M[0]), (self.db.M[-1]-s[7])])
        return np.asarray(cols)

    def integrate(self, x, controls, times, source, target, w0, rate, dt):
        """Re-integrate a prescribed control plan; returns ALL RK-stage constraint evidence."""
        horizon = float(times[-1])
        n = int(round(horizon/dt))
        if n <= 0 or abs(n*dt-horizon) > 1e-8:
            raise ValueError('Integration grid must divide horizon')
        tt = np.arange(n)*dt
        ii = np.clip(np.searchsorted(times, tt+1e-9, side='right')-1, 0, len(controls)-1)
        uu = np.asarray(controls)[ii]
        if n not in self._maps:
            # Numeric rollouts only; symbolic users keep the MX self.rk graph.
            self._maps[n] = codegen.compile_function(self.rk.mapaccum(f'rollout_{n}', n))
        xx, samples = self._maps[n](x, uu[None], w0, rate, self._dm(source), self._dm(target), dt, tt[None])
        xx, samples = np.asarray(xx), np.asarray(samples)
        margins = self.normalized_margins(samples)
        finite = np.isfinite(xx).all() and np.isfinite(samples).all()
        return {'x': xx.T, 'samples': samples, 'min_margin': float(np.min(margins)) if finite else -np.inf,
                'valid': bool(finite and np.min(margins) >= -1e-7), 'dt': dt, 'times': tt,
                'alpha': uu, 'margins': np.min(margins, axis=1).tolist()}
