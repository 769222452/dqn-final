"""v4c truth plant: the motion the vehicle really flies, with disturbances and a model mismatch the certifier does not know.

Same compiled point-mass model and RK4 grid as CompiledPlant (plant.py); only the aerodynamic forces differ:

    CL_true = k_rho k_L (1 + lift_error) CL
    CD_true = k_rho k_D (1 + extra_drag + drag_mach_slope (Mach - 6)) (CD_ref + morph_effect (CD - CD_ref))

CL, CD: the blended table coefficients of the active geometry pair; CD_ref: the reference (minimum-drag, fixed) geometry
at the same Mach and alpha, so morph_effect scales the drag a shape adds over the reference shape. k_rho: density ratio of
the scenario's density field at the downrange position at the start of the 3 s step (correlation length 200 km, the
position moves about 6 km per step); k_L, k_D: the scenario's lift and drag biases. Scaling CL and CD by k_rho equals
scaling the density in L and D; the reported dynamic pressure and heat flux are scaled by k_rho and sqrt(k_rho).
Disturbance generator and mismatch values: target_rl.scenario and the configuration's plant section (same as the fast
model of target_rl), so a scenario seed means the same draws in both.
"""
from __future__ import annotations

import math

import casadi as ca
import numpy as np

from . import codegen

N_PARAMS = 5        # total lift factor, total drag factor, morph_effect, extra_drag, drag_mach_slope


class TruthPair:
    def __init__(self, db, c0, c1, cref, p):
        self.db, self.c0, self.c1, self.cref, self.p = db, c0, c1, cref, p

    def coefficients(self, mach, alpha, eta):
        a = self.db.symbolic(mach, alpha, self.c0)
        b = self.db.symbolic(mach, alpha, self.c1)
        r = self.db.symbolic(mach, alpha, self.cref)
        q = (1-eta[0])*a+eta[0]*b
        p = self.p
        cd = (r[1]+p[2]*(q[1]-r[1]))*(1+p[3]+p[4]*(mach-6.))
        return p[0]*q[0], p[1]*cd, None


class TruthPlant:
    def __init__(self, nominal, reference_geometry):
        self.nominal = nominal
        self.db, self.mission = nominal.db, nominal.mission
        self.ref = int(self.db.index[reference_geometry])
        db = self.db
        x = ca.MX.sym('x', 4)
        u = ca.MX.sym('alpha')
        w = ca.MX.sym('fraction')
        c0, c1, cr = (ca.MX.sym(n, db.ncoef) for n in ('source', 'target', 'reference'))
        p = ca.MX.sym('p', N_PARAMS)
        model = self.mission.model(TruthPair(db, c0, c1, cr, p))
        rhs, q = model.rhs_with_quantities(x, [u], [w])
        qv = ca.vertcat(*[q[k] for k in ('q_dyn', 'heat_flux', 'n_load', 'mach', 'CL', 'CD', 'D', 'L')])
        fn = ca.Function('truth', [x, u, w, c0, c1, cr, p], [rhs, qv])
        w0, rate, dt, tau = (ca.MX.sym(n) for n in ('fraction0', 'rate', 'dt', 'tau'))
        frac = lambda t: ca.fmin(1.0, w0+rate*t)
        k1, q1 = fn(x, u, frac(tau), c0, c1, cr, p)
        x2 = x+dt*k1/2
        k2, q2 = fn(x2, u, frac(tau+dt/2), c0, c1, cr, p)
        x3 = x+dt*k2/2
        k3, q3 = fn(x3, u, frac(tau+dt/2), c0, c1, cr, p)
        x4 = x+dt*k3
        k4, q4 = fn(x4, u, frac(tau+dt), c0, c1, cr, p)
        nxt = x+dt*(k1+2*k2+2*k3+k4)/6
        _, qe = fn(nxt, u, frac(tau+dt), c0, c1, cr, p)
        samples = ca.horzcat(*[ca.vertcat(xx, qq) for xx, qq in ((x, q1), (x2, q2), (x3, q3), (x4, q4), (nxt, qe))])
        self.rk = ca.Function('truth_rk', [x, u, w0, rate, c0, c1, cr, p, dt, tau], [nxt, samples])
        self._maps = {}
        self.set_scenario(None, None)

    def set_scenario(self, field, scenario, plant=None):
        """field: callable s [m] -> density ratio (None: 1); scenario: object with k_lift, k_drag (None: 1, 1);
        plant: dict morph_effect, extra_drag, drag_mach_slope, lift_error (None: no mismatch)."""
        self.field = field
        self.k_lift = 1. if scenario is None else float(scenario.k_lift)
        self.k_drag = 1. if scenario is None else float(scenario.k_drag)
        pm = dict(plant or {})
        self.mm = (float(pm.get('lift_error', 0.)), float(pm.get('morph_effect', 1.)), float(pm.get('extra_drag', 0.)),
                   float(pm.get('drag_mach_slope', 0.)))

    def k_rho(self, s):
        return 1. if self.field is None else float(self.field(float(s)))

    def integrate(self, x, controls, times, source, target, w0, rate, dt):
        """Same arguments and result layout as CompiledPlant.integrate; 'valid' means finite only (truth margins are
        reported in min_margin / margins, never used to refuse)."""
        horizon = float(times[-1])
        n = int(round(horizon/dt))
        if n <= 0 or abs(n*dt-horizon) > 1e-8:
            raise ValueError('Integration grid must divide horizon')
        tt = np.arange(n)*dt
        ii = np.clip(np.searchsorted(times, tt+1e-9, side='right')-1, 0, len(controls)-1)
        uu = np.asarray(controls)[ii]
        if n not in self._maps:
            self._maps[n] = codegen.compile_function(self.rk.mapaccum(f'truth_rollout_{n}', n))
        kr = self.k_rho(x[3])
        le, eff, extra, slope = self.mm
        p = ca.DM([kr*self.k_lift*(1+le), kr*self.k_drag, eff, extra, slope])
        nom = self.nominal
        xx, samples = self._maps[n](x, uu[None], w0, rate, nom._dm(source), nom._dm(target), nom._dm(self.ref), p, dt,
                                    tt[None])
        xx, samples = np.asarray(xx), np.asarray(samples).copy()
        samples[4] *= kr                  # dynamic pressure with the true density
        samples[5] *= math.sqrt(kr)       # heat-flux proxy ~ sqrt(rho)
        margins = nom.normalized_margins(samples)
        finite = bool(np.isfinite(xx).all() and np.isfinite(samples).all())
        return {'x': xx.T, 'samples': samples, 'min_margin': float(np.min(margins)) if finite else -np.inf,
                'valid': finite, 'dt': dt, 'times': tt, 'alpha': uu, 'margins': np.min(margins, axis=1).tolist(),
                'k_rho': kr, 'truth': True}
