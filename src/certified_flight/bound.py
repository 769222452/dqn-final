"""Physical range bounds for the 1200 s mission (Phase 0 of experiments/improve).

relaxed_bound: an UPPER bound on the range of ANY admissible flight of this vehicle, i.e. of any CL
policy, morphing strategy, inner loop or surrogate. It is the full-horizon OCP

    max s(1200)  s.t.  exact point-mass dynamics (mission.model), every path limit at every RK stage
                       (h, V, gamma, q_dyn, heat flux, load factor, Mach support),
                       CD >= polar envelope(M, CL),

with CL itself as the control. The envelope is the lower convex hull of all (CL, CD) pairs of the 298
geometries over the alpha support. Two facts make this a relaxation:
- a morph blend is a convex combination of two endpoint (CL, CD) pairs at the same alpha, so it never
  lies below the hull;
- the alpha-rate limit, the 8 s morph time, the 3 s decision structure and every certification layer
  are dropped.
Heat flux depends only on (rho, V) and is exact. The hull is represented by its supporting lines at
fixed CL points (an epigraph, smooth for IPOPT). The lines at Mach knot i lower-bound the hull of all
samples on the two adjacent knot intervals, so their linear interpolation in Mach is a lower bound
everywhere. Validity limits (stated in every report):
- the hull is computed from samples (alpha step 0.05 deg, Mach step 0.02) and shifted down by
  ENVELOPE_SAFETY;
- the time grid is finite. Each solution is re-integrated on a 0.1 s grid and the difference reported.

fixed_geometry_reference: an ACHIEVABLE open-loop reference. The same horizon with the exact compiled
plant, one fixed geometry, alpha as control and the alpha-rate limit; its controls are re-integrated with
plant.integrate (0.1 s, all RK stages audited).
"""
from __future__ import annotations

import time

import casadi as ca
import numpy as np
from scipy.interpolate import PchipInterpolator

ENVELOPE_SAFETY = 2e-3          # relative downward shift of the sampled hull (sampling-gap allowance)
CL_POINTS = np.round(np.linspace(-0.01, 0.14, 31), 5)


def lower_hull(x, y):
    """Lower convex hull (monotone chain) of points; returns hull vertices sorted by x."""
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    hull = []
    for px, py in zip(x, y):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2-x1)*(py-y1)-(y2-y1)*(px-x1) <= 0:
                hull.pop()
            else:
                break
        hull.append((px, py))
    return np.asarray(hull)


def polar_samples(db, mach, alpha_step_deg=.05, indices=None):
    """(CL, CD) of every geometry on a dense alpha grid at one Mach (exact PCHIP of the database)."""
    alphas = np.clip(np.deg2rad(np.arange(np.rad2deg(db.A[0]), np.rad2deg(db.A[-1])+1e-9, alpha_step_deg)),
                     db.A[0], db.A[-1])
    y = PchipInterpolator(db.A, db.curves(mach, indices), axis=1, extrapolate=False)(alphas)   # [geom, alpha, 2]
    cl, cd = y[..., 0].ravel(), y[..., 1].ravel()
    ok = np.isfinite(cl) & np.isfinite(cd)
    return cl[ok], cd[ok]


def envelope(db, knots=None, mach_step=.02, indices=None):
    """Supporting lines (a_ij, b_ij) of the hull at CL_POINTS for every Mach knot i: CD >= a + b CL."""
    knots = db.M if knots is None else np.asarray(knots, float)
    lines = np.zeros((len(knots), len(CL_POINTS), 2))
    cl_range = np.zeros((len(knots), 2))
    for i in range(len(knots)):
        lo, hi = knots[max(0, i-1)], knots[min(len(knots)-1, i+1)]
        cl, cd = [], []
        for M in np.arange(lo, hi+1e-9, mach_step):
            a, b = polar_samples(db, min(M, db.M[-1]), indices=indices)
            cl.append(a)
            cd.append(b)
        cl, cd = np.concatenate(cl), np.concatenate(cd)
        hull = lower_hull(cl, cd*(1-ENVELOPE_SAFETY))
        cl_range[i] = cl.min(), cl.max()
        for j, c in enumerate(CL_POINTS):
            k = int(np.clip(np.searchsorted(hull[:, 0], c)-1, 0, len(hull)-2))
            (x1, y1), (x2, y2) = hull[k], hull[k+1]
            slope = (y2-y1)/(x2-x1)
            lines[i, j] = y1-slope*x1, slope
    return knots, lines, cl_range


class EnvelopeAero:
    """Coefficient provider for mission.model: the 'alpha' slot carries CL, eta carries CD."""

    def coefficients(self, mach, alpha, eta):
        return alpha, eta[0], None


def _grid(fine_until=240., fine_dt=1., coarse_dt=4., horizon=1200.):
    dts = [fine_dt]*int(round(fine_until/fine_dt))
    dts += [coarse_dt]*int(round((horizon-fine_until)/coarse_dt))
    return np.asarray(dts)


def _limits_constraints(model, x, q, lim, db, add):
    """Path limits on one stage (same normalisation as plant.normalized_margins)."""
    for j, key, scale in ((0, 'altitude_m', 1e4), (1, 'velocity_mps', 1e3), (2, 'gamma_rad', .1)):
        add((x[j]-lim[key][0])/scale, 0, ca.inf)
        add((lim[key][1]-x[j])/scale, 0, ca.inf)
    add(1-q['q_dyn']/lim['dynamic_pressure_pa_max'], 0, ca.inf)
    add(1-q['heat_flux']/lim['heat_proxy_wm2_max'], 0, ca.inf)
    add(1-q['n_load']/lim['load_factor_g_max'], 0, ca.inf)
    add(q['mach']-db.M[0], 0, ca.inf)
    add(db.M[-1]-q['mach'], 0, ca.inf)


def relaxed_bound(mission, db, x0, warm=None, dts=None, ipopt=None, indices=None):
    """Upper bound on s(1200)-s(0). warm: optional dict(t, x[n,4], CL[n]) to initialise."""
    started = time.perf_counter()
    knots, lines, cl_range = envelope(db, indices=indices)
    A = [ca.interpolant(f'a{j}', 'linear', [knots], lines[:, j, 0]) for j in range(len(CL_POINTS))]
    Bf = [ca.interpolant(f'b{j}', 'linear', [knots], lines[:, j, 1]) for j in range(len(CL_POINTS))]
    model = mission.model(EnvelopeAero())
    lim = mission.data['limits']
    dts = _grid() if dts is None else np.asarray(dts)
    n = len(dts)
    sx = np.array([1e4, 1e3, .1, 1e5])
    X = ca.MX.sym('X', 4, n+1)
    CL = ca.MX.sym('CL', n)
    CD = ca.MX.sym('CD', n)
    g, lb, ub = [], [], []

    def add(v, lo, hi):
        g.append(v)
        lb.append(lo)
        ub.append(hi)

    xs = lambda k: X[:, k]*sx
    add(X[:, 0]-np.asarray(x0)/sx, 0, 0)
    xx, uu, ee = ca.MX.sym('x', 4), ca.MX.sym('u'), ca.MX.sym('e')
    rhs, qty = model.rhs_with_quantities(xx, [uu], [ee])
    F = ca.Function('F', [xx, uu, ee], [rhs, ca.vertcat(qty['q_dyn'], qty['heat_flux'], qty['n_load'], qty['mach'])])
    for k, dt in enumerate(dts):
        state = xs(k)
        ns = 1 if dt <= 1+1e-9 else 2
        h = dt/ns
        for _ in range(ns):
            k1, q1 = F(state, CL[k], CD[k])
            x2 = state+h*k1/2
            k2, q2 = F(x2, CL[k], CD[k])
            x3 = state+h*k2/2
            k3, q3 = F(x3, CL[k], CD[k])
            x4 = state+h*k3
            k4, q4 = F(x4, CL[k], CD[k])
            stages = [(state, q1), (x2, q2), (x3, q3), (x4, q4)]
            state = state+h*(k1+2*k2+2*k3+k4)/6
            for xsg, qq in stages:
                _limits_constraints(model, xsg, dict(q_dyn=qq[0], heat_flux=qq[1], n_load=qq[2], mach=qq[3]),
                                    lim, db, add)
                for j in range(len(CL_POINTS)):
                    add((CD[k]-(A[j](qq[3])+Bf[j](qq[3])*CL[k]))*100, 0, ca.inf)
        add(X[:, k+1]-state/sx, 0, 0)
    _, qe = F(xs(n), CL[n-1], CD[n-1])
    _limits_constraints(model, xs(n), dict(q_dyn=qe[0], heat_flux=qe[1], n_load=qe[2], mach=qe[3]), lim, db, add)
    obj = -(X[3, n]-X[3, 0])
    v = ca.vertcat(ca.vec(X), CL, CD)
    G = ca.vertcat(*g)
    lbg = np.concatenate([np.full(e.numel(), lo) for e, lo in zip(g, lb)])
    ubg = np.concatenate([np.full(e.numel(), hi) for e, hi in zip(g, ub)])
    cl_lo, cl_hi = float(cl_range[:, 0].min()), float(cl_range[:, 1].max())
    lbx = np.r_[np.full(4*(n+1), -np.inf), np.full(n, cl_lo), np.full(n, 1e-4)]
    ubx = np.r_[np.full(4*(n+1), np.inf), np.full(n, cl_hi), np.full(n, .1)]
    opts = dict(print_time=False, ipopt=dict(max_iter=3000, tol=1e-8, print_level=0, mu_strategy='adaptive',
                                             **(ipopt or {})))
    solver = ca.nlpsol('bound', 'ipopt', dict(x=v, f=obj, g=G), opts)
    t = np.r_[0, np.cumsum(dts)]
    if warm is not None:
        xw = np.column_stack([np.interp(t, warm['t'], np.asarray(warm['x'])[:, j]) for j in range(4)])
        clw = np.interp(t[:-1], warm['t_ctrl'], warm['CL'])
    else:
        xw = np.tile(x0, (n+1, 1))
        clw = np.full(n, .027)
    v0 = np.r_[(xw/sx).ravel(), clw, np.full(n, .0018)]
    r = solver(x0=v0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    sol = np.asarray(r['x']).ravel()
    Xs = sol[:4*(n+1)].reshape(n+1, 4)*sx
    out = dict(status=solver.stats()['return_status'], iterations=int(solver.stats()['iter_count']),
               range_km=float((Xs[-1, 3]-Xs[0, 3])/1000), t=t.tolist(), x=Xs.tolist(),
               CL=sol[4*(n+1):4*(n+1)+n].tolist(), CD=sol[4*(n+1)+n:].tolist(), wall_s=time.perf_counter()-started,
               envelope=dict(safety=ENVELOPE_SAFETY, cl_points=CL_POINTS.tolist(), cl_range=[cl_lo, cl_hi],
                             geometries='all' if indices is None else [db.ids[i] for i in indices]))
    out['verification'] = verify_relaxed(mission, db, x0, out, knots, lines)
    return out


def hull_cd(knots, lines, mach, cl):
    i = np.clip(np.searchsorted(knots, mach)-1, 0, len(knots)-2)
    w = (mach-knots[i])/(knots[i+1]-knots[i])
    l = (1-w)*lines[i]+w*lines[i+1]
    return float(np.max(l[:, 0]+l[:, 1]*cl))


def verify_relaxed(mission, db, x0, sol, knots, lines, dt=.1):
    """Re-integrate the relaxed controls on a fine grid with CD = envelope(M, CL) (numeric RK4)."""
    model = mission.model(EnvelopeAero())
    lim = mission.data['limits']
    t = np.asarray(sol['t'])
    cl = np.asarray(sol['CL'])
    x = np.asarray(x0, float)
    worst = np.inf
    for k in range(len(cl)):
        steps = int(round((t[k+1]-t[k])/dt))
        for _ in range(steps):
            def fx(y):
                M = y[1]/float(model.atm.speed_of_sound(y[0]))
                cd = hull_cd(knots, lines, M, cl[k])
                r, q = model.rhs_with_quantities(y, [cl[k]], [cd])
                return np.asarray(r, float), q
            k1, q1 = fx(x)
            k2, q2 = fx(x+dt*k1/2)
            k3, q3 = fx(x+dt*k2/2)
            k4, q4 = fx(x+dt*k3)
            for q in (q1, q2, q3, q4):
                worst = min(worst, 1-float(q['q_dyn'])/lim['dynamic_pressure_pa_max'],
                            1-float(q['heat_flux'])/lim['heat_proxy_wm2_max'],
                            1-float(q['n_load'])/lim['load_factor_g_max'])
            x = x+dt*(k1+2*k2+2*k3+k4)/6
    return dict(dt=dt, range_km=float((x[3]-x0[3])/1000), end_state=x.tolist(), min_qhn_margin=float(worst),
                range_difference_km=float((x[3]-x0[3])/1000-sol['range_km']))
