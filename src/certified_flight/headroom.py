"""Headroom under uncertainty: how much can ANY feedback strategy gain over a well-tuned simple one?

The study works on the same exact point-mass dynamics and path limits as `bound.py`, with the polar
CD >= hull(M, CL) of all 298 geometries (no friction), but on a coarse, parametric time grid so that one
full-horizon solve takes seconds instead of minutes. All strategies share this one model, so only RELATIVE
numbers (gaps between strategies) are meaningful; the absolute range is validated against `relaxed_bound`
in `tests/test_headroom.py` and in experiments/headroom/README.md.

Truth model (what the vehicle actually flies in) differs from the planning model by
    rho_true(h, s) = k_rho(s) * rho_std(h)        density bias along the track (smooth random field)
    CL_true = k_L * CL_cmd,  CD_true = k_D * hull(M, CL_cmd)      constant aero biases

Strategies (src/certified_flight/headroom.py -> scripts/headroom_study.py):
    clairvoyant   full-horizon solve that knows the whole density field s -> k_rho(s) and (k_L, k_D):
                  an upper bound for ANY non-clairvoyant strategy, learned or not
    mpc           receding-horizon re-planning every few seconds from the measured state with parameters
                  estimated online (k_rho from the measured dynamic pressure, k_L/k_D from the measured
                  accelerations) and a constant back-off on the q / n / heat limits
    open loop     the nominal plan flown without feedback (shows why feedback matters at all)
Sensors are ideal (noise free), which makes MPC stronger than it would be in flight; the reported headroom
is therefore a conservative (low) estimate of what feedback can gain.
"""
from __future__ import annotations

import time

import casadi as ca
import numpy as np

from . import bound

CL_GRID = np.round(np.linspace(-0.01, 0.12, 53), 5)
CL_BOUNDS = (0.0, 0.10)
SX = np.array([1e4, 1e3, .1, 1e5])
STAGE_MARGINS = ('altitude_lo', 'altitude_hi', 'velocity_lo', 'velocity_hi', 'gamma_lo', 'gamma_hi', 'q', 'heat', 'n',
                 'mach_lo', 'mach_hi')


# --------------------------------------------------------------------------------------------- polar
def polar_table(db):
    """Hull CD on the (Mach knot, CL grid) mesh, from the same supporting lines as bound.relaxed_bound."""
    knots, lines, _ = bound.envelope(db)
    cd = np.max(lines[:, None, :, 0]+lines[:, None, :, 1]*CL_GRID[None, :, None], axis=2)
    return knots, cd


class PolarSurface:
    """Smooth (bspline) CD(M, CL) with exactly the hull values at the grid nodes."""

    def __init__(self, knots, cd):
        self.knots, self.cd = np.asarray(knots), np.asarray(cd)
        self.fn = ca.interpolant('polar', 'bspline', [self.knots, CL_GRID], self.cd.ravel(order='F'))

    def __call__(self, mach, cl):
        m = ca.fmin(ca.fmax(mach, self.knots[0]), self.knots[-1])
        c = ca.fmin(ca.fmax(cl, CL_GRID[0]), CL_GRID[-1])
        return self.fn(ca.vertcat(m, c))


class ScaledPolar:
    """Coefficient provider for mission.model: 'alpha' carries the commanded CL; true CL, CD are scaled."""

    def __init__(self, surface, k_lift, k_drag):
        self.surface, self.k_lift, self.k_drag = surface, k_lift, k_drag

    def coefficients(self, mach, cl_cmd, eta):
        return self.k_lift*cl_cmd, self.k_drag*self.surface(mach, cl_cmd), None


class ScaledAtmosphere:
    def __init__(self, base, k_rho):
        self.base, self.k_rho = base, k_rho

    def density(self, h):
        return self.k_rho*self.base.density(h)

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)


def dynamics(mission, surface):
    """F(x, c, p) -> (xdot, [q_dyn, heat, n, mach, CL, CD]) with p = (k_rho, k_L, k_D)."""
    x, c, p = ca.MX.sym('x', 4), ca.MX.sym('c'), ca.MX.sym('p', 3)
    model = mission.model(ScaledPolar(surface, p[1], p[2]))
    model.atm = ScaledAtmosphere(model.atm, p[0])
    rhs, q = model.rhs_with_quantities(x, [c], [ca.MX(0)])
    out = ca.vertcat(q['q_dyn'], q['heat_flux'], q['n_load'], q['mach'], q['CL'], q['CD'])
    return ca.Function('F', [x, c, p], [rhs, out]), model


def stage_margins(mission, mach_support, x, qty, back=1.0):
    """The path limits at one stage, normalised like plant.normalized_margins; >= 0 is feasible."""
    lim = mission.data['limits']
    return [(x[0]-lim['altitude_m'][0])/1e4, (lim['altitude_m'][1]-x[0])/1e4,
            (x[1]-lim['velocity_mps'][0])/1e3, (lim['velocity_mps'][1]-x[1])/1e3,
            (x[2]-lim['gamma_rad'][0])/.1, (lim['gamma_rad'][1]-x[2])/.1,
            1-qty[0]/(lim['dynamic_pressure_pa_max']*back), 1-qty[1]/(lim['heat_proxy_wm2_max']*back),
            1-qty[2]/(lim['load_factor_g_max']*back), qty[3]-mach_support[0], mach_support[1]-qty[3]]


# ------------------------------------------------------------------------------------------ planner
class Planner:
    """Full-horizon OCP (multiple shooting, RK4, every stage constrained) on a parametric time grid.

    The grid has n nodes whose steps are fixed fractions of the remaining time T, finer early (the pull-out)
    and coarser late, so the same NLP is re-solved for every receding-horizon plan (parameters change only).

    Density forecast inside the plan, kind='param':  k_rho(s) = exp(ln(k0) exp(-lam (s - s_now))), i.e.
    persistence for lam = 0 and mean reversion to the standard atmosphere for lam > 0.
    kind='profile': k_rho(s) is the given field (clairvoyant plan).
    """
    SPLIT = ((15, .10), (22, .30), (23, .60))
    IPOPT = dict(warm_start_init_point='yes', warm_start_bound_push=1e-8, warm_start_mult_bound_push=1e-8, mu_init=1e-5)
    BASE_BACKOFF = .998          # discretisation allowance: the nominal open-loop plan must stay feasible in the 0.5 s truth

    def __init__(self, mission, surface, mach_support, kind='param', profile=None, split=None, smooth=1e-4, ipopt=None):
        self.mission, self.kind = mission, kind
        self.F, _ = dynamics(mission, surface)
        split = split or self.SPLIT
        self.fractions = np.concatenate([np.full(k, f/k) for k, f in split])
        n = self.n = len(self.fractions)
        P = ca.MX.sym('P', 10)                  # x0(4), T, back, k0, lam, k_L, k_D
        X, C = ca.MX.sym('X', 4, n+1), ca.MX.sym('C', n)
        if kind == 'profile':
            s_grid, k_grid = profile
            kfun = ca.interpolant('field', 'linear', [np.asarray(s_grid)], np.asarray(k_grid))
            density = lambda s: kfun(ca.fmin(ca.fmax(s, s_grid[0]), s_grid[-1]))
        else:
            density = lambda s: ca.exp(ca.log(P[6])*ca.exp(-P[7]*(s-P[3])))
        g, lb, ub = [X[:, 0]-P[0:4]/SX], [np.zeros(4)], [np.zeros(4)]

        def add(v, lo, hi):
            v = ca.vertcat(v)
            g.append(v)
            lb.append(np.full(v.numel(), lo))
            ub.append(np.full(v.numel(), hi))

        def evaluate(state, c):
            return self.F(state, c, ca.vertcat(density(state[3]), P[8], P[9]))

        def limits(state, qty):
            add(ca.vertcat(*stage_margins(mission, mach_support, state, qty, P[5])), 0, ca.inf)

        for k in range(n):
            dt = self.fractions[k]*P[4]
            s0 = X[:, k]*SX
            k1, q1 = evaluate(s0, C[k])
            x2 = s0+dt*k1/2
            k2, q2 = evaluate(x2, C[k])
            x3 = s0+dt*k2/2
            k3, q3 = evaluate(x3, C[k])
            x4 = s0+dt*k3
            k4, q4 = evaluate(x4, C[k])
            for st, qq in ((s0, q1), (x2, q2), (x3, q3), (x4, q4)):
                limits(st, qq)
            add(X[:, k+1]-(s0+dt*(k1+2*k2+2*k3+k4)/6)/SX, 0, 0)
        _, qe = evaluate(X[:, n]*SX, C[n-1])
        limits(X[:, n]*SX, qe)
        obj = -X[3, n]*SX[3]/1e5+smooth*ca.sumsqr((C[1:]-C[:-1])/.01)
        options = dict(max_iter=400, tol=1e-7, print_level=0, mu_strategy='adaptive')
        options.update(self.IPOPT if ipopt is None else ipopt)
        opts = dict(print_time=False, ipopt=options)
        self.solver = ca.nlpsol('plan', 'ipopt', dict(x=ca.vertcat(ca.vec(X), C), f=obj, g=ca.vertcat(*g), p=P), opts)
        self.lbg, self.ubg = np.concatenate(lb), np.concatenate(ub)
        self.lbx = np.r_[np.full(4*(n+1), -np.inf), np.full(n, CL_BOUNDS[0])]
        self.ubx = np.r_[np.full(4*(n+1), np.inf), np.full(n, CL_BOUNDS[1])]
        self.last = None
        self.mach_support = mach_support

    def times(self, T):
        return np.r_[0, np.cumsum(self.fractions*T)]

    def _guess(self, x0, T, t_abs):
        """Warm start: the previous plan mapped onto the new grid, else a gliding flight at CL 0.026."""
        t = t_abs+self.times(T)
        if self.last is not None:
            tl, xl, cl = self.last
            tt = np.clip(t, tl[0], tl[-1])
            xs = np.column_stack([np.interp(tt, tl, xl[:, j]) for j in range(4)])
            cc = np.interp(0.5*(tt[:-1]+tt[1:]), 0.5*(tl[:-1]+tl[1:]), cl)
            xs[0] = x0
            return xs, cc
        xs, x = [np.asarray(x0, float)], np.asarray(x0, float)
        cc = np.full(self.n, .026)
        for k in range(self.n):
            dt, sub = self.fractions[k]*T, 4
            for _ in range(sub):
                h = dt/sub
                f = lambda y: np.asarray(self.F(y, cc[k], [1., 1., 1.])[0]).ravel()
                a = f(x)
                b = f(x+h*a/2)
                c2 = f(x+h*b/2)
                d = f(x+h*c2)
                x = x+h*(a+2*b+2*c2+d)/6
            xs.append(x)
        return np.asarray(xs), cc

    def plan(self, x0, T, t_abs=0., back=1., k0=1., lam=0., k_lift=1., k_drag=1.):
        started = time.perf_counter()
        xs, cc = self._guess(x0, T, t_abs)
        v0 = np.r_[(xs/SX).ravel(), cc]
        p = np.r_[np.asarray(x0, float), T, back, k0, lam, k_lift, k_drag]
        r = self.solver(x0=v0, p=p, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        stats = self.solver.stats()
        sol = np.asarray(r['x']).ravel()
        n = self.n
        x_plan = sol[:4*(n+1)].reshape(n+1, 4)*SX
        c_plan = sol[4*(n+1):]
        ok = stats['return_status'] in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
        out = dict(ok=ok, status=stats['return_status'], iterations=int(stats['iter_count']), t=t_abs+self.times(T),
                   x=x_plan, c=c_plan, range_km=float((x_plan[-1, 3]-x_plan[0, 3])/1000),
                   wall_s=time.perf_counter()-started)
        if ok:
            self.last = (out['t'], x_plan, c_plan)
        return out


# --------------------------------------------------------------------------------------- the truth
def density_field(sigma, corr_len_m, seed, length_m=3.2e6, step_m=5e3):
    """Mean-one multiplicative density bias along the track: exp(sigma z - sigma^2/2), z a unit AR(1) process."""
    rng = np.random.default_rng(seed)
    s = np.arange(0, length_m+step_m, step_m)
    a = np.exp(-step_m/corr_len_m)
    z = np.empty(len(s))
    z[0] = rng.standard_normal()
    for i in range(1, len(s)):
        z[i] = a*z[i-1]+np.sqrt(1-a*a)*rng.standard_normal()
    return s, np.exp(sigma*z-sigma**2/2)


class Truth:
    """The 'real' vehicle: 0.5 s RK4 on the perturbed model, every stage's limit margin recorded."""

    def __init__(self, mission, surface, mach_support, field, k_lift=1., k_drag=1., dt=.5):
        self.mission, self.mach_support, self.dt = mission, mach_support, dt
        self.F, self.model = dynamics(mission, surface)
        self.s_grid, self.k_grid = field
        self.k_lift, self.k_drag = k_lift, k_drag
        self.m, self.s_ref = mission.data['model']['mass_kg'], mission.data['model']['reference_area_m2']

    def k_rho(self, s):
        return float(np.interp(s, self.s_grid, self.k_grid))

    def eval(self, x, c, k_lift=None, k_drag=None, k_rho=None):
        p = [self.k_rho(x[3]) if k_rho is None else k_rho, self.k_lift if k_lift is None else k_lift,
             self.k_drag if k_drag is None else k_drag]
        r, q = self.F(x, c, p)
        return np.asarray(r).ravel(), np.asarray(q).ravel()

    def step(self, x, c):
        dt = self.dt
        k1, q1 = self.eval(x, c)
        x2 = x+dt*k1/2
        k2, q2 = self.eval(x2, c)
        x3 = x+dt*k2/2
        k3, q3 = self.eval(x3, c)
        x4 = x+dt*k3
        k4, q4 = self.eval(x4, c)
        margin = np.min([stage_margins(self.mission, self.mach_support, st, q) for st, q in
                         ((x, q1), (x2, q2), (x3, q3), (x4, q4))], axis=0)
        return x+dt*(k1+2*k2+2*k3+k4)/6, margin


class LoadEstimator:
    """Online k_L, k_D from the measured accelerations (ratio of integrals, exponential forgetting).

    k_rho is not estimated here: the dynamic pressure is measured directly (flush air data system)."""

    def __init__(self, truth, forget=.97, prior=2.):
        self.t, self.forget = truth, forget
        self.num = np.array([prior, prior])        # [L, D]: prior mass 'prior' at the nominal value 1
        self.den = np.array([prior, prior])

    def add(self, x0, x1, c, k_rho_measured):
        """One step of data; every term is evaluated at the mid-point state (the finite differences are centred)."""
        t = self.t
        xm = .5*(x0+x1)
        _, q = t.eval(xm, c, 1., 1., k_rho_measured)                  # nominal CL, CD at the measured density
        q_dyn, cl, cd = q[0], q[4], q[5]
        h, v, g = xm[0], xm[1], xm[2]
        grav = float(t.model.atm.gravity(h, spherical=True))
        r_e = t.model.atm.cfg.R_earth+h
        dt = t.dt
        l_hat = t.m*v*(x1[2]-x0[2])+t.m*(grav-v*v/r_e)*np.cos(g)*dt
        d_hat = -t.m*(x1[1]-x0[1])-t.m*grav*np.sin(g)*dt
        self.num += [l_hat, d_hat]
        self.den += [q_dyn*t.s_ref*cl*dt, q_dyn*t.s_ref*cd*dt]

    def decay(self):
        self.num *= self.forget
        self.den *= self.forget

    @property
    def values(self):
        return float(self.num[0]/self.den[0]), float(self.num[1]/self.den[1])


def fly(truth, x0, controller, horizon=1200., period=15.):
    """Closed loop. controller(t, x, estimator, truth) -> callable c(t') valid until t + period."""
    est = LoadEstimator(truth)
    x = np.asarray(x0, float)
    t, margin, log = 0., np.full(len(STAGE_MARGINS), np.inf), []
    wall = 0.
    while t < horizon-1e-9:
        started = time.perf_counter()
        schedule = controller(t, x, est, truth, horizon)
        wall += time.perf_counter()-started
        for _ in range(int(round(period/truth.dt))):
            if t >= horizon-1e-9:
                break
            c = schedule(t)
            k_meas = truth.k_rho(x[3])
            xn, m = truth.step(x, c)
            est.add(x, xn, c, k_meas)
            margin, x, t = np.minimum(margin, m), xn, t+truth.dt
        est.decay()
        log.append((t, x[0], x[1], x[3], c))
    return dict(range_km=float((x[3]-x0[3])/1000), min_margin=float(margin.min()), feasible=bool(margin.min() >= -1e-6),
                worst_by_limit=dict(zip(STAGE_MARGINS, margin.tolist())),
                end_state=x.tolist(), controller_wall_s=wall, log=np.asarray(log))


def schedule_from_plan(plan, t_abs_offset=0.):
    t, c = plan['t'], plan['c']
    return lambda tau: float(c[int(np.clip(np.searchsorted(t, tau+1e-9, side='right')-1, 0, len(c)-1))])


def mpc_controller(planner, back=1., lam=0.):
    def controller(t, x, est, truth, horizon):
        k_lift, k_drag = est.values
        plan = planner.plan(x, horizon-t, t_abs=t, back=back, k0=truth.k_rho(x[3]), lam=lam, k_lift=k_lift,
                            k_drag=k_drag)
        controller.plans = getattr(controller, 'plans', 0)+1
        controller.failures = getattr(controller, 'failures', 0)+(0 if plan['ok'] else 1)
        if not plan['ok'] and getattr(controller, 'last', None) is not None:
            return controller.last
        controller.last = schedule_from_plan(plan)
        return controller.last
    return controller


def open_loop_controller(plan):
    sched = schedule_from_plan(plan)
    return lambda t, x, est, truth, horizon: sched
