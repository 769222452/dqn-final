"""One disturbed flight towards a target (range D*, speed V* at t = 1200 s).

Truth: point-mass RK4 (inner_dt_s = 0.5 s) with the scenario's density field and lift/drag biases; every RK
stage is checked against every path limit; a violation ends the flight (failure). Lift CL_true = k_L CL, drag
CD_true = k_D CD(M, CL, w).

Control: every decision (12 s) the policy picks an action = (CL level, morph level w). Every inner step the
shared protection loop applies CL = max(CL command, law(q_ref = 0.95 q_max)), with the dynamic-pressure law
CL_level (1 + kp (q/q_ref - 1)) - kg gamma, clipped to the configured CL range and to the CL available at this
Mach. The measured quantities are the dynamic pressure (ideal sensor: density ratio of the current point) and
the lift/drag biases estimated online (LoadEstimator); the density field ahead and the true biases are unknown.

Return (configs/task.yaml reward): hit (no violation, |dD| <= 5 km, |dV| <= 25 m/s) +0.10; otherwise
-min(e, 3) with e = sqrt((dD/100 km)^2 + (dV/100 m/s)^2); a violation -4.
"""
import copy
import math

from . import aero, config, dynamics
from .scenario import Field

_W = {}


def world(morph=True):
    key = (config.library_name(), bool(morph))
    if key not in _W:
        cfg = config.load()
        m = cfg['mission']
        _W[key] = dict(cfg=cfg, model=dynamics.Model(cfg), limits=dynamics.Limits(cfg), drag=aero.Drag(morph, cfg),
                       x0=list(m['initial_state']), T=m['duration_s'], dt=m['inner_dt_s'], decision=m['decision_s'],
                       qmax=m['limits']['dynamic_pressure_pa_max'], cl_levels=list(cfg['actions']['cl_levels']),
                       w_levels=list(cfg['actions']['morph_levels']) if morph else [0.],
                       cl_clip=cfg['actions']['cl_clip'], prot=cfg['protection'], hit=cfg['hit'],
                       reward=cfg['reward'])
    return _W[key]


def n_actions(morph=True):
    w = world(morph)
    return len(w['cl_levels'])*len(w['w_levels'])


def decode(a, morph=True):
    """Action index -> (CL level index, w level index)."""
    nw = len(world(morph)['w_levels'])
    return int(a)//nw, int(a) % nw


def encode(i_cl, i_w, morph=True):
    return int(i_cl)*len(world(morph)['w_levels'])+int(i_w)


def law_cl(model, x, q, r, qmax, kp, kg):
    return model.level_cl(x, q)*(1+kp*(q/(r*qmax)-1))-kg*x[2]


class LoadEstimator:
    """Online k_L, k_D from measured accelerations (ratio of integrals with exponential forgetting)."""

    def __init__(self, forget=.97, prior=2.):
        self.forget = forget
        self.num = [prior, prior]
        self.den = [prior, prior]

    def add(self, w, x0, x1, cl, cd_nom, k_rho_measured, dt):
        model = w['model']
        xm = [.5*(a+b) for a, b in zip(x0, x1)]
        _, q = model.rhs(xm, cl, cd_nom, k_rho_measured)
        h, v, g = xm[0], xm[1], xm[2]
        grav = model.atm.gravity(h)
        r = model.atm.R_earth+h
        l_hat = model.m*v*(x1[2]-x0[2])+model.m*(grav-v*v/r)*math.cos(g)*dt
        d_hat = -model.m*(x1[1]-x0[1])-model.m*grav*math.sin(g)*dt
        self.num[0] += l_hat
        self.num[1] += d_hat
        self.den[0] += q['q_dyn']*model.S*cl*dt
        self.den[1] += q['q_dyn']*model.S*cd_nom*dt

    def decay(self):
        self.num = [v*self.forget for v in self.num]
        self.den = [v*self.forget for v in self.den]

    @property
    def values(self):
        return self.num[0]/self.den[0], self.num[1]/self.den[1]


class TargetFlight:
    def __init__(self, sc, target, morph=True, trace=False):
        self.w = world(morph)
        self.morph = bool(morph)
        self.sc, self.target = sc, target
        self.field = Field(sc, self.w['cfg'])
        self.x = [self.w['x0'][0]+sc.dh, self.w['x0'][1]+sc.dv, self.w['x0'][2], self.w['x0'][3]]
        self.s0 = self.x[3]
        self.t = 0.
        self.done = self.failed = False
        self.reason = None
        self.G = None
        self.est = LoadEstimator()
        self.rho_prev = self.field(self.x[3])
        self.cl = None
        self.last_action = None
        self.protect_steps = 0
        self.steps = 0
        self.min_margin = 1e9
        self.wl = 0.
        self.info = {}
        self.trace = [] if trace else None      # per inner step: t, h, V, gamma, range_km, CL, w, q/q_max, protected

    # ------------------------------------------------------------------ measurements
    def k_rho(self):
        return self.field(self.x[3])

    def q(self):
        model = self.w['model']
        return .5*model.atm.density(self.x[0])*self.k_rho()*self.x[1]**2

    def range_km(self):
        return (self.x[3]-self.s0)/1e3

    def features(self, extra=False):
        x, tg = self.x, self.target
        base = [(x[0]-40000)/1e4, (x[1]-2400)/800, x[2]/.1, self.t/self.w['T'],
                (tg.range_km-self.range_km())/1000., (x[1]-tg.speed_mps)/500.]
        if not extra:
            return base
        kl, kd = self.est.values
        return base+[(self.k_rho()-1)/.1, (self.q()/self.w['qmax']-.9)/.1, (kl-1)/.05, (kd-1)/.05,
                     (self.rho_prev-1)/.1]

    # ------------------------------------------------------------------ dynamics
    def _cd(self, mach, cl, wl):
        return self.w['drag'].cd(mach, cl, wl)

    def _deriv(self, y, cl, wl):
        model = self.w['model']
        kr = self.field(y[3])
        a = model.atm.speed_of_sound(y[0])
        cd = self._cd(y[1]/a, cl, wl)
        return model.rhs(y, self.sc.k_lift*cl, self.sc.k_drag*cd, kr)

    def command_cl(self, cl_cmd):
        """Protection loop and clipping applied to a CL command at the current state."""
        w = self.w
        model = w['model']
        q = self.q()
        p = w['prot']
        guard = law_cl(model, self.x, q, p['q_ratio'], w['qmax'], p['kp'], p['kg'])
        cl = max(cl_cmd, guard)
        protected = guard > cl_cmd
        lo, hi = w['cl_clip']
        mach = self.x[1]/model.atm.speed_of_sound(self.x[0])
        clo, chi = w['drag'].cl_limits(mach)
        return min(max(cl, lo, clo), hi, chi), protected

    def advance(self, duration, command):
        """Fly `duration` seconds; command(env) -> (CL command, w) is evaluated every inner step."""
        w = self.w
        dt, T = w['dt'], w['T']
        lim = w['limits']
        model = w['model']
        end = min(self.t+duration, T)
        while not self.done and self.t < end-1e-9:
            cl_cmd, wl = command(self)
            cl, protected = self.command_cl(cl_cmd)
            self.protect_steps += protected
            self.steps += 1
            self.cl, self.wl = cl, wl
            x = self.x
            k_meas = self.k_rho()
            k1, q1 = self._deriv(x, cl, wl)
            x2 = [a+dt*b/2 for a, b in zip(x, k1)]
            k2, q2 = self._deriv(x2, cl, wl)
            x3 = [a+dt*b/2 for a, b in zip(x, k2)]
            k3, q3 = self._deriv(x3, cl, wl)
            x4 = [a+dt*b for a, b in zip(x, k3)]
            k4, q4 = self._deriv(x4, cl, wl)
            worst, which = 1e9, None
            for y, q in ((x, q1), (x2, q2), (x3, q3), (x4, q4)):
                m_ = lim.margin(y, q)
                if m_ < worst:
                    worst, which = m_, (y, q)
            self.min_margin = min(self.min_margin, worst)
            xn = [a+dt*(b1+2*b2+2*b3+b4)/6 for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
            mach = x[1]/model.atm.speed_of_sound(x[0])
            self.est.add(w, x, xn, cl, self._cd(mach, cl, wl), k_meas, dt)
            self.est.decay()
            self.x = xn
            self.t += dt
            if self.trace is not None:
                self.trace.append([self.t, xn[0], xn[1], xn[2], (xn[3]-self.s0)/1e3, cl, wl, self.q()/w['qmax'],
                                   int(protected)])
            if not all(math.isfinite(v) for v in xn) or worst < -1e-6:
                self.failed = self.done = True
                self.reason = 'VIOLATION:'+(lim.which(*which) if which else 'nonfinite')
            elif self.t >= T-1e-9:
                self.done = True
                self.reason = 'END'
        if self.done:
            self._finish()
        return self

    def step(self, action):
        """One decision with a fixed (CL level, w level) action."""
        if self.done:
            raise RuntimeError('flight finished')
        i_cl, i_w = decode(action, self.morph)
        cl_cmd, wl = self.w['cl_levels'][i_cl], self.w['w_levels'][i_w]
        self.rho_prev = self.k_rho()
        self.last_action = int(action)
        return self.advance(self.w['decision'], lambda env: (cl_cmd, wl))

    # ------------------------------------------------------------------ outcome
    def errors(self):
        return self.range_km()-self.target.range_km, self.x[1]-self.target.speed_mps

    def hit(self):
        dD, dV = self.errors()
        h = self.w['hit']
        return (self.done and not self.failed and abs(dD) <= h['range_km'] and abs(dV) <= h['speed_mps'])

    def _finish(self):
        r = self.w['reward']
        if self.failed:
            self.G = r['violation']
        elif self.hit():
            self.G = r['hit_bonus']
        else:
            dD, dV = self.errors()
            self.G = -min(math.hypot(dD/r['range_scale_km'], dV/r['speed_scale_mps']), r['miss_cap'])

    def fork(self):
        c = copy.copy(self)
        c.x = list(self.x)
        c.est = copy.copy(self.est)
        c.est.num, c.est.den = list(self.est.num), list(self.est.den)
        c.trace = None                   # forks (branches, predictions) are never traced
        c.info = dict(self.info)
        return c

    def summary(self):
        dD, dV = self.errors()
        return dict(level=self.sc.level, seed=self.sc.seed, target=self.target.name, target_range_km=self.target.range_km,
                    target_speed_mps=self.target.speed_mps, G=self.G, hit=bool(self.hit()), feasible=not self.failed,
                    reason=self.reason, t_end=self.t, range_km=self.range_km(), V_end=self.x[1], h_end=self.x[0],
                    dD_km=dD, dV_mps=dV, protect_fraction=self.protect_steps/max(self.steps, 1),
                    min_margin=self.min_margin, **self.info)
