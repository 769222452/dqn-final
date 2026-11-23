"""One disturbed flight towards a target (range D*, speed V* at t = 1200 s; v3, mission.end = speed: range D* when the speed
has come down to V*, at most mission.duration_s).

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
                       kg=float(cfg.get('execution', {}).get('kg', 0.)),
                       update_s=float(cfg.get('execution', {}).get('update_s', 3.)),
                       morph_exec=dict(cfg.get('execution', {}).get('morph', {})),
                       end=m.get('end', 'time'), plant=dict(cfg.get('plant', {}) or {}),
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
    def __init__(self, sc, target, morph=True, trace=False, kg=None):
        self.w = world(morph)
        self.kg = self.w['kg'] if kg is None else float(kg)     # flight-path-angle feedback on the CL target
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
        # v2 execution (configs/v2.yaml execution.morph.realistic): the plant flies library shapes only and needs
        # transition_s to move to a new one. Predictions (forks made by predictors) set realistic = False.
        self.realistic = bool(morph and self.w['morph_exec'].get('realistic', False))
        self.mstate = dict(act=0., frm=0., goal=0., t0=-1e9, n=0)
        # v3 (configs/v3.yaml plant): the plant differs from the model of the vehicle that predictions, the initial CL map
        # and the bias estimator use. Predictions (forks) set plant = False and fly the nominal model.
        self.plant = bool(self.w['plant'])
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
                     (self.rho_prev-1)/.1, tg.range_km/1000., tg.speed_mps/1000.]

    # ------------------------------------------------------------------ dynamics
    def _cd(self, mach, cl, wl):
        return self.w['drag'].cd(mach, cl, wl)

    def _deriv(self, y, cl, wl):
        model = self.w['model']
        kr = self.field(y[3])
        a = model.atm.speed_of_sound(y[0])
        mach = y[1]/a
        if self.plant:       # model mismatch: morph drag authority, extra drag, drag slope with Mach, lift error
            pm = self.w['plant']
            cd = self._cd(mach, cl, wl*pm.get('morph_effect', 1.))*(1+pm.get('extra_drag', 0.)
                                                                    +pm.get('drag_mach_slope', 0.)*(mach-6.))
            return model.rhs(y, self.sc.k_lift*(1+pm.get('lift_error', 0.))*cl, self.sc.k_drag*cd, kr)
        cd = self._cd(mach, cl, wl)
        return model.rhs(y, self.sc.k_lift*cl, self.sc.k_drag*cd, kr)

    def command_cl(self, cl_cmd):
        """Protection loop and clipping applied to a CL command at the current state."""
        w = self.w
        model = w['model']
        q = self.q()
        p = w['prot']
        if p.get('use_lift_estimate'):
            # v3: the equilibrium CL of the guard is corrected by the online lift estimate (with less lift than modelled the
            # proportional law would otherwise settle above the dynamic-pressure limit)
            kl = min(max(self.est.values[0], .7), 1.3)
            guard = model.level_cl(self.x, q)/kl*(1+p['kp']*(q/(p['q_ratio']*w['qmax'])-1))-p['kg']*self.x[2]
        else:
            guard = law_cl(model, self.x, q, p['q_ratio'], w['qmax'], p['kp'], p['kg'])
        cl = max(cl_cmd, guard)
        protected = guard > cl_cmd
        buf = p.get('floor_buffer_m', 0.)
        if buf > 0:                      # v2: altitude-floor guard, raise CL within buf of the altitude floor
            h0 = w['limits'].h[0]+buf
            if self.x[0] < h0:
                need = model.level_cl(self.x, q)*(1+p.get('floor_gain', 2.)*(h0-self.x[0])/buf)-p['kg']*self.x[2]
                if need > cl:
                    cl, protected = need, True
        lo, hi = w['cl_clip']
        mach = self.x[1]/model.atm.speed_of_sound(self.x[0])
        clo, chi = w['drag'].cl_limits(mach)
        return min(max(cl, lo, clo), hi, chi), protected

    def _morph_exec(self, wl, cl):
        """Executed morph level: the requested level snapped to a library shape (at the current Mach and CL), reached by
        a linear transition of transition_s; a new shape is accepted only after the current transition is complete."""
        ex, ms = self.w['morph_exec'], self.mstate
        tr = float(ex.get('transition_s', 8.))
        goal = wl
        if ex.get('snap', True):
            goal = self.w['drag'].snap(self.x[1]/self.w['model'].atm.speed_of_sound(self.x[0]), cl, wl)
        if self.t-ms['t0'] >= tr-1e-9 and abs(goal-ms['goal']) > 1e-9:
            ms.update(frm=ms['act'], goal=goal, t0=self.t)
            ms['n'] += 1
        ms['act'] = ms['goal'] if tr <= 0 else ms['frm']+(ms['goal']-ms['frm'])*min(1., (self.t-ms['t0'])/tr)
        return ms['act']

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
            if self.realistic:
                wl = self._morph_exec(wl, cl)
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
            elif w['end'] == 'speed' and xn[1] <= self.target.speed_mps:
                # v3: the flight ends when the speed reaches the target speed (state interpolated to the crossing)
                u = (x[1]-self.target.speed_mps)/max(x[1]-xn[1], 1e-12)
                self.x = [a+u*(b-a) for a, b in zip(x, xn)]
                self.t += (u-1)*dt
                self.done = True
                self.reason = 'SPEED'
            elif self.t >= T-1e-9:
                self.done = True
                self.reason = 'END' if w['end'] != 'speed' else 'TIMEOUT'
        if self.done:
            self._finish()
        return self

    def step(self, action):
        """One decision: CL target level and w level held for the decision (12 s).

        Execution (round 2): every update_s the executed CL target is refreshed to CL target - kg * gamma (the
        certified chain refreshes its CL target every 3 s); the protection loop then applies every inner step.
        With kg = 0 this is the round-1 execution (the CL target held)."""
        if self.done:
            raise RuntimeError('flight finished')
        i_cl, i_w = decode(action, self.morph)
        cl_cmd, wl = self.w['cl_levels'][i_cl], self.w['w_levels'][i_w]
        self.rho_prev = self.k_rho()
        self.last_action = int(action)
        held = dict(next=self.t, cl=cl_cmd)
        upd = self.w['update_s']

        def command(env):
            if env.t >= held['next']-1e-9:
                held['cl'] = cl_cmd-env.kg*env.x[2]
                held['next'] = env.t+upd
            return held['cl'], wl
        return self.advance(self.w['decision'], command)

    # ------------------------------------------------------------------ outcome
    def errors(self):
        return self.range_km()-self.target.range_km, self.x[1]-self.target.speed_mps

    def hit(self):
        dD, dV = self.errors()
        h = self.w['hit']
        return (self.done and not self.failed and abs(dD) <= h['range_km'] and abs(dV) <= h['speed_mps']
                and (self.w['end'] != 'speed' or self.reason == 'SPEED'))

    def _finish(self):
        r = self.w['reward']
        if self.failed:
            self.G = r['violation']
        elif self.hit():
            self.G = r['hit_bonus']
        elif r.get('shape') == 'log_tol':
            # study R: z = error in units of the hit tolerance; 0 at the tolerance, -0.5 per doubling, capped
            dD, dV = self.errors()
            h = self.w['hit']
            z = math.hypot(dD/h['range_km'], dV/h['speed_mps'])
            self.G = -min(.5*math.log2(max(z, 1.)), r['miss_cap'])
        else:
            dD, dV = self.errors()
            self.G = -min(math.hypot(dD/r['range_scale_km'], dV/r['speed_scale_mps']), r['miss_cap'])

    def fork(self):
        c = copy.copy(self)
        c.x = list(self.x)
        c.est = copy.copy(self.est)
        c.est.num, c.est.den = list(self.est.num), list(self.est.den)
        c.trace = None                   # forks (branches, predictions) are never traced
        c.mstate = dict(self.mstate)
        c.plant = self.plant
        c.info = dict(self.info)
        return c

    def summary(self):
        dD, dV = self.errors()
        return dict(level=self.sc.level, seed=self.sc.seed, target=self.target.name, target_range_km=self.target.range_km,
                    target_speed_mps=self.target.speed_mps, G=self.G, hit=bool(self.hit()), feasible=not self.failed,
                    reason=self.reason, t_end=self.t, range_km=self.range_km(), V_end=self.x[1], h_end=self.x[0],
                    dD_km=dD, dV_mps=dV, protect_fraction=self.protect_steps/max(self.steps, 1),
                    min_margin=self.min_margin, n_switches=self.mstate['n'], **self.info)
