"""Longitudinal point-mass model of the original project, in plain floats (fast scalar evaluation).

State x = [h, V, gamma, s]. Non-rotating spherical Earth, zero bank, zero thrust:

    hdot = V sin(gamma)
    Vdot = -D/m - g sin(gamma)
    gdot = L/(m V) - (g/V - V/r) cos(gamma),     r = R_e + h
    sdot = V cos(gamma) R_e / r
    L = q S CL,  D = q S CD,  q = rho V^2 / 2,  g = g0 (R_e/r)^2
    heat flux = C sqrt(rho/R_n) V^3,  load factor n = sqrt(L^2 + D^2 + 1e-6)/(m g0)

Atmosphere: seven US-1976 layers to 86 km on geometric altitude, isothermal above, clamped to [0, 100] km.
Identical to morphing_waverider.dynamics (PointMass3DOF, Atmosphere) of the original repository;
tests/test_dynamics.py compares against values recorded from that code.
"""
import math

from . import config

_LAYERS = ((0.0, -0.0065), (11000.0, 0.0), (20000.0, 0.001), (32000.0, 0.0028), (47000.0, 0.0),
           (51000.0, -0.0028), (71000.0, -0.002))
_H_TOP = 86000.0


class Atmosphere:
    def __init__(self, cfg):
        self.c = c = dict(cfg)
        T, p = [c['T0']], [c['p0']]
        for i in range(1, len(_LAYERS)):
            hp, Lp = _LAYERS[i-1]
            hi = _LAYERS[i][0]
            Ti = T[-1]+Lp*(hi-hp)
            if abs(Lp) < 1e-12:
                pi = p[-1]*math.exp(-c['g0']*(hi-hp)/(c['R_gas']*T[-1]))
            else:
                pi = p[-1]*(Ti/T[-1])**(-c['g0']/(c['R_gas']*Lp))
            T.append(Ti)
            p.append(pi)
        self.Tb, self.pb = T, p
        self.R_earth, self.g0 = c['R_earth'], c['g0']

    def tp(self, h):
        c = self.c
        h = min(max(h, c['h_min_valid']), c['h_max_valid'])
        n = len(_LAYERS)
        i = n-1
        for j in range(n-1):
            if h < _LAYERS[j+1][0]:
                i = j
                break
        hb, L = _LAYERS[i]
        Tb, pb = self.Tb[i], self.pb[i]
        if i == n-1:
            he = min(h, _H_TOP)
            Ttop = Tb+L*(_H_TOP-hb)
            Ti = Tb+L*(he-hb)
            pi = pb*max(Ti/Tb, 1e-6)**(-c['g0']/(c['R_gas']*L))
            return Ti, pi*math.exp(-c['g0']*(h-he)/(c['R_gas']*Ttop))
        if abs(L) < 1e-12:
            return Tb, pb*math.exp(-c['g0']*(h-hb)/(c['R_gas']*Tb))
        Ti = Tb+L*(h-hb)
        return Ti, pb*max(Ti/Tb, 1e-6)**(-c['g0']/(c['R_gas']*L))

    def density(self, h):
        T, p = self.tp(h)
        return max(p/(self.c['R_gas']*T), self.c['rho_floor'])

    def speed_of_sound(self, h):
        return math.sqrt(self.c['gamma_gas']*self.c['R_gas']*self.tp(h)[0])

    def gravity(self, h):
        return self.g0*(self.R_earth/(self.R_earth+h))**2


class Model:
    """rhs(x, CL, CD, k_rho) -> (xdot, quantities); k_rho multiplies the standard density."""

    def __init__(self, cfg=None):
        cfg = cfg or config.load()
        self.atm = Atmosphere(cfg['atmosphere'])
        v = cfg['vehicle']
        self.m, self.S = v['mass_kg'], v['reference_area_m2']
        self.heat_C, self.heat_Rn = v['heat_C'], v['heat_R_n']

    def rhs(self, x, cl, cd, k_rho=1.):
        h, V, g_, _ = x
        atm = self.atm
        T, p = atm.tp(h)
        rho = max(p/(atm.c['R_gas']*T), atm.c['rho_floor'])*k_rho
        a = math.sqrt(atm.c['gamma_gas']*atm.c['R_gas']*T)
        r = atm.R_earth+h
        g = atm.g0*(atm.R_earth/r)**2
        q = .5*rho*V*V
        L = q*self.S*cl
        D = q*self.S*cd
        m = self.m
        sg, cg = math.sin(g_), math.cos(g_)
        Vs = max(V, 1.)
        xdot = (V*sg, -D/m-g*sg, L/(m*Vs)-(g/Vs-Vs/r)*cg, V*cg*atm.R_earth/r)
        qty = dict(rho=rho, a=a, mach=V/a, g=g, q_dyn=q, heat_flux=self.heat_C*math.sqrt(rho/self.heat_Rn)*V**3,
                   n_load=math.sqrt(L*L+D*D+1e-6)/(m*atm.g0))
        return xdot, qty

    def level_cl(self, x, q):
        """CL of a level (gamma-preserving) glide at the dynamic pressure q."""
        h, V = x[0], x[1]
        r = self.atm.R_earth+h
        return self.m*(self.atm.gravity(h)-V*V/r)/(q*self.S)


class Limits:
    """Normalised path-limit margins of the original screening code (negative = violated)."""

    def __init__(self, cfg=None):
        cfg = cfg or config.load()
        lim = cfg['mission']['limits']
        self.h, self.v, self.g = lim['altitude_m'], lim['velocity_mps'], lim['gamma_rad']
        self.q, self.heat, self.n = lim['dynamic_pressure_pa_max'], lim['heat_proxy_wm2_max'], lim['load_factor_g_max']
        self.mach = cfg['mission']['mach_support']

    def margin(self, x, qty):
        M = qty['mach']
        return min((x[0]-self.h[0])/1e4, (self.h[1]-x[0])/1e4, (x[1]-self.v[0])/1e3, (self.v[1]-x[1])/1e3,
                   (x[2]-self.g[0])/.1, (self.g[1]-x[2])/.1, 1-qty['q_dyn']/self.q, 1-qty['heat_flux']/self.heat,
                   1-qty['n_load']/self.n, M-self.mach[0], self.mach[1]-M)

    def which(self, x, qty):
        """Name of the most violated limit."""
        M = qty['mach']
        items = dict(h_lo=(x[0]-self.h[0])/1e4, h_hi=(self.h[1]-x[0])/1e4, V_lo=(x[1]-self.v[0])/1e3,
                     V_hi=(self.v[1]-x[1])/1e3, gamma_lo=(x[2]-self.g[0])/.1, gamma_hi=(self.g[1]-x[2])/.1,
                     q=1-qty['q_dyn']/self.q, heat=1-qty['heat_flux']/self.heat, load=1-qty['n_load']/self.n,
                     mach_lo=M-self.mach[0], mach_hi=self.mach[1]-M)
        return min(items, key=items.get)
