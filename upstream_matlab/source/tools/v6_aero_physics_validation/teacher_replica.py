"""Independent Python re-implementation of the formulas read in matlab/active (audit use only).

Nothing here is called by the project. It re-derives:
  * the prescribed shock surface of create_shock_functions.m and the leading-edge construction
    and shock-march direction of build_geometry_core.m (no method of characteristics);
  * the oblique-shock jump of qiujie_shock.m;
  * the Taylor-Maccoll weak attached cone of comptue_conicalfield.m / conical_flowfield.m;
  * the panel integration of offdesign_aero_v4.m (tangent cone + hypersonic expansion similarity).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat
from scipy.optimize import brentq, fsolve

G = 1.4
L = 5.0
C1 = C2 = 1.0331
AOA_NATIVE = -4.0
SREF = 6.931920786884801
LREF = 5.0


# ----------------------------------------------------------------------------- geometry side
def rotate_matrix(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    ang = np.arccos(np.clip(a @ b, -1, 1))
    ax = np.cross(a, b)
    if np.linalg.norm(ax) == 0:
        return np.eye(3)
    ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + (1 - np.cos(ang)) * K @ K + np.sin(ang) * K


class Shock:
    """Prescribed shock surface (create_shock_functions.m lines 18-20) and its rotation."""

    def __init__(self, a1, a2, b1, b2, c1=C1, c2=C2, aoa=AOA_NATIVE):
        self.ta1, self.ta2, self.tb1, self.tb2 = np.tan(np.radians([a1, a2, b1, b2]))
        self.c1, self.c2 = c1, c2
        self.V = np.array([np.cos(np.radians(aoa)), 0.0, np.sin(np.radians(aoa))])
        self.rat = rotate_matrix(self.V, [1, 0, 0])

    def Ry(self, u):
        return u * L * (u * self.ta2 - u * self.ta1 + self.ta1) + self.c1

    def Rz(self, u):
        return u * L * (u * self.tb2 - u * self.tb1 + self.tb1) + self.c2

    def dRy(self, u):
        return L * (2 * u * (self.ta2 - self.ta1) + self.ta1)

    def dRz(self, u):
        return L * (2 * u * (self.tb2 - self.tb1) + self.tb1)

    def base(self, u, v):
        return np.array([u * L, self.Ry(u) * np.cos(v), self.Rz(u) * np.sin(v)])

    def rot(self, u, v):
        return self.rat @ self.base(u, v)

    def normal_rot(self, u, v):
        ru = np.array([L, self.dRy(u) * np.cos(v), self.dRz(u) * np.sin(v)])
        rv = np.array([0.0, -self.Ry(u) * np.sin(v), self.Rz(u) * np.cos(v)])
        return self.rat @ np.cross(ru, rv)

    def beta(self, u, v):
        """Exactly build_geometry_core: pi/2 - acos(|Fx|/|F|) with F in the rotated (wind) frame."""
        F = self.normal_rot(u, v)
        return np.pi / 2 - np.arccos(min(abs(F[0]) / np.linalg.norm(F), 1.0))


def le_parameter_t(M=50):
    t = np.zeros(M)
    i1 = np.arange(0, M // 2)
    t[i1] = np.linspace(0, 0.09, len(i1))
    dy = abs(t[0] - t[1])
    i2 = np.arange(M // 2, M)
    t[i2] = np.linspace(0.09 + dy, 1, len(i2))
    return t[::-1]


def leading_edge(shock: Shock, d1, M=50):
    """Planform Bezier (build_geometry_core lines 69-140): x=L t^2, y=L(2t(1-t)d1+0.2 t^2) on the shock."""
    t = le_parameter_t(M)
    x = L * t ** 2
    y = L * (2 * t * (1 - t) * d1 + 0.2 * t ** 2)
    u = x / L
    v = np.arccos(np.clip(y / shock.Ry(u), -1, 1))
    return t, u, v, x, y


def shock_march(shock: Shock, u0, v0, x_fct1, n_l=40, aoa=AOA_NATIVE):
    """Shock trace of build_geometry_core lines 287-301 (rotated frame); returns u, v, beta per step."""
    dx = ((L + 0.1 - x_fct1) / (n_l - 1)) * np.cos(np.radians(aoa))
    us, vs, bs = [u0], [v0], [shock.beta(u0, v0)]
    p = shock.rot(u0, v0)
    for _ in range(n_l):
        F = shock.normal_rot(us[-1], vs[-1])
        D = np.array([F[2] ** 2 + F[1] ** 2, -F[0] * F[1], -F[0] * F[2]])
        tx, ty = p[0] + dx, p[1] + dx * D[1] / D[0]
        sol, info, ier, _ = fsolve(lambda uv: [shock.rot(*uv)[0] - tx, shock.rot(*uv)[1] - ty],
                                   [us[-1], vs[-1]], full_output=True, xtol=1e-13)
        if ier != 1:
            break
        us.append(sol[0]); vs.append(sol[1]); bs.append(shock.beta(*sol))
        p = shock.rot(*sol)
    return np.array(us), np.array(vs), np.array(bs)


def oblique(M, beta):
    """qiujie_shock.m jump relations. Returns (Mn, p2/p1, deflection rad)."""
    mn = M * np.sin(beta)
    pr = 2 * G / (G + 1) * mn ** 2 - (G - 1) / (G + 1)
    defl = beta - np.arctan(2 * np.tan(beta) * (1 / mn ** 2 + (G - 1) / 2) / (G + 1))
    return mn, pr, defl


# ----------------------------------------------------------------------------- aero side
def _tm_rhs(a, y):
    a2 = (G - 1) / 2 * (1 - y[0] ** 2 - y[1] ** 2)
    return [y[1], (a2 * (2 * y[0] + y[1] / np.tan(a)) - y[1] ** 2 * y[0]) / (y[1] ** 2 - a2)]


def _wall(M, b):
    mn = M * np.sin(b)
    rr = (G + 1) * mn ** 2 / (2 + (G - 1) * mn ** 2)
    vinf = np.sqrt((G - 1) * M ** 2 / (2 + (G - 1) * M ** 2))
    y0 = [vinf * np.cos(b), -vinf * np.sin(b) / rr]
    ev = lambda a, y: y[1]
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(_tm_rhs, [b, 1e-7], y0, rtol=1e-10, atol=1e-12, max_step=0.01, events=ev, first_step=1e-6)
    if s.t_events[0].size == 0:
        return np.nan, None
    return s.t_events[0][0], (y0, s.y_events[0][0])


@lru_cache(maxsize=200000)
def cone_pressure_ratio(M, theta):
    """Weak attached Taylor-Maccoll cone-surface p/p_inf for half-angle theta (rad)."""
    if theta <= 0:
        return 1.0
    mu = np.arcsin(1 / M)
    grid = np.linspace(mu + 1e-7, np.pi / 2 - 1e-4, 48)
    lo = mu
    hi = None
    for b in grid:
        w, _ = _wall(M, b)
        if np.isfinite(w) and w - theta >= 0:
            hi = b
            break
        if np.isfinite(w):
            lo = b
    if hi is None:
        raise ValueError("no attached cone")
    f = lambda b: (_wall(M, b)[0] - theta) if b > mu else -theta
    b = brentq(f, lo, hi, xtol=1e-14, rtol=1e-13)
    _, (y0, yw) = _wall(M, b)
    v2s, v2w = y0[0] ** 2 + y0[1] ** 2, yw[0] ** 2 + yw[1] ** 2
    ms = np.sqrt(2 * v2s / ((G - 1) * (1 - v2s)))
    mw = np.sqrt(2 * v2w / ((G - 1) * (1 - v2w)))
    mn = M * np.sin(b)
    prs = 1 + 2 * G / (G + 1) * (mn ** 2 - 1)
    p0r = prs * (1 + (G - 1) / 2 * ms ** 2) ** (G / (G - 1))
    return float(p0r / (1 + (G - 1) / 2 * mw ** 2) ** (G / (G - 1)))


def cone_beta(M, theta):
    mu = np.arcsin(1 / M)
    grid = np.linspace(mu + 1e-7, np.pi / 2 - 1e-4, 48)
    lo, hi = mu, None
    for b in grid:
        w, _ = _wall(M, b)
        if np.isfinite(w) and w - theta >= 0:
            hi = b
            break
        if np.isfinite(w):
            lo = b
    f = lambda b: (_wall(M, b)[0] - theta) if b > mu else -theta
    return brentq(f, lo, hi, xtol=1e-14)


def load_surfaces(mat):
    m = loadmat(mat)
    return {k: np.asarray(m[k], float) for k in ("wav_x", "wav_y", "wav_z", "wav_up_x", "wav_up_y", "wav_up_z")}


def panels(S):
    """offdesign_aero_v4 lines 11-31: body frame diag(1,-1,-1), outward normals, mirrored halves."""
    out = []
    for k, (X, Y, Z, sgn, lab) in enumerate([(S["wav_x"], -S["wav_y"], -S["wav_z"], -1, "lower"),
                                             (S["wav_up_x"], -S["wav_up_y"], -S["wav_up_z"], 1, "upper")]):
        P = np.stack([X, Y, Z], -1)
        p1, p2, p3, p4 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
        v = sgn * np.cross(p3 - p1, p2 - p4)
        mag = np.linalg.norm(v, axis=-1)
        n = v / mag[..., None]
        c = (p1 + p2 + p3 + p4) / 4
        for mirror in (1, -1):
            out.append({"n": (n * [1, mirror, 1]).reshape(-1, 3), "A": (mag / 2).reshape(-1),
                        "R": (c * [1, mirror, 1]).reshape(-1, 3), "surface": np.full(mag.size, lab),
                        "row": np.repeat(np.arange(mag.shape[0]), mag.shape[1]),
                        "col": np.tile(np.arange(mag.shape[1]), mag.shape[0])})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def aero(P, M, alpha_deg, lookup_n=33, detail=False):
    a = np.radians(alpha_deg)
    ed = np.array([np.cos(a), 0, np.sin(a)])
    el = np.array([-np.sin(a), 0, np.cos(a)])
    inc = -P["n"] @ ed
    th = np.arcsin(inc)
    pr = np.ones_like(th)
    comp, exp = th > 1e-9, th < -1e-9
    if comp.any():
        knots = np.linspace(0, th[comp].max(), lookup_n)
        vals = np.array([1.0] + [cone_pressure_ratio(float(M), float(k)) for k in knots[1:]])
        pr[comp] = PchipInterpolator(knots, vals)(th[comp])
    base = 1 + (G - 1) / 2 * M * th[exp]
    pr[exp] = base ** (2 * G / (G - 1))
    cp = 2 * (pr - 1) / (G * M ** 2)
    dF = -(cp * P["A"])[:, None] * P["n"] / SREF
    CF = dF.sum(0)
    res = {"CL": float(CF @ el), "CD": float(CF @ ed)}
    if detail:
        res.update({"theta": th, "cp": cp, "dF": dF, "ed": ed, "el": el})
    return res
