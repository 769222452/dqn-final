"""Skin friction of the candidate waveriders (experiments/aero_fidelity/README.md).

The aerodynamic snapshot (data/geometry_subset) is the P2 strip shock-expansion model with
`friction: none`. This module adds the viscous drag the snapshot omits, using the same correlations as the
project's own `provenance/offdesign_aero_v7.m` friction branch:
- Eckert reference temperature T* = Te (1 + 0.032 Me² + 0.58 (Tw/Te − 1));
- laminar 0.664/√Re*x and turbulent 0.0592/Re*x^0.2, both cell-averaged exactly;
- transition at edge Re_x = Re_tr;
- wall shear along the free-stream projection on each panel.

Edge conditions come from each panel's local inviscid state, using the P1 option of that code:
- compression: oblique shock (weak solution) referred to the free stream;
- expansion: Prandtl–Meyer;
- detached panels: normal-shock state.

The P2 database transfers the state along strips, so pressures here are close to, not identical with, the
database. `pressure_coefficients` reports them so the geometry handling can be checked against
data/geometry_subset/aerodynamics/p2_grid.npz.

Geometry (native_geometry.mat): half body, lower surface wav_*, upper surface wav_up_*, 50 × 50 grids,
rows = spanwise strips, columns = streamwise from the leading edge (column 0) to the base. Body axes =
diag(1, −1, −1) · file axes, as in offdesign_aero_v7.m. Forces of the mirrored half are added (x and z
components double, y cancels). Reference area 6.931920786884801 m² for every geometry, as in the snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

GAMMA = 1.4
R_AIR = 287.05287
SREF = 6.931920786884801
TW_K = 1000.
RE_TR = 5e6


def sutherland(T):
    return 1.458e-6*T**1.5/(T+110.4)


@dataclass
class Panels:
    centre: np.ndarray        # [n, 3] body axes
    normal: np.ndarray        # [n, 3] outward unit normals
    area: np.ndarray          # [n]
    s1: np.ndarray            # [n] streamwise distance from the leading edge, panel start
    s2: np.ndarray            # [n] panel end
    surface: np.ndarray       # [n] 0 lower, 1 upper


def _surface(P, outward_z_sign):
    a, b, c, d = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    cr = np.cross(c-a, d-b)
    area = .5*np.linalg.norm(cr, axis=-1)
    n = cr/np.maximum(2*area, 1e-300)[..., None]
    # Orient every panel of this surface consistently: lower surface outward = -z (down), upper = +z.
    flip = np.sign(np.nansum(n[..., 2]*area)) != outward_z_sign
    if flip:
        n = -n
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)                 # [rows, cols-1] streamwise segments
    s_end = np.cumsum(seg, axis=1)
    s_start = s_end-seg
    s1 = .5*(s_start[:-1]+s_start[1:])
    s2 = .5*(s_end[:-1]+s_end[1:])
    centre = .25*(a+b+c+d)
    return centre, n, area, s1, s2


def load_panels(path):
    z = loadmat(path)
    flip = np.array([1., -1., -1.])
    out = []
    for k, (prefix, sign) in enumerate((('wav_', -1.), ('wav_up_', 1.))):
        P = np.stack([z[prefix+'x'], z[prefix+'y'], z[prefix+'z']], -1)*flip
        centre, n, area, s1, s2 = _surface(P, sign)
        ok = area > 1e-12
        out.append([centre[ok], n[ok], area[ok], s1[ok], s2[ok], np.full(ok.sum(), k)])
    return Panels(*[np.concatenate([o[i] for o in out]) for i in range(6)])


# ------------------------------------------------------------------------------------- inviscid edge state
def _theta_beta(M, beta):
    return np.arctan(2/np.tan(beta)*(M**2*np.sin(beta)**2-1)/(M**2*(GAMMA+np.cos(2*beta))+2))


def oblique(M, theta):
    """Weak oblique shock for deflection theta (rad, > 0); detached panels get the normal-shock state.

    Returns pressure, temperature and density ratios, downstream Mach number and a detached flag."""
    M = np.broadcast_to(M, theta.shape).astype(float)
    mu = np.arcsin(1/M)
    # theta_max and its beta by golden-section-free dense search (vectorised, robust)
    bs = mu[..., None]+(np.pi/2-mu[..., None])*np.linspace(1e-6, 1-1e-6, 400)
    th = _theta_beta(M[..., None], bs)
    imax = np.argmax(th, axis=-1)
    thmax = np.take_along_axis(th, imax[..., None], -1)[..., 0]
    detached = theta >= thmax
    lo = mu.copy()
    hi = np.take_along_axis(bs, imax[..., None], -1)[..., 0]
    for _ in range(60):                                   # bisection on the weak branch
        mid = .5*(lo+hi)
        f = _theta_beta(M, mid)-theta
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f >= 0, mid, hi)
    beta = np.where(detached, np.pi/2, .5*(lo+hi))
    Mn = M*np.sin(beta)
    pr = 1+2*GAMMA/(GAMMA+1)*(Mn**2-1)
    rr = (GAMMA+1)*Mn**2/((GAMMA-1)*Mn**2+2)
    Tr = pr/rr
    Mn2 = np.sqrt((1+.5*(GAMMA-1)*Mn**2)/(GAMMA*Mn**2-.5*(GAMMA-1)))
    th_eff = np.where(detached, 0., theta)
    M2 = Mn2/np.maximum(np.sin(beta-th_eff), 1e-12)
    return pr, Tr, rr, M2, detached


def _nu(M):
    g = GAMMA
    return np.sqrt((g+1)/(g-1))*np.arctan(np.sqrt((g-1)/(g+1)*(M**2-1)))-np.arctan(np.sqrt(M**2-1))


def expansion(M, dtheta):
    """Prandtl–Meyer expansion by dtheta (rad, > 0) from M; vacuum beyond the maximum turning."""
    M = np.broadcast_to(M, dtheta.shape).astype(float)
    nu2 = _nu(M)+dtheta
    numax = .5*np.pi*(np.sqrt((GAMMA+1)/(GAMMA-1))-1)
    vac = nu2 >= numax*(1-1e-9)
    lo, hi = M.copy(), np.full_like(M, 1e4)
    target = np.minimum(nu2, numax*(1-1e-9))
    for _ in range(80):
        mid = np.sqrt(lo*hi)
        f = _nu(mid)-target
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f >= 0, mid, hi)
    M2 = np.sqrt(lo*hi)
    t = (1+.5*(GAMMA-1)*M**2)/(1+.5*(GAMMA-1)*M2**2)
    Tr = t
    pr = t**(GAMMA/(GAMMA-1))
    rr = t**(1/(GAMMA-1))
    return pr, Tr, rr, M2, vac


def directions(alpha):
    ed = np.array([np.cos(alpha), 0., np.sin(alpha)])     # drag direction (free-stream velocity, body axes)
    el = np.array([-np.sin(alpha), 0., np.cos(alpha)])
    return ed, el


def edge_state(p, M, alpha):
    """Panel-wise inviscid state referred to the free stream (tangent wedge / Prandtl–Meyer)."""
    ed, _ = directions(alpha)
    inc = -(p.normal@ed)                                 # sin of the local incidence (> 0 compression)
    th = np.arcsin(np.clip(inc, -1, 1))
    pr = np.ones_like(th)
    Tr = np.ones_like(th)
    rr = np.ones_like(th)
    Me = np.full_like(th, float(M))
    bad = np.zeros_like(th, bool)
    vac = np.zeros_like(th, bool)
    c = th > 1e-12
    e = th < -1e-12
    if c.any():
        pr[c], Tr[c], rr[c], Me[c], bad[c] = oblique(np.full(c.sum(), float(M)), th[c])
    if e.any():
        pr[e], Tr[e], rr[e], Me[e], vac[e] = expansion(np.full(e.sum(), float(M)), -th[e])
    return pr, Tr, rr, Me, bad, vac


def pressure_coefficients(p, M, alpha):
    """(CL, CD) of the pressure force, mirrored half body, P1-type panel state (validation only)."""
    pr, *_ = edge_state(p, M, alpha)
    Cp = 2*(pr-1)/(GAMMA*M**2)
    F = -(Cp*p.area)[:, None]*p.normal/SREF
    F = 2*F.sum(0)*np.array([1, 0, 1])
    ed, el = directions(alpha)
    return float(F@el), float(F@ed)


def _cell_average(s1, s2, power):
    """Exact average over [s1, s2] of s^-power, divided by the midpoint value (as offdesign_aero_v7.m)."""
    s1 = np.maximum(s1, 1e-9)
    s2 = np.maximum(s2, s1+1e-12)
    xm = .5*(s1+s2)
    return (s2**(1-power)-s1**(1-power))/((1-power)*(s2-s1))/xm**(-power)


def friction_coefficients(p, M, alpha, T_inf, rho_inf, Tw=TW_K, Re_tr=RE_TR, mode='transition', state=None):
    """(CL_f, CD_f) of the wall shear, mirrored half body, referred to the free-stream q and SREF.

    state: optional precomputed edge_state(p, M, alpha) (it does not depend on altitude)."""
    pr, Tr, rr, Me, bad, vac = edge_state(p, M, alpha) if state is None else state
    a_inf = np.sqrt(GAMMA*R_AIR*T_inf)
    V = M*a_inf
    q = .5*rho_inf*V*V
    Te = Tr*T_inf
    rhoe = rr*rho_inf
    ue = Me*np.sqrt(GAMMA*R_AIR*Te)
    Ts = Te*(1+.032*Me**2+.58*(Tw/Te-1))
    rs = rhoe*Te/Ts
    x = .5*(p.s1+p.s2)
    Rex = rs*ue*np.maximum(x, 1e-9)/sutherland(Ts)
    cfl = .664/np.sqrt(Rex)*_cell_average(p.s1, p.s2, .5)
    cft = .0592/Rex**.2*_cell_average(p.s1, p.s2, .2)
    if mode == 'laminar':
        cfs = cfl
    elif mode == 'turbulent':
        cfs = cft
    elif mode == 'transition':
        Ree = rhoe*ue*np.maximum(x, 1e-9)/sutherland(Te)
        cfs = np.where(Ree < Re_tr, cfl, cft)
    else:
        raise ValueError(mode)
    tau = cfs*rs/rhoe*.5*rhoe*ue**2
    tau = np.where(vac, 0., tau)                          # no shear in a Prandtl–Meyer vacuum
    ed, el = directions(alpha)
    nd = p.normal@ed
    t = ed[None]-nd[:, None]*p.normal
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]
    F = ((tau*p.area)[:, None]*t).sum(0)/(q*SREF)
    F = 2*F*np.array([1, 0, 1])
    return float(F@el), float(F@ed)


def load_all(root):
    """{geometry_id: Panels} for every candidate geometry of the snapshot."""
    root = Path(root)
    ids = [line.split(',')[1] for line in (root/'action_index.csv').read_text(encoding='utf-8-sig').splitlines()[1:]]
    return ids, {g: load_panels(root/'geometries'/g/'native_geometry.mat') for g in ids}
