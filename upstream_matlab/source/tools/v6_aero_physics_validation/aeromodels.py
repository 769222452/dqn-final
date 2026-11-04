"""Low-order inviscid pressure models evaluated on the Teacher's own panels (same areas, normals, Sref).

  A  tangent cone + hypersonic expansion similarity  -> the Teacher itself (teacher_replica.aero)
  B  tangent wedge: exact weak oblique shock from the free stream on each panel (attached-shock check;
     detached panels flagged and given modified-Newtonian Cp), Prandtl-Meyer expansion from the free stream
  C  strip shock-expansion: every native streamwise panel row is a 2-D strip; the first panel turns the free
     stream by an oblique shock / Prandtl-Meyer fan, every later panel by a shock (compression) or a
     Prandtl-Meyer fan (expansion) from the local state of the previous panel. No cross-flow, no
     shock-shock interaction, no entropy-layer effects.
  N  modified Newtonian (Cp_max sin^2 theta, Cp=0 on shadowed panels) as a hypersonic bound.

theta is the Teacher's panel incidence asin(-n . e_drag) for the actual free-stream direction.
"""
from __future__ import annotations

import numpy as np

G = 1.4
GM = (G - 1) / 2


# ----------------------------------------------------------------------------- oblique shock (vectorised)
def beta_max(M):
    M = np.asarray(M, float)
    s2 = ((G + 1) * M ** 2 / 4 - 1 + np.sqrt((G + 1) * (1 + GM * M ** 2 + (G + 1) * M ** 4 / 16))) / (G * M ** 2)
    return np.arcsin(np.sqrt(np.clip(s2, 0, 1)))


def theta_of_beta(M, b):
    return np.arctan(2 / np.tan(b) * (M ** 2 * np.sin(b) ** 2 - 1) / (M ** 2 * (G + np.cos(2 * b)) + 2))


def theta_max(M):
    return theta_of_beta(M, beta_max(M))


def weak_beta(M, theta, iters=70):
    """Weak-branch shock angle for deflection theta>0; NaN where theta > theta_max (detached)."""
    M, theta = np.broadcast_arrays(np.asarray(M, float), np.asarray(theta, float))
    lo = np.arcsin(1 / M) + 1e-12
    hi = beta_max(M)
    ok = theta <= theta_of_beta(M, hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        f = theta_of_beta(M, mid) - theta
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    b = (lo + hi) / 2
    return np.where(ok, b, np.nan)


def shock_state(M, theta):
    b = weak_beta(M, theta)
    mn1 = M * np.sin(b)
    pr = 1 + 2 * G / (G + 1) * (mn1 ** 2 - 1)
    mn2 = np.sqrt((1 + GM * mn1 ** 2) / (G * mn1 ** 2 - GM))
    M2 = mn2 / np.sin(b - theta)
    return pr, M2, b


# ----------------------------------------------------------------------------- Prandtl-Meyer
def nu(M):
    M = np.asarray(M, float)
    k = np.sqrt((G + 1) / (G - 1))
    return k * np.arctan(np.sqrt((M ** 2 - 1) / k ** 2)) - np.arctan(np.sqrt(M ** 2 - 1))


NU_MAX = np.pi / 2 * (np.sqrt((G + 1) / (G - 1)) - 1)
_MT = np.logspace(0, 3.5, 20001)
_NT = nu(_MT)


def nu_inv(v):
    v = np.asarray(v, float)
    return np.where(v >= _NT[-1], np.inf, np.interp(v, _NT, _MT))


def pm_state(M, dtheta_expand):
    """Isentropic turn by dtheta (>0 expansion, <0 isentropic compression); returns p2/p1, M2."""
    v2 = nu(M) + dtheta_expand
    M2 = nu_inv(np.clip(v2, 0, None))
    pr = np.where(np.isfinite(M2), ((1 + GM * M ** 2) / (1 + GM * M2 ** 2)) ** (G / (G - 1)), 0.0)
    return pr, M2


# ----------------------------------------------------------------------------- panel models
def cp_from_pr(M, pr):
    return 2 * (pr - 1) / (G * M ** 2)


def newtonian_cp(M, theta, modified=True):
    theta = np.asarray(theta, float)
    if modified:
        mn = M
        p02 = ((G + 1) ** 2 * mn ** 2 / (4 * G * mn ** 2 - 2 * (G - 1))) ** (G / (G - 1)) * (1 - G + 2 * G * mn ** 2) / (G + 1)
        cpmax = 2 * (p02 - 1) / (G * M ** 2)
    else:
        cpmax = 2.0
    return np.where(theta > 0, cpmax * np.sin(theta) ** 2, 0.0)


def wedge_cp(M, theta):
    """Tangent wedge (compression) + Prandtl-Meyer from free stream (expansion). Detached -> NaN, flagged."""
    theta = np.asarray(theta, float)
    cp = np.zeros_like(theta)
    detached = np.zeros(theta.shape, bool)
    c = theta > 1e-12
    if c.any():
        pr, _, b = shock_state(M, theta[c])
        cp[c] = cp_from_pr(M, pr)
        detached[c] = ~np.isfinite(b)
    e = theta < -1e-12
    if e.any():
        pr, _ = pm_state(M, -theta[e])
        cp[e] = cp_from_pr(M, pr)
    return {"cp": np.where(detached, np.nan, cp), "detached": detached}


def strip_shock_expansion(P, theta, M):
    """Model C on the Teacher's panel ordering (blocks of native rows x cols); returns Cp per panel + flags."""
    cp = np.full(len(theta), np.nan)
    detached = np.zeros(len(theta), bool)
    # teacher_replica.panels stacks four row-major (row x col) blocks: lower, lower mirror, upper, upper mirror
    nb = (P["row"].max() + 1) * (P["col"].max() + 1)
    assert len(theta) == 4 * nb
    grp = (np.arange(len(theta)) // nb) * 100000 + P["row"]
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    ends = np.r_[starts[1:], len(theta)]
    assert all(np.array_equal(P["col"][s:e], np.arange(e - s)) for s, e in zip(starts, ends))
    nc = int(P["col"].max() + 1)
    th = theta.reshape(-1, nc)  # one strip per row, columns run leading edge -> base
    ns = th.shape[0]
    Ml = np.full(ns, float(M))
    pl = np.ones(ns)
    prev = np.zeros(ns)
    dead = np.zeros(ns, bool)  # detached somewhere upstream on this strip
    cpa = np.full(th.shape, np.nan)
    Mloc = np.full(th.shape, np.nan)
    for j in range(nc):
        dth = th[:, j] - prev
        pr = np.ones(ns)
        M2 = Ml.copy()
        fin = np.isfinite(Ml) & ~dead
        c = fin & (dth > 1e-12)
        if c.any():
            p_, m_, b_ = shock_state(Ml[c], dth[c])
            bad = ~np.isfinite(b_)
            pr[c], M2[c] = np.where(bad, np.nan, p_), np.where(bad, np.nan, m_)
            dead[np.flatnonzero(c)[bad]] = True
        e = fin & (dth < -1e-12)
        if e.any():
            pr[e], M2[e] = pm_state(Ml[e], -dth[e])
        vac = ~np.isfinite(Ml) & ~dead
        pr[vac] = 0.0
        pl = np.where(dead, np.nan, pl * pr)
        Ml = np.where(dead, np.nan, np.where(vac, np.inf, M2))
        prev = th[:, j]
        cpa[:, j] = cp_from_pr(M, pl)
        Mloc[:, j] = Ml
        detached.reshape(-1, nc)[:, j] = dead
    cp[:] = cpa.ravel()
    return {"cp": cp, "detached": detached, "M_local": Mloc.ravel()}


def forces(P, cp, alpha_deg, sref):
    a = np.radians(alpha_deg)
    ed, el = np.array([np.cos(a), 0, np.sin(a)]), np.array([-np.sin(a), 0, np.cos(a)])
    dF = -(cp * P["A"])[:, None] * P["n"] / sref
    F = dF.sum(0)
    return float(F @ el), float(F @ ed)


def incidence(P, alpha_deg):
    a = np.radians(alpha_deg)
    ed = np.array([np.cos(a), 0, np.sin(a)])
    return np.arcsin(np.clip(-P["n"] @ ed, -1, 1))


def all_models(P, M, alpha_deg, sref, teacher_fn):
    """CL, CD of models A/B/C/N at one (M, alpha); detached panels fall back to modified Newtonian."""
    th = incidence(P, alpha_deg)
    out = {}
    if teacher_fn is not None:
        t = teacher_fn(P, M, alpha_deg)
        out["A_tangent_cone_teacher"] = (t["CL"], t["CD"], 0.0)
    cn = newtonian_cp(M, th)
    w = wedge_cp(M, th)
    cw = np.where(np.isfinite(w["cp"]), w["cp"], cn)
    out["B_tangent_wedge"] = (*forces(P, cw, alpha_deg, sref), float(P["A"][w["detached"]].sum() / P["A"].sum()))
    s = strip_shock_expansion(P, th, M)
    cs = np.where(np.isfinite(s["cp"]), s["cp"], cn)
    out["C_strip_shock_expansion"] = (*forces(P, cs, alpha_deg, sref), float(P["A"][s["detached"]].sum() / P["A"].sum()))
    out["N_modified_newtonian"] = (*forces(P, cn, alpha_deg, sref), 0.0)
    return out, {"theta": th, "cp_wedge": cw, "cp_strip": cs, "cp_newton": cn}
