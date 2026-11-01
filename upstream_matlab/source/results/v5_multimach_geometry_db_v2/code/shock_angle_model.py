"""Analytic local shock-angle model of the active generator's shock surface.

This is a diagnostic re-derivation of quantities that build_geometry_core
computes internally but never returns. It reads the same closed-form shock
surface as create_shock_functions.m and the same leading-edge Bezier as
build_geometry_core.m, so it predicts the generator's domain gate

    sin(beta_local) > 1 / Ma_design        (attached oblique shock exists)

without running MATLAB. It is used only to explain and to pre-screen; every
feasibility claim in this task is still decided by an actual MATLAB call.

Surface (program coordinates, before the angle-of-attack rotation):
    x = u*L
    y = (u*L*(u*(tan a2 - tan a1) + tan a1) + c1) * cos v
    z = (u*L*(u*(tan b2 - tan b1) + tan b1) + c2) * sin v
Leading edge (plane Bezier, before rotation):
    x_le = L*t**2
    y_le = L*(2*d1*t*(1-t) + 0.2*t**2)
build_geometry_core solves x_v(u,v) = x_le and y_v(u,v) = y_le on the
unrotated surface, so u = t**2 exactly and cos v = y_le / Ry(u).
"""
from __future__ import annotations

import numpy as np

L_DESIGN = 5.0
C1 = C2 = 1.0331
AOA_PROG_DEG = -4.0
M_STREAM_SURFACES = 50
TRANSITION_POINT = 0.09


def bezier_t():
    """Exactly the t distribution of build_geometry_core (M=50, flipped)."""
    m = M_STREAM_SURFACES
    n1 = int(np.floor(m * 0.5))
    t1 = np.linspace(0.0, TRANSITION_POINT, n1)
    dt = abs(t1[0] - t1[1])
    t2 = np.linspace(TRANSITION_POINT + dt, 1.0, m - n1)
    return np.flip(np.concatenate([t1, t2]))


def radii(u, a1, a2, b1, b2):
    ta1, ta2 = np.tan(np.radians(a1)), np.tan(np.radians(a2))
    tb1, tb2 = np.tan(np.radians(b1)), np.tan(np.radians(b2))
    ry = u * L_DESIGN * (u * (ta2 - ta1) + ta1) + C1
    rz = u * L_DESIGN * (u * (tb2 - tb1) + tb1) + C2
    dry = L_DESIGN * (2 * u * (ta2 - ta1) + ta1)
    drz = L_DESIGN * (2 * u * (tb2 - tb1) + tb1)
    return ry, rz, dry, drz


def sin_beta(u, v, z):
    """sin(local shock angle) = |Fx_rotated| / |F|, F = r_u x r_v."""
    a1, a2, b1, b2 = z[:4]
    ry, rz, dry, drz = radii(u, a1, a2, b1, b2)
    cv, sv = np.cos(v), np.sin(v)
    f0x = dry * rz * cv ** 2 + drz * ry * sv ** 2
    f0y = -L_DESIGN * rz * cv
    f0z = -L_DESIGN * ry * sv
    th = np.radians(AOA_PROG_DEG)
    fx = np.cos(th) * f0x + np.sin(th) * f0z
    norm = np.sqrt(f0x ** 2 + f0y ** 2 + f0z ** 2)
    return np.abs(fx) / norm


def leading_edge(z):
    """Returns (t, u, v, valid) for the 50 native stream-surface origins."""
    a1, a2, b1, b2, d1, _ = z
    t = bezier_t()
    u = t ** 2
    x_le = L_DESIGN * t ** 2
    y_le = L_DESIGN * (2 * d1 * t * (1 - t) + 0.2 * t ** 2)
    ry, _, _, _ = radii(u, a1, a2, b1, b2)
    ratio = y_le / ry
    valid = np.all(ratio <= 1.0) and np.all(ratio >= -1e-12)
    v = np.arccos(np.clip(ratio, -1.0, 1.0))
    return t, u, v, bool(valid), x_le, y_le


def leading_edge_shock_angles(z):
    t, u, v, valid, x_le, y_le = leading_edge(z)
    sb = sin_beta(u, v, z)
    return {"t": t, "u": u, "v": v, "leading_edge_solvable": valid,
            "sin_beta": sb, "beta_deg": np.degrees(np.arcsin(np.clip(sb, 0, 1))),
            "x_le": x_le, "y_le": y_le}


def predicted_first_failure(z, ma_design):
    """First stream-surface index (1-based) whose leading-edge shock angle is
    at or below the Mach angle, i.e. where qiujie_shock has no attached
    solution. None means the gate is satisfied on every stream surface."""
    r = leading_edge_shock_angles(z)
    bad = np.where(r["sin_beta"] <= 1.0 / ma_design)[0]
    return (int(bad[0]) + 1) if len(bad) else None


def min_required_mach(z):
    """Lowest Ma_design for which every stream surface has an attached shock."""
    r = leading_edge_shock_angles(z)
    return float(1.0 / r["sin_beta"].min())
