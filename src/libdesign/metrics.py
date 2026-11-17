"""Library metrics on equal-lift drag (the F8D2000 evaluation definition, datasets/F8D2000/README.md).

Conditions: Mach 5.75:0.25:8 (10) x CL 0.015:0.005:0.070 (12) = 120, equal weight. The equal-lift alpha of a
geometry is found by PCHIP in alpha on its monotone CL branch (no extrapolation).

static gain   1 - mean_c CD_selected(c) / mean_c CD_bestfixed(c); CD_selected = lowest drag at each condition,
              best fixed = the geometry with the lowest mean drag among those that reach every condition.
authority     mean_c CDmax(c)/CDmin(c) over the geometries that reach the condition (drag modulation at equal
              lift, the energy-management lever of the target-arrival task); also the 10-90 % quantile ratio.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

MACHS = np.arange(5.75, 8.0001, .25)
CLS = np.round(np.arange(.015, .0701, .005), 4)
MACH_GRID = np.array([3.5, 4, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25])
ALPHA_GRID_DEG = np.array([-1, 0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12.])


def _branch(cl, cd, cls):
    out = np.full(len(cls), np.nan)
    if not np.all(np.isfinite(cl)):
        return out
    k = int(np.argmax(cl))
    c, d = cl[:k+1], cd[:k+1]
    keep = np.r_[True, np.diff(np.maximum.accumulate(c)) > 0]
    c, d = c[keep], d[keep]
    inside = (cls >= c[0]) & (cls <= c[-1])
    out[inside] = np.interp(cls[inside], c, d)
    return out


def equal_lift(CL, CD, machs=MACHS, cls=CLS, alpha_step_deg=.02):
    """CL, CD [geometry, 18 Mach, 20 alpha] on the library grid -> CD [geometry, len(machs), len(cls)]."""
    A = np.deg2rad(ALPHA_GRID_DEG)
    a = np.deg2rad(np.arange(-1, 12.0001, alpha_step_deg))
    out = np.full((CL.shape[0], len(machs), len(cls)), np.nan)
    for i, M in enumerate(machs):
        k = int(np.argmin(abs(MACH_GRID-M)))
        assert abs(MACH_GRID[k]-M) < 1e-9
        y = PchipInterpolator(A, np.stack([CL[:, k], CD[:, k]], -1), axis=1)(a)
        for g in range(CL.shape[0]):
            out[g, i] = _branch(y[g, :, 0], y[g, :, 1], cls)
    return out


def static_gain(cd):
    full = np.all(np.isfinite(cd), axis=(1, 2))
    mean = np.where(full, np.nanmean(cd, axis=(1, 2)), np.inf)
    best = int(np.argmin(mean))
    sel = np.nanmin(cd, axis=0)
    winners = np.nanargmin(np.where(np.isfinite(cd), cd, np.inf), axis=0)
    per = 1-sel/cd[best]
    return dict(gain=float(1-sel.mean()/cd[best].mean()), best_fixed=best, n_winners=int(len(np.unique(winners))),
                per_condition_max=float(per.max()), per_condition_mean=float(per.mean()))


def authority(cd):
    with np.errstate(all='ignore'):
        lo, hi = np.nanmin(cd, 0), np.nanmax(cd, 0)
        q10, q90 = np.nanquantile(cd, .1, axis=0), np.nanquantile(cd, .9, axis=0)
    return dict(max_over_min=float(np.nanmean(hi/lo)), q90_over_q10=float(np.nanmean(q90/q10)),
                at_M7_CL03=float((hi/lo)[int(np.argmin(abs(MACHS-7))), int(np.argmin(abs(CLS-.03)))]))


def library_metrics(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    cd = equal_lift(z['CL'], z['CD_pressure'])
    return dict(n=int(z['CL'].shape[0]), static=static_gain(cd), authority=authority(cd)), cd
