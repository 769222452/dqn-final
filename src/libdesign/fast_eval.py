"""Equal-lift drag of one geometry at a few conditions (for sweeps and optimisation).

P2 on a 0.5 deg alpha grid from -1 to 12 deg at the requested Mach numbers, PCHIP in alpha on the monotone CL
branch, CD at the requested CLs (NaN if the CL is not reached or the model is invalid there).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from . import p2 as P

ALPHAS_DEG = np.arange(-1., 12.0001, .5)


def equal_lift_cd(lo, up, machs, cls, alphas_deg=ALPHAS_DEG):
    machs, cls = np.asarray(machs, float), np.asarray(cls, float)
    a = np.deg2rad(alphas_deg)
    MM, AA = np.meshgrid(machs, a, indexing='ij')
    r = P.p2(lo, up, MM.ravel(), AA.ravel())
    CL, CD = r['CL'].reshape(MM.shape), r['CD'].reshape(MM.shape)
    out = np.full((len(machs), len(cls)), np.nan)
    fine = np.deg2rad(np.arange(alphas_deg[0], alphas_deg[-1]+1e-9, .02))
    for i in range(len(machs)):
        ok = np.isfinite(CL[i]) & np.isfinite(CD[i])
        if ok.sum() < 4:
            continue
        # contiguous valid block starting at the lowest valid alpha
        idx = np.flatnonzero(ok)
        stop = idx[0]
        while stop+1 < len(ok) and ok[stop+1]:
            stop += 1
        sl = slice(idx[0], stop+1)
        if stop-idx[0] < 3:
            continue
        y = PchipInterpolator(a[sl], np.stack([CL[i, sl], CD[i, sl]], -1), axis=0)(fine[(fine >= a[sl][0]) & (fine <= a[sl][-1])])
        c, d = y[:, 0], y[:, 1]
        k = int(np.argmax(c))
        c, d = c[:k+1], d[:k+1]
        keep = np.r_[True, np.diff(np.maximum.accumulate(c)) > 0]
        c, d = c[keep], d[keep]
        inside = (cls >= c[0]) & (cls <= c[-1])
        out[i, inside] = np.interp(cls[inside], c, d)
    return out
