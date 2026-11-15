"""Screening G, step 1: how much can morphing change drag at a given lift?

For every Mach node of the database and CL in CL_GRID: CD of every geometry at that CL (alpha found on a dense
alpha grid, exact PCHIP of the database; geometries that cannot reach the CL inside the alpha support are
skipped). Reports min / max / quantiles over the library and the CD of the initial geometry (G42).

    python experiments/perturb/screening/drag_range.py --out experiments/perturb/screening/drag_range.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
CL_GRID = np.round(np.arange(.015, .0751, .005), 4)


def cd_at_cl(db, mach, cls, step_deg=.02):
    """[geometry, len(cls)] CD at the given CLs (NaN where the CL is outside the geometry's alpha support)."""
    a = np.deg2rad(np.arange(np.rad2deg(db.A[0]), np.rad2deg(db.A[-1])+1e-9, step_deg))
    y = PchipInterpolator(db.A, db.curves(mach), axis=1)(a)            # [g, alpha, 2]
    out = np.full((y.shape[0], len(cls)), np.nan)
    for g in range(y.shape[0]):
        cl, cd = y[g, :, 0], y[g, :, 1]
        k = np.argmax(cl)                                               # monotone branch from the lowest alpha
        cl, cd = cl[:k+1], cd[:k+1]
        if np.any(np.diff(cl) <= 0):
            keep = np.r_[True, np.diff(np.maximum.accumulate(cl)) > 0]
            cl, cd = cl[keep], cd[keep]
        inside = (cls >= cl[0]) & (cls <= cl[-1])
        out[g, inside] = np.interp(cls[inside], cl, cd)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    from certified_flight.aero import AeroDatabase
    from certified_flight.build import runtime_config
    db = AeroDatabase(ROOT/runtime_config(ROOT)['physical']['snapshot'])
    g42 = [i for i, x in enumerate(db.ids) if 'G42' in x]
    g42 = g42[0] if g42 else None
    rows = []
    for M in db.M:
        cd = cd_at_cl(db, float(M), CL_GRID)
        for j, c in enumerate(CL_GRID):
            v = cd[:, j]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            q = np.quantile(v[ok], [0, .1, .5, .9, 1])
            rows.append(dict(mach=float(M), CL=float(c), n=int(ok.sum()), cd_min=q[0], cd_p10=q[1], cd_med=q[2],
                             cd_p90=q[3], cd_max=q[4], argmin=db.ids[int(np.nanargmin(v))],
                             argmax=db.ids[int(np.nanargmax(v))],
                             cd_g42=None if g42 is None or not ok[g42] else float(v[g42])))
    Path(a.out).write_text(json.dumps(dict(cl_grid=CL_GRID.tolist(), g42=None if g42 is None else db.ids[g42],
                                           rows=rows), indent=1))
    print(' M    CL    n   CDmin   CDp10  CDmed  CDp90  CDmax  G42   max/min p90/p10 G42/min')
    for r in rows:
        if r['CL'] in (.02, .025, .03, .04, .05, .06) and r['mach'] in (5., 6., 6.5, 7., 7.5, 8.):
            print('%4.1f %.3f %3d %.5f %.5f %.5f %.5f %.5f %s %.2f %.2f %s' % (
                r['mach'], r['CL'], r['n'], r['cd_min'], r['cd_p10'], r['cd_med'], r['cd_p90'], r['cd_max'],
                '%.5f' % r['cd_g42'] if r['cd_g42'] else '   -   ', r['cd_max']/r['cd_min'], r['cd_p90']/r['cd_p10'],
                '%.3f' % (r['cd_g42']/r['cd_min']) if r['cd_g42'] else '-'))


if __name__ == '__main__':
    main()
