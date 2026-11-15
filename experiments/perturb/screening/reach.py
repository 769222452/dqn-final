"""Screening G, step 2: reachable (range, final speed) at t = 1200 s with and without morphing.

Same point-mass RK4 and path-limit checks as screen.py. Three phases (0-400, 400-800, 800-1200 s); in each phase
the dynamic-pressure law with set point r sets CL, and the drag at that CL is

  fixed   CD = polar of the initial geometry G42 (no morphing)
  morph   CD = CDmin + w (CDmax - CDmin), CDmin/CDmax over the 298 geometries at the same Mach and CL
          (w = 0 is the lowest-drag geometry, w = 1 the highest; the 8 s morph time is ignored)

Grid: r in R_SET per phase, w in W_SET per phase (morph only). Output: one row per flight.

    python experiments/perturb/screening/reach.py --out experiments/perturb/screening/reach.json
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen as S  # noqa: E402
from drag_range import cd_at_cl  # noqa: E402

R_SET = (.3, .45, .6, .8, 1.)
W_SET = (0., .33, .67, 1.)
PHASES = (400., 800.)
CLT = np.round(np.arange(0., .1001, .0025), 4)
_T = {}


def tables():
    if not _T:
        w = S.world('pressure')
        from certified_flight.aero import AeroDatabase
        from certified_flight.build import runtime_config
        db = AeroDatabase(S.ROOT/runtime_config(S.ROOT)['physical']['snapshot'])
        g42 = [i for i, x in enumerate(db.ids) if 'G42' in x][0]
        lo, hi, fx = [], [], []
        for M in db.M:
            cd = cd_at_cl(db, float(M), CLT)
            lo.append(np.nanmin(cd, 0))
            hi.append(np.nanmax(cd, 0))
            fx.append(cd[g42])
        lo, hi, fx = map(np.array, (lo, hi, fx))
        fx = np.where(np.isfinite(fx), fx, np.nan)
        _T.update(M=db.M, lo=lo, hi=hi, fx=fx, w=w)
    return _T


def interp2(tab, M, cl):
    T = tables()
    i = int(np.clip(np.searchsorted(T['M'], M)-1, 0, len(T['M'])-2))
    u = (M-T['M'][i])/(T['M'][i+1]-T['M'][i])
    row = (1-u)*tab[i]+u*tab[i+1]
    return float(np.interp(cl, CLT, row))


def fly(spec, x0=None):
    T = tables()
    w = T['w']
    model = w['model']
    qmax = w['lim']['dynamic_pressure_pa_max']
    x = np.array(w['x0'] if x0 is None else x0, float)
    s0, worst, t = x[3], np.inf, 0.
    rs, ws = spec['r'], spec.get('w')

    def phase(t):
        return 0 if t < PHASES[0] else (1 if t < PHASES[1] else 2)

    def fx(y, cl, k):
        M = y[1]/float(model.atm.speed_of_sound(y[0]))
        if ws is None:
            cd = interp2(T['fx'], M, cl)
        else:
            a, b = interp2(T['lo'], M, cl), interp2(T['hi'], M, cl)
            cd = a+ws[k]*(b-a)
        if not np.isfinite(cd):
            cd = 1.
        r, q = model.rhs_with_quantities(y, [cl], [cd])
        return np.asarray(r, float), q

    while t < S.T_END-1e-9:
        k = phase(t)
        h, V = x[0], x[1]
        qd = .5*float(model.atm.density(h))*V*V
        grav = float(model.atm.gravity(h, spherical=True))
        lvl = w['m']*(grav-V*V/(model.atm.cfg.R_earth+h))/(qd*w['S'])
        cl = float(np.clip(lvl*(1+spec['kp']*(qd/(rs[k]*qmax)-1))-spec['kg']*x[2], 0., CLT[-1]))
        k1, q1 = fx(x, cl, k)
        k2, q2 = fx(x+S.DT*k1/2, cl, k)
        k3, q3 = fx(x+S.DT*k2/2, cl, k)
        k4, q4 = fx(x+S.DT*k3, cl, k)
        for yy, qq in ((x, q1), (x+S.DT*k1/2, q2), (x+S.DT*k2/2, q3), (x+S.DT*k3, q4)):
            worst = min(worst, S.margins(w, yy, qq))
        x = x+S.DT*(k1+2*k2+2*k3+k4)/6
        t += S.DT
        if not np.all(np.isfinite(x)) or worst < -1e-6:
            return dict(spec, feasible=False, range_km=float((x[3]-s0)/1000), V_end=float(x[1]), h_end=float(x[0]),
                        t_end=t)
    return dict(spec, feasible=True, range_km=float((x[3]-s0)/1000), V_end=float(x[1]), h_end=float(x[0]), t_end=t)


def job(spec):
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    return fly(spec)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    gains = dict(kp=2., kg=1.)
    specs = [dict(mode='fixed', r=list(r), **gains) for r in itertools.product(R_SET, repeat=3)]
    specs += [dict(mode='morph', r=list(r), w=list(ww), **gains)
              for r in itertools.product(R_SET, repeat=3) for ww in itertools.product(W_SET, repeat=3)]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(job, specs, chunksize=16)
    Path(a.out).write_text(json.dumps(dict(R_SET=R_SET, W_SET=W_SET, phases=PHASES, rows=rows)))
    for m in ('fixed', 'morph'):
        ok = [r for r in rows if r['mode'] == m and r['feasible']]
        rg = [r['range_km'] for r in ok]
        print(m, len(ok), 'feasible; range %.0f-%.0f km; V_end %.0f-%.0f m/s' % (
            min(rg), max(rg), min(r['V_end'] for r in ok), max(r['V_end'] for r in ok)), flush=True)


if __name__ == '__main__':
    main()
