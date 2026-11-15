"""Where is the nominal headroom? (pressure-only aerodynamics)

  replay   fly the relaxed bound's CL(t) for the first T seconds, then hand over to the dynamic-pressure law
           (small grid of r, kp, kg; best feasible). Shows how much of the bound-minus-law gap sits in the
           first T seconds.
  states   for a grid of initial states: range of every law of a fixed family. Gives the best single law that
           is feasible on all states (robust law) and the best law per state (state-adaptive, same family).

    python experiments/perturb/screening/pullout.py replay --out experiments/perturb/screening/replay.json
    python experiments/perturb/screening/pullout.py states --out experiments/perturb/screening/states.json
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

ROOT = S.ROOT


def fly_ctrl(ctrl, x0=None, log=False):
    """screen.fly with an arbitrary controller ctrl(t, x, level, q) -> CL."""
    w = S.world('pressure')
    model, B = w['model'], w['B']
    x = np.array(w['x0'] if x0 is None else x0, float)
    s0, worst, t, path = x[3], np.inf, 0., []

    def fx(y, cl):
        M = y[1]/float(model.atm.speed_of_sound(y[0]))
        cd = B.hull_cd(w['knots'], w['lines'], M, cl, y[0], w['h_grid'])
        r, q = model.rhs_with_quantities(y, [cl], [cd])
        return np.asarray(r, float), q

    while t < S.T_END-1e-9:
        h, V = x[0], x[1]
        qd = .5*float(model.atm.density(h))*V*V
        grav = float(model.atm.gravity(h, spherical=True))
        lvl = w['m']*(grav-V*V/(model.atm.cfg.R_earth+h))/(qd*w['S'])
        cl = float(np.clip(ctrl(t, x, lvl, qd), 0., S.CL_MAX))
        k1, q1 = fx(x, cl)
        k2, q2 = fx(x+S.DT*k1/2, cl)
        k3, q3 = fx(x+S.DT*k2/2, cl)
        k4, q4 = fx(x+S.DT*k3, cl)
        for yy, qq in ((x, q1), (x+S.DT*k1/2, q2), (x+S.DT*k2/2, q3), (x+S.DT*k3, q4)):
            worst = min(worst, S.margins(w, yy, qq))
        x = x+S.DT*(k1+2*k2+2*k3+k4)/6
        t += S.DT
        if log:
            path.append((t, x[0], x[1], x[2], x[3], cl, qd))
        if not np.all(np.isfinite(x)) or worst < -1e-6:
            return dict(feasible=False, range_km=float((x[3]-s0)/1000), t=t, min_margin=float(worst), path=path)
    return dict(feasible=True, range_km=float((x[3]-s0)/1000), t=t, min_margin=float(worst), path=path)


def law_ctrl(r_of_t, kp, kg):
    qmax = S.world('pressure')['lim']['dynamic_pressure_pa_max']
    return lambda t, x, lvl, qd: lvl*(1+kp*(qd/(r_of_t(t)*qmax)-1))-kg*x[2]


def replay_job(a):
    T, r, kp, kg = a
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    sol = json.loads((ROOT/'experiments/improve/bounds/dh0_dv0.json').read_text())['relaxed']
    tb, cb = np.array(sol['t'][:-1]), np.array(sol['CL'])
    law = law_ctrl(lambda t: r, kp, kg)

    def ctrl(t, x, lvl, qd):
        return cb[min(np.searchsorted(tb, t+1e-9, side='right')-1, len(cb)-1)] if t < T else law(t, x, lvl, qd)
    o = fly_ctrl(ctrl)
    return dict(T=T, r=r, kp=kp, kg=kg, feasible=o['feasible'], range_km=o['range_km'], min_margin=o['min_margin'])


def family():
    R = [round(x, 2) for x in np.arange(.50, 1.0001, .05)]
    gains = list(itertools.product((1., 2., 4.), (.3, 1., 3.)))
    specs = [dict(kind='const', r=r, kp=kp, kg=kg) for r in R for kp, kg in gains]
    specs += [dict(kind='two-phase', r0=r0, r1=r1, T=T, kp=kp, kg=kg) for r0 in (.5, .6, .7, .8, .9)
              for r1 in (.9, .95, 1.) for T in (60., 120., 240.) for kp, kg in ((2., 1.), (4., 1.), (4., 3.))]
    return specs


def state_job(a):
    x0, i = a
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    spec = family()[i]
    o = fly_ctrl(law_ctrl(S.law(spec), spec['kp'], spec['kg']), x0=x0)
    return (o['range_km'] if o['feasible'] else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['replay', 'states'])
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    w = S.world('pressure')
    with mp.get_context('spawn').Pool(a.workers) as pool:
        if a.mode == 'replay':
            jobs = [(T, r, kp, kg) for T in (0., 30., 60., 90., 120., 180., 240., 480., 1200.)
                    for r in (.94, .96, .98, .99, 1.) for kp in (2., 4., 8.) for kg in (1., 3.)]
            rows = pool.map(replay_job, jobs)
            out = dict(rows=rows, best={})
            for T in sorted({r['T'] for r in rows}):
                ok = [r for r in rows if r['T'] == T and r['feasible']]
                out['best'][str(T)] = max(ok, key=lambda r: r['range_km']) if ok else None
                print(T, out['best'][str(T)], flush=True)
        else:
            x0 = np.array(w['x0'], float)
            states = [dict(dh=dh, dv=dv, dg=dg) for dh in (-4000., 0., 4000., 8000.) for dv in (-200., 0.)
                      for dg in (-.04, 0., .04)]
            states += [dict(dh=dh, dv=dv, dg=0.) for dh, dv in ((-2000., 0.), (2000., 0.), (0., -100.), (0., 100.))]
            n = len(family())
            out = dict(family=family(), states=[])
            for st in states:
                y = x0+np.array([st['dh'], st['dv'], st['dg'], 0.])
                res = pool.map(state_job, [(y.tolist(), i) for i in range(n)], chunksize=8)
                out['states'].append(dict(st, ranges=res))
                ok = [(v, i) for i, v in enumerate(res) if v is not None]
                print(st, 'feasible %d/%d' % (len(ok), n), 'best', max(ok) if ok else None, flush=True)
                Path(a.out).write_text(json.dumps(out))
        Path(a.out).write_text(json.dumps(out, indent=None if a.mode == 'states' else 1))


if __name__ == '__main__':
    main()
