"""Screening B: fixed-time target arrival (range D*, speed V* at t = 1200 s) with a classical predictor-corrector.

Truth: the point-mass RK4 of reach.py (0.5 s, every RK stage checked against all path limits) with the
disturbances of fastflight (scenario(level, seed): initial offset, along-track density field, lift/drag biases).
Control every 0.5 s: dynamic-pressure law with set point r (CL), drag fraction w (morph) as in reach.py.

Guidance (every CYCLE s): choose (r, w) held to the end so that the predicted (range, speed) at 1200 s hits the
target. Prediction: same model, RK4 2 s, density ratio frozen at the current measured value, the TRUE lift/drag
biases (a generous baseline). Correction: damped Newton with finite differences, NEWTON iterations per cycle,
warm-started from the previous cycle; a step is accepted only if its predicted trajectory keeps PRED_MARGIN on
every path limit and does not increase the predicted miss (backtracking 1, 1/2, 1/4).
  pc2   morphing: unknowns (r, w), targets (D*, V*)
  pc1   G42 only: unknown r, target D* only (speed not controllable)

Success: no limit violation, |dD| <= 5 km, |dV| <= 25 m/s (pc1: speed error reported, not required).

    python experiments/perturb/screening/target.py --level L2 --seeds 2000-2019 --out experiments/perturb/screening/target_L2.json
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reach as RC  # noqa: E402
import screen as S  # noqa: E402

TARGETS = {'T1': (2350., 1520.), 'T2': (2250., 1420.), 'T3': (2150., 1300.),     # inside the G42-only set
           'T4': (2300., 1250.), 'T5': (2100., 1150.), 'T6': (1950., 1100.)}     # morphing only
# T1-T3 are interior points of the nominal G42-only set (reach.json), T4-T6 lie only in the morphing set.
CYCLE, NEWTON, DT_PRED = 24., 2, 2.
PRED_MARGIN = .005           # predicted trajectories must keep this normalised margin on every limit
TOL_D, TOL_V = 5., 25.
_M = {}


def models():
    if not _M:
        from certified_flight import fastflight as FF
        from certified_flight.bound import EnvelopeAero
        from certified_flight.headroom import ScaledAtmosphere
        w = RC.tables()['w']
        m = w['mission'].model(EnvelopeAero())
        m.atm = ScaledAtmosphere(m.atm, 1.)
        _M.update(m=m, FF=FF)
    return _M


def deriv(m, y, cl, mode, w, kL, kD):
    M = y[1]/float(m.atm.speed_of_sound(y[0]))
    if mode == 'pc1':
        cd = RC.interp2(RC.tables()['fx'], M, cl)
    else:
        a, b = RC.interp2(RC.tables()['lo'], M, cl), RC.interp2(RC.tables()['hi'], M, cl)
        cd = a+w*(b-a)
    if not np.isfinite(cd):
        cd = 1.
    r, q = m.rhs_with_quantities(y, [kL*cl], [kD*cd])
    return np.asarray(r, float), q


def law_cl(m, x, r, kp=2., kg=1.):
    W = RC.tables()['w']
    h, V = x[0], x[1]
    qd = .5*float(m.atm.density(h))*V*V
    grav = float(m.atm.gravity(h, spherical=True))
    lvl = W['m']*(grav-V*V/(m.atm.cfg.R_earth+h))/(qd*W['S'])
    return float(np.clip(lvl*(1+kp*(qd/(r*W['lim']['dynamic_pressure_pa_max'])-1))-kg*x[2], 0., RC.CLT[-1]))


def rk4(m, x, cl, dt, mode, w, kL, kD, krho=None, check=False):
    W = RC.tables()['w']
    worst = np.inf
    stages = []

    def f(y):
        if krho is not None:
            m.atm.k_rho = krho(y[3])
        return deriv(m, y, cl, mode, w, kL, kD)
    k1, q1 = f(x)
    k2, q2 = f(x+dt*k1/2)
    k3, q3 = f(x+dt*k2/2)
    k4, q4 = f(x+dt*k3)
    if check:
        for yy, qq in ((x, q1), (x+dt*k1/2, q2), (x+dt*k2/2, q3), (x+dt*k3, q4)):
            worst = min(worst, S.margins(W, yy, qq))
    return x+dt*(k1+2*k2+2*k3+k4)/6, worst


def predict(x, t, r, w, mode, kL, kD, krho_now):
    m = models()['m']
    m.atm.k_rho = krho_now
    y = np.array(x, float)
    worst = np.inf
    while t < S.T_END-1e-9:
        dt = min(DT_PRED, S.T_END-t)
        cl = law_cl(m, y, r)
        y, wst = rk4(m, y, cl, dt, mode, w, kL, kD, check=True)
        worst = min(worst, wst)
        t += dt
        if not np.all(np.isfinite(y)) or y[0] < 15e3:
            return np.array([y[3]/1e3-1e4, 0.]), False      # crashed prediction: very short
    return np.array([y[3]/1e3, y[1]]), worst >= PRED_MARGIN


def correct(x, t, u, mode, target, kL, kD, krho_now, iterations=None):
    lo, hi = (np.array([.3]), np.array([1.])) if mode == 'pc1' else (np.array([.3, 0.]), np.array([1., 1.]))
    u = np.clip(np.array(u, float), lo, hi)
    tgt = np.array(target[:1] if mode == 'pc1' else target)
    scale = np.array([1., 10.])[:len(tgt)]                    # 1 km ~ 10 m/s
    args = lambda v: (list(v) if mode != 'pc1' else [v[0], 0.])
    for _ in range(NEWTON if iterations is None else iterations):
        f0, ok0 = predict(x, t, *args(u), mode, kL, kD, krho_now)
        f0 = f0[:len(tgt)]
        e = (f0-tgt)/scale
        J = np.zeros((len(tgt), len(u)))
        for j in range(len(u)):
            du = np.zeros(len(u))
            du[j] = .02 if j == 0 else .05
            uu = np.clip(u+du, lo, hi)
            if uu[j] == u[j]:
                uu = np.clip(u-du, lo, hi)
            fj = predict(x, t, *args(uu), mode, kL, kD, krho_now)[0][:len(tgt)]
            J[:, j] = ((fj-tgt)/scale-e)/(uu[j]-u[j])
        step = np.linalg.lstsq(J, -e, rcond=None)[0]
        step = np.clip(step, -np.array([.15, .3])[:len(u)], np.array([.15, .3])[:len(u)])
        for frac in (1., .5, .25):                       # backtrack: keep predicted trajectories inside the limits
            un = np.clip(u+frac*step, lo, hi)
            fn, okn = predict(x, t, *args(un), mode, kL, kD, krho_now)
            if okn and (not ok0 or np.sum(((fn[:len(tgt)]-tgt)/scale)**2) <= np.sum(e**2)):
                u = un
                break
    return u


def fly(mode, target_name, level, seed, log=False):
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    FF = models()['FF']
    m = models()['m']
    sc = FF.scenario(level, seed)
    s_grid, k_grid = FF.density_field(sc)
    krho = lambda s: float(np.interp(s, s_grid, k_grid))
    W = RC.tables()['w']
    x = np.array(W['x0'], float)+np.array([sc.dh, sc.dv, 0., 0.])
    target = TARGETS[target_name]
    u = np.array([.8]) if mode == 'pc1' else np.array([.8, 0.])
    t, worst, path, n_pred = 0., np.inf, [], 0
    next_g = 0.
    while t < S.T_END-1e-9:
        if t >= next_g-1e-9:
            u = correct(x, t, u, mode, target, sc.k_lift, sc.k_drag, krho(x[3]), iterations=8 if t == 0 else None)
            next_g += CYCLE
        m.atm.k_rho = krho(x[3])
        cl = law_cl(m, x, u[0])
        x, wst = rk4(m, x, cl, S.DT, mode, 0. if mode == 'pc1' else u[1], sc.k_lift, sc.k_drag, krho, check=True)
        worst = min(worst, wst)
        t += S.DT
        if log:
            path.append([t, *x.tolist(), cl, *u.tolist()])
        if not np.all(np.isfinite(x)) or worst < -1e-6:
            break
    ok = bool(t >= S.T_END-1e-9 and worst >= -1e-6)
    dD, dV = x[3]/1e3-target[0], x[1]-target[1]
    hit = ok and abs(dD) <= TOL_D and (mode == 'pc1' or abs(dV) <= TOL_V)
    out = dict(mode=mode, target=target_name, level=level, seed=seed, feasible=ok, t_end=t, dD_km=float(dD),
               dV_mps=float(dV), hit=bool(hit), min_margin=float(worst))
    if log:
        out['path'] = path
    return out


def job(a):
    return fly(*a)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', default='L2')
    p.add_argument('--seeds', default='2000-2019')
    p.add_argument('--targets', default=','.join(TARGETS))
    p.add_argument('--modes', default='pc2,pc1')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    lo, hi = map(int, a.seeds.split('-'))
    jobs = []
    for mode in a.modes.split(','):
        for tn in a.targets.split(','):
            if mode == 'pc1' and tn in ('T4', 'T5', 'T6'):
                continue
            jobs += [(mode, tn, a.level, s) for s in range(lo, hi+1)]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    Path(a.out).write_text(json.dumps(dict(level=a.level, cycle=CYCLE, tol=[TOL_D, TOL_V], targets=TARGETS,
                                           rows=rows), indent=1))
    for mode in a.modes.split(','):
        for tn in a.targets.split(','):
            rr = [r for r in rows if r['mode'] == mode and r['target'] == tn]
            if rr:
                fe = [r for r in rr if r['feasible']]
                print('%s %s hit %d/%d feasible %d  |dD| med %.1f max %.1f  |dV| med %.0f max %.0f' % (
                    mode, tn, sum(r['hit'] for r in rr), len(rr), len(fe),
                    np.median([abs(r['dD_km']) for r in fe]) if fe else np.nan,
                    max([abs(r['dD_km']) for r in fe]) if fe else np.nan,
                    np.median([abs(r['dV_mps']) for r in fe]) if fe else np.nan,
                    max([abs(r['dV_mps']) for r in fe]) if fe else np.nan), flush=True)


if __name__ == '__main__':
    main()
