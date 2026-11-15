"""Headroom screening (experiments/perturb/screening/README.md): how far do simple dynamic-pressure laws stay
below the physical bound, with pressure-only aerodynamics and with skin friction?

Numeric RK4 (0.5 s) of the mission's point-mass model with CD = lower hull of the 298 geometries (same envelope
as certified_flight.bound; with friction, interpolated in Mach and altitude). CL is the control, updated every
0.5 s by the law

    CL = CL_level (1 + kp (q/q_ref(t) - 1)) - kg gamma,   q_ref(t) = r(t) q_max,   clipped to [0, 0.12].

Every RK stage is checked against ALL path limits (altitude, speed, flight-path angle, q, heat, load, Mach support);
a violation makes the flight infeasible. Families of r(t), each tuned by grid search on the nominal state:
  const      r(t) = r
  linear     r(t) = r0 + (r1 - r0) t/1200
  two-phase  r(t) = r0 for t < T, r1 afterwards

    python experiments/perturb/screening/screen.py --aero friction --out experiments/perturb/screening/friction.json
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
HERE = Path(__file__).resolve().parent

DT, T_END, CL_MAX = .5, 1200., .12
_W = {}


def world(aero):
    if _W.get('aero') != aero:
        from certified_flight import bound as B
        from certified_flight.aero import AeroDatabase
        from certified_flight.build import runtime_config
        from certified_flight.mission import Mission
        physical = runtime_config(ROOT)['physical']
        mission = Mission(ROOT, physical['mission'])
        db = AeroDatabase(ROOT/physical['snapshot'])
        cache = HERE/f'envelope_{aero}.npz'
        if cache.exists():
            z = np.load(cache)
            knots, lines, h_grid = z['knots'], z['lines'], (z['h_grid'] if 'h_grid' in z.files else None)
        else:
            fr = B.FrictionTable(ROOT/'data/aero_friction/friction_table.npz', db) if aero == 'friction' else None
            knots, lines, _ = B.envelope(db, friction=fr)
            h_grid = None if fr is None else fr.h
            np.savez(cache, knots=knots, lines=lines, **({} if h_grid is None else dict(h_grid=h_grid)))
        _W.update(aero=aero, B=B, mission=mission, model=mission.model(B.EnvelopeAero()), knots=knots, lines=lines,
                  h_grid=h_grid, lim=mission.data['limits'], support=(float(db.M[0]), float(db.M[-1])),
                  x0=np.array(mission.data['initial_state']['x_h_V_gamma_s'], float),
                  m=mission.data['model']['mass_kg'], S=mission.data['model']['reference_area_m2'])
    return _W


def margins(w, y, q):
    lim = w['lim']
    M = float(q['mach'])
    return min((y[0]-lim['altitude_m'][0])/1e4, (lim['altitude_m'][1]-y[0])/1e4,
               (y[1]-lim['velocity_mps'][0])/1e3, (lim['velocity_mps'][1]-y[1])/1e3,
               (y[2]-lim['gamma_rad'][0])/.1, (lim['gamma_rad'][1]-y[2])/.1,
               1-float(q['q_dyn'])/lim['dynamic_pressure_pa_max'], 1-float(q['heat_flux'])/lim['heat_proxy_wm2_max'],
               1-float(q['n_load'])/lim['load_factor_g_max'], M-w['support'][0], w['support'][1]-M)


def fly(aero, r_of_t, kp, kg, x0=None, log=False):
    w = world(aero)
    model, B = w['model'], w['B']
    x = np.array(w['x0'] if x0 is None else x0, float)
    s0 = x[3]
    worst, t, path, heat = np.inf, 0., [], 0.
    qmax = w['lim']['dynamic_pressure_pa_max']

    def fx(y, cl):
        M = y[1]/float(model.atm.speed_of_sound(y[0]))
        cd = B.hull_cd(w['knots'], w['lines'], M, cl, y[0], w['h_grid'])
        r, q = model.rhs_with_quantities(y, [cl], [cd])
        return np.asarray(r, float), q

    while t < T_END-1e-9:
        h, V, g = x[0], x[1], x[2]
        rho = float(model.atm.density(h))
        qd = .5*rho*V*V
        grav = float(model.atm.gravity(h, spherical=True))
        rr = model.atm.cfg.R_earth+h
        lvl = w['m']*(grav-V*V/rr)/(qd*w['S'])
        cl = float(np.clip(lvl*(1+kp*(qd/(r_of_t(t)*qmax)-1))-kg*g, 0., CL_MAX))
        k1, q1 = fx(x, cl)
        k2, q2 = fx(x+DT*k1/2, cl)
        k3, q3 = fx(x+DT*k2/2, cl)
        k4, q4 = fx(x+DT*k3, cl)
        heat += DT*(float(q1['heat_flux'])+2*float(q2['heat_flux'])+2*float(q3['heat_flux'])+float(q4['heat_flux']))/6
        for yy, qq in ((x, q1), (x+DT*k1/2, q2), (x+DT*k2/2, q3), (x+DT*k3, q4)):
            worst = min(worst, margins(w, yy, qq))
        x = x+DT*(k1+2*k2+2*k3+k4)/6
        t += DT
        if log:
            path.append((t, x[0], x[1], x[2], x[3], cl, qd))
        if not np.all(np.isfinite(x)) or worst < -1e-6:
            return dict(feasible=False, range_km=float((x[3]-s0)/1000), t=t, min_margin=float(worst), path=path,
                        heat_load=heat)
    return dict(feasible=True, range_km=float((x[3]-s0)/1000), t=t, min_margin=float(worst), path=path, heat_load=heat)


def law(spec):
    kind = spec['kind']
    if kind == 'const':
        return lambda t: spec['r']
    if kind == 'linear':
        return lambda t: spec['r0']+(spec['r1']-spec['r0'])*t/T_END
    if kind == 'two-phase':
        return lambda t: spec['r0'] if t < spec['T'] else spec['r1']
    raise ValueError(kind)


def job(args):
    aero, spec = args
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[k] = '1'
    r = fly(aero, law(spec), spec['kp'], spec['kg'])
    return dict(spec, feasible=r['feasible'], range_km=r['range_km'], min_margin=r['min_margin'], t_end=r['t'],
                heat_load=r['heat_load'])


def best(rows):
    ok = [r for r in rows if r['feasible']]
    return max(ok, key=lambda r: r['range_km']) if ok else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aero', choices=['pressure', 'friction'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    world(a.aero)                                   # builds and caches the envelope once
    R = [round(x, 2) for x in np.arange(.30, 1.0001, .05)]
    gains = list(itertools.product((1., 2., 4.), (.3, 1., 3.)))
    t0 = time.time()
    out = dict(aero=a.aero)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        const = pool.map(job, [(a.aero, dict(kind='const', r=r, kp=kp, kg=kg)) for r in R for kp, kg in gains])
        out['const'] = dict(best=best(const), grid=const)
        b = out['const']['best']
        print('const', b, '%.0f s' % (time.time()-t0), flush=True)
        kp, kg = (b['kp'], b['kg']) if b else (2., 1.)
        lin = pool.map(job, [(a.aero, dict(kind='linear', r0=r0, r1=r1, kp=kp, kg=kg)) for r0 in R for r1 in R])
        out['linear'] = dict(best=best(lin), grid=lin)
        print('linear', out['linear']['best'], '%.0f s' % (time.time()-t0), flush=True)
        two = pool.map(job, [(a.aero, dict(kind='two-phase', r0=r0, r1=r1, T=T, kp=kp, kg=kg))
                             for r0 in R for r1 in R for T in (60., 120., 240., 480.)])
        out['two-phase'] = dict(best=best(two), grid=two)
        print('two-phase', out['two-phase']['best'], '%.0f s' % (time.time()-t0), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
