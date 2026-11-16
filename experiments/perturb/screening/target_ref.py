"""Screening B reference: could the target be hit in this scenario at all (clairvoyant, three-phase plan)?

For each (scenario, target): the disturbances are known exactly; the plan is (r, w) per phase (0-400, 400-800,
800-1200 s) as in reach.py. Warm start: the nominal reach.json flight closest to the target; then Nelder-Mead
(MAXFEV evaluations) on the miss  (dD/1 km)^2 + (dV/10 m/s)^2, with violations penalised. An inner approximation
of the reachable set: a miss here does not prove the target unreachable, a hit proves it reachable.

    python experiments/perturb/screening/target_ref.py --level L2 --seeds 2000-2009 --out experiments/perturb/screening/target_ref_L2.json
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reach as RC  # noqa: E402
import screen as S  # noqa: E402
import target as TG  # noqa: E402

MAXFEV = 150


def fly_plan(p, sc, krho, mode='pc2'):
    m = TG.models()['m']
    W = RC.tables()['w']
    x = np.array(W['x0'], float)+np.array([sc.dh, sc.dv, 0., 0.])
    t, worst = 0., np.inf
    while t < S.T_END-1e-9:
        k = 0 if t < RC.PHASES[0] else (1 if t < RC.PHASES[1] else 2)
        r, w = float(np.clip(p[k], .3, 1.)), float(np.clip(p[3+k], 0., 1.))
        m.atm.k_rho = krho(x[3])
        cl = TG.law_cl(m, x, r)
        x, wst = TG.rk4(m, x, cl, S.DT, mode, w, sc.k_lift, sc.k_drag, krho, check=True)
        worst = min(worst, wst)
        t += S.DT
        if not np.all(np.isfinite(x)) or worst < -1e-6:
            return False, t, x
    return True, t, x


def job(a):
    tn, level, seed = a
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    FF = TG.models()['FF']
    sc = FF.scenario(level, seed)
    s_grid, k_grid = FF.density_field(sc)
    krho = lambda s: float(np.interp(s, s_grid, k_grid))
    D, V = TG.TARGETS[tn]
    rows = [r for r in json.loads((HERE/'reach.json').read_text())['rows'] if r['mode'] == 'morph' and r['feasible']]
    start = min(rows, key=lambda r: (r['range_km']-D)**2+((r['V_end']-V)/10)**2)
    best = dict(cost=np.inf)

    def cost(p):
        ok, t, x = fly_plan(p, sc, krho)
        dD, dV = x[3]/1e3-D, x[1]-V
        c = dD**2+(dV/10)**2 if ok else 1e6+(S.T_END-t)*1e3
        if c < best['cost']:
            best.update(cost=c, p=[float(v) for v in p], feasible=ok, dD_km=float(dD), dV_mps=float(dV))
        return c
    minimize(cost, np.r_[start['r'], start['w']], method='Nelder-Mead',
             options=dict(maxfev=MAXFEV, initial_simplex=None, xatol=1e-3, fatol=1e-2))
    hit = best['feasible'] and abs(best['dD_km']) <= TG.TOL_D and abs(best['dV_mps']) <= TG.TOL_V
    return dict(target=tn, level=level, seed=seed, hit=bool(hit), **{k: best[k] for k in ('p', 'feasible', 'dD_km', 'dV_mps')})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--level', default='L2')
    p.add_argument('--seeds', default='2000-2009')
    p.add_argument('--targets', default=','.join(TG.TARGETS))
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    lo, hi = map(int, a.seeds.split('-'))
    out = Path(a.out)
    rows = json.loads(out.read_text())['rows'] if out.exists() else []          # resumable: finished jobs are kept
    done = {(r['target'], r['seed']) for r in rows}
    jobs = [(tn, a.level, s) for tn in a.targets.split(',') for s in range(lo, hi+1) if (tn, s) not in done]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows.append(r)
            out.write_text(json.dumps(dict(level=a.level, maxfev=MAXFEV, rows=rows), indent=1))
            print(r['target'], r['seed'], 'hit' if r['hit'] else 'miss', flush=True)
    for tn in a.targets.split(','):
        rr = [r for r in rows if r['target'] == tn]
        print(tn, 'reference hit %d/%d' % (sum(r['hit'] for r in rr), len(rr)), flush=True)


if __name__ == '__main__':
    main()
