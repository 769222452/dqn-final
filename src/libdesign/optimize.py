"""Study C: how large can the static morphing gain be in the fixed-leading-edge family? (P2 model)

For a parent, a parameter set and a bound scale k (k x the F8D2000 limits; flap/twist limits 0.03 m x k):
  specialist(c)  the shape minimising equal-lift CD at condition c
  compromise     the shape minimising the mean CD over all conditions (the best fixed shape)
static gain bound = 1 - mean_c CD_specialist(c) / mean_c CD_compromise(c), on CONDS (Mach 6, 7, 8 x CL 0.02,
0.035, 0.05, 0.065). Constraints: geometry valid (no fold-back, non-negative thickness), P2 valid at the
condition, volume >= VMIN (the smallest F8D2000 volume, 0.4687 m^3). Bounded Powell, MAXFEV evaluations,
started from the parent.

    python -m libdesign.optimize --workers 4      # runs/library_design/optimize.json
"""
import argparse
import itertools
import multiprocessing as mp
import os

import numpy as np
from scipy.optimize import minimize

from . import geometry as Gm
from .fast_eval import equal_lift_cd
from .paths import OUT, save

MACHS = (6., 7., 8.)
CLS = (.02, .035, .05, .065)
CONDS = list(itertools.product(MACHS, CLS))
VMIN = .4687
MAXFEV = 400
SETS = dict(f8d=('front', 'rear', 'tail', 'span'), f8d_flap=('front', 'rear', 'tail', 'span', 'flap', 'twist'))


def limits(names, k):
    out = []
    for surf in ('upper', 'lower'):
        for n in names:
            base = Gm.F8D_LIMITS[surf][Gm.F8D_NAMES.index(n)] if n in Gm.F8D_NAMES else .03
            out.append(k*base)
    return np.array(out)


def shape(par, x, names):
    n = len(names)
    return Gm.deform(par, x[:n], x[n:], names)


_FIXED = {}


def fixed(parent_name, par):
    if parent_name not in _FIXED:
        _FIXED[parent_name] = Gm.Fixed(par)
    return _FIXED[parent_name]


def evaluate(par, x, names, machs, cls, F):
    lo, up = shape(par, x, names)
    if not F.check(lo, up):
        return None, 0.
    v = F.volume(lo, up)
    if v < VMIN:
        return None, v
    return equal_lift_cd(lo, up, machs, cls), v


def run(args):
    parent, setname, k, target = args          # target: index into CONDS, or 'mean'
    os.environ['OMP_NUM_THREADS'] = '1'
    names = SETS[setname]
    par = Gm.parent(parent)
    F = fixed(parent, par)
    L = limits(names, k)
    if target == 'mean':
        machs, cls = MACHS, CLS
    else:
        machs, cls = (CONDS[target][0],), (CONDS[target][1],)

    def f(x):
        cd, _ = evaluate(par, x, names, machs, cls, F)
        if cd is None or not np.all(np.isfinite(cd)):
            return 1e3                      # invalid shape or condition not reached: far above any CD x 1e3
        return float(np.mean(cd))*1e3
    x0 = np.zeros(len(L))
    res = minimize(f, x0, method='Powell', bounds=list(zip(-L, L)),
                   options=dict(maxfev=MAXFEV, xtol=1e-4, ftol=1e-7, direc=np.diag(.1*L)))   # small first steps
    x = res.x
    if res.fun >= 1e3:                     # nothing valid found: keep the parent
        x = np.zeros(len(L))
    cd_all, vol = evaluate(par, x, names, MACHS, CLS, F)
    return dict(parent=parent, set=setname, k=k, target=target, x=x.tolist(), limits=L.tolist(),
                objective=float(res.fun)/1e3, nfev=int(res.nfev),
                cd_all=None if cd_all is None else cd_all.tolist(), volume=vol,
                at_bound=[bool(abs(abs(xi)-li) < 1e-3*li) for xi, li in zip(x, L)])


def summarise(rows):
    out = {}
    keys = sorted({(r['parent'], r['set'], r['k']) for r in rows})
    for key in keys:
        rs = [r for r in rows if (r['parent'], r['set'], r['k']) == key]
        comp = [r for r in rs if r['target'] == 'mean']
        spec = {r['target']: r for r in rs if r['target'] != 'mean'}
        if not comp or len(spec) < len(CONDS):
            continue
        cc = np.array(comp[0]['cd_all'], float).ravel()
        cs = np.array([np.array(spec[i]['cd_all'], float).ravel()[i] if spec[i]['cd_all'] is not None else np.nan
                       for i in range(len(CONDS))])
        cs = np.fmin(cs, cc)                            # a specialist is never worse than the compromise
        if not np.all(np.isfinite(cc)):
            continue
        out['%s|%s|%g' % key] = dict(gain_bound=float(1-cs.mean()/cc.mean()),
                                     per_condition=(1-cs/cc).round(4).tolist(), compromise_x=comp[0]['x'],
                                     compromise_volume=comp[0]['volume'])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--parents', default=','.join(Gm.PARENTS))
    p.add_argument('--sets', default='f8d,f8d_flap')
    p.add_argument('--ks', default='1,2,3')
    a = p.parse_args(argv)
    path = OUT/'optimize.json'
    from .paths import load
    rows = load(path)['rows'] if path.exists() else []
    done = {(r['parent'], r['set'], r['k'], r['target']) for r in rows}
    jobs = [(par, s, float(k), t) for par in a.parents.split(',') for s in a.sets.split(',')
            for k in a.ks.split(',') for t in ['mean']+list(range(len(CONDS)))]
    jobs = [j for j in jobs if j not in done]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for r in pool.imap_unordered(run, jobs):
            rows.append(r)
            save(path, dict(conds=CONDS, rows=rows, summary=summarise(rows)))
    s = summarise(rows)
    for k, v in sorted(s.items()):
        print(k, 'gain bound %.2f%%' % (100*v['gain_bound']), 'volume %.3f' % v['compromise_volume'])


if __name__ == '__main__':
    main()
