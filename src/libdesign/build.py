"""Build a geometry library: shapes -> full P2 tables in the F8D2000/L298 npz format.

A design is a list of shapes (parent, names, upper coefficients, lower coefficients). Each shape is screened
(geometry valid, volume >= VMIN) and evaluated on the 18 x 20 Mach-alpha grid; shapes with any P2-invalid grid
point are dropped (the flight code needs complete tables). Output npz keys: geometry_ids, mach_grid,
alpha_deg_grid, CL, CD_pressure, Cm_pressure, model_valid, usable_p2, parents, names, coefficients, volume.

Designs (DESIGNS):
  f8d_x1     F8D2000 method, Latin hypercube in the 8 F8D coefficients, bounds x1 (re-sampled here)
  f8d_x3     same, bounds x3
  ladder     per parent: the parent, and a ladder of thickness-preserving camber 'brakes' (front, rear, flap
             camber modes at graded amplitudes) and aft-thinning shapes (upper tail +, lower tail -)
  hybrid     f8d_x3 samples + ladder

    python -m libdesign.build --design f8d_x3 --n 600 --workers 4
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
from scipy.stats import qmc

from . import geometry as Gm
from . import p2 as P
from .paths import OUT

VMIN = .4687
F8D = Gm.F8D_NAMES


def lhs_design(k, n, seed):
    lim = np.array(Gm.F8D_LIMITS['upper']+Gm.F8D_LIMITS['lower'])*k
    out = []
    for pi, par in enumerate(Gm.PARENTS):
        x = qmc.LatinHypercube(d=8, seed=seed+pi).random(int(np.ceil(n/len(Gm.PARENTS))*3))
        x = (2*x-1)*lim
        out += [(par, F8D, list(v[:4]), list(v[4:])) for v in x]
    return out


def ladder_design():
    out = []
    for par in Gm.PARENTS:
        out.append((par, F8D, [0]*4, [0]*4))
        for mode in ('front', 'rear', 'flap'):
            for c in (.02, .04, .06, .08, .1, .13, .16, -.02, -.04):
                out.append((par, (mode,), [c], [c]))
        for k in (.5, 1, 1.5, 2, 2.5, 3):
            out.append((par, F8D, [0, 0, k*.04, 0], [0, 0, -k*.035, 0]))
    return out


def evaluate(shape):
    parent, names, cu, cl = shape
    os.environ['OMP_NUM_THREADS'] = '1'
    par = Gm.parent(parent)
    lo, up = Gm.deform(par, cu, cl, names)
    F = Gm.Fixed(par)
    if not F.check(lo, up):
        return dict(ok=False, why='geometry')
    v = F.volume(lo, up)
    if v < VMIN:
        return dict(ok=False, why='volume')
    t = P.grid_table(lo, up)
    if not np.all(t['valid']):
        return dict(ok=False, why='p2_invalid')
    return dict(ok=True, CL=t['CL'], CD=t['CD'], Cm=t['Cm'], volume=v, parent=parent, names=list(names),
                cu=list(map(float, cu)), cl=list(map(float, cl)))


def _run(shapes, workers):
    with mp.get_context('spawn').Pool(workers) as pool:
        return pool.map(evaluate, shapes, chunksize=2)


def build(design, n=600, seed=20261117, workers=4, out=None):
    """LHS parts keep the first n accepted shapes (parents interleaved); ladders keep all accepted shapes."""
    parts = []
    if design in ('f8d_x1', 'f8d_x3', 'hybrid'):
        k = 1. if design == 'f8d_x1' else 3.
        sh = lhs_design(k, n, seed)
        order = np.argsort([i % (len(sh)//len(Gm.PARENTS)) for i in range(len(sh))], kind='stable')
        parts.append(('lhs', [sh[i] for i in order]))
    if design in ('ladder', 'hybrid'):
        parts.append(('ladder', ladder_design()))
    if not parts:
        raise ValueError(design)
    ok, res = [], []
    for kind, shapes in parts:
        r = _run(shapes, workers)
        res += r
        good = [x for x in r if x['ok']]
        ok += good[:n] if kind == 'lhs' else good
    shapes = [s for _, sh in parts for s in sh]
    why = {}
    for r in res:
        if not r['ok']:
            why[r['why']] = why.get(r['why'], 0)+1
    ids = np.array([f'{design}_{i:05d}' for i in range(len(ok))])
    path = out or OUT/'libraries'/f'{design}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.ones((len(ok), 18, 20), bool)
    names = [','.join(r['names']) for r in ok]
    width = max(len(r['cu']) for r in ok)
    pad = lambda v: list(v)+[0.]*(width-len(v))
    np.savez(path, geometry_ids=ids, mach_grid=P.MACH_GRID, alpha_deg_grid=P.ALPHA_GRID_DEG,
             CL=np.array([r['CL'] for r in ok]), CD_pressure=np.array([r['CD'] for r in ok]),
             Cm_pressure=np.array([r['Cm'] for r in ok]), model_valid=valid, usable_p2=valid,
             parents=np.array([r['parent'] for r in ok]), names=np.array(names),
             upper=np.array([pad(r['cu']) for r in ok]), lower=np.array([pad(r['cl']) for r in ok]),
             volume=np.array([r['volume'] for r in ok]))
    return path, dict(design=design, tried=len(shapes), kept=len(ok), rejected=why)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--design', required=True)
    p.add_argument('--merge', default=None, help='comma list of npz paths to concatenate into --design')
    p.add_argument('--n', type=int, default=600)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    if a.merge:
        print(merge(a.design, a.merge.split(',')))
        return
    path, info = build(a.design, a.n, workers=a.workers)
    print(path, info)



def merge(name, parts):
    """Concatenate library npz files (same grids) into runs/library_design/libraries/<name>.npz."""
    tabs = [np.load(p, allow_pickle=False) for p in parts]
    for t in tabs[1:]:
        assert np.array_equal(t['mach_grid'], tabs[0]['mach_grid']) and np.array_equal(t['alpha_deg_grid'], tabs[0]['alpha_deg_grid'])
    ids = np.concatenate([np.array([f'{i}:{g}' for g in t['geometry_ids']]) for i, t in enumerate(tabs)])
    out = OUT/'libraries'/f'{name}.npz'
    keys = ('CL', 'CD_pressure', 'Cm_pressure', 'model_valid', 'usable_p2')
    np.savez(out, geometry_ids=ids, mach_grid=tabs[0]['mach_grid'], alpha_deg_grid=tabs[0]['alpha_deg_grid'],
             **{k: np.concatenate([t[k] for t in tabs]) for k in keys})
    return out


if __name__ == '__main__':
    main()
