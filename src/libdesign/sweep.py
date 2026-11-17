"""Study A: one-parameter sweeps beyond the F8D2000 limits.

For each parent, surface and basis function: coefficient c = s * L for s in SCALES (L = the F8D2000 limit of
that parameter; flap/twist use 0.03 m), all other coefficients zero. Equal-lift CD at Mach {6, 7, 8} and
CL {0.02, 0.035, 0.05, 0.065}, validity (fold-back, non-negative thickness, P2 validity), volume.

    python -m libdesign.sweep --workers 4      # writes runs/library_design/sweep.json
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

from . import geometry as Gm
from .fast_eval import equal_lift_cd
from .paths import OUT, save

MACHS = (6., 7., 8.)
CLS = (.02, .035, .05, .065)
SCALES = (-5, -4, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 4, 5)
BASES = ('front', 'rear', 'tail', 'span', 'flap', 'twist')
EXTRA_LIMIT = .03


def limit(surface, name):
    if name in Gm.F8D_NAMES:
        return Gm.F8D_LIMITS[surface][Gm.F8D_NAMES.index(name)]
    return EXTRA_LIMIT


def job(args):
    parent, surface, name, s = args
    os.environ['OMP_NUM_THREADS'] = '1'
    par = Gm.parent(parent)
    coef = [s*limit(surface, name)]
    up_c, lo_c = (coef, [0.]) if surface == 'upper' else ([0.], coef)
    lo, up = Gm.deform(par, up_c, lo_c, (name,))
    ok, why = Gm.check(lo, up)
    cd = equal_lift_cd(lo, up, MACHS, CLS)
    return dict(parent=parent, surface=surface, basis=name, scale=s, coef_m=coef[0], geometry_ok=ok, why=why,
                volume=Gm.volume(lo, up), cd=cd.tolist())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    jobs = [(par, surf, b, s) for par in Gm.PARENTS for surf in ('upper', 'lower') for b in BASES for s in SCALES]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=4)
    save(OUT/'sweep.json', dict(machs=MACHS, cls=CLS, scales=SCALES, rows=rows))


if __name__ == '__main__':
    main()
