"""Study B: thickness-preserving camber modes (upper and lower surface moved by the same displacement).

Mode k moves both surfaces by c * B_k (B_k of the F8D2000 basis or flap/twist); the thickness distribution and
volume of the parent are unchanged, so large amplitudes stay geometrically valid. Sweep c in AMPS (m), equal-lift
CD at Mach {6, 7, 8} x CL {0.02, 0.035, 0.05, 0.065}.

    python -m libdesign.sweep_camber --workers 4     # runs/library_design/sweep_camber.json
"""
import argparse
import multiprocessing as mp
import os

from . import geometry as Gm
from .fast_eval import equal_lift_cd
from .paths import OUT, save

MACHS = (6., 7., 8.)
CLS = (.02, .035, .05, .065)
AMPS = (-.3, -.2, -.15, -.1, -.06, -.03, 0., .03, .06, .1, .15, .2, .3)
MODES = ('front', 'rear', 'tail', 'span', 'flap', 'twist')


def job(args):
    parent, mode, c = args
    os.environ['OMP_NUM_THREADS'] = '1'
    lo, up = Gm.deform(Gm.parent(parent), [c], [c], (mode,))
    ok, why = Gm.check(lo, up)
    cd = equal_lift_cd(lo, up, MACHS, CLS)
    return dict(parent=parent, mode=mode, coef_m=c, geometry_ok=ok, why=why, cd=cd.tolist())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    jobs = [(par, m, c) for par in Gm.PARENTS for m in MODES for c in AMPS]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=4)
    save(OUT/'sweep_camber.json', dict(machs=MACHS, cls=CLS, amps=AMPS, rows=rows))


if __name__ == '__main__':
    main()
