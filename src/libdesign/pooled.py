"""Pooled static-gain bound: robust to individual optimiser failures.

For (parent, parameter set, bound scale K), the candidate shapes are the parent and every optimised shape
(specialists and compromises) of the same parent whose parameters lie within K x the limits - including the
runs of smaller scales and, for f8d_flap, the f8d runs (their flap/twist are zero). On the 12 conditions:
best fixed = the candidate with the lowest mean CD; selected = the lowest CD per condition; bound = 1 - mean
selected / mean best fixed.

    python -m libdesign.pooled
"""
import numpy as np

from . import geometry as Gm
from .fast_eval import equal_lift_cd
from .optimize import CLS, MACHS
from .paths import OUT, load, save


def main():
    d = load(OUT/'optimize.json')
    rows = [r for r in d['rows'] if r['cd_all'] is not None]
    out = {}
    for parent in Gm.PARENTS:
        base = np.asarray(equal_lift_cd(*Gm.parent(parent), MACHS, CLS)).ravel()
        for s in ('f8d', 'f8d_flap'):
            for K in (1., 2., 3.):
                cands = [('parent', base)]
                for r in rows:
                    if r['parent'] != parent or r['k'] > K+1e-9:
                        continue
                    if s == 'f8d' and r['set'] != 'f8d':
                        continue
                    cands.append((f"{r['set']}|k{r['k']:g}|{r['target']}", np.asarray(r['cd_all'], float).ravel()))
                C = np.array([c for _, c in cands])
                ok = np.all(np.isfinite(C), 1)
                C, names = C[ok], [n for n, o in zip(cands, ok) if o]
                fixed = int(np.argmin(C.mean(1)))
                sel = C.min(0)
                per = 1-sel/C[fixed]
                out[f'{parent}|{s}|{K:g}'] = dict(gain_bound=float(1-sel.mean()/C[fixed].mean()), candidates=len(C),
                                                   best_fixed=names[fixed][0], per_condition=per.round(4).tolist(),
                                                   fixed_vs_parent=float(1-C[fixed].mean()/base.mean()))
    save(OUT/'pooled.json', out)
    for k, v in out.items():
        print(k, 'bound %.2f%%' % (100*v['gain_bound']), 'max cell %.1f%%' % (100*max(v['per_condition'])),
              'best fixed %s' % v['best_fixed'], 'fixed vs parent %.1f%%' % (100*v['fixed_vs_parent']), v['candidates'])


if __name__ == '__main__':
    main()
