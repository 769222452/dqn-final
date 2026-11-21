import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
AIMS = ['center', 'fast', 'far+fast', 'far', 'near']
keys = [(s, fp) for s in range(1500, 1596) for fp in (True, False)]
jobs = [('known', 0., k) for k in keys]+[(a, t, k) for t in (400., 700.) for a in AIMS for k in keys]
assert len(jobs) == len(rows)
tab = {(a, t, k): r for (a, t, k), r in zip(jobs, rows)}
def summ(rr, label):
    M = [r for (k, r) in rr if not k[1]]
    print('  %-34s success %5.1f%% (morph-only %5.1f%%) violations %4.1f%%  mean return %+.3f' % (label, 100*np.mean([r['hit'] for _, r in rr]), 100*np.mean([r['hit'] for r in M]), 100*np.mean([not r['feasible'] for _, r in rr]), np.mean([r['G'] for _, r in rr])))
print('== target known from t = 0'); summ([(k, tab[('known', 0., k)]) for k in keys], 'predictor-corrector')
for t in (400., 700.):
    print('== target revealed at %.0f s' % t)
    for a in AIMS: summ([(k, tab[(a, t, k)]) for k in keys], 'aim ' + a)
    summ([(k, max((tab[(a, t, k)] for a in AIMS), key=lambda r: r['G'])) for k in keys], 'hindsight-best aim per case')
