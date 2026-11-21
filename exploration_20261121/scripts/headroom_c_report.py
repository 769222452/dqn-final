import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
for lv in ('L2', 'L3'):
    R = [r for r in rows if r['level'] == lv]
    key = lambda r: (r['seed'] if 'seed' in r else None, r['target'])
    by = {}
    for r in R: by.setdefault((r['kind'], r['margin']), {})[r['target']] = r
    names = sorted(by)
    tg = sorted(by[('pc_oracle', 0.005)])
    print('== %s (%d cases)' % (lv, len(tg)))
    for k in names:
        rr = [by[k][t] for t in tg]
        print('  %-10s margin %.3f: success %5.1f%%  violations %5.1f%%  mean return %+.3f' % (k, k[1], 100*np.mean([r['hit'] for r in rr]), 100*np.mean([not r['feasible'] for r in rr]), np.mean([r['G'] for r in rr])))
    pcs = [k for k in names if k[0] == 'pc_strong']
    best = [max((by[k][t] for k in pcs), key=lambda r: r['G']) for t in tg]
    print('  hindsight-best margin per case: success %5.1f%%  violations %5.1f%%  mean return %+.3f' % (100*np.mean([r['hit'] for r in best]), 100*np.mean([not r['feasible'] for r in best]), np.mean([r['G'] for r in best])))
    diff = np.mean([len({round(by[k][t]['G'], 6) for k in pcs}) > 1 for t in tg])
    print('  cases where the margin changes the outcome: %.0f%%' % (100*diff))
