"""Does hindsight (counterfactual-goal) relabelling add usable signal to the advantage labels?
Held-out states: pick the action with the largest predicted advantage for the REAL goal; score by the exact outcome."""
import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
rows = json.load(open('/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad/outcome_rows.json'))
CL = [-.005, -.0025, 0., .0025, .005]; WL = [0., 1/3, 2/3, 1.]
def act(a): return [CL[a//4], WL[a % 4]]
TOL = np.array([5., 25.])
def reward(end, g, fail, linear=False):
    z = np.hypot((end[0]-g[0])/TOL[0], (end[1]-g[1])/TOL[1])
    if fail: return -4.
    if linear: return -z
    hit = abs(end[0]-g[0]) <= TOL[0] and abs(end[1]-g[1]) <= TOL[1]
    return 1. if hit else -min(.5*np.log2(max(z, 1.)), 3.)
IDX = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16]                 # BASE + GUIDE (the network inputs of study R)
def shifted(f, tg_old, tg_new):
    f = list(f); dD, dV = tg_new[0]-tg_old[0], tg_new[1]-tg_old[1]
    f[4] += dD/1000.; f[5] -= dV/500.; f[13] -= dD/100.; f[14] -= dV/100.
    return [f[i] for i in IDX]
groups = {}
for i, r in enumerate(rows): groups.setdefault((r['case'], r['t']), []).append(i)
cases = sorted({r['case'] for r in rows}); rng = np.random.default_rng(0)
test_c = set(rng.choice(cases, len(cases)//3, replace=False))

def build(mode):
    X, Y = [], []
    for (c, t), idx in groups.items():
        if c in test_c: continue
        r0 = rows[idx[0]]
        for i in idx:                                     # real-goal labels
            r = rows[i]
            X.append(shifted(r['f'], r['tgt'], r['tgt'])+act(r['a'])+act(r['a0']))
            Y.append(r['alt_G']-r['base_G'])
        if mode == 'real': continue
        ends = [(r0['base_end'], r0['base_fail'])]+[(rows[i]['alt_end'], rows[i]['alt_fail']) for i in idx]
        goals = [e for e, fl in ends if not fl]
        for g in goals:                                   # hindsight goals: where this flight (or a branch) ended
            lin = mode == 'hind_linear'
            g0 = reward(r0['base_end'], g, r0['base_fail'], lin)
            for i in idx:
                r = rows[i]
                X.append(shifted(r['f'], r['tgt'], g)+act(r['a'])+act(r['a0']))
                Y.append(reward(r['alt_end'], g, r['alt_fail'], lin)-g0)
    return np.array(X), np.array(Y)

def evaluate(mode, seeds=3):
    X, Y = build(mode)
    sx = StandardScaler().fit(X)
    ms = [MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, early_stopping=True, random_state=s, alpha=1e-3).fit(sx.transform(X), np.clip(Y, -1.5, 1.5)) for s in range(seeds)]
    zs, hits = [], []
    for (c, t), idx in groups.items():
        if c not in test_c: continue
        idx = [i for i in idx if not rows[i]['alt_fail']]
        if not idx: continue
        r0 = rows[idx[0]]
        Z = np.array([shifted(rows[i]['f'], rows[i]['tgt'], rows[i]['tgt'])+act(rows[i]['a'])+act(rows[i]['a0']) for i in idx])
        p = np.mean([m.predict(sx.transform(Z)) for m in ms], 0)
        cand = [r0['base_end']]+[rows[i]['alt_end'] for i in idx]
        pick = int(np.argmax(np.r_[0., p]))
        z = lambda e: np.hypot((e[0]-r0['tgt'][0])/TOL[0], (e[1]-r0['tgt'][1])/TOL[1])
        zs.append(z(cand[pick])); hits.append(z(cand[pick]) <= 1.)
    return len(X), np.mean(np.log2(np.maximum(zs, 1e-3))), np.median(zs), 100*np.mean(hits), len(zs)
for mode in ('real', 'hind', 'hind_linear'):
    n, ml, med, h, ns = evaluate(mode)
    print('%-12s labels %6d | held-out states %d: mean log2(miss/tol) %.2f, median miss %.1f x tol, hit %.0f%%' % (mode, n, ns, ml, med, h))
print('(reference from before: keep 1.07 / 2.4x / 33%;  best single change in hindsight (oracle) 0.76 / 2.1x / 36%)')
