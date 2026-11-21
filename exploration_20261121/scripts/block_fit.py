import json, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
rows = json.load(open('/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad/block_rows.json'))
CL = [-.005, -.0025, 0., .0025, .005]; WL = [0., 1/3, 2/3, 1.]
ok = np.array([not (r['alt_fail'] or r['base_fail']) for r in rows])
d = np.array(rows and [[r['alt_end'][0]-r['base_end'][0], r['alt_end'][1]-r['base_end'][1]] for r in rows])
names = np.array([r['name'] for r in rows]); ns = np.array([r['n'] for r in rows])
print('effect on the end point (median |change|, violation-free), tolerance 5 km / 25 m/s')
for nm in ['brake1/3', 'brake2/3', 'brake1', 'cl-0.005', 'cl+0.005', 'hold']:
    for n in (1, 4):
        m = ok & (names == nm) & (ns == n)
        print('  %-9s x%d decisions: range %6.1f km (p90 %6.1f)  speed %6.1f m/s (p90 %6.1f)   violations %.0f%%' % (nm, n, np.median(abs(d[m, 0])), np.percentile(abs(d[m, 0]), 90), np.median(abs(d[m, 1])), np.percentile(abs(d[m, 1]), 90), 100*np.mean([ (rows[i]['alt_fail']) for i in np.where((names == nm) & (ns == n))[0]])))
X = np.array([r['f']+[CL[r['a']//4], WL[r['a'] % 4], r['n']] for r in rows])
cases = np.array([r['case'] for r in rows]); uc = np.unique(cases); rng = np.random.default_rng(0)
tc = set(rng.choice(uc, len(uc)//3, replace=False)); te = np.array([c in tc for c in cases]); tr = ~te
sx = StandardScaler().fit(X[tr & ok])
for k, nm, u in ((0, 'range', 'km'), (1, 'speed', 'm/s')):
    y = d[:, k]/10.
    ms = [MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True, random_state=s, alpha=1e-3).fit(sx.transform(X[tr & ok]), y[tr & ok]) for s in range(3)]
    p = np.mean([m.predict(sx.transform(X[te & ok])) for m in ms], 0)*10.
    e = p-d[te & ok, k]
    print('outcome model for block effects, held-out %s: RMSE %.1f %s (effect std %.1f), R2 %.2f' % (nm, np.sqrt(np.mean(e**2)), u, d[te & ok, k].std(), 1-np.var(e)/np.var(d[te & ok, k])))
    for n in (1, 4):
        m4 = (ns[te & ok] == n); ee = e[m4]; dd = d[te & ok, k][m4]
        print('      %d-decision blocks only: R2 %.2f' % (n, 1-np.var(ee)/np.var(dd)))
