import json, numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
rows = json.load(open('/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad/outcome_rows.json'))
CL = [-.005, -.0025, 0., .0025, .005]; WL = [0., 1/3, 2/3, 1.]
def act(a): return [CL[a//4], WL[a % 4]]
X = np.array([r['f']+act(r['a'])+act(r['a0']) for r in rows])
tgt = np.array([r['tgt'] for r in rows]); be = np.array([r['base_end'] for r in rows]); ae = np.array([r['alt_end'] for r in rows])
fail = np.array([r['alt_fail'] or r['base_fail'] for r in rows])
d = (ae-be)                                           # exact effect of the change on the end point
Gadv = np.array([r['alt_G']-r['base_G'] for r in rows])
cases = np.array([r['case'] for r in rows]); uc = np.unique(cases)
rng = np.random.default_rng(0); test_c = set(rng.choice(uc, len(uc)//3, replace=False)); te = np.array([c in test_c for c in cases]); tr = ~te
ok = ~fail
print('rows %d (train %d, test %d); alternatives without violation %d' % (len(rows), tr.sum(), te.sum(), ok.sum()))
print('|effect| of one change on the end point: range median %.1f km (p90 %.1f), speed median %.1f m/s (p90 %.1f)' % (np.median(abs(d[ok, 0])), np.percentile(abs(d[ok, 0]), 90), np.median(abs(d[ok, 1])), np.percentile(abs(d[ok, 1]), 90)))
sx = StandardScaler().fit(X[tr & ok])
def fit(Y, seeds=3):
    ms = [MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True, random_state=s, alpha=1e-3).fit(sx.transform(X[tr & ok]), Y[tr & ok]) for s in range(seeds)]
    return lambda Z: np.mean([m.predict(sx.transform(Z)) for m in ms], 0)
# outcome model: effect on (range km, speed m/s)
f_out = fit(np.stack([d[:, 0]/10., d[:, 1]/10.], 1))
pred = f_out(X)*10.
m = te & ok
for k, nm, u in ((0, 'range', 'km'), (1, 'speed', 'm/s')):
    e = pred[m, k]-d[m, k]
    print('outcome model, held-out %s: RMSE %.2f %s (effect std %.2f), R2 %.2f' % (nm, np.sqrt(np.mean(e**2)), u, d[m, k].std(), 1-np.var(e)/np.var(d[m, k])))
# advantage regression on the scalar return (the current approach)
f_adv = fit(Gadv[:, None]); padv = f_adv(X)
print('advantage regression, held-out: corr %.2f, sign agreement %.0f%%' % (np.corrcoef(padv[m], Gadv[m])[0, 1], 100*np.mean(np.sign(padv[m]) == np.sign(Gadv[m]))))
# decision quality per held-out state: choose among {keep a0, alternatives}
tol = np.array([5., 25.])
def zlog(end, tg): z = np.hypot((end[:, 0]-tg[:, 0])/tol[0], (end[:, 1]-tg[:, 1])/tol[1]); return z
res = dict(keep=[], oracle=[], outcome=[], adv=[]); hit = {k: [] for k in res}
gkey = {}
for i in np.where(te)[0]: gkey.setdefault((rows[i]['case'], rows[i]['t']), []).append(i)
for idx in gkey.values():
    idx = [i for i in idx if ok[i]]
    if not idx: continue
    z_base = zlog(be[idx[0]:idx[0]+1], tgt[idx[0]:idx[0]+1])[0]
    z_alt = zlog(ae[idx], tgt[idx])
    zo = zlog(be[idx]+pred[idx], tgt[idx])           # predicted miss of each alternative
    cand_true = np.r_[z_base, z_alt]
    pick_o = np.argmin(np.r_[z_base, zo])            # model-based choice (keep = predicted z of the baseline end)
    pick_a = int(np.argmax(np.r_[0., padv[idx]]))
    for k, j in (('keep', 0), ('oracle', int(np.argmin(cand_true))), ('outcome', pick_o), ('adv', pick_a)):
        res[k].append(cand_true[j]); hit[k].append(cand_true[j] <= 1.)
print('held-out states: %d' % len(res['keep']))
for k in ('keep', 'adv', 'outcome', 'oracle'):
    z = np.array(res[k]); print('%-8s mean log2(miss/tol) %.2f  median miss %.1f x tol  hit %.0f%%' % (k, np.mean(np.log2(np.maximum(z, 1e-3))), np.median(z), 100*np.mean(hit[k])))
