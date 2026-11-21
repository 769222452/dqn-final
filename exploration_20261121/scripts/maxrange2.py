"""Max range, L2, F8D2000B (+altitude guard). (1) corrected constant-CL baseline (absolute CL levels, round-2 execution);
(2) scalar adaptive dynamic-pressure ratio: q-law r0 = 0.7 for 120 s, then r2 chosen from what is observable at 120 s
(initial offsets, estimated lift/drag biases, measured density ratio) by a linear model fitted on training scenarios;
oracle r2 (best constant r2 after 120 s, knows the scenario) as the bound for any scalar rule."""
import sys, json, numpy as np, multiprocessing as mp
from target_rl import baselines as B, flight as F, guided as G
from target_rl.scenario import scenario, Target
PH = [1e9, 1e9]; GRID = [.5, .6, .7, .75, .8, .85, .9, .95]
def cmd(r, morph=True): return B._command(np.array([r]*3+[0.]*3), PH, morph)
def const_cl(a):
    seed, cl = a; e = F.TargetFlight(scenario('L2', seed), Target('x', 0., 0.), True); e.advance(e.w['T'], G._hold(cl, 0.)); return e.range_km(), bool(e.failed)
def stage(a):
    seed = a; sc = scenario('L2', seed)
    e = F.TargetFlight(sc, Target('x', 0., 0.), True); e.advance(120., cmd(.7))
    kl, kd = e.est.values
    f = [sc.dh/1000., sc.dv/100., kl-1, kd-1, e.k_rho()-1, e.rho_prev-1, (e.x[0]-39000)/1000., (e.x[1]-2400)/100.]
    outs = {}
    for r2 in GRID:
        c = e.fork(); c.advance(c.w['T']-c.t, cmd(r2)); outs[r2] = (c.range_km(), bool(c.failed))
    return seed, f, outs
if __name__ == '__main__':
    train, test = list(range(1100, 1196)), list(range(1500, 1596))
    with mp.Pool(4) as p:
        cs = [(s, c) for s in train[:48] for c in G.ABS_CL]; rc = p.map(const_cl, cs, chunksize=8)
        tab = {}
        for (s, c), (rg, fl) in zip(cs, rc): tab.setdefault(c, []).append((rg, fl))
        okc = [c for c, v in tab.items() if sum(x[1] for x in v) == 0] or list(tab); cbest = max(okc, key=lambda c: np.mean([x[0] for x in tab[c]]))
        rt = p.map(const_cl, [(s, cbest) for s in test], chunksize=4)
        print('best constant CL (training, no violations) %.3f -> test mean range %.1f km, violations %d' % (cbest, np.mean([x[0] for x in rt]), sum(x[1] for x in rt)), flush=True)
        TR = p.map(stage, train, chunksize=4); TE = p.map(stage, test, chunksize=4)
    def table(R): return np.array([[o[r][0] if not o[r][1] else -1e4 for r in GRID] for _, _, o in R])
    Ytr, Yte = table(TR), table(TE)
    valid_te = Yte.max(1) > 0; valid_tr = Ytr.max(1) > 0
    best_tr = np.array([GRID[int(np.argmax(y))] for y in Ytr]); X = np.array([f for _, f, _ in TR]); Xt = np.array([f for _, f, _ in TE])
    mu, sd = X.mean(0), X.std(0)+1e-9; Z = np.c_[(X-mu)/sd, np.ones(len(X))]
    w = np.linalg.solve(Z[valid_tr].T@Z[valid_tr]+3.*np.eye(Z.shape[1]), Z[valid_tr].T@best_tr[valid_tr])
    pred = np.clip(np.c_[(Xt-mu)/sd, np.ones(len(Xt))]@w, .5, .95)
    ti = lambda r: GRID.index(min(GRID, key=lambda g: abs(g-r)))
    # a fixed r2 selected on TRAINING scenarios by mean range, ignoring the scenarios that violate for every r2
    fixr = max(GRID, key=lambda r: np.mean(np.where(Ytr[valid_tr, GRID.index(r)] < 0, -300, Ytr[valid_tr, GRID.index(r)])))
    cte = np.array([x[0] if not x[1] else -1e4 for x in rt])
    arrs = {'best constant CL (original baseline)': cte, 'fixed q-law r2 = %.2f' % fixr: Yte[:, GRID.index(fixr)],
            'adaptive scalar rule (linear in what is observable at 120 s)': np.array([Yte[i, ti(pred[i])] for i in range(len(pred))]), 'oracle r2 (knows the scenario, scalar)': Yte.max(1)}
    print('test scenarios %d valid (at least one safe r2); %d violate for every r2 (initial transient)' % (valid_te.sum(), (~valid_te).sum()))
    for lab, v in arrs.items():
        print('%-62s mean range %.1f km  violations (valid scenarios) %d' % (lab, v[valid_te].mean() if (v[valid_te] > 0).all() else np.mean(np.where(v[valid_te] < 0, np.nan, v[valid_te])) if False else np.nanmean(np.where(v[valid_te] < 0, np.nan, v[valid_te])), int((v[valid_te] < 0).sum())))
    ref = arrs['best constant CL (original baseline)']
    for lab, v in arrs.items():
        if lab.startswith('best constant'): continue
        d = (v-ref)[valid_te]; d = d[(v[valid_te] > 0) & (ref[valid_te] > 0)]; rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        print('%-62s vs best constant CL: %+.1f km  95%% CI [%+.1f, %+.1f]' % (lab, d.mean(), *np.percentile(bs, [2.5, 97.5])))
    yb = np.array([GRID[int(np.argmax(y))] for y in Yte])[valid_te]
    print('R2 of the linear rule for the best r2 (test): %.2f ; std of best r2 across scenarios %.3f' % (1-np.var(pred[valid_te]-yb)/np.var(yb), yb.std()))
    json.dump(dict(Yte=Yte.tolist(), pred=pred.tolist(), rt=[list(x) for x in rt]), open('/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad/maxrange2.json', 'w'))
