"""Bootstrap ensemble of tanh MLPs predicting action advantages (ported from rollout_improvement.ensemble).

Targets: advantages A(s, a) times label_scale (configs/task.yaml: units of 0.01 return), clipped from below at
-label_clip (and from above at label_clip_upper when that key is set: study A, symmetric clipping). Each member: bootstrap sample of states, Adam, early stopping on its out-of-bag states. Action
selection uses the lower confidence bound mean - std over the members.
"""
import numpy as np

from . import config

L2 = 1e-3
MEMBERS = 5
STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 100
LABEL_DECAY = .5
LABEL_KEEP = 7


def mlp_init(rng, sizes):
    return [(rng.normal(0, 1/np.sqrt(a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def mlp_forward(params, x, keep=False):
    hs = [x]
    for j, (W, b) in enumerate(params):
        x = x@W+b
        if j < len(params)-1:
            x = np.tanh(x)
        hs.append(x)
    return (x, hs) if keep else x


def as_arrays(members):
    return [[(np.asarray(W), np.asarray(b)) for W, b in m] for m in members]


def lower_confidence_bound(members, f, inputs=None):
    """(mean - std, mean, std) over members for one feature vector; members as arrays (as_arrays)."""
    f = np.asarray(f, float)
    if inputs is not None:
        f = f[inputs]
    outs = np.array([mlp_forward(m, f[None])[0] for m in members])
    mu, sd = outs.mean(0), outs.std(0)
    return mu-sd, mu, sd


def fit_member(X, A, M, Xo, Ao, Mo, seed, sizes, Wt=None, Wo=None, steps=STEPS, lr=LEARNING_RATE, l2=L2):
    rng = np.random.default_rng(seed)
    p = mlp_init(rng, sizes)
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in p]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in p]
    Wt = M.astype(float) if Wt is None else np.where(M, Wt, 0.)
    Wo = (Mo.astype(float) if Wo is None else np.where(Mo, Wo, 0.)) if Mo is not None else None
    has_oob = Mo is not None and Mo.any()

    def mse(xx, aa, mm, ww=None):
        if not mm.any():
            return None
        e = (mlp_forward(p, xx)-aa)**2
        return float(np.mean(e[mm])) if ww is None else float((e*ww)[mm].sum()/ww[mm].sum())

    best = (mse(Xo, Ao, Mo, Wo) if has_oob else None, 0, [(W.copy(), b.copy()) for W, b in p])
    for t in range(1, steps+1):
        out, hs = mlp_forward(p, X, keep=True)
        g = (out-A)*Wt/max(Wt.sum(), 1e-12)
        grads = [None]*len(p)
        for j in range(len(p)-1, -1, -1):
            W, b = p[j]
            grads[j] = (hs[j].T@g+l2*W, g.sum(0))
            if j:
                g = (g@W.T)*(1-hs[j]**2)
        for j, ((W, b), (gW, gb)) in enumerate(zip(p, grads)):
            mW, mb = m[j]
            vW, vb = v[j]
            mW, mb = .9*mW+.1*gW, .9*mb+.1*gb
            vW, vb = .999*vW+.001*gW*gW, .999*vb+.001*gb*gb
            m[j], v[j] = (mW, mb), (vW, vb)
            c1, c2 = 1-.9**t, 1-.999**t
            p[j] = (W-lr*(mW/c1)/(np.sqrt(vW/c2)+1e-8), b-lr*(mb/c1)/(np.sqrt(vb/c2)+1e-8))
        if has_oob and t % EVAL_EVERY == 0:
            o = mse(Xo, Ao, Mo, Wo)
            if o < best[0]:
                best = (o, t, [(W.copy(), b.copy()) for W, b in p])
    if has_oob:
        p = best[2]
    diag = dict(best_step=best[1] if has_oob else steps, oob_mse=mse(Xo, Ao, Mo, Wo) if has_oob else None,
                train_mse=mse(X, A, M, Wt), oob_states=int(len(Xo)) if Xo is not None else 0)
    return [(W.tolist(), b.tolist()) for W, b in p], diag


def fit_member_job(kw):
    return fit_member(*[np.asarray(kw[k]) if kw[k] is not None else None for k in ('X', 'A', 'M', 'Xo', 'Ao', 'Mo')],
                      kw['seed'], kw['sizes'], Wt=np.asarray(kw['Wt']), Wo=np.asarray(kw['Wo']))


def label_matrix(labels, n_actions, weights=None, cfg=None):
    tr = (cfg or config.load())['training']
    keys = sorted({tuple(r['features']) for r in labels})
    index = {k: i for i, k in enumerate(keys)}
    X = np.array(keys)
    S = np.zeros((len(keys), n_actions))
    Wt = np.zeros((len(keys), n_actions))
    for r, w in zip(labels, weights if weights is not None else [1.]*len(labels)):
        i = index[tuple(r['features'])]
        y = max(r['advantage']*tr['label_scale'], -tr['label_clip'])
        if tr.get('label_clip_upper') is not None:
            y = min(y, tr['label_clip_upper'])
        S[i, r['action']] += w*y
        Wt[i, r['action']] += w
    M = Wt > 0
    A = np.where(M, S/np.where(M, Wt, 1.), 0.)
    return X, A, M, Wt


def retained_labels(archive, version, keep=LABEL_KEEP, decay=LABEL_DECAY):
    kept = [x for x in archive if version-x['v'] < keep]
    return kept, [decay**(version-x['v']) for x in kept]


def fit_ensemble(labels, seed, n_actions, inputs, hidden, weights=None, workers=1):
    """MEMBERS networks on bootstrap samples of the labelled states. Returns (members, diagnostics)."""
    from .parallel import pool_map
    X, A, M, Wt = label_matrix(labels, n_actions, weights=weights)
    X = X[:, inputs]
    rng = np.random.default_rng(seed)
    sizes = (len(inputs),)+tuple(hidden)+(n_actions,)
    jobs = []
    for k in range(MEMBERS):
        boot = rng.integers(0, len(X), len(X))
        oob = np.setdiff1d(np.arange(len(X)), np.unique(boot))
        jobs.append(('fit_member', dict(X=X[boot], A=A[boot], M=M[boot], Xo=X[oob], Ao=A[oob], Mo=M[oob],
                                        seed=seed*100+k, sizes=sizes, Wt=Wt[boot], Wo=Wt[oob])))
    res = pool_map(jobs, workers)
    members = [r[0] for r in res]
    diags = [r[1] for r in res]
    return members, dict(members=diags, states=int(len(X)), labels=int(M.sum()), raw_labels=len(labels))
