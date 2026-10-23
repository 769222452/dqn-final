"""Bootstrap ensemble of tanh MLPs that predict action advantages.

Targets are advantages A(s, a) scaled by LABEL_SCALE (units of 1e-4 return) and clipped from below at -CLIP:
only the sign region matters for policy improvement. Each member is trained on a bootstrap sample of states
with Adam and early-stopped on its out-of-bag states. The lower confidence bound mean - std over members is
used for action selection.
"""
import numpy as np

from .features import FEATURE_NAMES

N_ACTIONS = 19
LABEL_SCALE = 1e4
CLIP = 2.
L2 = 1e-3
HIDDEN = (64, 64, 64, 64)   # hidden layer widths of the proposed method
MEMBERS = 5
STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 100
LABEL_DECAY = .5      # weight factor per policy version of label age
LABEL_KEEP = 7        # policy versions (current included) whose labels are used for fitting


def parse_hidden(hidden):
    """Hidden layer widths from '64,64,64,64' or a sequence of ints."""
    widths = tuple(int(x) for x in (hidden.split(',') if isinstance(hidden, str) else hidden))
    if not widths or min(widths) < 1:
        raise ValueError('hidden layer widths must be positive integers')
    return widths


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


def lower_confidence_bound(members, f, inputs=None):
    """(mean - std, mean, std) of the member predictions for one feature vector."""
    f = np.asarray(f) if inputs is None else np.asarray(f)[inputs]
    outs = np.array([mlp_forward([(np.asarray(W), np.asarray(b)) for W, b in m], f[None])[0] for m in members])
    return outs.mean(0)-outs.std(0), outs.mean(0), outs.std(0)


def fit_member(X, A, M, Xo, Ao, Mo, seed, sizes, Wt=None, Wo=None, steps=STEPS, lr=LEARNING_RATE, l2=L2):
    """Weighted masked MSE regression; returns the parameters with the lowest out-of-bag error."""
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
            mW = .9*mW+.1*gW
            mb = .9*mb+.1*gb
            vW = .999*vW+.001*gW*gW
            vb = .999*vb+.001*gb*gb
            m[j] = (mW, mb)
            v[j] = (vW, vb)
            c1 = 1-.9**t
            c2 = 1-.999**t
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


def label_matrix(labels, weights=None, clip=CLIP):
    """States x actions targets, mask and weights; repeated (state, action) labels are weight-averaged."""
    keys = sorted({tuple(r['features']) for r in labels})
    index = {k: i for i, k in enumerate(keys)}
    X = np.array(keys)
    S = np.zeros((len(keys), N_ACTIONS))
    Wt = np.zeros((len(keys), N_ACTIONS))
    for r, w in zip(labels, weights if weights is not None else [1.]*len(labels)):
        i = index[tuple(r['features'])]
        S[i, r['action']] += w*max(r['advantage']*LABEL_SCALE, -clip)
        Wt[i, r['action']] += w
    M = Wt > 0
    A = np.where(M, S/np.where(M, Wt, 1.), 0.)
    return X, A, M, Wt


def retained_labels(archive, version, keep=LABEL_KEEP, decay=LABEL_DECAY):
    """Labels of the last `keep` policy versions with weight decay**age."""
    kept = [x for x in archive if version-x['v'] < keep]
    return kept, [decay**(version-x['v']) for x in kept]


def fit_ensemble(labels, seed, weights=None, inputs=None, hidden=HIDDEN):
    """Fit MEMBERS networks on bootstrap samples of the labelled states. Returns (members, diagnostics)."""
    X, A, M, Wt = label_matrix(labels, weights=weights)
    rng = np.random.default_rng(seed)
    members = []
    diags = []
    sizes = ((len(FEATURE_NAMES) if inputs is None else len(inputs)),) + tuple(hidden) + (N_ACTIONS,)
    if inputs is not None:
        X = X[:, inputs]
    for k in range(MEMBERS):
        boot = rng.integers(0, len(X), len(X))
        oob = np.setdiff1d(np.arange(len(X)), np.unique(boot))
        mem, d = fit_member(X[boot], A[boot], M[boot], X[oob], A[oob], M[oob], seed*100+k, sizes, Wt=Wt[boot], Wo=Wt[oob])
        members.append(mem)
        diags.append(d)
    var = float(np.average((A[M]-np.average(A[M], weights=Wt[M]))**2, weights=Wt[M])) if M.any() else None
    return members, dict(members=diags, target_var_clipped=var, states=int(len(X)), labels=int(M.sum()),
                         raw_labels=len(labels), weight_sum=float(Wt.sum()))


def _ranks(z):
    """Average ranks (ties share the mean rank; clipped targets tie at -CLIP)."""
    z = np.asarray(z, float)
    order = np.argsort(z, kind='mergesort')
    r = np.empty(len(z))
    r[order] = np.arange(len(z), dtype=float)
    for v in np.unique(z):
        idx = z == v
        if idx.sum() > 1:
            r[idx] = r[idx].mean()
    return r


def prediction_check(members, new, min_advantage, inputs=None):
    """Out-of-sample check: an ensemble fitted before these labels existed predicts the new alternative pairs.

    Reports precision/recall of positive pairs, Spearman rank correlation and R^2 on clipped targets.
    """
    alts = [x for x in new if x['action'] != x['base_action']]
    if not members or not alts:
        return None
    mean = []
    lcb = []
    for x in alts:
        l, mu, _ = lower_confidence_bound(members, x['features'], inputs)
        mean.append(mu[x['action']]-mu[x['base_action']])
        lcb.append(l[x['action']]-mu[x['base_action']])
    mean = np.array(mean)
    lcb = np.array(lcb)
    y = np.array([x['advantage']*LABEL_SCALE for x in alts])
    pos = y > min_advantage*LABEL_SCALE

    def pr(pred):
        tp = int((pred & pos).sum())
        return dict(predicted=int(pred.sum()), true_positive=tp, precision=tp/pred.sum() if pred.sum() else None,
                    recall=tp/pos.sum() if pos.sum() else None)

    rho = float(np.corrcoef(_ranks(mean), _ranks(y))[0, 1]) if len(y) > 2 and np.std(y) > 0 and np.std(mean) > 0 else None
    yc = np.maximum(y, -CLIP)
    mse = float(np.mean((mean-yc)**2))
    k = max(1, int(pos.sum()))
    top = np.argsort(-mean)[:k]
    return dict(pairs=len(y), positives=int(pos.sum()), mean_positive=pr(mean > 0), lcb_positive=pr(lcb > 0),
                spearman=rho, mse_clipped=mse, r2_clipped=1-mse/float(np.var(yc)) if np.var(yc) > 0 else None,
                top_k_hits=int(pos[top].sum()), top_k=k)
