"""How predictable are the advantage labels at all? (offline; does not change the method)

1. Repeats: test pairs of the forward-chaining study whose network inputs (h, V, gamma, t) and action exactly
   match a label of an earlier policy version. Any model of these inputs predicts both alike, so the agreement
   between the two labels bounds what any model can reach on these pairs.
2. Interpolation: 5-fold cross-validation by state within the labels retained at the end of training.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import sys

import numpy as np

sys.path.insert(0, r'D:\dqn-final\src')
from rollout_improvement import ensemble as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POL = r'D:\v6-mission-morph-clean\_runtime\heldout\policies'
INPUTS = [0, 1, 2, 4]
SIZES = {'16-16': (16, 16), '32-32-32-32': (32,)*4, '64-64-64-64': (64,)*4}


def spearman(a, b):
    rk = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable')
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(rk(a), rk(b))[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def key(x):
    return tuple(np.round(np.asarray(x['features'])[INPUTS], 12)) + (x['action'],)


def repeats(seed_name):
    arch = json.load(open(os.path.join(POL, seed_name+'.json'), encoding='utf-8-sig'))['label_archive']
    blocks = [arch[i:i+75] for i in range(0, len(arch), 75)]
    net = {}
    for f in ('netsize_%s.json' % seed_name, 'netsize_%s_ext2.json' % seed_name, 'netsize_%s_ext1.json' % seed_name):
        for r in json.load(open(os.path.join(HERE, f))):
            net[(r['block'], r['size'])] = r
    out = []
    for b in range(2, len(blocks)):
        last = {}
        for blk in blocks[:b]:
            for x in blk:
                last[key(x)] = x          # latest earlier label with these inputs and action
        test = [x for x in blocks[b] if x['action'] != x['base_action']]
        for i, x in enumerate(test):
            p = last.get(key(x))
            if p is not None:
                out.append(dict(block=b, y=x['advantage']*E.LABEL_SCALE, y_prev=p['advantage']*E.LABEL_SCALE,
                                same_version=p['v'] == x['v'],
                                net={s: net[(b, s)]['mean'][i] for s in ('16-16 (used)', '32-32-32-32', '64-64-64-64')
                                     if (b, s) in net}))
    return out


def fit(labels, weights, seed, hidden, idx):
    X, A, M, Wt = E.label_matrix(labels, weights=weights)
    X = X[:, INPUTS]
    tr = np.setdiff1d(np.arange(len(X)), idx)
    rng = np.random.default_rng(seed)
    sizes = (len(INPUTS),)+tuple(hidden)+(E.N_ACTIONS,)
    members = []
    for k in range(E.MEMBERS):
        boot = tr[rng.integers(0, len(tr), len(tr))]
        oob = np.setdiff1d(tr, np.unique(boot))
        m, _ = E.fit_member(X[boot], A[boot], M[boot], X[oob], A[oob], M[oob], seed*100+k, sizes, Wt=Wt[boot], Wo=Wt[oob])
        members.append(m)
    return members, X


def crossval(seed_name, folds=5):
    s = json.load(open(os.path.join(POL, seed_name+'.json'), encoding='utf-8-sig'))
    arch = s['label_archive']
    v = max(x['v'] for x in arch)
    kept, w = E.retained_labels(arch, v)
    X, A, M, Wt = E.label_matrix(kept, weights=w)
    states = {}
    for x in kept:
        states.setdefault(tuple(x['features']), []).append(x)
    order = list(states)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(order))
    res = {}
    for name, hidden in SIZES.items():
        y, pred = [], []
        for f in range(folds):
            hold = [order[j] for j in perm[f::folds]]
            hold_set = set(hold)
            train = [x for x in kept if tuple(x['features']) not in hold_set]
            tw = [wt for x, wt in zip(kept, w) if tuple(x['features']) not in hold_set]
            Xt, At, Mt, Wtt = E.label_matrix(train, weights=tw)
            rng2 = np.random.default_rng(f)
            sizes = (len(INPUTS),)+tuple(hidden)+(E.N_ACTIONS,)
            members = []
            for k in range(E.MEMBERS):
                boot = rng2.integers(0, len(Xt), len(Xt))
                oob = np.setdiff1d(np.arange(len(Xt)), np.unique(boot))
                m, _ = E.fit_member(Xt[boot][:, INPUTS], At[boot], Mt[boot], Xt[oob][:, INPUTS], At[oob], Mt[oob],
                                    f*100+k, sizes, Wt=Wtt[boot], Wo=Wtt[oob])
                members.append(m)
            for st in hold:
                for x in states[st]:
                    if x['action'] != x['base_action']:
                        _, mu, _ = E.lower_confidence_bound(members, x['features'], INPUTS)
                        y.append(x['advantage']*E.LABEL_SCALE)
                        pred.append(mu[x['action']]-mu[x['base_action']])
        y, pred = np.array(y), np.array(pred)
        res[name] = dict(n=len(y), spearman=spearman(y, pred),
                         mse_clipped=float(np.mean((pred-np.maximum(y, -E.CLIP))**2)))
        print(seed_name, 'cv', name, res[name], flush=True)
    return dict(states=len(order), labels=len(kept), cv=res)


if __name__ == '__main__':
    seed_name, part = sys.argv[1], sys.argv[2]
    out = os.path.join(HERE, f'ceiling_{part}_{seed_name}.json')
    r = repeats(seed_name) if part == 'repeats' else crossval(seed_name)
    json.dump(r, open(out, 'w'))
    if part == 'repeats':
        print(seed_name, 'repeat pairs', len(r))
