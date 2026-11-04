"""Network-size study on the real label archives (offline; does not change the method).

Forward chaining: for every iteration block b, fit an ensemble on the labels collected before b (retention and
weights as in training: last 7 policy versions, weight 0.5**age), then predict the alternative pairs of block b.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\dqn-final\src')
from rollout_improvement import ensemble as E  # noqa: E402

INPUTS = [0, 1, 2, 4]
SIZES = {'linear': (), '8': (8,), '16': (16,), '16-16 (used)': (16, 16), '32-32': (32, 32), '64-64': (64, 64),
         '128-128': (128, 128), '64-64-64': (64, 64, 64)}
GROUPS = {'ext1': {'16-16-16': (16, 16, 16), '32-32-32': (32, 32, 32), '32-32-32-32': (32, 32, 32, 32)},
          'ext2': {'128-128-128': (128, 128, 128), '64-64-64-64': (64, 64, 64, 64)},
          'd64x5': {'64x5': (64,)*5}, 'd64x6': {'64x6': (64,)*6},
          'd32x5': {'32x5': (32,)*5}, 'd32x6': {'32x6': (32,)*6}}
if len(sys.argv) > 2:
    SIZES = GROUPS[sys.argv[2]]
POL = r'D:\v6-mission-morph-clean\_runtime\heldout\policies'
HERE = os.path.dirname(os.path.abspath(__file__))


def fit(labels, weights, seed, hidden):
    X, A, M, Wt = E.label_matrix(labels, weights=weights)
    X = X[:, INPUTS]
    rng = np.random.default_rng(seed)
    sizes = (len(INPUTS),)+tuple(hidden)+(E.N_ACTIONS,)
    members, train = [], []
    for k in range(E.MEMBERS):
        boot = rng.integers(0, len(X), len(X))
        oob = np.setdiff1d(np.arange(len(X)), np.unique(boot))
        m, d = E.fit_member(X[boot], A[boot], M[boot], X[oob], A[oob], M[oob], seed*100+k, sizes,
                            Wt=Wt[boot], Wo=Wt[oob])
        members.append(m)
        train.append(d['train_mse'])
    return members, float(np.mean(train)), int(len(X)), int(M.sum())


def evaluate(members, test):
    y, mean, lcb = [], [], []
    for x in test:
        l, mu, _ = E.lower_confidence_bound(members, x['features'], INPUTS)
        mean.append(mu[x['action']]-mu[x['base_action']])
        lcb.append(l[x['action']]-mu[x['base_action']])
        y.append(x['advantage']*E.LABEL_SCALE)
    return np.array(y), np.array(mean), np.array(lcb)


def main():
    seed_name = sys.argv[1]
    out = os.path.join(HERE, f'netsize_{seed_name}' + (f'_{sys.argv[2]}' if len(sys.argv) > 2 else '') + '.json')
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(r['block'], r['size']) for r in rows}
    if True:
        s = json.load(open(os.path.join(POL, seed_name+'.json'), encoding='utf-8-sig'))
        arch = s['label_archive']
        blocks = [arch[i:i+75] for i in range(0, len(arch), 75)]
        for b in range(2, len(blocks)):
            v = blocks[b][0]['v']
            past = [x for blk in blocks[:b] for x in blk]
            kept, w = E.retained_labels(past, v)
            seen = {tuple(x['features']) for x in kept}
            test = [x for x in blocks[b] if x['action'] != x['base_action']]
            new_state = np.array([tuple(x['features']) not in seen for x in test])
            states = labels = None
            for name, hidden in SIZES.items():
                if (b, name) in done:
                    continue
                tick = time.perf_counter()
                members, train_mse, states, labels = fit(kept, w, 1000+b, hidden)
                y, mean, lcb = evaluate(members, test)
                rows.append(dict(seed=seed_name, block=b, version=v, size=name, states=states, labels=labels,
                                 train_mse=train_mse, y=y.tolist(), mean=mean.tolist(), lcb=lcb.tolist(),
                                 new_state=new_state.tolist(), wall=time.perf_counter()-tick))
            print(seed_name, 'block', b, 'train states', states, 'labels', labels, 'new-state tests',
                  int(new_state.sum()), '/', len(test), flush=True)
            json.dump(rows, open(out, 'w'))


if __name__ == '__main__':
    main()
