import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = 2.
ORDER = ['linear', '8', '16', '16-16 (used)', '32-32', '64-64', '128-128', '16-16-16', '32-32-32', '64-64-64',
         '128-128-128', '32-32-32-32', '64-64-64-64', '32x5', '32x6', '64x5', '64x6']


def ranks(x):
    return np.argsort(np.argsort(x))


def metrics(rows, subset):
    rho, mse, tr = [], [], []
    hits = {.25: [0, 0, 0], 1.: [0, 0, 0]}   # predicted, true positive, positives
    for r in rows:
        y, m, l = np.array(r['y']), np.array(r['mean']), np.array(r['lcb'])
        sel = np.array(r['new_state']) if subset == 'new' else np.ones(len(y), bool)
        if sel.sum() < 5:
            continue
        y, m, l = y[sel], m[sel], l[sel]
        if np.std(m) > 0 and np.std(y) > 0:
            rho.append(np.corrcoef(ranks(m), ranks(y))[0, 1])
        mse.append(np.mean((m-np.maximum(y, -CLIP))**2))
        tr.append(r['train_mse'])
        for k in hits:
            pred = l > k
            hits[k][0] += int(pred.sum())
            hits[k][1] += int((pred & (y > 0)).sum())
            hits[k][2] += int((y > k).sum())
    out = dict(blocks=len(mse), spearman=float(np.mean(rho)), test_mse=float(np.mean(mse)), train_mse=float(np.mean(tr)))
    for k, (p, tp, pos) in hits.items():
        out[f'switch@{k:g}'] = p
        out[f'precision@{k:g}'] = tp/p if p else float('nan')
    return out


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'netsize_bc_seed*.json'))):
        rows += json.load(open(f))
    seeds = sorted({r['seed'] for r in rows})
    print('seeds', seeds, 'blocks', {s: len({r['block'] for r in rows if r['seed'] == s}) for s in seeds})
    result = {}
    for subset in ('all', 'new'):
        print(f'\n== test labels: {subset}')
        print(f"{'size':14s} {'params':>7s} {'blocks':>6s} {'spearman':>8s} {'test MSE':>8s} {'train MSE':>9s} "
              f"{'switch>.25':>10s} {'prec':>5s} {'switch>1':>8s} {'prec':>5s}")
        for size in ORDER:
            rs = [r for r in rows if r['size'] == size]
            if not rs:
                continue
            name = size.split()[0]
            h = [] if size == 'linear' else ([int(name.split('x')[0])]*int(name.split('x')[1]) if 'x' in name
                                             else [int(x) for x in name.split('-')])
            dims = [4]+h+[19]
            params = sum(a*b+b for a, b in zip(dims[:-1], dims[1:]))
            m = metrics(rs, subset)
            result[(subset, size)] = m
            print(f"{size:14s} {params:7d} {m['blocks']:6d} {m['spearman']:8.3f} {m['test_mse']:8.3f} {m['train_mse']:9.3f} "
                  f"{m['switch@0.25']:10d} {m['precision@0.25']:5.2f} {m['switch@1']:8d} {m['precision@1']:5.2f}")
    json.dump({f'{k[0]}|{k[1]}': v for k, v in result.items()}, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
