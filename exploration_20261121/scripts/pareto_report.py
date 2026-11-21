import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
fam = rows[-66:] if False else None
rand = [r for r in rows if len(set(np.round(r['u'][:3], 6))) > 1 or len(set(np.round(r['u'][3:], 6))) > 1]
famr = [r for r in rows if len(set(np.round(r['u'][:3], 6))) == 1 and all(abs(x) < 1e-9 for x in r['u'][3:])]
print('valid plans: random %d, one-parameter family (constant r, no morph) %d' % (len(rand), len(famr)))
A = np.array([[r['R'], -r['H'], r['E'], r['V']] for r in rows]); names = ['range km', 'heat load (lower better)', 'terminal energy', 'terminal speed']
print('spread of each objective over valid plans:')
for k, n in enumerate(names):
    v = A[:, k]*(-1 if k == 1 else 1); print('  %-28s min %9.1f  max %9.1f' % (n, v.min(), v.max()))
C = np.corrcoef(A.T); print('correlations (heat sign flipped so that higher = better):'); print(np.round(C, 2))
def pareto(P):
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = np.all(P >= P[i], 1) & np.any(P > P[i], 1)
            if dom.any(): keep[i] = False
    return keep
def hv(P, ref, n=60000, seed=0):
    rng = np.random.default_rng(seed); lo = ref; hi = B.max(0)
    X = rng.uniform(lo, hi, (n, P.shape[1])); ok = np.zeros(n, bool)
    for p in P: ok |= np.all(X <= p, 1)
    return ok.mean()
obj = [0, 1, 2]                                   # range, -heat, energy
B = A[:, obj]; ref = B.min(0)
mask = pareto(B)
use_morph = np.array([max(r['u'][3:]) > 1e-9 for r in rows])
print('\nPareto set over (range, heat, energy): %d of %d plans; of those %.0f%% use the brake (w > 0)' % (mask.sum(), len(rows), 100*use_morph[mask].mean()))
nom = np.array([all(abs(x) < 1e-9 for x in r['u'][3:]) for r in rows])
for lab, sel in (('all plans (morph allowed)', np.ones(len(rows), bool)), ('no morph (w = 0)', nom), ('one-parameter family (constant r, w = 0)', np.array([r in famr for r in rows]))):
    P = B[sel]; P = P[pareto(P)]; print('  hypervolume %-42s %.3f (%d points)' % (lab, hv(P, ref), len(P)))
for k, n in enumerate(['range', 'heat', 'energy']):
    j = int(np.argmax(B[:, k])); r = rows[j]
    print('best %-6s plan: R %.0f km, H %.0f, E %.2f, V %.0f | r=%s w=%s' % (n, r['R'], r['H'], r['E'], r['V'], np.round(r['u'][:3], 2), np.round(r['u'][3:], 2)))
best_nom = max([r for r in rows if r in rand and all(abs(x) < 1e-9 for x in r['u'][3:])], key=lambda r: r['R'])
best_mor = max(rand, key=lambda r: r['R']); print('best range without brake %.0f km, with brake allowed %.0f km' % (best_nom['R'], best_mor['R']))
