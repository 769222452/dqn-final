import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
seeds = sorted({r['seed'] for r in rows}); aims = sorted({tuple(r['aim']) for r in rows})
tab = {(r['seed'], tuple(r['aim'])): r for r in rows}
TG = {'T4': (2226.4, 1168.4), 'T5': (2335.0, 1229.5), 'T2': (2335.0, 1528.6)}
def near(aim, name): return abs(aim[0]-TG[name][0]) < 1 and abs(aim[1]-TG[name][1]) < 1
half = len(seeds)//2; train, test = seeds[:half], seeds[half:]
print('scenarios %d (selection on first %d, evaluation on last %d), aims %d' % (len(seeds), len(train), len(test), len(aims)))
# ceilings: target known from t = 0 (aim = the target itself, then reveal the same target)
for name in ('T4', 'T5', 'T2'):
    a = [x for x in aims if near(x, name)][0]
    print('known-target ceiling %s: %.1f%%' % (name, 100*np.mean([tab[(s, a)][name] for s in seeds])))
for A, Bt in (('T4', 'T2'), ('T5', 'T2')):
    print('\n=== pair A=%s %s, B=%s %s ===' % (A, TG[A], Bt, TG[Bt]))
    def score(aim, ss): 
        hA = np.array([tab[(s, aim)][A] for s in ss], float); hB = np.array([tab[(s, aim)][Bt] for s in ss], float); return hA, hB
    table = []
    for aim in aims:
        hA, hB = score(aim, seeds); table.append((aim, hA.mean(), hB.mean(), (hA*hB).mean()))
    print('%-16s %8s %8s %8s %8s' % ('aim (km, m/s)', 'P(A)', 'P(B)', 'avg', 'both'))
    for aim, pa, pb, both in sorted(table, key=lambda r: -(r[1]+r[2]))[:6]:
        print('(%.0f, %.0f)'.ljust(16) % aim, '%7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (100*pa, 100*pb, 50*(pa+pb), 100*both))
    # best common aim chosen on train scenarios, evaluated on test scenarios
    best = max(aims, key=lambda a: score(a, train)[0].mean()+score(a, train)[1].mean())
    hA, hB = score(best, test)
    print('best common aim on train (%.0f, %.0f): test avg %.1f%%, P(A) %.1f%%, P(B) %.1f%%, both %.1f%%' % (best[0], best[1], 50*(hA.mean()+hB.mean()), 100*hA.mean(), 100*hB.mean(), 100*(hA*hB).mean()))
    # hindsight best aim per scenario (knows the disturbance, not A/B), test scenarios
    hs = []
    for s in test:
        cand = [(tab[(s, a)][A]+tab[(s, a)][Bt], tab[(s, a)][A]*tab[(s, a)][Bt]) for a in aims]; k = max(cand); hs.append(k)
    print('per-scenario hindsight aim (test): avg %.1f%%, both %.1f%%' % (100*np.mean([k[0] for k in hs])/2, 100*np.mean([k[1] for k in hs])))
    # naive aims: always A, always B
    for lab, nm in (('always aim at A', A), ('always aim at B', Bt)):
        a = [x for x in aims if near(x, nm)][0]; hA, hB = score(a, test)
        print('%-16s test avg %.1f%%, P(A) %.1f%%, P(B) %.1f%%, both %.1f%%' % (lab, 50*(hA.mean()+hB.mean()), 100*hA.mean(), 100*hB.mean(), 100*(hA*hB).mean()))
