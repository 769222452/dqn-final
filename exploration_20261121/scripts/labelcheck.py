"""Open-loop replay labels (current method) vs closed-loop rollout labels (continue with the policy) for the same
branches, on the interim study-R morphing policy and its training cases."""
import json, sys, time, numpy as np, multiprocessing as mp
from target_rl import train as T, guided as G, policy as P, config
ST = sys.argv[1]
TIMES = (192., 504., 804., 996.)

def job(ci):
    st = json.load(open(ST)); chain = st['chain']; sc, tg = T.case_from_json(st['cases'][ci])
    dec = []
    base = T.fly(G.make(sc, tg, True), chain, log=dec)
    sched = {int(round(t)): a for t, a in dec}
    rows = []
    for t_b in TIMES:
        e = G.make(sc, tg, True); T.fly(e, chain, schedule=sched, until=t_b)
        if e.done: continue
        a0 = sched[int(round(t_b))]
        for a in P.alternatives(a0, t_b, 0, True):
            bo = T.fly(e.fork(), chain, override=(t_b, a), schedule=sched)
            bc = T.fly(e.fork(), chain, override=(t_b, a))
            rows.append(dict(case=ci, t=t_b, a0=a0, a=a, Gb=base.G, Go=bo.G, Gc=bc.G))
    return rows

if __name__ == '__main__':
    st = json.load(open(ST)); n = len(st['cases'])
    ids = list(range(0, n, max(1, n//40)))[:40]
    t0 = time.time()
    with mp.Pool(4) as p: rows = [r for rr in p.map(job, ids, chunksize=1) for r in rr]
    print('branches', len(rows), 'cases', len(ids), 'time %.0f s' % (time.time()-t0))
    ao = np.array([r['Go']-r['Gb'] for r in rows]); ac = np.array([r['Gc']-r['Gb'] for r in rows])
    print('corr(open, closed) = %.2f   Spearman-like sign agreement (|adv|>0.02 in either): %.0f%%' % (np.corrcoef(ao, ac)[0, 1], 100*np.mean(np.sign(ao[(abs(ao) > .02) | (abs(ac) > .02)]) == np.sign(ac[(abs(ao) > .02) | (abs(ac) > .02)]))))
    print('share of alternatives that improve (adv > 0.02): open %.1f%%  closed %.1f%%' % (100*np.mean(ao > .02), 100*np.mean(ac > .02)))
    print('share that hurt (adv < -0.02):                    open %.1f%%  closed %.1f%%' % (100*np.mean(ao < -.02), 100*np.mean(ac < -.02)))
    print('mean |adv|: open %.3f closed %.3f ; median adv: open %.3f closed %.3f' % (np.mean(abs(ao)), np.mean(abs(ac)), np.median(ao), np.median(ac)))
    groups = {}
    for r in rows: groups.setdefault((r['case'], r['t']), []).append(r)
    agree = both = imp_c_only = imp_o_only = 0
    for g in groups.values():
        bo = max(g, key=lambda r: r['Go']); bc = max(g, key=lambda r: r['Gc'])
        io, ic = bo['Go'] > bo['Gb']+.02, bc['Gc'] > bc['Gb']+.02
        both += io and ic; imp_c_only += ic and not io; imp_o_only += io and not ic
        agree += (bo['a'] == bc['a'])
    n = len(groups)
    print('states %d: best alternative same under both: %.0f%% ; improvable under both %d, only closed-loop %d, only open-loop %d' % (n, 100*agree/n, both, imp_c_only, imp_o_only))
    json.dump(rows, open(ST.replace('.json', '_labelcheck.json'), 'w'))
    from target_rl.parallel import close; close()
