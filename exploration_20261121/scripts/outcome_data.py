"""Exact closed-loop counterfactual outcomes: for states along the interim policy's flights, the end point (range, speed)
of the baseline and of each alternative action followed by the policy."""
import json, sys, time, numpy as np, multiprocessing as mp
from target_rl import train as T, guided as G, policy as P, flight as F
ST = sys.argv[1]
TIMES = tuple(12.*k for k in (8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96))

def job(ci):
    st = json.load(open(ST)); chain = st['chain']; sc, tg = T.case_from_json(st['cases'][ci])
    dec = []
    base = T.fly(G.make(sc, tg, True), chain, log=dec)
    sched = {int(round(t)): a for t, a in dec}
    bs = base.summary()
    rows = []
    for t_b in TIMES:
        e = G.make(sc, tg, True); T.fly(e, chain, schedule=sched, until=t_b)
        if e.done: continue
        a0 = sched[int(round(t_b))]
        f = e.features(extra=True)
        for a in P.alternatives(a0, t_b, 0, True):
            b = T.fly(e.fork(), chain, override=(t_b, a)); s = b.summary()
            rows.append(dict(case=ci, t=t_b, f=[float(x) for x in f], a0=int(a0), a=int(a), tgt=[tg.range_km, tg.speed_mps],
                             base_end=[bs['range_km'], bs['V_end']], base_G=bs['G'], base_fail=not bs['feasible'],
                             alt_end=[s['range_km'], s['V_end']], alt_G=s['G'], alt_fail=not s['feasible']))
    return rows

if __name__ == '__main__':
    st = json.load(open(ST)); n = len(st['cases']); ids = list(range(0, n, 3))[:100]
    t0 = time.time()
    with mp.Pool(4) as p: rows = [r for rr in p.map(job, ids, chunksize=1) for r in rr]
    json.dump(rows, open(sys.argv[2], 'w')); print('rows', len(rows), 'cases', len(ids), '%.0f s' % (time.time()-t0))
    from target_rl.parallel import close; close()
