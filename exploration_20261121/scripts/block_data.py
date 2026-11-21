"""Block counterfactuals: force one action for N consecutive decisions, then continue with the policy (closed loop)."""
import json, sys, time, numpy as np, multiprocessing as mp
from target_rl import train as T, guided as G, policy as P, flight as F
ST = sys.argv[1]
TIMES = tuple(12.*k for k in (8, 16, 24, 32, 40, 48, 56, 64, 72, 80))
BLOCKS = {'brake1/3': (2, 1), 'brake2/3': (2, 2), 'brake1': (2, 3), 'cl-0.005': (0, None), 'cl+0.005': (4, None), 'hold': (2, None)}
def job(ci):
    st = json.load(open(ST)); chain = st['chain']; sc, tg = T.case_from_json(st['cases'][ci])
    dec = []; base = T.fly(G.make(sc, tg, True), chain, log=dec); bs = base.summary()
    sched = {int(round(t)): a for t, a in dec}; rows = []
    for t_b in TIMES:
        e = G.make(sc, tg, True); T.fly(e, chain, schedule=sched, until=t_b)
        if e.done: continue
        f = e.features(extra=True); w_cur = int(round(e.w_cur*3))
        for name, (icl, iw) in BLOCKS.items():
            for n in (1, 4):
                b = e.fork(); a = F.encode(icl, w_cur if iw is None else iw)
                for _ in range(n):
                    if b.done: break
                    b.step(a)
                T.fly(b, chain); s = b.summary()
                rows.append(dict(case=ci, t=t_b, f=[float(x) for x in f], name=name, n=n, a=int(a), tgt=[tg.range_km, tg.speed_mps],
                                 base_end=[bs['range_km'], bs['V_end']], base_fail=not bs['feasible'], alt_end=[s['range_km'], s['V_end']], alt_fail=not s['feasible']))
    return rows
if __name__ == '__main__':
    st = json.load(open(ST)); n = len(st['cases']); ids = list(range(0, n, 4))[:70]; t0 = time.time()
    with mp.Pool(4) as p: rows = [r for rr in p.map(job, ids, chunksize=1) for r in rr]
    json.dump(rows, open(sys.argv[2], 'w')); print('rows', len(rows), 'cases', len(ids), '%.0f s' % (time.time()-t0))
    from target_rl.parallel import close; close()
