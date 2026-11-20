"""Calibrate the target distribution of v2: random targets at least m (normalised: range/100 km, speed/100 m/s) inside
the nominal reachable region. Full-information planner (pc_oracle, cheap settings) and the normal strong predictor-
corrector, both with the altitude-floor guard, on training scenarios 1100-1195."""
import sys, json, time, multiprocessing as mp, numpy as np
from target_rl import baselines as B, region as RG, config
from target_rl.scenario import scenario, Target
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
reg = RG.load_region(); R = np.asarray(reg['region']); SCL = np.array([100., 100.])
def seg_dist(p, a, b):
    ab = b-a; t = np.clip(np.dot(p-a, ab)/max(np.dot(ab, ab), 1e-12), 0, 1); return np.linalg.norm(p-(a+t*ab))
def depth(p):
    q = np.asarray(p)/SCL; P = R/SCL
    return min(seg_dist(q, P[i], P[(i+1) % len(P)]) for i in range(len(P)))
def target(seed, m):
    rng = np.random.default_rng([int(seed), 11, int(m*1000)]); (x0, y0), (x1, y1) = R.min(0), R.max(0)
    for _ in range(100000):
        p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if RG.inside(R, p) and depth(p) >= m: return Target(f's{seed}m{m}', round(p[0], 3), round(p[1], 3))
def job(a):
    kind, seed, m = a; tg = target(seed, m); sc = scenario('L2', seed)
    pc = B.PredictorCorrector(kind); pc.pc = dict(pc.pc, **CHEAP); e = pc.fly(sc, tg); s = e.summary()
    return dict(kind=kind, m=m, seed=seed, hit=s['hit'], feasible=s['feasible'], dD=s['dD_km'], dV=s['dV_mps'])
if __name__ == '__main__':
    seeds = range(1100, 1196); ms = (0.0, 0.25, 0.5, 0.75)
    t0 = time.time()
    with mp.Pool(4) as p: rows = p.map(job, [(k, s, m) for m in ms for k in ('pc_oracle', 'pc_strong') for s in seeds], chunksize=2)
    json.dump(rows, open(sys.argv[1], 'w'))
    # share of the region area kept and of the morph-only band kept
    rng = np.random.default_rng(0); (x0, y0), (x1, y1) = R.min(0), R.max(0)
    pts = [p for p in ((rng.uniform(x0, x1), rng.uniform(y0, y1)) for _ in range(20000)) if RG.inside(R, p)]
    dd = np.array([depth(p) for p in pts])
    print('time %.0f s' % (time.time()-t0))
    for m in ms:
        r = [x for x in rows if x['m'] == m]
        o = [x for x in r if x['kind'] == 'pc_oracle']; s = [x for x in r if x['kind'] == 'pc_strong']
        print('margin %.2f (%3.0f km / %3.0f m/s): area kept %3.0f%% | full-info planner hits %2d/96 viol %d | strong PC hits %2d/96 viol %d' % (
            m, m*100, m*100, 100*np.mean(dd >= m), sum(x['hit'] for x in o), sum(not x['feasible'] for x in o), sum(x['hit'] for x in s), sum(not x['feasible'] for x in s)))
    for t in reg['targets']:
        print('  %s depth %.2f' % (t['name'], depth((t['range_km'], t['speed_mps']))))
