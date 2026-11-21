"""Late target assignment: the target (v2 distribution) is revealed at t_a; before that the vehicle flies the
predictor-corrector towards a fixed aim point. Upper bound: target known from t = 0. L2, v2 execution
(altitude guard via config, library shapes + 8 s transition)."""
import sys, json, numpy as np, multiprocessing as mp
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import baselines as B, flight as F, region as RG
from target_rl.scenario import scenario, Target
from newdist import target, depth
import snap_pc
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
reg = RG.load_region(); R = np.asarray(reg['region'])
rng = np.random.default_rng(5)
samp = np.array([[t.range_km, t.speed_mps] for t in (target(s, fp) for s in range(9000, 9300) for fp in (True, False))])
Dq, Vq = np.percentile(samp[:, 0], [20, 50, 80]), np.percentile(samp[:, 1], [20, 50, 80])
AIMS = {'center': (Dq[1], Vq[1]), 'fast': (Dq[1], Vq[2]), 'far+fast': (Dq[2], Vq[2]), 'far': (Dq[2], Vq[1]), 'near': (Dq[0], Vq[1])}
AIMS = {k: v for k, v in AIMS.items() if RG.inside(R, v)}
def fly(sc, tg, aim, t_a):
    env = F.TargetFlight(sc, Target('aim', *aim) if aim else tg, True)
    pc = B.PredictorCorrector('pc_strong'); pc.pc = dict(pc.pc, **CHEAP)
    nxt, switched = 0., aim is None
    while not env.done:
        if not switched and env.t >= t_a-1e-9:
            env.target = tg; switched = True; pc.correct(env, 4); nxt = env.t+pc.pc['cycle_s']
        elif env.t >= nxt-1e-9:
            pc.correct(env, pc.pc['newton_first'] if env.t == 0 else pc.pc['newton']); nxt += pc.pc['cycle_s']
        stop = min(nxt, t_a) if not switched and t_a > env.t else nxt
        env.advance(max(stop-env.t, env.w['dt']), B._command(pc.u, pc.phases, True))
    return env.summary()
def job(a):
    seed, fp, aim_name, t_a = a
    B._command = snap_pc.make_cmd(True, 8.)
    s = fly(scenario('L2', seed), target(seed, fp), AIMS.get(aim_name), t_a)
    s.update(seed=seed, aim=aim_name, t_a=t_a); return s
if __name__ == '__main__':
    print('aims', {k: (round(v[0]), round(v[1])) for k, v in AIMS.items()}, flush=True)
    jobs = [(s, fp, 'known', 0.) for s in range(1500, 1596) for fp in (True, False)]
    jobs += [(s, fp, a, t) for t in (400., 700.) for a in AIMS for s in range(1500, 1596) for fp in (True, False)]
    with mp.Pool(4) as p: rows = p.map(job, jobs, chunksize=4)
    json.dump(rows, open(sys.argv[1], 'w')); print('rows', len(rows), flush=True)
