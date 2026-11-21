"""Late-confirmed handover (A or B) headroom. 0-408 s: strong PC (cheap settings, v2 execution: altitude guard from config,
library shapes snapped, 8 s transition) flies towards a common aim point without knowing A or B. 408 s: the final target is
announced; the same PC continues to 1200 s. One prep flight per (scenario, aim); A / B continuations are forks of the same
408 s state. L2 disturbance, scenarios 1500-1547."""
import sys, copy, json, numpy as np, multiprocessing as mp
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import baselines as B, flight as F, region as RG
from target_rl.scenario import scenario, Target
import snap_pc
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
T_REVEAL = 408.
REG = RG.load_region(); TG = {t['name']: (t['range_km'], t['speed_mps']) for t in REG['targets']}
DS = [TG['T4'][0], 2280., TG['T2'][0]]
VS = [TG['T4'][1], TG['T5'][1], 1300., 1400., TG['T2'][1]]
AIMS = [(d, v) for d in DS for v in VS]
CONT = ['T4', 'T5', 'T2']
def new_pc(kind, morph):
    pc = B.PredictorCorrector(kind); pc.pc = dict(pc.pc, **CHEAP); return pc
def job(a):
    seed, aim, morph = a
    B._command = snap_pc.make_cmd(True, 8.) if morph else B._command0
    sc = scenario('L2', seed); kind = 'pc_strong' if morph else 'pc_fixed'
    env = F.TargetFlight(sc, Target('aim', *aim), morph); pc = new_pc(kind, morph); nxt = 0.
    while env.t < T_REVEAL-1e-9 and not env.done:
        if env.t >= nxt-1e-9: pc.correct(env, pc.pc['newton_first'] if env.t == 0 else pc.pc['newton']); nxt += 48.
        env.advance(max(min(nxt, T_REVEAL)-env.t, env.w['dt']), B._command(pc.u, pc.phases, morph))
    out = dict(seed=seed, aim=list(aim), morph=morph, prep_failed=bool(env.failed))
    for name in CONT:
        if env.done:
            out[name] = False; continue
        e = env.fork(); 
        if hasattr(env, 'mstate'): e.mstate = dict(env.mstate)
        p = copy.deepcopy(pc); e.target = Target(name, *TG[name]); p.correct(e, 4); nxt2 = e.t+48.
        while not e.done:
            if e.t >= nxt2-1e-9: p.correct(e, 1); nxt2 += 48.
            e.advance(max(nxt2-e.t, e.w['dt']), B._command(p.u, p.phases, morph))
        out[name] = bool(e.hit())
    return out
if __name__ == '__main__':
    B._command0 = B._command
    morph = sys.argv[2] == 'morph'
    jobs = [(s, aim, morph) for s in range(1500, 1548) for aim in AIMS]
    with mp.Pool(4) as p: rows = p.map(job, jobs, chunksize=3)
    json.dump(rows, open(sys.argv[1], 'w')); print('rows', len(rows), flush=True)
