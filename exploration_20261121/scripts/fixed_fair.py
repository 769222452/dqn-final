"""Fair fixed-geometry baseline: the same strong predictor-corrector (3 phases, matches BOTH range and speed) but the morph level
is forced to 0 (the minimum-drag shape; static morph gain is ~0.07%, so this is a proxy for a fixed geometry that optimises
its lift/dynamic-pressure schedule). v2 target distribution, L2, v2 execution (altitude guard)."""
import sys, json, numpy as np, multiprocessing as mp
from target_rl import baselines as B
from target_rl.scenario import scenario
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from newdist import target
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
_cmd0 = B._command
def cmd_w0(params, phases, morph):
    base = _cmd0(params, phases, morph)
    def command(env):
        cl, w = base(env); return cl, 0.
    return command
def job(a):
    w0, seed, fp = a
    B._command = cmd_w0 if w0 else _cmd0
    pc = B.PredictorCorrector('pc_strong'); pc.pc = dict(pc.pc, **CHEAP)
    e = pc.fly(scenario('L2', seed), target(seed, fp)); s = e.summary(); s['w0'] = w0; return s
if __name__ == '__main__':
    with mp.Pool(4) as p: rows = p.map(job, [(w0, s, fp) for w0 in (True, False) for s in range(1500, 1596) for fp in (True, False)], chunksize=4)
    json.dump(rows, open(sys.argv[1], 'w'))
    for w0 in (True, False):
        r = [x for x in rows if x['w0'] == w0]; Fx = [x for x in r if x['target'].startswith('F')]; Mo = [x for x in r if x['target'].startswith('M')]
        print('%-34s success %5.1f%%  | targets reachable without morph %5.1f%% | morph-only targets %5.1f%% | violations %4.1f%%' % ('w = 0 (fixed-like, matches D and V)' if w0 else 'free morph (continuous)', 100*np.mean([x['hit'] for x in r]), 100*np.mean([x['hit'] for x in Fx]), 100*np.mean([x['hit'] for x in Mo]), 100*np.mean([not x['feasible'] for x in r])))
