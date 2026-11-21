"""Success map in the (range, terminal speed) plane. Grid points inside the nominal morph-reachable hull. L2 disturbance, 12 scenarios
per point. fixed-like: strong PC with the morph level forced to 0 (matches D and V); morph: strong PC with library shapes snapped and an
8 s transition. Altitude guard from config. Cheap PC settings."""
import sys, json, numpy as np, multiprocessing as mp
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import baselines as B, region as RG
from target_rl.scenario import scenario, Target
import snap_pc, fixed_fair
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
REG = RG.load_region(); MH = np.asarray(REG['morph_hull']); FXH = np.asarray(REG['fixed_hull'])
DS = np.arange(1900., 2561., 50.); VS = np.arange(1050., 1851., 75.)
PTS = [(float(d), float(v)) for d in DS for v in VS if RG.inside(MH, (d, v))]
_c0 = B._command
def job(a):
    morph, pt, seed = a
    B._command = snap_pc.make_cmd(True, 8.) if morph else fixed_fair.cmd_w0
    pc = B.PredictorCorrector('pc_strong'); pc.pc = dict(pc.pc, **CHEAP)
    e = pc.fly(scenario('L2', seed), Target('g', *pt)); return (morph, pt[0], pt[1], seed, bool(e.hit()), bool(e.failed))
if __name__ == '__main__':
    print('grid points inside the morph hull: %d (%d also inside the fixed hull)' % (len(PTS), sum(RG.inside(FXH, p) for p in PTS)), flush=True)
    jobs = [(m, p, s) for m in (False, True) for p in PTS for s in range(1500, 1512)]
    with mp.Pool(4) as p: rows = p.map(job, jobs, chunksize=6)
    json.dump(dict(pts=PTS, rows=rows), open(sys.argv[1], 'w')); print('rows', len(rows))
