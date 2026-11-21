"""Task C headroom: strong predictor-corrector at several safety margins (its only conservatism knob), and the
full-information planner, at disturbance levels L2 and L3. Execution as v2: altitude-floor guard, library shapes
snapped at the current Mach/CL, 8 s linear transition. New target distribution (>= 0.5 from the edge, stratified),
exploration scenarios 1500-1595 (192 cases per level)."""
import sys, json, numpy as np, multiprocessing as mp
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import baselines as B
from target_rl.scenario import scenario
from newdist import target
import snap_pc
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
MARGINS = (0.005, 0.02, 0.05, 0.1)
def job(a):
    level, kind, margin, seed, fp = a
    B._command = snap_pc.make_cmd(True, 8.)
    pc = B.PredictorCorrector(kind); pc.pc = dict(pc.pc, **CHEAP, margin=margin)
    e = pc.fly(scenario(level, seed), target(seed, fp)); s = e.summary()
    s.update(level=level, kind=kind, margin=margin); return s
if __name__ == '__main__':
    jobs = [(lv, k, m, s, fp) for lv in ('L2', 'L3') for (k, m) in [('pc_strong', m) for m in MARGINS]+[('pc_oracle', 0.005)]
            for s in range(1500, 1596) for fp in (True, False)]
    with mp.Pool(4) as p: rows = p.map(job, jobs, chunksize=4)
    json.dump(rows, open(sys.argv[1], 'w')); print('rows', len(rows))
