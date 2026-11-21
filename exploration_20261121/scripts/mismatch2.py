"""Model mismatch headroom. The PLANT differs from the model used inside the predictor-corrector:
  morph effectiveness (drag authority of w), extra viscous drag, drag slope with Mach, lift error.
The estimator of the lift/drag biases still compares against the NOMINAL model (honest: it learns only a constant factor).
pc_strong: predictions with the nominal model (frozen density, estimated biases). pc_oracle: true density, true biases AND the
true plant. v2 execution (altitude guard, library shapes, 8 s transition) in the plant. v2 target distribution, L2, seeds 1500-1595."""
import sys, json, numpy as np, multiprocessing as mp
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import baselines as B, flight as F
from target_rl.scenario import scenario
from newdist import target
import snap_pc
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
LEVELS = {'M0 none': dict(eff=1., visc=0., slope=0., lift=0.), 'M1 mild': dict(eff=.8, visc=.10, slope=.05, lift=-.03),
          'M2 moderate': dict(eff=.6, visc=.20, slope=.10, lift=-.05), 'M3 strong': dict(eff=.4, visc=.30, slope=.15, lift=-.08)}
MM = dict(eff=1., visc=0., slope=0., lift=0.)
_cd0 = F.TargetFlight._cd; _init0 = F.TargetFlight.__init__; _pred0 = B.predict
def _deriv(self, y, cl, wl):
    model = self.w['model']; kr = self.field(y[3]); a = model.atm.speed_of_sound(y[0]); M = y[1]/a
    if getattr(self, 'mm', False):
        cd = _cd0(self, M, cl, wl*MM['eff'])*(1+MM['visc']+MM['slope']*(M-6.)); kL = self.sc.k_lift*(1+MM['lift'])
    else:
        cd = _cd0(self, M, cl, wl); kL = self.sc.k_lift
    return model.rhs(y, kL*cl, self.sc.k_drag*cd, kr)
def init(self, *a, **k):
    _init0(self, *a, **k); self.mm = True
def predict(env, params, phases, pc, oracle=False):
    if oracle: return _pred0(env, params, phases, pc, oracle)
    mm = env.mm; env.mm = False
    try: return _pred0(env, params, phases, pc, oracle)
    finally: env.mm = mm
def job(a):
    level, kind, seed, fp = a
    MM.update(LEVELS[level]); F.TargetFlight._deriv = _deriv; F.TargetFlight.__init__ = init; B.predict = predict
    B._command = snap_pc.make_cmd(True, 8.)
    pc = B.PredictorCorrector(kind); pc.pc = dict(pc.pc, **CHEAP)
    e = pc.fly(scenario('L2', seed), target(seed, fp)); s = e.summary(); s.update(level=level, kind=kind); return s
if __name__ == '__main__':
    jobs = [(lv, k, s, fp) for lv in LEVELS for k in ('pc_strong', 'pc_oracle') for s in range(1500, 1596) for fp in (True, False)]
    with mp.Pool(4) as p: rows = p.map(job, jobs, chunksize=4)
    json.dump(rows, open(sys.argv[1], 'w'))
    for lv in LEVELS:
        for k in ('pc_strong', 'pc_oracle'):
            r = [x for x in rows if x['level'] == lv and x['kind'] == k]; M = [x for x in r if x['target'].startswith('M')]
            print('%-12s %-10s success %5.1f%% (morph-only targets %5.1f%%) violations %4.1f%%  mean return %+.3f' % (lv, k, 100*np.mean([x['hit'] for x in r]), 100*np.mean([x['hit'] for x in M]), 100*np.mean([not x['feasible'] for x in r]), np.mean([x['G'] for x in r])), flush=True)
