"""Strong predictor-corrector with the real morph mechanism in the plant: only shapes of the 2091 library (snap the
requested drag level to the nearest available shape at the current Mach and CL), 8 s linear transition, a new shape
accepted only after the current transition, unlimited switches. Its own predictions stay continuous (relax-and-snap)."""
import sys, json, numpy as np, multiprocessing as mp
from target_rl import baselines as B, flight as F
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from newdist import target
from target_rl.scenario import scenario
T = np.load('/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad/snap_table.npz')
MM, CLS, W = T['M'], T['cls'], T['W']
CHEAP = dict(cycle_s=48., newton=1, newton_first=4, predict_dt_s=4.)
_cmd0 = B._command
def make_cmd(snap, trans):
    def _command(params, phases, morph):
        base = _cmd0(params, phases, morph)
        def command(env):
            cl, w = base(env)
            if env.w['dt'] > .75: return cl, w
            if snap:
                mach = env.x[1]/env.w['model'].atm.speed_of_sound(env.x[0])
                i = int(np.argmin(abs(MM-mach))); j = int(np.argmin(abs(CLS-cl)))
                ws = W[i, j]; ws = ws[np.isfinite(ws)]
                if len(ws): w = float(ws[np.argmin(abs(ws-w))])
            if trans > 0:
                st = env.__dict__.setdefault('mstate', dict(act=0., frm=0., goal=0., t0=-1e9, n=0))
                if env.t-st['t0'] >= trans and abs(w-st['goal']) > 1e-9:
                    st.update(frm=st['act'], goal=w, t0=env.t, n=st['n']+1)
                st['act'] = st['frm']+(st['goal']-st['frm'])*min(1., (env.t-st['t0'])/trans)
                w = st['act']
            return cl, w
        return command
    return _command
VAR = {'continuous, instant': (False, 0.), 'continuous, 8 s transition': (False, 8.), 'library shapes, 8 s transition': (True, 8.)}
def job(a):
    name, seed, fp = a
    B._command = make_cmd(*VAR[name])
    pc = B.PredictorCorrector('pc_strong'); pc.pc = dict(pc.pc, **CHEAP)
    e = pc.fly(scenario('L2', seed), target(seed, fp)); s = e.summary(); s['var'] = name
    s['switches'] = e.__dict__.get('mstate', {}).get('n'); return s
if __name__ == '__main__':
    with mp.Pool(4) as p:
        for name in VAR:
            rows = p.map(job, [(name, s, fp) for s in range(1500, 1596) for fp in (True, False)], chunksize=2)
            Mo = [r for r in rows if r['target'].startswith('M')]; Fx = [r for r in rows if r['target'].startswith('F')]
            sw = [r['switches'] for r in rows if r['switches'] is not None]
            print('%-32s success %3.0f%% (fixed-reachable %3.0f%%, morph-only %3.0f%%) violations %2d  shape switches/flight %s' % (name, 100*np.mean([r['hit'] for r in rows]), 100*np.mean([r['hit'] for r in Fx]), 100*np.mean([r['hit'] for r in Mo]), sum(not r['feasible'] for r in rows), ('%.0f' % np.median(sw)) if sw else '-'), flush=True)
