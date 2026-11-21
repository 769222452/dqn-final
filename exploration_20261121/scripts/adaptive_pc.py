"""Adaptive predictor-corrector under plant/model mismatch (levels of mismatch2). The drag model used for prediction is
CD = a cd0 + b dcd_w + c cd0 (M-6) + d dcd_w (M-6), parameters identified online by recursive least squares from the measured
drag along the flown trajectory (batch update every cycle); lift factor from the existing estimator. v2 execution."""
import sys, json, dataclasses, numpy as np, multiprocessing as mp
sys.path.insert(0, '/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad')
from target_rl import baselines as B, flight as F
from target_rl.scenario import scenario
from newdist import target
import snap_pc, mismatch2 as M2
CHEAP = M2.CHEAP; _cd0 = M2._cd0
def _deriv(self, y, cl, wl):
    model = self.w['model']; kr = self.field(y[3]); a = model.atm.speed_of_sound(y[0]); M = y[1]/a
    ad = getattr(self, 'adapt', None)
    if getattr(self, 'mm', False):
        cd = _cd0(self, M, cl, wl*M2.MM['eff'])*(1+M2.MM['visc']+M2.MM['slope']*(M-6.)); kL = self.sc.k_lift*(1+M2.MM['lift']); kD = self.sc.k_drag
    elif ad is not None:
        c0 = _cd0(self, M, cl, 0.); cw = _cd0(self, M, cl, wl)-c0
        cd = ad[0]*c0+ad[1]*cw+ad[2]*c0*(M-6.)+ad[3]*cw*(M-6.); kL = self.sc.k_lift; kD = 1.
    else:
        cd = _cd0(self, M, cl, wl); kL = self.sc.k_lift; kD = self.sc.k_drag
    return model.rhs(y, kL*cl, kD*cd, kr)
def predict_adapt(env, params, phases, pc, oracle=False):
    p = env.fork(); p.trace = None; p.mm = False
    kl, _ = env.est.values
    p.field = B._Const(env.k_rho()); p.sc = dataclasses.replace(env.sc, k_lift=float(np.clip(kl, .7, 1.3)), k_drag=1.)
    p.adapt = tuple(env.theta); p.w = dict(env.w, dt=pc['predict_dt_s'])
    p.advance(env.w['T']-env.t, B._command(params, phases, env.morph))
    return np.array([p.range_km(), p.x[1]]), (not p.failed) and p.min_margin >= pc['margin']
def rls_update(env, lam=.998):
    A = env.trace; m = env.w['model']; S = m.S; dt = env.w['dt']; th, P = env.theta, env.P
    for k in range(max(env.n_proc, 1), len(A)):
        t, h1, V1, g1, r1, cl, wl, qr, pr = A[k]; _, h0, V0, g0, r0 = A[k-1][:5]
        hm, vm, gm = .5*(h0+h1), .5*(V0+V1), .5*(g0+g1); sm = env.s0+1e3*.5*(r0+r1)
        kmeas = env.field(env.s0+1e3*r0)
        _, q = m.rhs([hm, vm, gm, sm], cl, 0., kmeas)
        d_hat = -m.m*(V1-V0)-m.m*m.atm.gravity(hm)*np.sin(gm)*dt
        cd_meas = d_hat/(q['q_dyn']*S*dt)
        M = vm/m.atm.speed_of_sound(hm); c0 = _cd0(env, M, cl, 0.); cw = _cd0(env, M, cl, wl)-c0
        phi = np.array([c0, cw, c0*(M-6.), cw*(M-6.)])
        if not np.isfinite(cd_meas) or abs(cd_meas) > .2: continue
        K = P@phi/(lam+phi@P@phi); th = th+K*(cd_meas-phi@th); P = (P-np.outer(K, phi@P))/lam
    env.theta = np.clip(th, [.4, .1, -.6, -.6], [2.5, 2.5, .6, .6]); env.P = P; env.n_proc = len(A)
def setup(level, adaptive):
    M2.MM.update(M2.LEVELS[level]); F.TargetFlight._deriv = _deriv; F.TargetFlight.__init__ = M2.init
    B.predict = predict_adapt if adaptive else M2.predict
    B._command = snap_pc.make_cmd(True, 8.)
def fly(level, adaptive, seed, fp):
    setup(level, adaptive)
    env = F.TargetFlight(scenario('L2', seed), target(seed, fp), True, trace=adaptive)
    env.theta = np.array([1., 1., 0., 0.]); env.P = np.eye(4)*2e3; env.n_proc = 0; env.adapt = None
    pc = B.PredictorCorrector('pc_strong'); pc.pc = dict(pc.pc, **CHEAP); nxt = 0.
    while not env.done:
        if env.t >= nxt-1e-9:
            if adaptive and env.t > 0: rls_update(env)
            pc.correct(env, pc.pc['newton_first'] if env.t == 0 else pc.pc['newton']); nxt += 48.
        env.advance(max(nxt-env.t, env.w['dt']), B._command(pc.u, pc.phases, True))
    s = env.summary(); s['theta'] = [float(x) for x in env.theta]; return s
def job(a):
    s = fly(*a); s.update(level=a[0], adaptive=a[1]); return s
if __name__ == '__main__':
    if sys.argv[1] == 'test':
        for fp in (True, False):
            for ad in (False, True):
                s = fly('M2 moderate', ad, 1500, fp); print(fp, ad, 'hit', s['hit'], 'feasible', s['feasible'], 'dD %.1f dV %.0f' % (s['dD_km'], s['dV_mps']), 'theta', np.round(s['theta'], 2))
    else:
        levels = sys.argv[2].split(',')
        import os
        AD = os.environ.get('ADAPT', '1') == '1'
        jobs = [(lv, AD, s, fp) for lv in levels for s in range(1500, 1596) for fp in (True, False)]
        with mp.Pool(4) as p: rows = p.map(job, jobs, chunksize=4)
        json.dump(rows, open(sys.argv[1], 'w'))
        for lv in levels:
            r = [x for x in rows if x['level'] == lv]; Mo = [x for x in r if x['target'].startswith('M')]
            print('%-12s PC (adaptive=' + str(AD) + '): success %5.1f%% (morph-only %5.1f%%) violations %4.1f%% mean return %+.3f  theta median %s' % (lv, 100*np.mean([x['hit'] for x in r]), 100*np.mean([x['hit'] for x in Mo]), 100*np.mean([not x['feasible'] for x in r]), np.mean([x['G'] for x in r]), np.round(np.median([x['theta'] for x in r], 0), 2)), flush=True)
