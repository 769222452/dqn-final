"""Quasi-Newton predictor-corrector: the sensitivity matrix J is computed by finite differences at t = 0 and refreshed only
every `refresh` cycles; every other cycle costs ONE prediction (the miss of the current plan) and a linear step."""
import sys, time, numpy as np, multiprocessing as mp
from target_rl import baselines as B, flight as F, study_r as R
from target_rl.study_a import cases
SC = np.array([1., 10.])
class PCQ(B.PredictorCorrector):
    def __init__(self, refresh):
        super().__init__('pc_strong'); self.refresh = refresh; self.J = None; self.cyc = 0
        self.pc = dict(self.pc, cycle_s=48., predict_dt_s=4.)
    def _fd(self, env, u):
        pc = self.pc; rb = pc['r_bounds']; hi = np.array([rb[1]]*3+[1.]*3)
        f0, ok0 = self._predict(env, u); e = (f0-self._tgt(env))/SC
        J = np.zeros((2, 6))
        for j in range(6):
            d = .02 if j < 3 else .05; uu = u.copy(); uu[j] = u[j]+d if u[j]+d <= hi[j] else u[j]-d
            fj, _ = self._predict(env, uu); J[:, j] = ((fj-self._tgt(env))/SC-e)/(uu[j]-u[j])
        return J
    def _tgt(self, env): return np.array([env.target.range_km, env.target.speed_mps])
    def correct(self, env, iterations):
        pc = self.pc; rb = pc['r_bounds']
        lo = np.array([rb[0]]*3+[0.]*3); hi = np.array([rb[1]]*3+[1.]*3)
        u = np.clip(self.u.copy(), lo, hi); free = self._free(env.t)
        n_it = pc['newton_first'] if env.t == 0 else 1
        for it in range(n_it):
            if self.J is None or it == 0 and (env.t == 0 or (self.refresh and self.cyc % self.refresh == 0)):
                self.J = self._fd(env, u)
            f0, ok0 = self._predict(env, u); e = (f0-self._tgt(env))/SC
            step = np.linalg.lstsq(self.J[:, free], -e, rcond=None)[0]
            lim = np.where(np.array(free) < 3, .15, .3); step = np.clip(step, -lim, lim)
            un = u.copy(); un[free] = np.clip(u[free]+step, lo[free], hi[free]); u = un
        self.u = u; self.cyc += 1
        return u
def fly(a):
    refresh, sc, tg = a
    t0 = time.time(); pc = PCQ(refresh); e = pc.fly(sc, tg); s = e.summary(); s.update(wall_s=time.time()-t0, npred=pc.n_predictions); return s
if __name__ == '__main__':
    cs = cases('validation')
    with mp.Pool(4) as p:
        for refresh in (0, 10, 5, 2):
            rows = p.map(fly, [(refresh, sc, tg) for sc, tg in cs], chunksize=1); ok = [r for r in rows if r['feasible']]
            print('Jacobian refresh every %-2s cycles: hits %3d/96 viol %2d meanG %6.3f |dD| %5.1f |dV| %4.0f preds %4d wall %.2f' % ('never' if refresh == 0 else refresh, sum(r['hit'] for r in rows), len(rows)-len(ok), np.mean([r['G'] for r in rows]), np.median([abs(r['dD_km']) for r in ok]), np.median([abs(r['dV_mps']) for r in ok]), np.median([r['npred'] for r in rows]), np.median([r['wall_s'] for r in rows])), flush=True)
