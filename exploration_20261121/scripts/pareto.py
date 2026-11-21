"""Trajectory-output objectives on random 3-phase (r, w) plans, nominal conditions (L0), 1200 s fixed time:
range [km], total heat load [MJ/m^2], terminal specific energy [MJ/kg], terminal speed [m/s]."""
import sys, json, math, numpy as np, multiprocessing as mp
from target_rl import baselines as B, flight as F
from target_rl.scenario import Scenario, Target
NOM = Scenario('L0', 0, 0., 0., 1., 1.); PH = [400., 800.]
def run(u):
    env = F.TargetFlight(NOM, Target('x', 0., 0.), True, trace=True)
    env.advance(1200., B._command(np.array(u, float), PH, True))
    if env.failed: return None
    A = np.array(env.trace); m = env.w['model']; dt = env.w['dt']
    hf = np.array([m.heat_C*math.sqrt(m.atm.density(h)/m.heat_Rn)*V**3 for h, V in A[:, 1:3]])
    h, V = env.x[0], env.x[1]
    E = .5*V*V+m.atm.gravity(h)*h
    return dict(u=list(map(float, u)), R=env.range_km(), H=float(hf.sum()*dt/1e6), Hpk=float(hf.max()/1e6), E=float(E/1e6), V=float(V), h=float(h))
def job(u): return run(u)
if __name__ == '__main__':
    rng = np.random.default_rng(0); N = int(sys.argv[2])
    U = [list(np.r_[rng.uniform(.3, .95, 3), rng.choice([0., 1/3, 2/3, 1.], 3)]) for _ in range(N)]
    U += [[r]*3+[0.]*3 for r in np.linspace(.3, .95, 66)]                  # one-parameter family: constant q-law, no morph
    with mp.Pool(4) as p: res = p.map(job, U, chunksize=20)
    json.dump([r for r in res if r], open(sys.argv[1], 'w')); print('flown', len(U), 'valid', sum(r is not None for r in res))
