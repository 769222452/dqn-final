"""Fly the certified chain with a constant CL action for 1200 s; print the final state and per-checkpoint states."""
import sys, time, json, os
sys.path.insert(0, '/home/user/dqn-final/src')
import numpy as np
from certified_flight import CL_ACTIONS, build_flight
from rollout_improvement.policy import ChainPolicy
from rollout_improvement.rollouts import certified_fly
def main(idx, out):
    t0 = time.time()
    env, certifier = build_flight(out, f'const{idx}')
    print('built in %.0f s' % (time.time()-t0), flush=True)
    from rollout_improvement.policy import constant_stage
    pol = ChainPolicy([constant_stage(CL_ACTIONS[idx])], env)
    rec = []
    for until in (240., 480., 720., 960., 1200.):
        t1 = time.time(); G, stop, acts = certified_fly(certifier, pol, until=until)
        rec.append(dict(t=until, G=float(G), stop=str(stop), x=[float(v) for v in env.x], wall=time.time()-t1)); print(rec[-1], flush=True)
        if stop is not None: break
    json.dump(rec, open(out+'/summary.json', 'w'))
if __name__ == '__main__': main(int(sys.argv[1]), sys.argv[2])
