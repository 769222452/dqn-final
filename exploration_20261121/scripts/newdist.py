"""Interim study-R policies and references on the v2-style target distribution (>= 0.5 from the region edge,
one fixed-reachable and one morph-only target per scenario), exploration scenarios 1500-1595."""
import json, numpy as np
from target_rl import region as RG, study_r as R
from target_rl.parallel import pool_map, close
from target_rl.scenario import scenario, Target
from target_rl.train import case_to_json
reg = RG.load_region(); RR = np.asarray(reg['region']); FX = np.asarray(reg['fixed_hull']); SCL = np.array([100., 100.])
def seg(p, a, b):
    ab = b-a; t = np.clip(np.dot(p-a, ab)/max(np.dot(ab, ab), 1e-12), 0, 1); return np.linalg.norm(p-(a+t*ab))
def depth(p):
    q = np.asarray(p)/SCL; P = RR/SCL; return min(seg(q, P[i], P[(i+1) % len(P)]) for i in range(len(P)))
def target(seed, fixed_part, m=.5):
    rng = np.random.default_rng([int(seed), 23, int(fixed_part)]); (x0, y0), (x1, y1) = RR.min(0), RR.max(0)
    while True:
        p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if RG.inside(RR, p) and depth(p) >= m and RG.inside(FX, p) == fixed_part:
            return Target(('F' if fixed_part else 'M')+str(seed), round(p[0], 3), round(p[1], 3))
if __name__ == '__main__':
    cases = [(scenario('L2', s), target(s, fp)) for s in range(1500, 1596) for fp in (True, False)]
    methods = ['pc_fixed', 'rule_fixed', 'rl_fixed:17', 'rule_morph', 'rl_morph:17', 'pc_strong']
    res = {}
    for m in methods:
        chain, morph = R.spec(m)
        res[m] = pool_map([('study_r', dict(case=case_to_json(c), chain=chain, morph=morph, method=m)) for c in cases], 4)
        print('done', m, flush=True)
    json.dump(res, open('/tmp/claude-0/-home-user-dqn-final/87c281b1-63f7-56ca-b436-68c23ccd16ca/scratchpad/newdist_rows.json', 'w'))
    close()
