"""Study R flight, start rule and training entry (run in a subprocess with configs/study_r.yaml)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = r'''
import numpy as np
from target_rl import guided as G, region as RG, policy as P, flight as F, config, train as T
assert config.load()['guidance']['kind'] == 'rel_pred' and F.n_actions(True) == 20 and F.n_actions(False) == 5
sc, tg = RG.test_cases(which='dev')[1]
rule = dict(kind='pfb', morph=True, **config.load()['start_rule'])
e = G.make(sc, tg, True)
while not e.done:
    e.step(P.chain_action([rule], e.features(extra=True), True))
assert e.npred == 25 and e.info['n_predictions'] == 25 and e.G is not None
e = G.make(sc, tg, True)
for _ in range(20):
    e.step(F.encode(2, 0))
c = e.fork(); c.step(F.encode(4, 3)); e.step(F.encode(4, 3))
assert c.x == e.x and c.cl_t == e.cl_t and np.allclose(c.pred, e.pred)
f = e.fork(); [f.step(F.encode(0, 0)) for _ in range(3)]; e.step(F.encode(2, 0))
assert f.x != e.x
st = T.VARIANTS['rel']; assert st['start'] == 'rule' and not T.VARIANTS['relfixed']['morph']
print('ok')
'''


def test_study_r_flight_and_rule():
    env = dict(os.environ, TARGET_RL_CONFIG='configs/study_r.yaml', PYTHONPATH=str(ROOT/'src'))
    r = subprocess.run([sys.executable, '-c', CODE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-2000:]
