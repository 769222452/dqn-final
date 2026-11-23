"""v4c: F8D2000B snapshot, drag-level generator, truth plant, and the target-arrival flight on the certified chain.
The chain checks run in a subprocess with TARGET_RL_CONFIG=configs/v4c.yaml and CF_RUNTIME=configs/runtime_v4c.json."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def test_snapshot_and_level_generator():
    from certified_flight.aero import AeroDatabase
    db = AeroDatabase(ROOT/'data'/'geometry_f8d2000b')
    assert len(db.ids) == 2091 and 'F8D_01351' in db.index and sum(not g.startswith('F8D') for g in db.ids) == 91
    _, cd, ok = db.equal_lift(6., .05)
    db.level = None
    k0 = db.generate(6., .05)['geometry_index']
    assert cd[k0] == cd[ok].min()                                   # default: the original minimum-drag generator
    db.level = 1.
    k1 = db.generate(6., .05)['geometry_index']
    assert cd[k1] == cd[ok].max()
    db.level = .5
    g = db.generate(6., .05)
    levels = (cd[ok]-cd[ok].min())/(cd[ok].max()-cd[ok].min())
    lv = (cd[g['geometry_index']]-cd[ok].min())/(cd[ok].max()-cd[ok].min())
    assert abs(abs(lv-.5)-np.min(np.abs(levels-.5))) < 1e-12 and g['selection'] == 'DRAG_LEVEL_AT_CL'   # nearest level
    db.level = None


def test_truth_plant_neutral_equals_nominal_and_mismatch_adds_drag():
    from certified_flight.mission import Mission
    from certified_flight.aero import AeroDatabase
    from certified_flight.plant import CompiledPlant
    from certified_flight.truth import TruthPlant
    m = Mission(ROOT, 'configs/mission.yaml')
    db = AeroDatabase(ROOT/'data'/'geometry_f8d2000b')
    plant = CompiledPlant(m, db)
    tp = TruthPlant(plant, 'F8D_01351')
    x = np.array(m.data['initial_state']['x_h_V_gamma_s'], float)
    k = db.index['F8D_01351']
    brake = [i for i, g in enumerate(db.ids) if not g.startswith('F8D')][-1]
    args = (x, np.full(6, .06), np.arange(7)*.5, k, brake, 0., 1/8, .5)
    assert np.array_equal(plant.integrate(*args)['x'], tp.integrate(*args)['x'])

    class Sc:
        k_lift, k_drag = 1., 1.
    tp.set_scenario(lambda s: 1., Sc(), dict(extra_drag=.2))
    assert tp.integrate(*args)['x'][-1, 1] < plant.integrate(*args)['x'][-1, 1]   # more drag: slower


def test_chain_flight_runs_forks_deterministically():
    env = dict(os.environ, TARGET_RL_CONFIG='configs/v4c.yaml', TARGET_RL_LIBRARY='F8D2000B',
               CF_RUNTIME='configs/runtime_v4c.json', PYTHONPATH=str(ROOT/'src'))
    code = '''
import numpy as np
from target_rl import config, flight as F
from target_rl.chain import ChainFlight
from target_rl.scenario import scenario, Target
e = ChainFlight(scenario('L2', 1100), Target('x', 2100., 1400.), True, physics_controller=True)
a_brake = F.encode(2, 5, True)
e.step(F.encode(2, 0, True)); e.step(a_brake)
f = e.fork()
e.step(a_brake); f.step(a_brake)
assert e.x == f.x and e.t == f.t == 36.                                  # a fork replays exactly
assert e.n_switches >= 1 and not e.failed and e.min_margin > 0
assert sum(e.channels.values()) == 12                                    # 3 s chain decisions
g = ChainFlight(scenario('L2', 1100), Target('x', 2100., 1400.), True, physics_controller=True)
for _ in range(3):
    g.step(F.encode(2, 0, True))
assert g.x != e.x                                                         # braking changed the motion
print('ok')
'''
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-3000:]
