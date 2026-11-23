"""v3: plant mismatch (predictions fly the nominal model), flight ends at the target speed, time scale, start rule, splits.
The checks run in a subprocess with TARGET_RL_CONFIG=configs/v3.yaml (the default configuration is the round-2 task)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(code):
    env = dict(os.environ, TARGET_RL_CONFIG='configs/v3.yaml', TARGET_RL_LIBRARY='F8D2000B', PYTHONPATH=str(ROOT/'src'))
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-3000:]


def test_config_region_and_splits():
    run('''
import numpy as np
from target_rl import config, region as RG
c = config.load()
assert c['experiment']['tag'] == 'v3' and c['mission']['end'] == 'speed' and c['plant']['extra_drag'] > 0
assert config.runs_dir().parts[-3:] == ('runs', 'v3', 'F8D2000B') and config.region_path().parts[-4:] == ('runs', 'v3', 'F8D2000B', 'region.json')
R = lambda k: set(config.seed_range(c['seeds'][k]))
assert not (R('train') & R('validation')) and not (R('train') & R('test')) and not (R('validation') & R('test'))
reg = RG.load_region()
va = RG.v2_cases('validation')
assert len(va) == 192 and {t.name[0] for _, t in va} == {'F', 'M'}
assert all(RG.inside(np.asarray(reg['region']), (t.range_km, t.speed_mps)) for _, t in va)
assert len(RG.domain_grid()) > 20
print('ok')
''')


def test_flight_ends_at_target_speed():
    run('''
from target_rl import flight as F, guided as G
from target_rl.scenario import Scenario, Target
sc = Scenario('L0', 0, 0., 0., 1., 1.)
e = F.TargetFlight(sc, Target('x', 2500., 1500.), True)
e.advance(e.w['T'], G._hold(.05, 0.))
assert e.reason == 'SPEED' and abs(e.x[1]-1500.) < 1e-6 and e.t < e.w['T'] and abs(e.errors()[1]) < 1e-6
assert e.hit() == (abs(e.errors()[0]) <= 5.)
e = F.TargetFlight(sc, Target('x', 2500., 300.), True)                 # never reaches 300 m/s: limit or timeout, no hit
e.advance(e.w['T'], G._hold(.05, 0.))
assert e.reason != 'SPEED' and not e.hit()
print('ok')
''')


def test_plant_mismatch_and_nominal_prediction():
    run('''
from target_rl import flight as F, guided as G
from target_rl.scenario import Scenario, Target
sc = Scenario('L0', 0, 0., 0., 1., 1.)
rng = {}
for plant in (True, False):
    e = F.TargetFlight(sc, Target('x', 2500., 1300.), True)
    assert e.plant
    e.plant = plant
    e.advance(e.w['T'], G._hold(.05, 0.))
    rng[plant] = e.range_km()
assert rng[True] < .9*rng[False]                            # the plant has more drag than the model
tg = Target('x', 2600., 1300.)
p1 = G.RelPredFlight(sc, tg, True).pred.copy()
w = F.world(True); saved = w['plant']; w['plant'] = {}
try:
    p2 = G.RelPredFlight(sc, tg, True).pred.copy()
finally:
    w['plant'] = saved
assert abs(p1[0]-p2[0]) < 1e-9 and abs(p1[1]-p2[1]) < 1e-9      # predictions fly the nominal model whatever the plant
print('ok')
''')


def test_initial_cl_time_scale_and_start_rule():
    run('''
import numpy as np
from target_rl import config, flight as F, guided as G, policy as P, train as T
from target_rl.scenario import Target
c = config.load()
cl = [G.initial_cl_speed(Target('x', d, 1400.)) for d in (1800., 2400., 3000.)]
assert cl[0] > cl[1] > cl[2]                                 # longer range at the same speed: less CL
assert P.horizon() == 2400. and max(T.state_times(0, c['training']['states'], 48., 12., 2400.)) > 2000.
assert F.world(True)['w_levels'] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
rule = dict(kind='pfb', morph=True, **c['start_rule'])
f = np.zeros(len(P.NAMES))
f[P.NAMES.index('pred_range_error')] = .30                  # 30 km overshoot predicted, CL target 0: CL up, no brake yet
assert F.decode(P.initial_action(rule, f), True) == (4, 0)
f[P.NAMES.index('r_plan')] = 1.                              # CL target 0.10 >= cl_brake: CL up and full brake
assert F.decode(P.initial_action(rule, f), True) == (4, 5)
f[P.NAMES.index('pred_range_error')] = -.30                 # 30 km short: CL down, no brake
assert F.decode(P.initial_action(rule, f), True) == (0, 0)
print('ok')
''')


def test_v3b_held_out_acceptance_and_guard():
    env = dict(os.environ, TARGET_RL_CONFIG='configs/v3b.yaml', TARGET_RL_LIBRARY='F8D2000B', PYTHONPATH=str(ROOT/'src'))
    code = '''
from target_rl import config, flight as F, guided as G, region as RG
from target_rl.scenario import Scenario, Target
c = config.load()
assert c['experiment']['tag'] == 'v3b' and c['training']['held_out_acceptance'] and c['protection']['use_lift_estimate']
assert config.region_path().parts[-4:] == ('runs', 'v3', 'F8D2000B', 'region.json')
R = lambda k: set(config.seed_range(c['seeds'][k]))
for k in ('train', 'validation', 'test', 'domain'):
    assert not (R('acceptance') & R(k)), k
acc = RG.v2_cases('accept')
assert len(acc) == 200 and acc[0][1].name.startswith('acc')
sc = Scenario('L0', 0, 0., 0., .9, 1.)              # 10 % less lift than modelled (plus the plant's 5 %)
e = F.TargetFlight(sc, Target('x', 2500., 1300.), True)
e.advance(600., G._hold(.0, 0.))                     # the policy asks for no lift: the guard alone holds the limit
assert not e.failed and e.min_margin > 0
print('ok')
'''
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-3000:]
