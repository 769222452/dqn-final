"""v2 execution layer (altitude guard, library shapes, 8 s transition), v2 target distribution and splits, w0 baseline.
The checks run in a subprocess with TARGET_RL_CONFIG=configs/v2.yaml (the default configuration is the round-2 task)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(code):
    env = dict(os.environ, TARGET_RL_CONFIG='configs/v2.yaml', TARGET_RL_LIBRARY='F8D2000B', PYTHONPATH=str(ROOT/'src'))
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-3000:]


def test_config_and_splits():
    run('''
from target_rl import config
c = config.load()
assert c['experiment']['tag'] == 'v2' and c['protection']['floor_buffer_m'] == 1500. and c['execution']['morph']['realistic']
R = lambda k: set(config.seed_range(c['seeds'][k]))
assert not (R('train') & R('validation')) and not (R('train') & R('test')) and not (R('validation') & R('test'))
assert not (R('test') & R('dev')) and R('domain') <= R('validation')
assert not (R('validation') & set(range(1400, 1600))), 'seeds looked at during exploration are not decision seeds'
assert str(config.runs_dir()).endswith('runs/v2/F8D2000B')
print('ok')
''')


def test_target_distribution():
    run('''
import numpy as np
from target_rl import config, region as RG
reg = RG.load_region(); FX = np.asarray(reg['fixed_hull'])
margin = config.load()['targets']['v2']['margin']
tr, va, te, dm = (RG.v2_cases(k) for k in ('train', 'validation', 'test', 'domain'))
assert (len(tr), len(va), len(te)) == (300, 192, 320) and len(dm) == len(RG.domain_grid()) * 12 and len(RG.domain_grid()) == 72
for _, t in va:
    p = (t.range_km, t.speed_mps)
    assert RG.edge_depth(p, reg) >= margin-1e-9 and RG.inside(FX, p) == (t.name[0] == 'F'), t
assert {t.name[0] for _, t in va} == {'F', 'M'} and all(RG.inside(np.asarray(reg['morph_hull']), (g.range_km, g.speed_mps)) for _, g in dm)
assert [t for _, t in RG.v2_cases('validation')][:6] == [t for _, t in va][:6]          # reproducible
print('ok')
''')


def test_morph_execution_snap_and_transition():
    run('''
from target_rl import flight as F
from target_rl.scenario import Scenario, Target
sc = Scenario('L0', 0, 0., 0., 1., 1.)
env = F.TargetFlight(sc, Target('x', 2300., 1400.), True)
assert env.realistic
d = env.w['drag']
assert d.snap(6., .03, 0.) == 0. and abs(d.snap(6., .03, 1.)-1.) < 1e-6 and abs(d.snap(6., .03, .5)-.5) < .06
env.advance(4., lambda e: (.03, 1.))
assert .4 < env.mstate['act'] < .6 and env.mstate['n'] == 1                   # halfway through the 8 s transition
env.advance(4.5, lambda e: (.03, 0.))                                           # the new request waits for the current shape
assert env.mstate['n'] == 2 and .85 < env.mstate['act'] < 1.0
env.advance(8., lambda e: (.03, 0.))
assert env.mstate['act'] == 0. and env.mstate['n'] == 2 and env.summary()['n_switches'] == 2
f = env.fork(); f.advance(30., lambda e: (.03, 1.))
assert env.mstate['n'] == 2, 'a fork does not change the mstate of the original'
print('ok')
''')


def test_predictions_are_continuous_and_instantaneous():
    run('''
import numpy as np
from target_rl import baselines as B, flight as F
from target_rl.scenario import Scenario, Target
sc = Scenario('L0', 0, 0., 0., 1., 1.)
env = F.TargetFlight(sc, Target('x', 2300., 1400.), True)
pc = B.PredictorCorrector('pc_strong')
f, ok = B.predict(env, np.array([.8, .8, .8, 0., .5, .5]), [400., 800.], pc.pc)
assert np.all(np.isfinite(f)) and env.mstate['n'] == 0 and env.realistic
print('ok')
''')


def test_altitude_guard_raises_lift_near_the_floor():
    run('''
from target_rl import flight as F
from target_rl.scenario import Scenario, Target
sc = Scenario('L0', 0, 0., 0., 1., 1.)
a = F.TargetFlight(sc, Target('x', 2300., 1400.), True)
b = F.TargetFlight(sc, Target('x', 2300., 1400.), True)
b.w = dict(b.w, prot=dict(b.w['prot'], floor_buffer_m=0.))
for e in (a, b):
    e.x[0] = a.w['limits'].h[0]+400.
    e.x[1] = 1100.                     # slow and low: the dynamic-pressure law alone would not lift the vehicle
cl_a, pa = a.command_cl(0.); cl_b, pb = b.command_cl(0.)
assert cl_a > cl_b+1e-4 and pa, (cl_a, cl_b)
a.x[0] = 40000.; b.x[0] = 40000.
assert abs(a.command_cl(.03)[0]-b.command_cl(.03)[0]) < 1e-12      # no effect away from the floor
print('ok')
''')


def test_fixed_like_baseline_keeps_the_morph_level_at_zero():
    run('''
from target_rl import baselines as B, flight as F, region as RG
pc = B.PredictorCorrector('pc_strong_w0')
assert pc.strong and pc.w0 and pc._free(0.) == [0, 1, 2]
sc, tg = RG.v2_cases('validation')[0]
env = pc.fly(sc, tg)
assert env.summary()['n_switches'] == 0 and env.mstate['goal'] == 0. and not env.failed
print('ok')
''')
