"""Flight environment, policy helpers, region geometry, baselines and a tiny training run."""
import numpy as np
import pytest

from target_rl import baselines as B, flight as F, policy as P, region as RG, train as T
from target_rl.scenario import Scenario, Target, scenario

NOMINAL = Scenario('L0', 0, 0., 0., 1., 1.)


def constant_flight(sc, tg, a, morph=True):
    env = F.TargetFlight(sc, tg, morph)
    while not env.done:
        env.step(a)
    return env


def test_actions_roundtrip():
    assert F.n_actions(True) == 19*4 and F.n_actions(False) == 19
    for a in range(F.n_actions(True)):
        assert F.encode(*F.decode(a)) == a


def test_flight_is_deterministic_and_fork_is_independent():
    sc, tg = scenario('L2', 1100), Target('t', 2300., 1450.)
    a = F.encode(6, 1)
    e1, e2 = constant_flight(sc, tg, a), constant_flight(sc, tg, a)
    assert e1.summary() == e2.summary()
    env = F.TargetFlight(sc, tg)
    for _ in range(20):
        env.step(a)
    c = env.fork()
    c.step(F.encode(8, 3))
    while not env.done:
        env.step(a)
    assert env.summary() == e1.summary()


def test_hit_and_return():
    a = F.encode(7, 2)
    end = constant_flight(NOMINAL, Target('x', 0., 0.), a)
    hit = constant_flight(NOMINAL, Target('h', end.range_km()+3., end.x[1]-20.), a)
    assert hit.hit() and hit.G == pytest.approx(.10)
    miss = constant_flight(NOMINAL, Target('m', end.range_km()+50., end.x[1]), a)
    assert not miss.hit() and miss.G == pytest.approx(-.5, rel=1e-6)


def test_violation_ends_flight():
    env = F.TargetFlight(NOMINAL, Target('x', 2000., 1200.))
    env.advance(1200., lambda e: (.12, 1.))      # maximum lift: pull-up beyond a limit
    assert env.failed and env.done and env.G == -4. and env.reason.startswith('VIOLATION')


def test_protection_holds_dynamic_pressure():
    env = constant_flight(NOMINAL, Target('x', 2000., 1200.), F.encode(1, 0))   # CL 0: dive without protection
    assert not env.failed and env.summary()['protect_fraction'] > .5


def test_neighbourhood_and_alternatives():
    a = F.encode(9, 1)
    nb = P.neighbourhood(a)
    assert len(nb) == 5*3 and a in nb
    alts = P.alternatives(a, 0., 0)
    assert {F.decode(x) for x in alts} == {(8, 1), (10, 1), (9, 0), (9, 2)}
    assert {F.decode(x) for x in P.alternatives(a, 96., 0)} == {(7, 1), (11, 1), (9, 0), (9, 2)}
    assert P.alternatives(F.encode(0, 0, False), 0., 0, False) == [F.encode(1, 0, False)]


def test_region_geometry():
    sq = RG.hull([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)])
    assert RG.inside(sq, (1, 1)) and not RG.inside(sq, (3, 1))
    assert RG.chord(sq, 1.) == (0., 2.)
    assert np.allclose(RG.shrink(sq, .5).min(0), [.5, .5])


def test_predictor_corrector_hits_a_nominal_target():
    plan = dict(r=[.8, .8, .8], w=[.3, .3, .3])
    end = RG.fly_plan(NOMINAL, plan, True)
    tg = Target('p', end.range_km()-20., end.x[1]-30.)
    env = B.PredictorCorrector('pc_fair').fly(NOMINAL, tg)
    assert env.hit(), env.summary()


def test_tiny_training_run_is_resumable(tmp_path):
    cases = [(scenario('L2', s), Target(f'c{s}', 2250.+10*k, 1400.+20*k)) for k, s in enumerate((1100, 1101, 1102))]
    kw = dict(workers=1, out=tmp_path, cases=cases, pi0=F.encode(6, 1), overrides=dict(states=3), log=lambda m: None)
    st = T.train('full', 17, iterations=1, **kw)
    assert st['iteration'] == 1 and len(st['history']) == 1
    st = T.train('full', 17, iterations=1, **kw)
    assert st['iteration'] == 2
    chain = st['chain']
    env = T.fly(F.TargetFlight(*cases[0]), chain)
    assert env.done and env.G is not None
