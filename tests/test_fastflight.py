"""Fast disturbed flight (certified_flight.fastflight) and the scenario training helpers."""
import numpy as np
import pytest

from certified_flight import fastflight as F
from rollout_improvement.features import ALL_FEATURE_NAMES, FEATURE_NAMES, features, input_names, parse_inputs
from rollout_improvement.scenario_training import FULL_INPUTS, branch_job, candidate_job, constant, fly

R90 = F.R_LEVELS.index(.90)


def flown(sc, a=R90, **inner):
    env = F.FastFlight(sc, **inner)
    while not env.done:
        env.step(a)
    return env


def test_nominal_reproduces_the_pilot():
    # experiments/perturb pilot: q-tracking law b = 0.10, kp 2, kg 1, nominal state -> 2480.96 km
    env = flown(F.Scenario('L0', 0, 0., 0., 1., 1.))
    assert env.summary()['success']
    assert env.range_km == pytest.approx(2480.963, abs=1e-2)


def test_scenarios_are_deterministic_and_disjoint():
    assert F.scenario('L2', 1003) == F.scenario('L2', 1003)
    assert not set(F.TRAIN_SEEDS) & set(F.TEST_SEEDS)
    a, b = F.scenario('L1', 1003), F.scenario('L3', 1003)
    assert (a.dh, a.dv) == (b.dh, b.dv)                       # same offset at every level
    assert np.sign(a.k_lift-1) == np.sign(b.k_lift-1)         # same standard-normal draw, scaled by the level
    sc = F.scenario('L2', 1003)
    assert abs(sc.dh) <= 2000 and abs(sc.dv) <= 100 and abs(sc.k_lift-1) <= .1+1e-12


def test_flight_is_deterministic_and_fork_is_independent():
    sc = F.scenario('L2', 1001)
    e1, e2 = flown(sc), flown(sc)
    assert e1.range_km == e2.range_km and e1.G == e2.G
    env = F.FastFlight(sc)
    for _ in range(10):
        env.step(R90)
    fork = env.fork()
    x = env.x.copy()
    while not fork.done:
        fork.step(R90-3)
    assert np.array_equal(env.x, x)                          # the original is untouched
    while not env.done:
        env.step(R90)
    assert env.range_km == pytest.approx(e1.range_km, abs=1e-9)


def test_violation_ends_the_flight_as_failure():
    # a set point at the limit with no damping on a strong-disturbance scenario overshoots q
    bad = [s for s in range(1000, 1016) if not flown(F.scenario('L3', s), F.R_LEVELS.index(.98), kp=1., kg=0.).summary()['success']]
    assert bad, 'expected at least one violation'
    env = flown(F.scenario('L3', bad[0]), F.R_LEVELS.index(.98), kp=1., kg=0.)
    s = env.summary()
    assert not s['success'] and s['reason'].startswith('VIOLATION') and env.t < F.MISSION_S


def test_features_and_inputs():
    env = F.FastFlight(F.scenario('L2', 1002))
    f = features(env)
    assert len(f) == len(ALL_FEATURE_NAMES) == len(FEATURE_NAMES)+5
    idx = parse_inputs(FULL_INPUTS)
    assert input_names(idx) == FULL_INPUTS.split(',')
    assert parse_inputs('h,V,gamma,alpha,t,prev') is None   # old meaning of 'all six' unchanged


def test_branch_labels_are_exact_counterfactuals():
    inner = dict(kp=2., kg=1.)
    chain = [constant(R90)]
    nom = candidate_job(('L2', 1004, inner, chain))
    res = branch_job(('L2', 1004, inner, chain, [96.], 0, nom['schedule']))
    assert 'error' not in res and len(res['rows']) == 2
    for r in res['rows']:
        # replaying the alternative directly gives the same end return
        env = fly(F.FastFlight(F.scenario('L2', 1004), **inner), chain, override=(96., r['action']),
                  schedule={int(round(t)): a for t, a in nom['schedule']})
        assert env.G == pytest.approx(r['G_end'], abs=1e-12)
