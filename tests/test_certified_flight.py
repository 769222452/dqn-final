"""Integration tests of the certified flight (about one minute; the first run also compiles the plant)."""
import hashlib

import numpy as np
import pytest

from certified_flight import CL_ACTIONS, build_flight, integrity
from certified_flight.build import ROOT
from rollout_improvement.policy import ChainPolicy
from rollout_improvement.rollouts import certified_fly

# Constant CL 0.04 for the first 240 s of the mission.
G_240 = 0.23273813720731978
X_240 = [28339.074031398977, 2254.5447558374167, -0.0020314152458305616, 558571.5292975676]  # h, V, gamma, s


def state_hash(env):
    return hashlib.sha256(np.asarray(env.x, np.float64).tobytes()).hexdigest()[:16]


def test_input_integrity():
    assert integrity.verify(ROOT) > 300


def test_action_set():
    assert len(CL_ACTIONS) == 19 and CL_ACTIONS[8] == .04 and CL_ACTIONS == sorted(CL_ACTIONS)


@pytest.fixture(scope='module')
def flight_240(tmp_path_factory):
    env, certifier = build_flight(tmp_path_factory.mktemp('flight'), 'test')
    G1, stop1, _ = certified_fly(certifier, ChainPolicy([], env), until=120.)
    branch = certifier.fork(tmp_path_factory.mktemp('branch'))
    G2, stop2, acts = certified_fly(certifier, ChainPolicy([], env), until=240.)
    G3, stop3, _ = certified_fly(branch, ChainPolicy([], branch.env), until=240.)
    return dict(env=env, branch=branch, G_trunk=G1+G2, G_branch=G1+G3, stops=(stop1, stop2, stop3), acts=acts)


def test_constant_policy_reproduces_reference(flight_240):
    assert flight_240['stops'] == (None, None, None)
    assert flight_240['G_trunk'] == pytest.approx(G_240, rel=0, abs=1e-12)
    assert flight_240['env'].t == 240. and set(flight_240['acts']) == {8}


def test_fork_is_independent_and_exact(flight_240):
    assert flight_240['G_branch'] == flight_240['G_trunk']
    assert state_hash(flight_240['branch'].env) == state_hash(flight_240['env'])
    assert flight_240['branch'].env is not flight_240['env']


def test_reference_state(flight_240):
    """State at 240 s. Other CPUs may differ in the last digits, so each component is compared relative to its
    typical magnitude (h, V, gamma, s); the flight-path angle is close to zero at 240 s."""
    scale = np.array([1e4, 1e3, 1e-1, 1e5])
    err = np.abs(np.asarray(flight_240['env'].x)-X_240)/scale
    assert np.all(err < 1e-11), err
