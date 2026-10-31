"""The new CL screen must never use ZOH witnesses as continuous-control proof."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from certified_flight.certification import CL_ACTIONS
from certified_flight.reachability import CLReachability
from certified_flight.transition import Transition
from continuous_surrogate.reachability import ContinuousCLReachability, _WitnessPlant


@pytest.fixture
def physical(monkeypatch):
    # Isolate the constructive witness from the unchanged conservative box.
    monkeypatch.setattr(CLReachability, '_mach_box', lambda *args: (5., 5.))
    monkeypatch.setattr(CLReachability, '_coefficient_box',
                        lambda self, ml, mh, al, ah, ids:
                        (np.full(len(ids), -.1), np.full(len(ids), .2)))
    db = SimpleNamespace(A=np.array([-.1, .2]), M=np.array([3.5, 8.25]), ids=['G0', 'G1'])
    mission = SimpleNamespace(data=dict(plant_dt_s=.5, decision_dt_s=3., duration_s=1200.,
        limits=dict(alpha_rad=[-.1, .2], alpha_rate_radps=.1)))
    calls = []
    bad_grid = None

    def integrate(x, nodes, times, src, tgt, frac, rate, dt):
        calls.append(dict(nodes=np.asarray(nodes).copy(), times=np.asarray(times).copy(), dt=dt,
                          source=src, target=tgt, fraction=frac, rate=rate))
        starts = np.arange(round(times[-1]/dt))*dt
        evaluation_times = np.array([0., .5, .5, 1., 1.])*dt+starts[:, None]
        alpha = np.interp(evaluation_times.ravel(), times, nodes)
        samples = np.zeros((12, len(alpha)))
        samples[8] = alpha  # Linear exact CL law makes the witness integral checkable.
        return dict(valid=dt != plant.bad_grid, samples=samples, times=starts, dt=dt)

    plant = SimpleNamespace(db=db, mission=mission, atmosphere=object(), calls=calls,
        bad_grid=bad_grid, normalized_margins=lambda samples: np.ones((11, samples.shape[1])),
        base=SimpleNamespace(quantities=lambda *args: np.array([1., 1., 1., 5., 0., .01, 1., 1.])),
        integrate=integrate)
    return plant


def test_projection_uses_continuous_witness_mean_and_both_grids(physical):
    z = Transition(0, 1, started=0., duration=8.)
    before = deepcopy(vars(z))
    result = ContinuousCLReachability(physical).check(np.zeros(4), 0., z, 1., 2., enabled=True)
    assert result['unreachable_certified'] and result['projection_applied']
    # [0,.05,.1,.15,.2,.2,.2] has mean .1333, unlike the .15 ZOH endpoint mean.
    assert result['CL_cmd'] == pytest.approx(2/15, abs=1e-14)
    assert result['CL_cmd'] != pytest.approx(.15)
    assert len(physical.calls) == 4
    assert [v['dt'] for v in physical.calls] == [.1, .5, .1, .5]
    for call in physical.calls:
        assert len(call['nodes']) == len(call['times']) == 7
        assert call['nodes'][0] == 0.
        assert np.all(abs(np.diff(call['nodes'])/.5) <= .1+1e-12)
        assert (call['source'], call['target'], call['fraction'], call['rate']) == (0, 1, .125, .125)
    assert vars(z) == before


@pytest.mark.parametrize('bad_grid', [.1, .5])
def test_one_failed_physical_grid_prevents_projection(physical, bad_grid):
    physical.bad_grid = bad_grid
    result = ContinuousCLReachability(physical).check(np.zeros(4), 0., Transition(0, 1), 1., 2., enabled=True)
    assert result['unreachable_certified']
    assert not result['projection_applied'] and result['CL_cmd'] == 2.
    assert result['projection_reason'] == 'OUTSIDE_OUTER_ENVELOPE_EXTREME_WITNESS_UNSAFE'


def test_pending_generator_target_never_projects_from_fixed_pair(physical):
    result = ContinuousCLReachability(physical).check(np.zeros(4), 0., Transition(0, 0), 0., 2., enabled=True)
    assert result['projection_reason'] == 'GENERATOR_TARGET_PENDING_NO_PROJECTION'
    assert result['CL_cmd'] == 2. and not result['projection_applied']
    assert not physical.calls


def test_all_nineteen_absolute_cl_values_remain_exact(physical):
    reachability = ContinuousCLReachability(physical)
    for cl in CL_ACTIONS:
        result = reachability.check(np.zeros(4), 0., Transition(0, 1), 1., cl, enabled=True)
        assert result['CL_raw'] == result['CL_cmd'] == cl
        assert not result['projection_applied']


def test_projection_switch_preserves_old_disabled_behavior(physical):
    result = ContinuousCLReachability(physical).check(np.zeros(4), 0., Transition(0, 1), 1., 2., enabled=False)
    assert result['projection_reason'] == 'CERTIFIED_UNREACHABLE_PROJECTION_DISABLED'
    assert result['CL_cmd'] == 2. and not result['projection_applied']


@pytest.mark.parametrize('alpha,endpoints', [(0., np.full(6, .2)), (.3, np.full(6, .2))])
def test_physical_pass_cannot_override_alpha_rate_or_boundary_limits(physical, alpha, endpoints):
    wrapper = _WitnessPlant(physical, alpha)
    result = wrapper.integrate(np.zeros(4), endpoints, np.arange(7)*.5, 0, 1, .125, .125, .5)
    assert not result['valid']
