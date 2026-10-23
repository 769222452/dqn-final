import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rollout_improvement import evaluation

_spec = importlib.util.spec_from_file_location('compare', Path(__file__).parents[1]/'scripts'/'compare.py')
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def _summary(ranges, ref=100., test_set='held-out'):
    rows = [dict(dh=0., dV=0., range_km=ref+1, gain_km=1., stop='DONE')]
    rows += [dict(dh=float(i+1), dV=0., range_km=r, gain_km=r-ref, stop='DONE') for i, r in enumerate(ranges)]
    return dict(test_set=test_set, reference_cl=.025, initial_conditions=rows, nominal_range_km=ref+1,
                nominal_gain_km=1., probe_mean_A_closed=-1., probe_mean_A_open=-2., probe_closed_not_worse=1,
                probes=[{}])


def test_compare_pairs_policies_on_the_same_states():
    r = compare.compare({'a': _summary([101., 103., 99.]), 'b': _summary([100., 100., 100.])})
    assert r['states'] == 3
    assert r['policies']['a']['gain_over_reference_km']['mean'] == pytest.approx(1.)
    assert r['policies']['a']['gain_over_reference_km']['wins'] == 2
    d = r['pairs']['a - b']
    assert d['mean'] == pytest.approx(1.) and d['worst'] == -1. and d['wins'] == 2


def test_compare_rejects_mismatched_tests():
    with pytest.raises(ValueError):
        compare.compare({'a': _summary([1., 2.]), 'b': _summary([1., 2.], test_set='development')})
    with pytest.raises(ValueError):
        compare.compare({'a': _summary([1., 2.]), 'b': _summary([1., 2., 3.])})


def test_held_out_set_is_fixed_and_disjoint():
    offsets = evaluation.held_out_offsets()
    times = evaluation.held_out_probe_times()
    assert offsets == evaluation.held_out_offsets() and times == evaluation.held_out_probe_times()
    assert len(offsets) == evaluation.HELD_OUT_STATES == len(set(offsets))
    assert not set(offsets) & set(evaluation.INITIAL_OFFSETS) and (0., 0.) not in offsets
    assert len(times) == len(evaluation.PROBE_TIMES) and not set(times) & set(evaluation.PROBE_TIMES)
    assert all(t % 12 == 0 and 12 <= t <= 1176 for t in times)
    a = np.array(offsets)
    assert np.all(np.abs(a) <= evaluation.MAX_OFFSET)


def test_held_out_set_values():
    assert evaluation.held_out_offsets()[:3] == [(-738., -63.1), (386., -66.7), (1800., -74.)]
    assert evaluation.held_out_probe_times() == (72., 300., 336., 540., 720., 960., 1164.)
    offsets, times = evaluation.test_set_points('held-out-1')
    assert offsets == [(0., 0.)]+evaluation.held_out_offsets() and times == evaluation.held_out_probe_times()


def test_test_sets_are_disjoint():
    sets = {name: evaluation.test_set_points(name) for name in evaluation.TEST_SETS}
    assert evaluation.TEST_SETS[0] == 'held-out-2'
    for name, (offsets, times) in sets.items():
        perturbed = [d for d in offsets if d != (0., 0.)]
        assert len(perturbed) == len(set(perturbed)) and len(times) == len(set(times)) == 7
        assert np.all(np.abs(np.array(perturbed)) <= evaluation.MAX_OFFSET)
        assert all(t % 12 == 0 and 12 <= t <= 1176 for t in times)
        for other, (o2, t2) in sets.items():
            if other != name:
                assert not (set(perturbed) & set(o2)-{(0., 0.)}) and not set(times) & set(t2)
    assert len(sets['held-out-2'][0]) == 41
    with pytest.raises(ValueError):
        evaluation.test_set_points('held-out')


def test_compare_reports_states_where_the_reference_completed():
    a = _summary([101., 103., 99.])
    a['initial_conditions'][2]['reference_stop'] = 'NEXT_STATE_NOT_CERTIFIED'
    r = compare.compare({'a': a})['policies']['a']
    assert r['reference_completed'] == 2 and r['gain_where_reference_completed_km']['n'] == 2
    assert r['gain_where_reference_completed_km']['mean'] == pytest.approx(0.)


def test_paired_bootstrap_ci():
    lo, hi = evaluation.paired_bootstrap_ci([1., 2., 3., 4.])
    assert 1. <= lo < 2.5 < hi <= 4.
    assert evaluation.paired_bootstrap_ci([2.]*5) == [2., 2.]


def test_folder_names_of_the_development_set():
    assert evaluation._name((-500., -20.)) == 'dh-500_dv-20'
    assert evaluation._name((0., 0.)) == 'dh0_dv0'
    assert evaluation._name((-738., -63.1)) == 'dh-738_dv-63.1'
