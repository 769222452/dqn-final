"""The bounded checker is exercised with jobs replaced; no physical flight."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from continuous_recovery import training as original_training
from rollout_improvement.policy import constant_stage
from supported_recovery import rollouts


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1]/'scripts'/'check_supported_recovery.py'
    spec = importlib.util.spec_from_file_location('check_supported_recovery_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounded_pair(args):
    arms = {}
    for name, action in (('base', args['base_action']), ('candidate', args['action'])):
        trace = [dict(t=float(t), requested_action=action, executed_action=action, committed=True)
                 for t in range(48, 60, 3)]
        arms[name] = dict(initial_snapshot_sha256=args['binding']['state_sha256'], steps=4, time_s=60.,
            first_12s_requested_actions=[action]*4, first_12s_actual_executed_actions=[action]*4,
            G_rest=.1, G_sum=.1, execution_trace=trace)
    return dict(arms, valid=False, advantage=None, completed=False, support_label=dict(valid=False),
        baseline_recomputed=True, G_base_rest=.1, G_candidate_rest=.1,
        action_semantics='certified_policy_request', certificates_reused=False)


def test_checker_uses_two_sources_and_two_bounded_same_reference_pairs(tmp_path, monkeypatch, checker):
    flights, pairs = [], []
    reference = [constant_stage(.04)]

    def prepare(out, model, **kwargs):
        out.mkdir()
        return {'source': 'frozen'}, reference, {'sha256': 'binding'}

    def candidate(args):
        flights.append(deepcopy(args))
        name = 'reference' if args['chain'] == reference else 'off_nominal'
        prefix = {str(t): dict(t=float(t), action=8, features=[0., 0., 0., .5, t/1200, 0.],
                               state_sha256=name+'_hash', G_prefix=.05)
                  for t in (0, 48)}
        return dict(time_s=60., steps=20, stop='MAX_DECISIONS', decision_prefix=prefix,
                    completed_1200s=False)

    def pair(args):
        pairs.append(deepcopy(args))
        return bounded_pair(args)

    monkeypatch.setattr(original_training, 'prepare_runtime', prepare)
    monkeypatch.setattr(rollouts, 'candidate_job', candidate)
    monkeypatch.setattr(rollouts, 'paired_job', pair)
    out = tmp_path/'new_check'
    result = checker.run_check(out, tmp_path/'model.npz', tmp_path/'policy.json')
    assert result['passed'], result['errors']
    assert len(flights) == len(pairs) == 2
    assert all(flight['max_decisions'] == 20 for flight in flights)
    assert all(pair['max_decisions'] == 4 and pair['continuation'] == 'closed-loop' for pair in pairs)
    assert all(pair['reference_chain'] == reference for pair in pairs)
    assert all(pair['G_base_rest'] == checker.CACHE_SENTINEL for pair in pairs)
    assert all(pair['binding']['G_prefix'] == -checker.CACHE_SENTINEL for pair in pairs)
    assert pairs[0]['binding']['state_sha256'] != pairs[1]['binding']['state_sha256']
    assert flights[1]['chain'] == reference+[constant_stage(.03)]
    assert result['network_updates'] == result['valid_training_labels'] == 0
    assert not result['full_mission_completed'] and not result['training_performed']
    saved = json.loads((out/'validation.json').read_text(encoding='utf-8'))
    assert saved['passed'] and saved['protocol'] == checker.PROTOCOL


def test_checker_preserves_failed_source_evidence(tmp_path, monkeypatch, checker):
    def prepare(out, model, **kwargs):
        out.mkdir()
        return {}, [constant_stage(.04)], {'sha256': 'binding'}

    monkeypatch.setattr(original_training, 'prepare_runtime', prepare)
    monkeypatch.setattr(rollouts, 'candidate_job', lambda args: dict(time_s=3., steps=1, stop='UNKNOWN'))
    monkeypatch.setattr(rollouts, 'paired_job', lambda args: pytest.fail('No pair may run after failed source'))
    result = checker.run_check(tmp_path/'new_check', tmp_path/'model.npz', tmp_path/'policy.json')
    assert not result['passed'] and 'Reference source' in result['errors'][0]
    assert (tmp_path/'new_check'/'validation.json').is_file()


def test_checker_refuses_existing_directory_before_preparation(tmp_path, monkeypatch, checker):
    monkeypatch.setattr(original_training, 'prepare_runtime', lambda *a, **kw: pytest.fail('Existing path'))
    with pytest.raises(ValueError, match='new directory'):
        checker.run_check(tmp_path, tmp_path/'model.npz', tmp_path/'policy.json')


@pytest.mark.parametrize('mutation,key', [
    ('origin', 'identical_restored_origins'), ('short', 'both_four_steps'),
    ('request', 'candidate_request_held_12s'), ('label', 'no_valid_training_label'),
    ('cache', 'cached_returns_not_used'), ('commit', 'candidate_committed_trace'),
])
def test_pair_checks_reject_misleading_success(checker, mutation, key):
    result = bounded_pair(dict(binding={'state_sha256': 'origin'}, base_action=8, action=9))
    if mutation == 'origin':
        result['candidate']['initial_snapshot_sha256'] = 'other'
    elif mutation == 'short':
        result['candidate']['steps'] = 3
    elif mutation == 'request':
        result['candidate']['first_12s_requested_actions'] = [8]*4
    elif mutation == 'label':
        result['valid'] = True
    elif mutation == 'cache':
        result['base']['G_rest'] = checker.CACHE_SENTINEL
        result['G_base_rest'] = checker.CACHE_SENTINEL
    else:
        result['candidate']['execution_trace'][0]['committed'] = False
    checks = checker._pair_checks(result, 'origin', 8, 9)
    assert checks[key] is False
