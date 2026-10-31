"""One supported-policy iteration: all flights and neural fitting are mocked."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from supported_recovery import training
from supported_recovery.support import (canonical_hash, context_hash, continuation_identity,
                                       prefix_chain_hash)
from test_supported_recovery_support import snapshot


def binding(t=12., alpha=.05, base=8):
    state = snapshot()
    state['state'].update(t=float(t), alpha=alpha,
                         transition=dict(source=2, target=2, started=0., duration=8.))
    physical = state['state']
    x = physical['x']
    features = [(x[0]-40000)/10000, (x[1]-2400)/800, x[2]/.1,
                alpha/.1, float(t)/1200, 0.]
    return dict(t=float(t), action=base, G_prefix=.2, features=features,
                state=state, state_sha256=canonical_hash(state))


def pair(*, reference=None, continuation=None, at=None, action=9, advantage=.01, valid=True):
    reference = [] if reference is None else reference
    continuation = continuation_identity(reference) if continuation is None else continuation
    at = binding() if at is None else deepcopy(at)
    return dict(valid=valid, completed=valid, binding=at, base_action=at['action'], action=action,
        advantage=advantage if valid else None,
        reference_prefix_sha256=prefix_chain_hash(reference), continuation_sha256=continuation['sha256'],
        action_semantics='certified_policy_request', query_origin='nominal',
        evidence_path='mock-pair-evidence',
        base=dict(completed=valid, time_s=1200. if valid else 24., G_rest=.4),
        candidate=dict(completed=valid, time_s=1200. if valid else 24., G_rest=.4+advantage),
        failure_semantics=None if valid else 'UNKNOWN_NOT_CERTIFIED')


def rows(*, at=None, action=9, advantage=.01):
    return training.labels_from_pairs([pair(at=at, action=action, advantage=advantage)],
                                     [], continuation_identity([]))


def candidate(*, completed=True, reward=1., logs=None, prefixes=None):
    prefixes = {str(t): binding(t) for t in range(0, 1200, 12)} if prefixes is None else prefixes
    return dict(G=reward, range_km=reward*1000., success=completed, completed=completed,
        completed_1200s=completed, time_s=1200. if completed else 24.,
        stop='DONE' if completed else 'UNKNOWN_NOT_CERTIFIED', support_logs=[] if logs is None else logs,
        decision_prefix=prefixes, schedule=[[float(t), 8] for t in range(0, 1200, 12)],
        failure_semantics=None if completed else 'UNKNOWN_NOT_CERTIFIED', wall_s=.01)


def proposal(t, scores, *, supported=(), base=8, reference=None, state=None):
    reference = [] if reference is None else reference
    at = binding(t) if state is None else deepcopy(state)
    candidates = [dict(action=a, supported=a in supported, score=score,
                       reason='EXACT_SUPPORT' if a in supported else 'NO_CANDIDATE_ACTION_LABEL')
                  for a, score in scores.items()]
    stage = dict(stage_index=0, reference_prefix_sha256=prefix_chain_hash(reference),
                 base_action=base, proposed_action=max(scores, key=scores.get),
                 selected_action=base, candidates=candidates)
    return dict(t=float(t), stages=[stage]), at


def test_budget_charges_launches_and_rejects_overrun_without_mutation():
    budget = training.Budget(3, 2)
    budget.charge('baseline')
    budget.charge('pair', 2)
    budget.charge('candidate', 2)
    before = budget.state()
    assert before['paired_continuations_launched'] == 4
    assert before['baseline_and_candidates_separate']
    for kind, amount in [('baseline', 1), ('pair', 2), ('candidate', 1), ('pair', -1)]:
        with pytest.raises(ValueError, match='budget'):
            budget.charge(kind, amount)
        assert budget.state() == before
    budget.charge('pair')
    assert budget.state()['launched'] == dict(baseline=1, pair=3, candidate=2)


@pytest.mark.parametrize('count', [2, 5, 20, 25])
def test_nominal_queries_cover_whole_48_second_sweep(count):
    nominal = candidate()
    selected = training.nominal_queries(nominal, count)
    ticks = [q['binding']['t'] for q in selected]
    assert len(ticks) == len(set(ticks)) == count
    assert ticks[0] == 0. and ticks[-1] == 1152.
    assert all(t % 48. == 0 for t in ticks)
    assert all(0 < abs(q['action']-q['base_action']) <= 2 for q in selected)
    selected[0]['binding']['state']['state']['alpha'] = 123.
    assert nominal['decision_prefix']['0']['state']['state']['alpha'] == .05


@pytest.mark.parametrize('count', [0, 26])
def test_invalid_nominal_sample_count_rejected(count):
    with pytest.raises(ValueError, match='1 to 25'):
        training.nominal_queries(candidate(), count)


def test_pair_labels_keep_request_action_and_base_zero_with_fresh_contract():
    source = pair(advantage=-.01)
    source['candidate']['executed'] = [8, 8, 8, 8]  # The measured action is still request 9.
    produced = training.labels_from_pairs([source], [], continuation_identity([]))
    assert [(r['action'], r['advantage']) for r in produced] == [(8, 0.), (9, -.01)]
    assert all(r['reference_version'] == 0 for r in produced)
    assert [r['support']['label_kind'] for r in produced] == ['baseline_zero', 'branch']
    for row in produced:
        assert row['support']['base_action'] == 8
        assert row['support']['candidate_action'] == row['action']
        assert row['support']['advantage'] == row['advantage']
        assert row['support']['context_sha256'] == context_hash(source['binding']['state'])
        assert row['support']['action_semantics'] == 'certified_policy_request'


@pytest.mark.parametrize('field,value', [
    ('reference_prefix_sha256', 'old-version'),
    ('continuation_sha256', 'different-tail'),
    ('action_semantics', 'executed_action'),
])
def test_pair_wrong_version_continuation_or_semantics_cannot_be_imported(field, value):
    result = pair()
    result[field] = value
    with pytest.raises(ValueError, match='another reference/continuation/semantics'):
        training.labels_from_pairs([result], [], continuation_identity([]))


def test_unknown_pair_is_not_negative_label_and_infrastructure_error_is_not_unknown():
    unknown = pair(valid=False)
    assert training.labels_from_pairs([unknown], [], continuation_identity([])) == []
    error = dict(unknown, error='worker failed before audited outcome')
    with pytest.raises(RuntimeError, match='infrastructure error'):
        training.labels_from_pairs([error], [], continuation_identity([]))


def test_nonfinite_valid_advantage_is_rejected():
    result = pair()
    result['advantage'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        training.labels_from_pairs([result], [], continuation_identity([]))


def test_exact_repeat_deduplicates_but_conflicting_return_is_not_averaged():
    first = rows()[1]
    assert len(training.fitting_rows([first, deepcopy(first)], [0, 1, 2, 4])) == 1
    conflict = dict(deepcopy(first), advantage=.02)
    with pytest.raises(ValueError, match='conflicting measured returns'):
        training.fitting_rows([first, conflict], [0, 1, 2, 4])


def test_hidden_alpha_alias_conflict_rejected_before_old_ensemble_can_average():
    first = rows(at=binding(alpha=.05), advantage=.01)[1]
    hidden = rows(at=binding(alpha=.051), advantage=.02)[1]
    assert first['support']['context_sha256'] != hidden['support']['context_sha256']
    with pytest.raises(ValueError, match='Network-input alias'):
        training.fitting_rows([first, hidden], [0, 1, 2, 4])
    assert len(training.fitting_rows([first, hidden], None)) == 2


def test_same_network_action_different_base_is_not_averaged_even_if_target_equal():
    first = rows()[1]
    different = deepcopy(first)
    different['base_action'] = 7
    with pytest.raises(ValueError, match='Network-input alias'):
        training.fitting_rows([first, different], [0, 1, 2, 4])


def test_no_labels_skips_fitting_and_stage_creation(monkeypatch):
    fitter = Mock(side_effect=AssertionError('No labels must not call fitter'))
    monkeypatch.setattr(training, 'fit_ensemble', fitter)
    chains, result = training.make_stages([], [], continuation_identity([]), 1, [0, 1, 2, 4], [.25])
    assert chains == [] and result['fitted'] is False
    fitter.assert_not_called()


def test_mixed_controller_identity_rejected_before_fitting(monkeypatch):
    labels = rows()
    labels[-1]['support']['runtime_identity'] = 'another-controller'
    fitter = Mock(return_value=([], {}))
    monkeypatch.setattr(training, 'fit_ensemble', fitter)
    with pytest.raises(ValueError, match='[Mm]ixed.controller'):
        training.make_stages([], labels, continuation_identity([]), 1, [0, 1, 2, 4], [.25])
    fitter.assert_not_called()


@pytest.mark.parametrize('field,value', [
    ('reference_prefix_sha256', 'other-reference'),
    ('continuation_sha256', 'other-continuation'),
    ('action_semantics', 'executed_action'),
])
def test_make_stages_rechecks_label_contract_before_fitting(monkeypatch, field, value):
    labels = rows()
    labels[-1]['support'][field] = value
    fitter = Mock(return_value=([], {}))
    monkeypatch.setattr(training, 'fit_ensemble', fitter)
    with pytest.raises(ValueError):
        training.make_stages([], labels, continuation_identity([]), 1, [0, 1, 2, 4], [.25])
    fitter.assert_not_called()


def test_active_queries_exclude_measured_supported_and_base_then_deduplicate_states():
    first, a = proposal(12., {8: 100., 9: 90., 10: 80., 7: 70.}, supported=(7,))
    second, b = proposal(24., {8: 100., 9: 60., 10: 50.})
    cand = candidate(logs=[first, second], prefixes={'12': a, '24': b})
    measured = [pair(at=a, action=9, valid=False)]  # An attempted UNKNOWN also consumes this query.
    queries = training.active_queries([cand, deepcopy(cand)], [], measured, 9)
    assert [(q['binding']['t'], q['action']) for q in queries] == [(12., 10), (24., 9)]
    assert all(q['origin'] == 'active_visited_state' and q['prediction_only'] for q in queries)
    assert len({context_hash(q['binding']['state']) for q in queries}) == len(queries)
    assert training.active_queries([cand], [], measured, 0) == []
    assert len(training.active_queries([cand], [], measured, 1)) == 1


def test_active_queries_require_real_binding_and_matching_reference():
    entry, at = proposal(12., {8: 0., 9: 2.})
    with pytest.raises(ValueError, match='real visited-state'):
        training.active_queries([candidate(logs=[entry], prefixes={})], [], [], 1)
    entry['stages'][0]['reference_prefix_sha256'] = 'other-reference'
    assert training.active_queries([candidate(logs=[entry], prefixes={'12': at})], [], [], 1) == []


@pytest.fixture
def mock_run(tmp_path, monkeypatch):
    calls, fits = [], []
    options = dict(baseline=candidate(), candidates=[candidate(reward=.99), candidate(reward=.98)],
                   pair_valid=True, failure_kind=None, free=10_000_000_000)
    reference = []
    out = tmp_path/'run'

    def prepare(path, *_args):
        Path(path).mkdir(parents=True, exist_ok=False)
        return {'mock_runtime': True}, deepcopy(reference), {'mock_experiment': True}

    def fit(labels, seed, weights, inputs, hidden):
        fits.append(deepcopy(labels))
        return [[(np.zeros((len(inputs), 19)).tolist(), np.zeros(19).tolist())]], {'mock_fit': True}

    def dispatch(batch, workers):
        saved = json.loads((out/'state.json').read_text())
        phase = saved['phase']
        launched = saved['budget']['launched']
        assert phase.startswith('RUNNING_')
        assert launched[phase[len('RUNNING_'):].lower()] >= len(batch)
        if options.get('launch_observer'):
            options['launch_observer'](batch, saved)
        results = []
        for kind, args in batch:
            calls.append((kind, deepcopy(args)))
            if options['failure_kind'] == kind:
                results.append({'error': 'mock infrastructure failure'})
            elif kind == 'pair':
                cont = continuation_identity(args['reference_chain'], args['continuation'], args['schedule'])
                result = pair(reference=args['reference_chain'], continuation=cont,
                              at=args['binding'], action=args['action'], valid=options['pair_valid'])
                result['query_origin'] = args['query_origin']
                results.append(result)
            elif Path(args['folder']).name == 'baseline':
                results.append(deepcopy(options['baseline']))
            else:
                result = options['candidates'].pop(0)
                results.append(deepcopy(result))
        return results

    monkeypatch.setattr(training, 'prepare_runtime', prepare)
    monkeypatch.setattr(training, 'fit_ensemble', fit)
    monkeypatch.setattr(training, 'jobs', dispatch)
    monkeypatch.setattr(training.shutil, 'disk_usage', lambda _: SimpleNamespace(free=options['free']))
    kwargs = dict(out=out, model=tmp_path/'unused-model.npz', pairs=2, active_pairs=0,
                  candidates=2, workers=1, log=lambda _: None)
    return SimpleNamespace(kwargs=kwargs, options=options, calls=calls, fits=fits,
                           out=out, reference=reference)


def test_incomplete_baseline_stops_before_pairs_fits_or_candidates(mock_run):
    mock_run.options['baseline'] = candidate(completed=False, reward=100.)
    state = training.train(**mock_run.kwargs)
    assert state['phase'] == 'BASELINE_NOT_COMPLETE' and state['iteration'] == 0
    assert state['chain'] == mock_run.reference
    assert len(mock_run.calls) == 1 and mock_run.fits == []
    assert state['budget']['launched'] == dict(pair=0, candidate=0, baseline=1)
    assert state['labels'] == [] and state['pairs'] == []


def test_incomplete_candidate_even_with_huge_reward_cannot_be_accepted(mock_run):
    mock_run.options['candidates'] = [candidate(completed=False, reward=100.), candidate(reward=.9)]
    state = training.train(**mock_run.kwargs)
    assert state['phase'] == 'COMPLETE' and not state['accepted']
    assert state['iteration'] == 1 and state['policy_version'] == 0
    assert state['chain'] == mock_run.reference and state['final_result']['G'] == 1.
    assert state['budget']['launched'] == dict(pair=2, candidate=2, baseline=1)
    assert len(mock_run.calls) == 5
    assert state['convergence_proven'] is False


def test_better_complete_candidate_is_accepted_after_full_mock_iteration(mock_run):
    mock_run.options['candidates'] = [candidate(reward=1.02), candidate(reward=1.01)]
    state = training.train(**mock_run.kwargs)
    assert state['accepted'] and state['policy_version'] == state['iteration'] == 1
    assert state['chain'][:-1] == mock_run.reference
    assert state['chain'][-1]['kind'] == 'supported_model'
    assert state['chain'][-1]['action_semantics'] == 'certified_policy_request'
    assert state['final_result']['G'] == 1.02
    assert len(state['labels']) == 4 and state['label_valid_pairs'] == 2
    assert len(mock_run.fits) == 1 and all(r['reference_version'] == 0 for r in mock_run.fits[0])
    summary = json.loads((mock_run.out/'summary.json').read_text())
    assert summary['reward_delta'] == pytest.approx(.02)
    assert summary['range_delta_km'] == pytest.approx(20.)
    assert summary['budget']['paired_continuations_launched'] == 4
    assert not summary['local_generalization_validated'] and not summary['convergence_proven']


def test_all_unknown_pairs_generate_no_targets_and_do_not_fit(mock_run):
    mock_run.options['pair_valid'] = False
    state = training.train(**mock_run.kwargs)
    assert not state['accepted'] and state['labels'] == [] and state['label_valid_pairs'] == 0
    assert mock_run.fits == []
    assert state['budget']['launched'] == dict(pair=2, candidate=0, baseline=1)
    assert state['support_limited']
    assert state['stopping_reason'] == 'NO_COMPLETE_PAIRED_LABELS'
    assert state['iteration'] == 0 and state['phase'] == 'NO_COMPLETE_PAIRED_LABELS'
    summary = json.loads((mock_run.out/'summary.json').read_text())
    assert summary['iterations'] == 0 and summary['phase'] == 'NO_COMPLETE_PAIRED_LABELS'
    assert summary['labels'] == 0 and not summary['accepted']


def test_active_visited_state_pair_refits_within_shared_pair_and_candidate_budgets(mock_run):
    log, visited = proposal(12., {8: 0., 9: 3., 10: 2.}, supported=(8,))
    first = candidate(reward=1.01, logs=[log], prefixes={'12': visited})
    mock_run.options['candidates'] = [first, candidate(reward=1.02)]
    state = training.train(**dict(mock_run.kwargs, active_pairs=1))
    assert state['accepted'] and state['final_result']['G'] == 1.02
    assert state['budget']['launched'] == dict(pair=2, candidate=2, baseline=1)
    assert len(mock_run.fits) == 2 and [len(batch) for batch in mock_run.fits] == [2, 4]
    pair_launches = [args for kind, args in mock_run.calls if kind == 'pair']
    assert [args['query_origin'] for args in pair_launches] == ['nominal', 'active_visited_state']
    assert pair_launches[1]['binding'] == visited and pair_launches[1]['action'] == 9
    assert state['support_limited']
    assert state['candidates'][0]['result']['support_statistics']['proposed_changes_blocked'] == 1


def test_no_active_queries_uses_reserved_candidate_for_independent_verification(mock_run):
    mock_run.options['candidates'] = [candidate(reward=.99), candidate(reward=.98)]
    state = training.train(**dict(mock_run.kwargs, active_pairs=1))
    assert not state['accepted'] and len(mock_run.fits) == 1
    assert state['budget']['launched'] == dict(pair=1, candidate=2, baseline=1)
    launched = [args for kind, args in mock_run.calls if kind == 'candidate']
    assert 'initial_candidates' in launched[1]['folder']
    assert 'final_verification' in launched[2]['folder']
    assert launched[1]['chain'] == launched[2]['chain']


@pytest.mark.parametrize('verification,accepted,final_reward', [
    (candidate(completed=False, reward=100.), False, 1.),
    (candidate(reward=.99), False, 1.),
    (candidate(reward=1.01), True, 1.01),
])
def test_latest_same_chain_verification_supersedes_initial_success(mock_run, verification, accepted, final_reward):
    mock_run.options['candidates'] = [candidate(reward=1.03), deepcopy(verification)]
    state = training.train(**dict(mock_run.kwargs, active_pairs=1))
    assert state['accepted'] is accepted
    assert state['final_result']['G'] == final_reward
    attempts = state['candidates']
    assert len(attempts) == 2
    assert canonical_hash(attempts[0]['chain']) == canonical_hash(attempts[1]['chain'])
    assert attempts[0]['result']['G'] == 1.03
    if not accepted:
        assert state['chain'] == mock_run.reference and state['policy_version'] == 0


def test_complete_candidate_weights_and_pending_pointer_exist_before_flight_launch(mock_run):
    observed = []

    def inspect(batch, saved):
        if saved['phase'] != 'RUNNING_CANDIDATE':
            return
        pending = saved['pending_candidates']
        assert len(pending) == len(batch)
        for item, (kind, args) in zip(pending, batch):
            assert kind == 'candidate'
            request_path = Path(item['request_file'])
            requests = json.loads(request_path.read_text())
            assert requests[item['index']] == json.loads(json.dumps(args))
            assert item['chain_sha256'] == canonical_hash(args['chain'])
            stage = requests[item['index']]['chain'][-1]
            assert stage['members'] and np.asarray(stage['members'][0][0][0]).shape == (4, 19)
            assert stage['support_records']
            assert not (Path(args['folder']).parent/'results.json').exists()
            observed.append(request_path)

    mock_run.options['launch_observer'] = inspect
    state = training.train(**mock_run.kwargs)
    assert len(observed) == 2
    assert all(path.is_file() for path in observed)
    assert state['pending_candidates'] == []


def test_candidate_infrastructure_failure_retains_prelaunch_weights_and_pending_pointer(mock_run):
    def fail_candidate_launch(batch, saved):
        if saved['phase'] == 'RUNNING_CANDIDATE':
            mock_run.options['failure_kind'] = 'candidate'

    mock_run.options['launch_observer'] = fail_candidate_launch
    with pytest.raises(RuntimeError, match='mock infrastructure failure'):
        training.train(**mock_run.kwargs)
    state = json.loads((mock_run.out/'state.json').read_text())
    assert state['phase'] == 'FAILED_INFRASTRUCTURE'
    assert len(state['pending_candidates']) == 2
    for item in state['pending_candidates']:
        request = json.loads(Path(item['request_file']).read_text())[item['index']]
        assert item['chain_sha256'] == canonical_hash(request['chain'])
        assert request['chain'][-1]['members'] and request['chain'][-1]['support_records']
    assert state['candidates'] == [] and state['policy_version'] == 0


def test_infrastructure_failure_still_charges_attempt_and_preserves_checkpoint(mock_run):
    mock_run.options['failure_kind'] = 'pair'
    with pytest.raises(RuntimeError, match='mock infrastructure failure'):
        training.train(**mock_run.kwargs)
    state = json.loads((mock_run.out/'state.json').read_text())
    assert state['phase'] == 'FAILED_INFRASTRUCTURE'
    assert state['budget']['launched'] == dict(pair=2, candidate=0, baseline=1)
    assert state['labels'] == [] and mock_run.fits == []
    assert (mock_run.out/'infrastructure_failure.json').exists()


def test_disk_reserve_blocks_before_any_launch_or_budget_charge(mock_run):
    mock_run.options['free'] = 1_499_999_999
    with pytest.raises(RuntimeError, match='Insufficient free disk'):
        training.train(**mock_run.kwargs)
    state = json.loads((mock_run.out/'state.json').read_text())
    assert state['phase'] == 'PAUSED_DISK_SPACE'
    assert state['budget']['launched'] == dict(pair=0, candidate=0, baseline=0)
    assert mock_run.calls == []


def test_disk_batch_estimate_reserves_both_paired_continuations_before_launch(mock_run):
    # Baseline needs 1 GB plus reserve; the t=0 and t=1152 pair batch
    # needs (2 + .08) GB plus reserve, so 3 GB must stop before the pairs.
    mock_run.options['free'] = 3_000_000_000
    with pytest.raises(RuntimeError, match='Insufficient free disk'):
        training.train(**mock_run.kwargs)
    state = json.loads((mock_run.out/'state.json').read_text())
    assert state['phase'] == 'PAUSED_DISK_SPACE'
    assert state['budget']['launched'] == dict(pair=0, candidate=0, baseline=1)
    assert len(mock_run.calls) == 1 and state['labels'] == []


def test_disk_pair_allowance_uses_remaining_horizon_and_candidate_weights_survive_pause(mock_run, monkeypatch):
    mock_run.options['free'] = 2_600_000_000
    # A late pair needs only 80 MB. Two full candidate flights need 2 GB.
    monkeypatch.setattr(training, 'nominal_queries', lambda _nominal, _count: [
        dict(binding=binding(1152.), base_action=8, action=9, origin='nominal')])
    with pytest.raises(RuntimeError, match='Insufficient free disk'):
        training.train(**dict(mock_run.kwargs, pairs=1))
    state = json.loads((mock_run.out/'state.json').read_text())
    assert state['phase'] == 'PAUSED_DISK_SPACE'
    assert state['budget']['launched'] == dict(pair=1, candidate=0, baseline=1)
    assert len(state['labels']) == 2 and len(state['pending_candidates']) == 2
    assert all(Path(item['request_file']).is_file() for item in state['pending_candidates'])


@pytest.mark.parametrize('change', [dict(pairs=0), dict(pairs=True), dict(active_pairs=2),
    dict(pairs=26, active_pairs=0), dict(candidates=1), dict(workers=0), dict(continuation_mode='ambiguous')])
def test_invalid_run_budgets_or_mode_do_not_prepare_runtime(mock_run, change):
    kwargs = dict(mock_run.kwargs, **change)
    with pytest.raises(ValueError):
        training.train(**kwargs)
    assert not mock_run.out.exists() and mock_run.calls == []
