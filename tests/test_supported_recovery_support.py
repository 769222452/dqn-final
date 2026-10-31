"""Exact support is controller-, continuation-, state-, and action-specific."""
from copy import deepcopy

import numpy as np
import pytest

from supported_recovery.support import (build_support_label, canonical_hash, context_hash,
    continuation_identity, prefix_chain_hash, support_context, support_decision)


def snapshot():
    return dict(protocol='continuous-recovery-training-flight/1', identity='controller-runtime',
        state=dict(x=[40000., 2400., -.01, 3000.], alpha=.05, t=12.,
            transition=dict(source=1, target=2, started=9., duration=8.),
            history_valid=True, previous_cl_raw=.04, previous_cl_cmd=.039,
            previous_cl_actual=.038, reward=.3, switches=2),
        controller_memory=dict(protocol='continuous-recovery/1', identity='controller-runtime',
            memory={'controls': [0.05, 0.06], 'ends_at_s': 60.}, epoch=2, exit_passes=1),
        certification_conditions=dict(guard={'budget': 2., 'horizon': 30.},
                                      dead_end_retry={'enabled': True}))


def support_stage(state=None, *, candidate=9, base=8, label_kind='branch'):
    state = snapshot() if state is None else state
    cont = continuation_identity([])
    row = build_support_label(state, reference_prefix_sha256=prefix_chain_hash([]),
        continuation_sha256=cont['sha256'], base_action=base, candidate_action=candidate,
        label_kind=label_kind)
    return dict(kind='supported_model', support_mode='exact', action_semantics='certified_policy_request',
        reference_prefix_sha256=prefix_chain_hash([]),
        runtime_identity=state['identity'], continuation=cont, continuation_sha256=cont['sha256'],
        support_records=[row])


def check(stage, state=None, candidate=9, base=8):
    return support_decision(stage, support_context(snapshot() if state is None else state),
                            reference_prefix_sha256=prefix_chain_hash([]),
                            base_action=base, candidate_action=candidate)


@pytest.mark.parametrize('change', [
    lambda s: s['state'].__setitem__('alpha', .051),
    lambda s: s['state']['x'].__setitem__(3, 3001.),
    lambda s: s['state']['transition'].__setitem__('target', 3),
    lambda s: s['state']['transition'].__setitem__('started', 8.),
    lambda s: s['state'].__setitem__('previous_cl_cmd', .04),
    lambda s: s['state'].__setitem__('previous_cl_actual', .04),
    lambda s: s['controller_memory']['memory']['controls'].__setitem__(1, .061),
    lambda s: s['controller_memory'].__setitem__('exit_passes', 0),
    lambda s: s['controller_memory'].__setitem__('epoch', 3),
    lambda s: s['certification_conditions']['guard'].__setitem__('budget', 3.),
])
def test_same_time_different_future_control_context_is_not_supported(change):
    original = snapshot()
    other = deepcopy(original)
    change(other)
    assert original['state']['t'] == other['state']['t']
    assert check(support_stage(original), original)['supported']
    answer = check(support_stage(original), other)
    assert not answer['supported']
    assert 'EXACT_CONTEXT_MISMATCH' in answer['reasons']


def test_context_normalizes_settled_geometry_but_retains_active_fraction():
    a, b = snapshot(), snapshot()
    a['state']['transition'] = dict(source=1, target=2, started=0., duration=8.)
    b['state']['transition'] = dict(source=2, target=2, started=11., duration=8.)
    assert context_hash(a) == context_hash(b)
    assert support_context(a)['state']['transition'] == dict(source=2, target=2, fraction=1., rate=0., duration_s=8.)


def test_context_ignores_only_audited_accumulated_totals_and_does_not_mutate():
    a = snapshot()
    before = deepcopy(a)
    support_context(a)
    assert a == before
    b = deepcopy(a)
    b['state'].update(reward=99., switches=200)
    assert context_hash(a) == context_hash(b)
    b['state']['new_controller_variable'] = 1
    with pytest.raises(ValueError, match='Unknown or missing'):
        support_context(b)


@pytest.mark.parametrize('field,value,reason', [
    ('base_action', 7, 'LABEL_BASE_ACTION_MISMATCH'),
    ('reference_prefix_sha256', 'older-policy', 'LABEL_REFERENCE_PREFIX_MISMATCH'),
    ('runtime_identity', 'older-controller', 'LABEL_RUNTIME_IDENTITY_MISMATCH'),
    ('continuation_sha256', 'other-tail', 'LABEL_CONTINUATION_MISMATCH'),
    ('valid', False, 'LABEL_NOT_VALID'),
    ('action_semantics', 'executed_action', 'LABEL_ACTION_SEMANTICS_MISMATCH'),
])
def test_label_group_mismatch_rejects(field, value, reason):
    stage = support_stage()
    stage['support_records'][0][field] = value
    answer = check(stage)
    assert not answer['supported'] and reason in answer['reasons']


def test_absent_action_and_baseline_zero_do_not_borrow_support():
    stage = support_stage(candidate=8, label_kind='baseline_zero')
    assert check(stage, candidate=8)['supported']
    assert check(stage, candidate=9)['reason'] == 'NO_CANDIDATE_ACTION_LABEL'
    stage['support_records'][0]['candidate_action'] = 9
    assert 'BASELINE_ZERO_OTHER_ACTION' in check(stage)['reasons']
    with pytest.raises(ValueError, match='another action'):
        support_stage(candidate=9, label_kind='baseline_zero')


def test_prefix_is_predecessor_only_and_continuation_metadata_is_validated():
    stage = support_stage()
    stage['reference_prefix_sha256'] = prefix_chain_hash([stage])
    assert check(stage)['reason'] == 'REFERENCE_PREFIX_MISMATCH'
    stage = support_stage()
    stage['continuation']['mode'] = 'open-loop'
    assert 'CONTINUATION_METADATA_INVALID' in check(stage)['reasons']
    stage = support_stage()
    stage['continuation_sha256'] = 'tampered'
    assert 'CONTINUATION_HASH_MISMATCH' in check(stage)['reasons']


def test_local_mode_rejected_even_with_an_arbitrary_radius():
    stage = support_stage()
    stage.update(support_mode='local', radius=100., calibration={'passed': True})
    with pytest.raises(ValueError, match='no validated calibration'):
        check(stage)


def test_controller_identity_and_nonfinite_values_fail_closed():
    s = snapshot()
    s['controller_memory']['identity'] = 'different'
    with pytest.raises(ValueError, match='identities must agree'):
        support_context(s)
    s = snapshot()
    s['state']['alpha'] = np.nan
    with pytest.raises(ValueError):
        support_context(s)


def test_continuation_normalization_and_finite_hash_compatibility():
    from continuous_surrogate.experiment import digest
    a = continuation_identity([], 'open-loop', {24: 9, 12: 8})
    b = continuation_identity([], 'open-loop', [[12., 8], [24., 9]])
    assert a == b and a['sha256'] == digest({k: v for k, v in a.items() if k != 'sha256'})
    assert canonical_hash({'x': np.array([1., 2.])}) == digest({'x': [1., 2.]})
    with pytest.raises(ValueError, match='actual replay'):
        continuation_identity([], 'open-loop')
    with pytest.raises(ValueError, match='cannot carry'):
        continuation_identity([], 'closed-loop', {12: 8})


def test_negative_measured_advantage_is_supported_and_reported_without_averaging():
    stage = support_stage()
    stage['support_records'][0]['advantage'] = -.01
    answer = check(stage)
    assert answer['supported'] and answer['measured_advantage'] == -.01
    stage['support_records'].append(dict(stage['support_records'][0], advantage=.02))
    answer = check(stage)
    assert answer['measured_advantage_conflict']
    assert answer['measured_advantages'] == [-.01, .02]
    assert 'measured_advantage' not in answer
