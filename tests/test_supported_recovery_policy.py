"""Support filtering changes only the opt-in improvement stage and scheduler."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from continuous_recovery.training_backend import snapshot_policy
from rollout_improvement.features import features
from rollout_improvement.policy import ChainPolicy, chain_action, constant_stage
from supported_recovery.policy import SupportPolicy, make_policy, policy_action, restore_policy
from supported_recovery.support import (build_support_label, canonical_hash,
                                       continuation_identity, prefix_chain_hash)
from test_supported_recovery_support import snapshot


def environment():
    data = snapshot()
    env = SimpleNamespace(x=np.asarray(data['state']['x']), t=12., alpha=.05,
                          history_valid=True, previous_cl_raw=.04)
    env.snapshot = lambda: deepcopy(dict(data, state=dict(data['state'], t=env.t,
                                x=env.x.tolist(), alpha=env.alpha)))
    return env


def model(scores, kind='model'):
    bias = np.zeros(19)
    for action, score in scores.items():
        bias[action] = score
    return dict(kind=kind, members=[[(np.zeros((4, 19)).tolist(), bias.tolist())]],
                inputs=[0, 1, 2, 4], margin=.25, times=[12.], window=48.)


def improved_chain(env, scores=None, supported=(9,), prefix=None):
    prefix = [] if prefix is None else deepcopy(prefix)
    cont = continuation_identity(prefix)
    base = policy_action(prefix, env)
    records = [build_support_label(env, reference_prefix_sha256=prefix_chain_hash(prefix),
        continuation_sha256=cont['sha256'], base_action=base, candidate_action=a,
        advantage=-.005 if a == 9 else .01) for a in supported]
    stage = model({9: 2., 10: 3.} if scores is None else scores, 'supported_model')
    stage.update(reference_prefix_sha256=prefix_chain_hash(prefix), continuation=cont,
                 continuation_sha256=cont['sha256'], runtime_identity=env.snapshot()['identity'],
                 support_mode='exact', action_semantics='certified_policy_request', support_records=records)
    return prefix + [stage]


def test_filter_then_original_lcb_rule_and_explicit_missing_candidate_logs():
    env = environment()
    chain = improved_chain(env)
    action, log = policy_action(chain, env, record=True)
    assert action == 9
    row = log['stages'][0]
    assert (row['base_action'], row['proposed_action'], row['selected_action']) == (8, 10, 9)
    candidates = {x['action']: x for x in row['candidates']}
    assert candidates[9]['supported'] and candidates[9]['score'] == 2.
    assert candidates[9]['measured_advantage'] == -.005  # Support is not a benefit claim.
    assert not candidates[10]['supported'] and candidates[10]['reason'] == 'NO_CANDIDATE_ACTION_LABEL'
    assert log['executed_action'] is None


def test_all_rejected_falls_back_to_incoming_and_query_is_pure():
    env = environment()
    chain = improved_chain(env, supported=())
    saved_chain, saved_state = canonical_hash(chain), canonical_hash(env.snapshot())
    policy = make_policy(chain, env)
    assert policy_action(chain, env) == 8
    assert policy_action(chain, env, record=True)[0] == 8
    assert policy.logs == [] and policy.decisions == [] and policy.action is None
    assert canonical_hash(chain) == saved_chain and canonical_hash(env.snapshot()) == saved_state
    policy.decide()
    assert policy.logs[-1]['stages'][0]['reason'] == 'NO_SUPPORTED_ALTERNATIVE'


@pytest.mark.parametrize('scores', [{9: .25}, {8: 3., 9: 2.}, {8: 2., 9: 2.}])
def test_original_strict_margin_and_base_comparison_remain(scores):
    env = environment()
    assert policy_action(improved_chain(env, scores=scores), env) == 8


def test_legacy_chain_output_exact_and_does_not_require_support_snapshot():
    rng = np.random.default_rng(13)
    chain = [constant_stage(.035), model({9: 2., 8: 3.}),
             dict(kind='override', entries=[dict(t=12., base_action=8, action=6)], match='t'),
             model({5: 2., 7: 4.})]
    for i in range(100):
        env = SimpleNamespace(x=rng.normal(size=4), alpha=float(rng.normal()), t=float(i*12),
                              history_valid=True, previous_cl_raw=.025)
        old = ChainPolicy(chain, env)
        new = make_policy(chain, env)
        assert new.decide() == old.decide() == chain_action(chain, features(env))
        assert np.array_equal(new.preference(), old.preference())


def test_prefix_hash_excludes_new_stage_and_does_not_ignore_legacy_stages():
    env = environment()
    chain = improved_chain(env, prefix=[constant_stage(.04)])
    assert policy_action(chain, env) == 9
    assert chain[-1]['reference_prefix_sha256'] == prefix_chain_hash(chain[:-1])
    assert chain[-1]['reference_prefix_sha256'] != prefix_chain_hash(chain)
    chain[0] = constant_stage(.045)
    action, log = policy_action(chain, env, record=True)
    assert action == 9
    assert all('REFERENCE_PREFIX_MISMATCH' in c['reasons'] for c in log['stages'][0]['candidates'])


def test_new_state_requires_new_exact_context_despite_identical_four_network_inputs():
    env = environment()
    chain = improved_chain(env)
    old_inputs = features(env)[[0, 1, 2, 4]].copy()
    env.alpha += .001
    assert np.array_equal(features(env)[[0, 1, 2, 4]], old_inputs)
    action, log = policy_action(chain, env, record=True)
    assert action == 8
    assert 'EXACT_CONTEXT_MISMATCH' in next(c for c in log['stages'][0]['candidates'] if c['action'] == 9)['reasons']


def test_scheduler_override_schedule_holding_and_execution_are_separate():
    env = environment()
    policy = make_policy(improved_chain(env), env, override=(12., 10), schedule={24: 7})
    assert policy.decide() == 10
    env.t = 15.
    assert policy.decide() == 10 and len(policy.logs) == 1
    assert policy.logs[0]['executed_action'] is None
    policy.record_execution(9, t=12.)
    assert policy.logs[0]['executions'] == [dict(t=12., requested_action=10, executed_action=9)]
    env.t = 24.
    assert policy.decide() == 7 and policy.logs[-1]['reason'] == 'CONTINUATION_SCHEDULE'


def test_snapshot_restore_preserves_new_class_and_held_action_without_requery():
    env = environment()
    chain = improved_chain(env)
    policy = make_policy(chain, env)
    assert policy.decide() == 9
    saved = snapshot_policy(policy)
    env.t = 15.
    env.alpha += .02
    restored = restore_policy(chain, env, saved)
    assert isinstance(restored, SupportPolicy)
    assert snapshot_policy(restored) == saved
    assert restored.decide() == 9 and restored.logs == []
    restored.record_execution(8, t=15.)
    assert restored.logs[-1]['reason'] == 'RESTORED_HELD_DECISION'
    env.t = 24.
    assert restored.decide() == 8
    assert restored.logs[-1]['stages'][0]['reason'] == 'NO_SUPPORTED_ALTERNATIVE'


def test_prepared_supported_weights_are_owned_immutable_and_hash_identical():
    env = environment()
    chain = improved_chain(env)
    original = canonical_hash(chain)
    policy = make_policy(chain, env)
    assert canonical_hash(policy.chain) == original
    weight = policy.chain[-1]['members'][0][0][0]
    assert not weight.flags.writeable
    with pytest.raises(ValueError):
        weight[0, 0] = 4.
    chain[-1]['members'][0][0][0][0][0] = 4.
    assert weight[0, 0] == 0.
