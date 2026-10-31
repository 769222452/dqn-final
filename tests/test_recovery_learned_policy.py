"""Frozen feedback inference and isolated first-PASS action trials (no OCP)."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from certified_flight import CL_ACTIONS
from certified_flight.certification import PASS, UNKNOWN
from continuous_recovery.learned_policy import FrozenChainPolicy, certify_ranked
from continuous_surrogate.safety import SafetyResult
from rollout_improvement.features import features
from rollout_improvement.policy import ChainPolicy, preference


def checkpoint(tmp_path, chain=None):
    if chain is None:
        W = np.zeros((6, 19))
        W[0, 7] = 1.
        W[0, 9] = -1.
        chain = [dict(kind='model', members=[[[W.tolist(), np.zeros(19).tolist()]]],
                      margin=0., window=1200., times=[0.], inputs=None)]
    path = tmp_path/'policy.json'
    path.write_text(json.dumps(dict(chain=chain)), encoding='utf-8')
    return path


def state(t=0., h=50000.):
    return dict(x=[h, 2400., 0., 0.], alpha=.05, t=t,
                transition=dict(source=0, target=0, started=0., duration=8.))


def test_original_feedback_every_twelve_seconds_and_absolute_history(tmp_path):
    path = checkpoint(tmp_path)
    content = path.read_bytes()
    adapter = FrozenChainPolicy(path)
    first = adapter.choose(state(), False, None)
    assert first['action'] == 7 and first['new_policy_decision']
    held = adapter.choose(state(3., 30000.), True, .025)
    assert held['action'] == 7 and not held['new_policy_decision']
    assert held['features'][-1] == pytest.approx((.025-.04)/.04)
    changed = adapter.choose(state(12., 30000.), True, .02)
    assert changed['action'] == 9 and changed['new_policy_decision']
    env = SimpleNamespace(x=np.array(state(12., 30000.)['x']), t=12., alpha=.05,
                          previous_cl_raw=.02, history_valid=True)
    original = ChainPolicy(json.loads(content)['chain'], env)
    assert original.decide() == changed['action']
    np.testing.assert_array_equal(original.preference(), changed['preference'])
    np.testing.assert_array_equal(features(env), changed['features'])
    assert path.read_bytes() == content
    W, b = adapter.policy.chain[0]['members'][0][0]
    assert not W.flags.writeable and not b.flags.writeable


def test_frozen_policy_resume_and_bad_dimensions(tmp_path):
    path = checkpoint(tmp_path)
    adapter = FrozenChainPolicy(path)
    adapter.choose(state())
    adapter.choose(state(9.))
    saved = json.loads(json.dumps(adapter.state_dict()))
    restored = FrozenChainPolicy(path)
    restored.load_state_dict(saved)
    assert restored.choose(state(12., 30000.)) == adapter.choose(state(12., 30000.))
    bad = deepcopy(saved)
    bad['identity'] = 'other-model'
    with pytest.raises(ValueError, match='Incompatible'):
        restored.load_state_dict(bad)
    with pytest.raises(ValueError, match='backwards'):
        restored.choose(state(3.))
    broken = json.loads(path.read_text())
    broken['chain'][0]['members'][0][0][1] = [0.]*18
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='layer'):
        FrozenChainPolicy(path)


class FakeController:
    def __init__(self, responses, retry_pass=None, memory=None):
        self.guard = SimpleNamespace(budget=30., max_evaluations=12)
        self.system = dict(safety=self.guard)
        self.identity, self.memory = 'same-model', deepcopy(memory)
        self.epoch, self.exit_passes = 3, 1 if memory else 0
        self.responses, self.retry_pass = responses, retry_pass
        self.calls = []  # External instrumentation, not controller state.

    def state_dict(self):
        return deepcopy(dict(identity=self.identity, memory=self.memory, epoch=self.epoch,
                             exit_passes=self.exit_passes))

    def decide(self, measured, cl, folder):
        a = CL_ACTIONS.index(cl)
        self.calls.append(dict(action=a, budget=self.guard.budget,
                              memory=deepcopy(self.memory), exit_passes=self.exit_passes))
        measured['x'][0] = -99.
        self.memory = dict(request_CL=cl, marker=a)
        self.epoch += 1
        reason = self.responses.get(a, 'UNKNOWN_TRAJECTORY')
        if self.guard.budget == 60. and a == self.retry_pass:
            reason = 'PASS'
        if reason == 'ERROR':
            raise ValueError('injected numerical refusal')
        accepted = reason == 'PASS'
        selected = SafetyResult(accepted, reason, np.full(7, .05) if accepted else None,
            np.arange(7)*.5 if accepted else None, dict(valid=True) if accepted else None,
            {} if accepted else None, dict(proof_status=PASS if accepted else UNKNOWN))
        return selected, SimpleNamespace(source=0, target=0), cl, dict(events=[]), self.state_dict()


def test_ranked_pool_rollback_and_changed_cl_invalidates_memory(tmp_path):
    controller = FakeController({8: 'UNKNOWN_TRAJECTORY', 9: 'PASS'},
                                memory=dict(request_CL=.04, marker='old'))
    before, measured = controller.state_dict(), state()
    original_state = deepcopy(measured)
    q = np.arange(19, dtype=float)*-1.
    q[8], q[9] = 3., 2.
    pool = certify_ranked(controller, measured, q, tmp_path/'pool')
    assert pool['requested_action'] == 8 and pool['action'] == 9
    assert pool['checked'] == [8, 9]
    assert pool['statuses'][8] == UNKNOWN and pool['statuses'][9] == PASS
    assert pool['reasons'][10] == 'NOT_CHECKED'
    assert controller.state_dict() == before and measured == original_state
    assert controller.calls[0]['memory'] == before['memory']
    assert controller.calls[1]['memory'] is None and controller.calls[1]['exit_passes'] == 0
    assert pool['details']['events'][0]['event'] == 'RECOVERY_INVALIDATED_RAW_CL_CHANGE'
    assert pool['manager_after'].memory['request_CL'] == .045
    assert pool['manager_after'].epoch == before['epoch']+1
    assert sum(pool['mask']) == 1


def test_all_nineteen_checked_before_budget_only_retry(tmp_path):
    controller = FakeController({2: 'SEARCH_BUDGET_EXHAUSTED', 4: 'ERROR'}, retry_pass=2)
    before = controller.state_dict()
    options = dict(enabled=True, wall_s=60., max_evaluations=24)
    pool = certify_ranked(controller, state(), preference(.04), tmp_path/'pool', options)
    assert len(pool['checked']) == 19
    assert len(pool['attempts']) == 20
    assert pool['attempts'][-1]['action'] == 2 and pool['attempts'][-1]['retry']
    assert pool['action'] == 2 and pool['retry']['actions_eligible'] == [2]
    assert pool['manager_after'].guard.budget == controller.guard.budget == 30.
    assert pool['manager_after'].guard.max_evaluations == controller.guard.max_evaluations == 12
    assert controller.state_dict() == before
    assert pool['attempts'][-1]['details']['events'] == []


def test_unknown_is_not_physical_infeasibility_or_automatic_retry(tmp_path):
    controller = FakeController({3: 'REAL_OCP_RECOVERY_NOT_ACCEPTED'})
    pool = certify_ranked(controller, state(), preference(.02), tmp_path/'pool',
                          dict(enabled=True, wall_s=60., max_evaluations=24))
    assert pool['action'] is None and pool['manager_after'] is None
    assert pool['certification_exhausted'] and not pool['universal_infeasibility_proven']
    assert len(pool['checked']) == len(pool['attempts']) == 19
    assert pool['statuses'] == [UNKNOWN]*19 and pool['retry'] is None
    assert pool['failure_semantics'] == 'RUNTIME_CERTIFICATION_EXHAUSTED'


def test_bad_preference_does_not_start_any_trial(tmp_path):
    controller = FakeController({})
    with pytest.raises(ValueError, match='nineteen'):
        certify_ranked(controller, state(), [0.]*18, tmp_path/'pool')
    assert controller.calls == []
