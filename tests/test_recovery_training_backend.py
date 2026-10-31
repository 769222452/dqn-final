"""Recovery training transactions and branch isolation, without an OCP solve."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from certified_flight import CL_ACTIONS
from certified_flight.transition import Transition
from continuous_recovery import training_backend as backend
from continuous_surrogate.experiment import context_from_state
from continuous_surrogate.safety import SafetyResult
from rollout_improvement.policy import ChainPolicy, constant_stage, preference
from recovery_audit_fixture import current_audit


@pytest.fixture
def factory(monkeypatch):
    mission = SimpleNamespace(data=dict(duration_s=1200., initial_state=dict(
        x_h_V_gamma_s=[40000., 2400., -.01, 0.], alpha_rad=.05)),
        outcome=lambda t, x: 'SUCCESS' if t == 1200. else 'INCOMPLETE',
        reward=lambda ds, outcome: dict(reward=ds/2.4e6+(.1 if outcome == 'SUCCESS' else 0.)))
    flags = dict(fail=False, audit_fail=False, fallback=False)
    visits = []

    def audit(c, nodes, times, cl):
        return current_audit(c, nodes, times, cl, displacement=(cl, -.01, 0., 6000.+cl),
                             actual_cl=cl+.001, valid=not flags['audit_fail'])

    class Controller:
        def __init__(self, system, model, identity):
            self.system, self.model, self.identity = system, model, identity
            self.guard, self.db = system['safety'], system['db']
            self.memory, self.epoch, self.exit_passes = None, 0, 0
        def state_dict(self):
            return deepcopy(dict(identity=self.identity, memory=self.memory,
                                 epoch=self.epoch, exit_passes=self.exit_passes))
        def load_state_dict(self, data):
            if data['identity'] != self.identity:
                raise ValueError('Incompatible recovery checkpoint')
            self.memory, self.epoch, self.exit_passes = (
                deepcopy(data['memory']), data['epoch'], data['exit_passes'])
        def decide(self, state, raw, folder):
            visits.append(dict(raw=raw, incoming=deepcopy(self.state_dict())))
            previous = self.memory['counter'] if self.memory is not None else 0
            self.memory = dict(request_CL=raw, counter=previous+1, nested={'history': [raw]})
            self.epoch += 1
            self.exit_passes = 1
            cmd = raw+.0005
            nodes = np.full(7, state['alpha'])
            au, arrays = audit(context_from_state(state, cmd), nodes, np.arange(7)*.5, cmd)
            accepted = not flags['fail'] and not (flags['fallback'] and raw == .025)
            checked = SafetyResult(accepted, 'PASS_MOCK' if accepted else 'NO_WITNESS',
                nodes, np.arange(7)*.5, au, arrays, {'proof_status': 'VERIFIED_PASS' if accepted else 'UNKNOWN_NOT_CERTIFIED'})
            z = Transition(**state['transition'])
            switched = z.command(1, state['t']) if raw >= .04 else False
            detail = dict(mach=7., generator={'geometry_index': z.target}, switched=switched,
                          events=[], selected_source='MOCK_CERTIFIED', ocp_called=False)
            return checked, z, cmd, detail, self.state_dict()

    monkeypatch.setattr('continuous_recovery.controller.RecoveryController', Controller)

    def make(folder):
        system = dict(mission=mission, db=SimpleNamespace(ids=['G42', 'G43'], index={'G42': 0}),
            plant=SimpleNamespace(audit=audit), safety=SimpleNamespace(budget=30., max_evaluations=12),
            original={'dead_end_retry': {'enabled': True, 'wall_s': 60., 'max_evaluations': 24}},
            config={'recovery': {'exit_confirmations': 2}}, identity={'sha256': 'source'})
        env = backend.TrainingFlight(system, SimpleNamespace(metadata={'frozen': True}), 'identity', {'v': 1})
        return env, backend.TrainingCertifier(env, folder)
    return make, flags, visits


def fly(certifier, cl=.025):
    pool = certifier.pool(preference(cl))
    action = int(np.argmax(np.where(pool['mask'], pool['preference'], -np.inf)))
    return certifier.commit(pool, action)


def test_same_action_fork_and_restored_snapshot_match_exactly(factory, tmp_path):
    make, _, _ = factory
    env, cert = make(tmp_path/'nominal')
    fly(cert)
    snapshot = env.snapshot()
    fork = cert.fork(tmp_path/'fork')
    restored, restored_cert = make(tmp_path/'restored')
    restored.load_state_dict(snapshot)
    a, b, c = fly(cert), fly(fork), fly(restored_cert)
    assert a[:2] == b[:2] == c[:2]
    assert env.snapshot() == fork.env.snapshot() == restored.snapshot()
    assert env.controller.memory['counter'] == 2
    assert not (tmp_path/'fork/decision_0001').exists()
    assert len(fork.pool_records) == 1
    assert fork.env.rows[0]['actual_execution_reintegrated'] is False
    assert fork.env.rows[0]['certified_trajectory_reused'] is True


def test_different_forks_own_recovery_model_and_guard_state(factory, tmp_path):
    make, _, _ = factory
    env, cert = make(tmp_path/'nominal')
    fly(cert)
    origin = env.snapshot_hash()
    left, right = cert.fork(tmp_path/'left'), cert.fork(tmp_path/'right')
    left.env.controller.memory['nested']['history'].append(99)
    left.env.controller.model.metadata['frozen'] = False
    left.env.controller.guard.budget = 99
    assert env.snapshot_hash() == origin == right.env.snapshot_hash()
    assert right.env.controller.model.metadata['frozen'] is True
    assert env.controller.guard.budget == right.env.controller.guard.budget == 30.
    fly(right, .04)
    assert env.snapshot_hash() == origin
    assert right.env.switches == 1 and env.switches == 0


def test_raw_cl_change_invalidates_old_tail_before_new_certification(factory, tmp_path):
    make, _, visits = factory
    env, cert = make(tmp_path/'base')
    fly(cert, .025)
    before = env.snapshot_hash()
    branch = cert.fork(tmp_path/'changed')
    pool = branch.pool(preference(.04))
    assert visits[-1]['incoming']['memory'] is None
    assert visits[-1]['incoming']['exit_passes'] == 0
    assert visits[-1]['incoming']['epoch'] == 1
    record = json.loads((tmp_path/'changed/decision_0000/action_pool.json').read_text())
    assert record['attempts'][0]['details']['events'][0]['event'] == 'RECOVERY_INVALIDATED_RAW_CL_CHANGE'
    assert env.snapshot_hash() == before == branch.env.snapshot_hash()
    branch.commit(pool, CL_ACTIONS.index(.04))
    assert env.snapshot_hash() == before


def test_unknown_pool_never_changes_state_or_fabricates_transition(factory, tmp_path):
    make, flags, _ = factory
    env, cert = make(tmp_path/'base')
    initial = env.snapshot()
    flags['fail'] = True
    pool = cert.pool(preference(.025))
    assert not pool['mask'].any() and len(pool['checked']) == 19
    assert set(pool['statuses']) == {'UNKNOWN_NOT_CERTIFIED'}
    assert env.snapshot() == initial and not env.history_valid
    with pytest.raises(ValueError, match='PASS'):
        cert.commit(pool, 5)
    assert not (tmp_path/'base/decision_0000/executed.npz').exists()
    assert (tmp_path/'base/stop_state.json').exists()


@pytest.mark.parametrize('change', ['state', 'memory', 'identity', 'budget', 'mask', 'policy', 'wrong_action'])
def test_stale_or_mismatched_pool_rejected_before_execution(factory, tmp_path, change):
    make, _, _ = factory
    env, cert = make(tmp_path/'base')
    cert.bind_policy_state({'action': 5})
    pool = cert.pool(preference(.025))
    action = 5
    if change == 'state':
        env._state['x'][0] += .001
    elif change == 'memory':
        env.controller.epoch += 1
    elif change == 'identity':
        env.identity = 'different'
    elif change == 'budget':
        env.controller.guard.budget += 1.
    elif change == 'mask':
        pool['mask'][4] = True
    elif change == 'policy':
        cert.bind_policy_state({'action': 4})
    else:
        action = 4
    with pytest.raises(ValueError):
        cert.commit(pool, action)
    assert env.t == 0 and env.total_reward == 0
    assert not (tmp_path/'base/decision_0000/executed.npz').exists()


@pytest.mark.parametrize('change', ['nodes', 'times', 'cl', 'transition', 'manager',
                                  'mission', 'config', 'guard_witness', 'switch_details'])
def test_sealed_pool_inputs_cannot_change_before_execution(factory, tmp_path, monkeypatch, change):
    make, _, _ = factory
    env, cert = make(tmp_path/'base')
    before = env.snapshot()
    pool = cert.pool(preference(.025))
    answer = cert._pending['answer']
    if change == 'nodes':
        answer['selected'].alpha_nodes[3] += .001
    elif change == 'times':
        answer['selected'].times[1] += .1
    elif change == 'cl':
        answer['cmd'] += .0001
    elif change == 'transition':
        answer['z'].target = 1
    elif change == 'manager':
        answer['manager_after'].epoch += 1
    elif change == 'mission':
        env.system['mission'].data['duration_s'] += 3.
    elif change == 'config':
        env.system['config']['recovery']['exit_confirmations'] += 1
    elif change == 'switch_details':
        answer['details']['switched'] = not answer['details']['switched']
    else:
        answer['selected'].diagnostics['extra_claim'] = True
    forbidden = Mock(side_effect=AssertionError('invalid execution cannot touch the plant'))
    monkeypatch.setattr(env.system['plant'], 'audit', forbidden)
    monkeypatch.setattr(env.system['plant'], 'integrate', forbidden, raising=False)
    with pytest.raises(ValueError):
        cert.commit(pool, 5)
    forbidden.assert_not_called()
    assert env.snapshot() == before
    assert env.rows == []
    assert not (tmp_path/'base/decision_0000/executed.npz').exists()


def test_pool_cannot_use_another_flights_sealed_proof(factory, tmp_path, monkeypatch):
    make, _, _ = factory
    env, cert = make(tmp_path/'base')
    _, other_cert = make(tmp_path/'other')
    before = env.snapshot()
    pool = cert.pool(preference(.025))
    other_cert.pool(preference(.025))
    cert._pending['evidence'] = other_cert._pending['evidence']
    forbidden = Mock(side_effect=AssertionError('mismatched proof cannot touch the plant'))
    monkeypatch.setattr(env.system['plant'], 'audit', forbidden)
    monkeypatch.setattr(env.system['plant'], 'integrate', forbidden, raising=False)
    with pytest.raises(ValueError):
        cert.commit(pool, 5)
    forbidden.assert_not_called()
    assert env.snapshot() == before
    assert not (tmp_path/'base/decision_0000/executed.npz').exists()


def test_pool_cannot_cross_fork_and_cannot_execute_twice(factory, tmp_path):
    make, _, _ = factory
    env, cert = make(tmp_path/'base')
    sibling = cert.fork(tmp_path/'sibling')
    pool = cert.pool(preference(.025))
    with pytest.raises(ValueError):
        sibling.commit(pool, 5)
    cert.commit(pool, 5)
    with pytest.raises(ValueError):
        cert.commit(pool, 5)
    assert env.t == 3. and sibling.env.t == 0.


def test_actual_fallback_action_updates_cl_history_and_certificate(factory, tmp_path):
    make, flags, _ = factory
    env, cert = make(tmp_path/'base')
    flags['fallback'] = True
    reward, done, info = fly(cert, .025)
    assert info['requested_action'] == 5 and info['action'] == 4
    assert env.previous_cl_raw == .02 and env.previous_cl_cmd == .0205
    assert env.previous_cl_actual == .0205+.001 and env.history_valid
    assert env.total_reward == reward and not done
    proof = json.loads((tmp_path/'base/decision_0000/certificate.json').read_text())
    assert proof['future_witness']['selected_action'] == 4
    assert proof['future_witness']['requested_action'] == 5


def test_commit_consumes_certified_trajectory_without_new_plant_calls(factory, tmp_path, monkeypatch):
    make, _, _ = factory
    env, cert = make(tmp_path/'base')
    pool = cert.pool(preference(.025))
    selected = cert._pending['answer']['selected']
    endpoint = selected.arrays['0.5']['x'][-1].copy()
    forbidden_audit = Mock(side_effect=AssertionError('commit must not call audit'))
    forbidden_integrate = Mock(side_effect=AssertionError('commit must not call integrate'))
    monkeypatch.setattr(env.system['plant'], 'audit', forbidden_audit)
    monkeypatch.setattr(env.system['plant'], 'integrate', forbidden_integrate, raising=False)
    reward, done, info = cert.commit(pool, 5)
    forbidden_audit.assert_not_called()
    forbidden_integrate.assert_not_called()
    assert env.t == 3. and env.total_reward == reward and not done
    assert np.array_equal(env.x, endpoint)
    assert info['actual_execution_reintegrated'] is False
    assert info['certified_trajectory_reused'] is True
    executed = np.load(tmp_path/'base/decision_0000/executed.npz')
    assert np.array_equal(executed['0.5_x'][-1], endpoint)


def test_invalid_initial_audit_cannot_be_sealed_or_change_flight(factory, tmp_path):
    make, flags, _ = factory
    env, cert = make(tmp_path/'base')
    before = env.snapshot()
    flags['audit_fail'] = True
    with pytest.raises(ValueError, match='audit'):
        cert.pool(preference(.025))
    assert env.snapshot() == before
    assert not (tmp_path/'base/decision_0000/executed.npz').exists()


def test_snapshot_rejects_identity_history_and_transition_transactionally(factory, tmp_path):
    make, _, _ = factory
    env, _ = make(tmp_path/'base')
    before = env.snapshot()
    for kind in ('identity', 'history', 'transition', 'controller'):
        state = deepcopy(before)
        if kind == 'identity':
            state['identity'] = 'old-controller'
        elif kind == 'history':
            state['state']['previous_cl_raw'] = .025
        elif kind == 'transition':
            state['state']['transition']['duration'] = 3.
        else:
            state['controller_memory']['identity'] = 'old-controller'
        with pytest.raises(ValueError):
            env.load_state_dict(state)
        assert env.snapshot() == before


def test_policy_restore_holds_action_and_preserves_next_boundary_override(factory, tmp_path):
    make, _, _ = factory
    env, cert = make(tmp_path/'base')
    chain = [constant_stage(.025)]
    policy = ChainPolicy(chain, env)
    assert policy.decide() == 5
    fly(cert, .025)
    saved = backend.snapshot_policy(policy)
    branch = cert.fork(tmp_path/'branch')
    held = backend.restore_policy(chain, branch.env, saved, override=(12., 4))
    assert held.decide() == 5 and len(held.decisions) == 1
    for _ in range(3):
        fly(branch, .025)
    assert held.decide() == 4 and len(held.decisions) == 2
    assert policy.decisions == [(0., 5)]
    with pytest.raises(ValueError, match='policy snapshot'):
        backend.restore_policy([constant_stage(.02)], branch.env, saved)


def test_runtime_hash_mismatch_rejected_before_build(tmp_path):
    config, model = tmp_path/'config.json', tmp_path/'model.npz'
    config.write_text('{}')
    model.write_bytes(b'weights')
    model.with_suffix('.json').write_text('{}')
    with pytest.raises(ValueError, match='config_sha256'):
        backend.build_training_flight(tmp_path/'out', dict(config=str(config), model=str(model),
                                                        config_sha256='different'))
    assert not (tmp_path/'out').exists()
