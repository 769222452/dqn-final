"""Frozen-policy runner evidence, scheduling and transaction tests without OCP."""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from certified_flight import CL_ACTIONS
from certified_flight.transition import Transition
from continuous_surrogate.experiment import context_from_state
from continuous_surrogate.safety import SafetyResult
from continuous_recovery.policy_experiment import run_policy_flight
from recovery_audit_fixture import current_audit


@pytest.fixture
def policy_system(monkeypatch):
    mission = SimpleNamespace(
        data=dict(initial_state=dict(x_h_V_gamma_s=[40000., 2400., -.01, 0.], alpha_rad=.05), duration_s=1200.),
        outcome=lambda t, x: 'SUCCESS' if t >= 1200. else 'INCOMPLETE',
        reward=lambda ds, outcome: dict(reward=ds/2.4e6+(.1 if outcome == 'SUCCESS' else 0.)))

    def audit(context, nodes, times, cl):
        return current_audit(context, nodes, times, cl)

    system = dict(mission=mission, db=SimpleNamespace(index={'G42': 0}, ids=['G42']),
        config={'flight': {'CL': .025}}, identity={'sha256': 'source'}, model_hash='model',
        plant=SimpleNamespace(audit=audit),
        original=dict(dead_end_retry={'enabled': True, 'wall_s': 60., 'max_evaluations': 24}))
    flags = dict(fail=False, mutate=False)

    class Controller:
        def __init__(self, system, model, identity):
            self.memory = {'counter': 0}
        def state_dict(self):
            return deepcopy(self.memory)

    module = ModuleType('continuous_recovery.controller')
    module.RecoveryController = Controller
    monkeypatch.setitem(sys.modules, 'continuous_recovery.controller', module)

    class Policy:
        metadata = dict(checkpoint_sha256='frozen', chain_sha256='chain')
        def __init__(self):
            self.decisions, self.calls = [], []
            self.decision_t = None
        def state_dict(self):
            return dict(decisions=deepcopy(self.decisions), decision_t=self.decision_t)
        def choose(self, state, history_valid=False, previous_cl_raw=None):
            self.calls.append((state['t'], history_valid, previous_cl_raw))
            new = self.decision_t is None or state['t'] >= self.decision_t+12.
            if new:
                self.decision_t = 12.*np.floor(state['t']/12.)
                self.decisions.append([self.decision_t, 5])
            return dict(action=5, CL_raw=.025, preference=(-abs(np.asarray(CL_ACTIONS)-.025)).tolist(),
                features=[0., 0., 0., .5, state['t']/1200., 0.],
                policy_decision_time_s=self.decision_t, new_policy_decision=new)

    def certify(controller, state, preference, folder, retry_options=None):
        assert retry_options == {'enabled': True, 'wall_s': 60., 'max_evaluations': 24}
        folder.mkdir()
        if flags['mutate']:
            controller.memory['counter'] += 99
        selected = Controller(None, None, None)
        selected.memory = deepcopy(controller.memory)
        selected.memory['counter'] += 1
        cl, action = .02, 4
        nodes = np.full(7, state['alpha'])
        a, arrays = audit(context_from_state(state, cl), nodes, np.arange(7)*.5, cl)
        status = 'UNKNOWN_NOT_CERTIFIED' if flags['fail'] else 'VERIFIED_PASS'
        result = SafetyResult(not flags['fail'], 'NOMINAL_H30_WITNESS_FOUND',
            nodes, np.arange(7)*.5, a, arrays, dict(quality={'passed': True}))
        detail = dict(mach=7., generator={'geometry_index': 0}, switched=False,
            pre={'supported': False}, post={'supported': True}, selected_source='NOMINAL',
            ocp_called=False, events=['TRY_NOMINAL'], nominal_status='NOMINAL_H30_WITNESS_FOUND')
        statuses = ['UNKNOWN_NOT_CERTIFIED']*19
        statuses[action] = status
        mask = [False]*19
        mask[action] = not flags['fail']
        return dict(action=None if flags['fail'] else action, selected=result,
            z=Transition(**state['transition']), cmd=cl, details=detail,
            manager_after=selected, controller_state=selected.state_dict(), checked=[action],
            statuses=statuses, reasons=[None]*19, mask=mask,
            attempts=[dict(action=action, status=status, accepted=not flags['fail'], details=detail)],
            retry=None, wall_s=.001)

    adapter = ModuleType('continuous_recovery.learned_policy')
    adapter.certify_ranked = certify
    monkeypatch.setitem(sys.modules, 'continuous_recovery.learned_policy', adapter)
    return system, SimpleNamespace(metadata={}), Policy(), flags


def test_policy_12s_choices_and_actual_3s_history_are_distinct(policy_system, tmp_path):
    system, model, policy, _ = policy_system
    result = run_policy_flight(system, model, policy, tmp_path/'run', max_decisions=5)
    assert result['stop_reason'] == 'MAX_DECISIONS' and not result['completed_1200s']
    assert result['policy_decisions'] == 2 and result['steps'] == 5
    assert result['requested_action_counts'][5] == 2
    assert result['executed_action_counts'][4] == 5
    assert result['safety_substitution_steps'] == 5
    assert policy.calls[0] == (0., False, None)
    assert all(valid and cl == .02 for t, valid, cl in policy.calls[1:])
    assert result['counts']['pre_ood'] == result['counts']['executed_pre_ood'] == 5
    assert result['action_certification'][4]['PASS'] == 5
    assert all(row['actual_execution_reintegrated'] is False for row in result['rows'])
    assert all(row['certified_trajectory_reused'] is True for row in result['rows'])
    checkpoint = json.loads((tmp_path/'run/checkpoint.json').read_text())
    assert checkpoint['controller_memory'] == {'counter': 5}
    assert checkpoint['state']['previous_cl_raw'] == .02
    assert checkpoint['policy_metadata']['checkpoint_sha256'] == 'frozen'
    cert = json.loads((tmp_path/'run/decision_0004/certificate.json').read_text())
    witness = cert['future_witness']
    assert witness['requested_action'] == 5 and witness['selected_action'] == 4
    assert witness['manager_before'] == {'counter': 4}
    assert witness['manager_after'] == {'counter': 5}
    assert witness['policy_identity']['chain_sha256'] == 'chain'


def test_all_unknown_never_commit_candidate_or_history(policy_system, tmp_path):
    system, model, policy, flags = policy_system
    flags['fail'] = True
    result = run_policy_flight(system, model, policy, tmp_path/'run', max_decisions=2)
    assert result['stop_reason'] == 'NO_CERTIFIED_ACTION'
    assert result['steps'] == 0 and result['time_s'] == 0.
    assert result['final_state']['history_valid'] is False
    stop = json.loads((tmp_path/'run/stop_state.json').read_text())
    checkpoint = json.loads((tmp_path/'run/checkpoint.json').read_text())
    assert stop['controller_memory'] == checkpoint['controller_memory'] == {'counter': 0}
    assert not (tmp_path/'run/decision_0000/executed.npz').exists()


def test_policy_runner_executes_current_audit_without_new_plant_calls(policy_system, tmp_path, monkeypatch):
    system, model, policy, _ = policy_system
    forbidden = Mock(side_effect=AssertionError('execution must consume the certified trajectory'))
    monkeypatch.setattr(system['plant'], 'audit', forbidden)
    monkeypatch.setattr(system['plant'], 'integrate', forbidden, raising=False)
    result = run_policy_flight(system, model, policy, tmp_path/'run', max_decisions=1)
    forbidden.assert_not_called()
    assert result['steps'] == 1 and result['time_s'] == 3.
    assert result['rows'][0]['certified_trajectory_reused'] is True


def test_candidate_live_memory_mutation_is_rejected(policy_system, tmp_path):
    system, model, policy, flags = policy_system
    flags['mutate'] = True
    with pytest.raises(ValueError, match='live controller memory'):
        run_policy_flight(system, model, policy, tmp_path/'run', max_decisions=1)
    result = json.loads((tmp_path/'run/result.json').read_text())
    assert result['steps'] == 0
    assert not (tmp_path/'run/decision_0000/executed.npz').exists()


def test_runner_refuses_overwrite(policy_system, tmp_path):
    system, model, policy, _ = policy_system
    with pytest.raises(FileExistsError):
        run_policy_flight(system, model, policy, tmp_path)
