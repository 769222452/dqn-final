"""Runner evidence and transactional memory contracts, without flight/OCP."""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from certified_flight.transition import Transition
from continuous_surrogate.experiment import context_from_state
from continuous_surrogate.safety import SafetyResult
from continuous_recovery.experiment import bind_proof, commit_audited, initial_state, run_flight
from recovery_audit_fixture import current_audit


@pytest.fixture
def mock_system(monkeypatch):
    nominal = dict(x_h_V_gamma_s=[40000., 2400., -.01, 0.], alpha_rad=.05)
    mission = SimpleNamespace(data=dict(initial_state=nominal, duration_s=1200.),
        outcome=lambda t, x: 'SUCCESS' if t >= 1200. else 'INCOMPLETE',
        reward=lambda ds, outcome: dict(reward=ds/2.4e6+(.1 if outcome == 'SUCCESS' else 0.)))
    def audit(context, nodes, times, cl):
        return current_audit(context, nodes, times, cl, margin_name='q')
    system = dict(mission=mission, db=SimpleNamespace(index={'G42': 0}, ids=['G42']),
                  config={'flight': {'CL': .025}}, identity={'sha256': 'source'}, model_hash='model',
                  plant=SimpleNamespace(audit=audit))
    fail = dict(enabled=False)
    class Controller:
        def __init__(self, system, model, identity): self.memory = {'counter': 0}
        def state_dict(self): return deepcopy(self.memory)
        def load_state_dict(self, memory): self.memory = deepcopy(memory)
        def decide(self, state, cl, out):
            self.memory['counter'] += 1
            z = Transition(**state['transition'])
            c = context_from_state(state, cl)
            nodes = np.full(7, state['alpha'])
            a, arrays = audit(c, nodes, np.arange(7)*.5, cl)
            ok = not fail['enabled']
            checked = SafetyResult(ok, 'PASS' if ok else 'TEACHER_AUDIT_REJECTED',
                nodes if ok else None, np.arange(7)*.5 if ok else None, a if ok else None,
                arrays if ok else None, {'quality': {'passed': True}})
            details = dict(reachable={'CL_cmd': cl}, mach=7., generator={'geometry_index': 0},
                switched=False, pre={'supported': True}, post={'supported': True},
                selected_source='NOMINAL', ocp_called=not ok, events=['TRY_NOMINAL'],
                reason=None if ok else 'DUAL_GRID_AUDIT_REJECTED')
            return checked, z, cl, details, self.state_dict()
    module = ModuleType('continuous_recovery.controller')
    module.RecoveryController = Controller
    monkeypatch.setitem(sys.modules, 'continuous_recovery.controller', module)
    return system, SimpleNamespace(metadata={}), fail


def test_short_segment_checkpoint_is_not_mission_success(mock_system, tmp_path):
    system, model, _ = mock_system
    result = run_flight(system, model, tmp_path/'run', max_decisions=2)
    assert result['stop_reason'] == 'MAX_DECISIONS'
    assert not result['completed_1200s'] and result['steps'] == 2
    checkpoint = json.loads((tmp_path/'run/checkpoint.json').read_text())
    assert checkpoint['controller_memory'] == {'counter': 2}
    assert checkpoint['state']['t'] == 6.
    cert = json.loads((tmp_path/'run/decision_0001/certificate.json').read_text())
    assert cert['future_witness']['manager_before'] == {'counter': 1}
    assert cert['future_witness']['manager_after'] == {'counter': 2}
    assert result['selected_source_counts'] == {'NOMINAL': 2}
    assert result['events'] == {'TRY_NOMINAL': 2}
    assert all(row['actual_execution_reintegrated'] is False for row in result['rows'])
    assert all(row['certified_trajectory_reused'] is True for row in result['rows'])


def test_refusal_saves_actual_reason_and_never_commits(mock_system, tmp_path):
    system, model, fail = mock_system
    fail['enabled'] = True
    result = run_flight(system, model, tmp_path/'run', max_decisions=2)
    assert result['steps'] == 0 and result['time_s'] == 0.
    assert result['stop_reason'] == 'UNKNOWN_NOT_CERTIFIED'
    stop = json.loads((tmp_path/'run/stop_state.json').read_text())
    assert stop['last_reason'] == 'DUAL_GRID_AUDIT_REJECTED'
    assert stop['controller_memory'] == {'counter': 1}
    assert stop['last_committed_controller_memory'] == {'counter': 0}
    assert not (tmp_path/'run/decision_0000/executed.npz').exists()
    checkpoint = json.loads((tmp_path/'run/checkpoint.json').read_text())
    assert checkpoint['controller_memory'] == {'counter': 0}


def test_runner_executes_current_audit_without_another_plant_call(mock_system, tmp_path, monkeypatch):
    system, model, _ = mock_system
    forbidden = Mock(side_effect=AssertionError('execution must consume the certified trajectory'))
    monkeypatch.setattr(system['plant'], 'audit', forbidden)
    monkeypatch.setattr(system['plant'], 'integrate', forbidden, raising=False)
    # The mock controller creates its current certificate through the local
    # fixture audit; the production execution path only sees these spies.
    result = run_flight(system, model, tmp_path/'run', max_decisions=1)
    forbidden.assert_not_called()
    assert result['steps'] == 1 and result['time_s'] == 3.
    assert result['rows'][0]['certified_trajectory_reused'] is True


def test_checkpoint_restore_has_separate_segment_totals(mock_system, tmp_path):
    system, model, _ = mock_system
    state, _ = initial_state(system)
    state.update(t=1197., reward=.9, switches=2)
    state['x'][3] = 2400000.
    result = run_flight(system, model, tmp_path/'run',
        start_state={'state': state, 'controller_memory': {'counter': 8}}, max_decisions=1)
    assert result['completed_1200s'] and result['time_s'] == 1200.
    assert result['segment_range_km'] == 6. and result['range_km'] == 2406.
    assert result['segment_switches'] == 0 and result['switches'] == 2
    assert result['controller_memory_summary']['state_sha256']
    checkpoint = json.loads((tmp_path/'run/checkpoint.json').read_text())
    assert checkpoint['controller_memory'] == {'counter': 9}


def test_manager_mutation_invalidates_certificate(mock_system):
    system, _, _ = mock_system
    state, _ = initial_state(system)
    z = Transition(**state['transition'])
    nodes = np.full(7, state['alpha'])
    _, arrays = system['plant'].audit(context_from_state(state, .025), nodes, np.arange(7)*.5, .025)
    witness = dict(guard={'passed': True}, manager_before={'epoch': 1}, manager_after={'epoch': 2})
    proof = bind_proof(state, nodes, 'identity', z, .025, arrays['0.5']['x'][-1], witness)
    witness['manager_after']['epoch'] = 3
    with pytest.raises(ValueError, match='incompatible'):
        commit_audited(system, state, z, nodes, proof, 'identity', .025, witness)
    assert state['t'] == 0.


def test_runner_refuses_existing_directory(mock_system, tmp_path):
    system, model, _ = mock_system
    with pytest.raises(FileExistsError):
        run_flight(system, model, tmp_path, max_decisions=1)
