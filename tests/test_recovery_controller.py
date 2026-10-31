"""Recovery controller protocol tests; no flight database or real OCP calls."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import continuous_recovery.controller as module
from continuous_recovery.controller import RecoveryController
from continuous_recovery.plan import create_recovery_plan
from continuous_surrogate.experiment import context_from_state
from continuous_surrogate.safety import SafetyResult


def result(passed=True, source='NOMINAL', c=None):
    c = c or dict(x=[40000., 2400., 0., 0.], alpha_prev=.05)
    x = np.asarray(c['x'], float)
    arrays = {'0.5': dict(x=np.asarray([x+[1., 0., 0., 100.]])),
              '0.1': dict(x=np.asarray([x+[2., 0., 0., 100.]]))}
    return SafetyResult(passed, 'PASS' if passed else 'UNKNOWN',
                        np.full(7, c['alpha_prev']) if passed else None,
                        np.arange(7)*.5 if passed else None,
                        dict(valid=True) if passed else None,
                        arrays if passed else None,
                        dict(selected_source=source, quality={}))


@pytest.fixture
def case():
    state = dict(x=[40000., 2400., 0., 0.], alpha=.05, t=0.,
                 transition=dict(source=0, target=0, started=0., duration=8.))
    calls = dict(model=[], generator=[], correction=[], dense=[], teacher=[])
    control = dict(project=None, fail_prediction=False, current_pass=True, teacher_accepted=False)

    def predict(c, head):
        calls['model'].append((head, deepcopy(c)))
        if control['fail_prediction'] and c['t'] > 0.:
            raise FloatingPointError('bad future numeric input')
        return (np.array([6.25]) if head == 'pre' else np.full(6, .05)), dict(supported=True)

    def correct(c, nodes):
        calls['correction'].append(deepcopy(c))
        return result(control['current_pass'], c=c)

    def audit(c, nodes, times):
        calls['dense'].append(deepcopy(c))
        answer = result(c=c)
        return answer.audit, answer.arrays

    def generate(mach, cl, current):
        calls['generator'].append((mach, cl, current))
        return dict(geometry_index=current)

    def reachable(x, alpha, z, t, raw, enabled):
        return dict(CL_raw=raw, CL_cmd=raw if control['project'] is None else control['project'])

    def solve(c, head, output_dir):
        calls['teacher'].append(deepcopy(c))
        plan = SimpleNamespace(success=True, status='SUCCESS', times=np.arange(121)*.5,
                               alpha_nodes=np.full(121, c['alpha_prev']))
        return dict(accepted=control['teacher_accepted'], reason='STUB_SOLVE'), plan

    guard = SimpleNamespace(budget=30., horizon=6., duration=1200., correct=correct, audit=audit,
                            candidate=lambda c: np.full(7, c['alpha_prev']))
    system = dict(safety=guard, plant=object(), db=SimpleNamespace(generate=generate),
                  reachability=SimpleNamespace(check=reachable),
                  teacher=SimpleNamespace(solve_audited=solve), config={})
    controller = RecoveryController(system, SimpleNamespace(predict=predict), {'source': 'test'})
    return controller, state, calls, control


def memory_for(controller, state, epoch=1, identity=None):
    c = context_from_state(state, .025)
    plan = SimpleNamespace(success=True, status='SUCCESS', times=np.arange(121)*.5,
                           alpha_nodes=np.full(121, state['alpha']))
    return create_recovery_plan(plan, c, controller.identity if identity is None else identity, epoch)


def test_prepare_preserves_raw_cl_projection_and_input_state(case):
    controller, state, calls, control = case
    control['project'] = .027
    original = deepcopy(state)
    c, z, detail = controller.prepare(state, .025)
    assert state == original
    assert c['CL_raw'] == .025 and c['CL_des'] == .027
    assert all(row['CL_raw'] == .025 and row['CL_des'] == .027 for _, row in calls['model'])
    assert calls['generator'] == [(6.25, .027, 0)]
    assert detail['proposal_nodes'][0] == state['alpha']
    assert c['proposal_origin'] == 'LOCAL_POST_SURROGATE'


def test_nominal_rollout_uses_live_control_and_independent_dense_history(case):
    controller, state, calls, control = case
    c, z, detail = controller.prepare(state, .025)
    checked = controller.nominal_witness(c, detail['proposal_nodes'])
    assert checked.accepted
    assert [c['t'] for c in calls['correction']] == [0., 3., 6.]
    # Coarse state advances by 1 at each decision; dense advances by 2 and
    # must not be reset to the coarse path at either future decision.
    assert [c['x'][0] for c in calls['correction']] == [40000., 40001., 40002.]
    assert [c['x'][0] for c in calls['dense']] == [40002., 40004.]
    assert checked.diagnostics['final_coarse_state'][0] == 40003.
    assert checked.diagnostics['final_dense_state'][0] == 40006.
    assert checked.diagnostics['witness_duration_s'] == 9.
    assert [row['t'] for head, row in calls['model'] if head == 'post'] == [0., 3., 6.]
    assert calls['teacher'] == []


def test_future_numerical_failure_is_unknown_not_a_current_only_pass(case):
    controller, state, calls, control = case
    c, _, detail = controller.prepare(state, .025)
    control['fail_prediction'] = True
    checked = controller.nominal_witness(c, detail['proposal_nodes'])
    assert not checked.accepted and checked.alpha_nodes is None
    assert checked.status == 'NOMINAL_NUMERICAL_OR_DOMAIN_FAILURE'
    assert checked.diagnostics['proof_status'] == 'UNKNOWN_NOT_CERTIFIED'


@pytest.mark.parametrize('failure_time,expected_status,phase', [
    (0., 'NOMINAL_CURRENT_REQUIRES_SAFETY_RECOVERY', 'current'),
    (3., 'NOMINAL_FUTURE_REQUIRES_SAFETY_RECOVERY', 'future'),
])
def test_safe_p1_constraint_avoidance_requests_persistent_recovery(
        case, failure_time, expected_status, phase):
    controller, state, calls, control = case
    original = controller.guard.correct

    def physical_avoidance(c, nodes):
        checked = original(c, nodes)
        baseline_audit = dict(valid=c['t'] != failure_time,
                              grids={'0.1': {'valid': c['t'] != failure_time},
                                     '0.5': {'valid': c['t'] != failure_time}},
                              rejection_reason='DYNAMIC_PRESSURE')
        checked.diagnostics['attempts'] = [dict(source='CONTINUOUS_EQUAL_LIFT',
                                                audit=baseline_audit)]
        return checked

    controller.guard.correct = physical_avoidance
    c, _, detail = controller.prepare(state, .025)
    checked = controller.nominal_witness(c, detail['proposal_nodes'])
    assert not checked.accepted and checked.alpha_nodes is None
    assert checked.status == expected_status
    trigger = checked.diagnostics['recovery_trigger']
    assert trigger['time_s'] == failure_time and trigger['phase'] == phase
    assert trigger['corrected_audit']['valid']  # Physical P1 PASS does not imply nominal tracking.
    assert not trigger['baseline_audit']['valid']
    assert trigger['potentially_feasible_with_recovery']
    assert not trigger['physical_infeasibility_proven']
    assert trigger['hard_constraints_unchanged']
    assert checked.diagnostics['proof_status'] == 'UNKNOWN_NOT_CERTIFIED'
    assert len(calls['correction']) == 1+int(failure_time/3.)


def test_safe_baseline_does_not_trigger_recovery_from_tracking_quality_flags(case):
    controller, state, calls, control = case
    original = controller.guard.correct

    def tracking_deviation(c, nodes):
        checked = original(c, nodes)
        checked.diagnostics['quality'] = dict(passed=False, rmse=.02)
        checked.diagnostics['attempts'] = [dict(source='CONTINUOUS_EQUAL_LIFT',
                                                audit=dict(valid=True))]
        return checked

    controller.guard.correct = tracking_deviation
    c, _, detail = controller.prepare(state, .025)
    checked = controller.nominal_witness(c, detail['proposal_nodes'])
    assert checked.accepted and 'recovery_trigger' not in checked.diagnostics


def test_nominal_deadline_cannot_release_late_pass(case, monkeypatch):
    controller, state, calls, control = case
    clock = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    original = controller.guard.correct

    def late(c, nodes):
        checked = original(c, nodes)
        clock[0] += 31.
        return checked

    controller.guard.correct = late
    c, _, detail = controller.prepare(state, .025)
    checked = controller.nominal_witness(c, detail['proposal_nodes'])
    assert not checked.accepted and checked.status == 'SEARCH_BUDGET_EXHAUSTED'


def test_current_pass_only_cannot_exit_recovery(case, tmp_path, monkeypatch):
    controller, state, calls, control = case
    controller.memory, controller.epoch = memory_for(controller, state), 1
    controller.nominal_witness = lambda *args: result(False)
    checks = []

    def tail(guard, c, memory, identity, pre):
        checks.append(deepcopy(c))
        return result(True, 'TAIL', c)

    monkeypatch.setattr(module, 'certify_tail', tail)
    selected, _, _, detail, memory = controller.decide(state, .025, tmp_path)
    assert selected.accepted and selected.diagnostics['selected_source'] == 'TAIL'
    assert checks and not calls['teacher']
    assert memory['memory'] is not None and memory['exit_passes'] == 0


def test_two_nominal_horizon_passes_exit_recovery_and_clear_memory(case, tmp_path, monkeypatch):
    controller, state, calls, control = case
    controller.memory, controller.epoch = memory_for(controller, state), 1
    controller.nominal_witness = lambda *args: result(True)
    monkeypatch.setattr(module, 'certify_tail', lambda *args: result(True, 'TAIL'))
    (tmp_path/'first').mkdir()
    (tmp_path/'second').mkdir()
    first = controller.decide(state, .025, tmp_path/'first')
    assert first[0].diagnostics['selected_source'] == 'TAIL'
    assert controller.exit_passes == 1 and controller.memory is not None
    second = controller.decide(state, .025, tmp_path/'second')
    assert second[0].diagnostics['selected_source'] == 'NOMINAL'
    assert controller.memory is None and controller.exit_passes == 0
    assert not calls['teacher']


def test_pending_exit_with_invalid_old_tail_replans_and_only_second_pass_exits(case, tmp_path, monkeypatch):
    controller, state, calls, control = case
    controller.memory, controller.epoch = memory_for(controller, state), 1
    controller.nominal_witness = lambda *args: result(True)
    control['teacher_accepted'] = True
    checks = []

    def certify(guard, c, memory, identity, pre):
        checks.append(memory['epoch'])
        # The old tail is expired; newly solved replacement passes its audits.
        return result(memory['epoch'] == 2, 'FRESH_TAIL', c)

    monkeypatch.setattr(module, 'certify_tail', certify)
    (tmp_path/'first').mkdir()
    (tmp_path/'second').mkdir()
    first = controller.decide(state, .025, tmp_path/'first')
    assert first[0].accepted and first[0].diagnostics['selected_source'] == 'FRESH_TAIL'
    assert first[3]['ocp_called'] and len(calls['teacher']) == 1
    assert controller.memory is not None and controller.epoch == 2 and controller.exit_passes == 1
    assert checks == [1, 2]
    assert not any(event['event'] == 'RECOVERY_EXIT_NOMINAL_H30' for event in first[3]['events'])
    assert any(event['event'] == 'RECOVERY_REPLAN_PENDING_EXIT' for event in first[3]['events'])
    second = controller.decide(state, .025, tmp_path/'second')
    assert second[0].accepted and second[0].diagnostics['selected_source'] == 'NOMINAL'
    assert controller.memory is None and controller.exit_passes == 0
    assert len(calls['teacher']) == 1


@pytest.mark.parametrize('teacher_accepted', [False, True])
def test_pending_exit_failed_replacement_cannot_execute_nominal_pass(case, tmp_path, monkeypatch, teacher_accepted):
    controller, state, calls, control = case
    old = memory_for(controller, state)
    controller.memory, controller.epoch = deepcopy(old), 1
    controller.nominal_witness = lambda *args: result(True)
    control['teacher_accepted'] = teacher_accepted
    monkeypatch.setattr(module, 'certify_tail', lambda *args: result(False, 'REJECTED_TAIL'))
    selected, _, _, detail, persisted = controller.decide(state, .025, tmp_path)
    assert not selected.accepted and selected.alpha_nodes is None
    assert len(calls['teacher']) == 1 and detail['ocp_called']
    assert controller.memory == old and controller.epoch == 1 and controller.exit_passes == 1
    assert persisted['memory'] is not None
    assert not any(event['event'] == 'RECOVERY_EXIT_NOMINAL_H30' for event in detail['events'])
    restored = RecoveryController(controller.system, controller.model, controller.identity)
    restored.load_state_dict(persisted)
    assert restored.state_dict() == persisted
    assert restored.exit_passes < restored.exit_confirmations


@pytest.mark.parametrize('teacher_accepted', [False, True])
def test_failed_fresh_recovery_helper_preserves_pending_state_transactionally(
        case, tmp_path, monkeypatch, teacher_accepted):
    controller, state, calls, control = case
    controller.memory, controller.epoch = memory_for(controller, state), 1
    controller.exit_passes = 1
    before = controller.state_dict()
    control['teacher_accepted'] = teacher_accepted
    monkeypatch.setattr(module, 'certify_tail', lambda *args: result(False, 'REJECTED_TAIL'))
    detail = dict(ocp_called=False, fallbacks=[], events=[])
    selected = controller._fresh_recovery(context_from_state(state, .025), tmp_path, detail)
    assert not selected.accepted
    assert controller.state_dict() == before
    restored = RecoveryController(controller.system, controller.model, controller.identity)
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == before


def test_nominal_failure_resets_pending_exit_count(case, tmp_path, monkeypatch):
    controller, state, calls, control = case
    controller.memory, controller.epoch = memory_for(controller, state), 1
    controller.exit_passes = 1
    controller.nominal_witness = lambda *args: result(False)
    monkeypatch.setattr(module, 'certify_tail', lambda *args: result(True, 'TAIL'))
    selected, _, _, _, persisted = controller.decide(state, .025, tmp_path)
    assert selected.accepted and selected.diagnostics['selected_source'] == 'TAIL'
    assert controller.exit_passes == persisted['exit_passes'] == 0
    assert controller.memory is not None and not calls['teacher']


def test_fresh_ocp_solver_acceptance_does_not_override_reaudit_rejection(case, tmp_path, monkeypatch):
    controller, state, calls, control = case
    old = memory_for(controller, state)
    controller.memory, controller.epoch = deepcopy(old), 1
    controller.nominal_witness = lambda *args: result(False)
    control['teacher_accepted'] = True
    monkeypatch.setattr(module, 'certify_tail', lambda *args: result(False, 'TAIL_REJECTED'))
    selected, _, _, detail, persisted = controller.decide(state, .025, tmp_path)
    assert not selected.accepted and len(calls['teacher']) == 1
    assert controller.memory == old and controller.epoch == 1
    assert persisted['memory']['epoch'] == persisted['epoch']
    assert any(event['event'] == 'FRESH_RECOVERY_REJECTED' for event in detail['events'])


def test_fresh_recovery_commits_memory_epoch_after_certification(case, tmp_path, monkeypatch):
    controller, state, calls, control = case
    controller.nominal_witness = lambda *args: result(False)
    control['teacher_accepted'] = True
    calls_at_certification = []

    def certify(guard, c, memory, identity, pre):
        calls_at_certification.append((controller.epoch, controller.memory, memory['epoch']))
        return result(True, 'FRESH_TAIL', c)

    monkeypatch.setattr(module, 'certify_tail', certify)
    selected, _, _, _, persisted = controller.decide(state, .025, tmp_path)
    assert selected.accepted
    assert calls_at_certification == [(0, None, 1)]
    assert controller.epoch == 1 and persisted['memory']['epoch'] == 1


def test_checkpoint_roundtrip_preserves_recovery_and_exit_state(case):
    controller, state, calls, control = case
    controller.memory, controller.epoch = memory_for(controller, state, epoch=2), 2
    controller.exit_passes = 1
    data = controller.state_dict()
    controller.memory, controller.epoch, controller.exit_passes = None, 0, 0
    controller.load_state_dict(data)
    assert controller.state_dict() == data
    data['memory']['controls']['alpha_nodes_rad'][0] += .01
    assert controller.memory['controls']['alpha_nodes_rad'][0] == .05


@pytest.mark.parametrize('mutation', ['identity', 'epoch', 'negative', 'exit', 'without_plan'])
def test_inconsistent_checkpoint_rejected_without_mutating_controller(case, mutation):
    controller, state, calls, control = case
    original = controller.state_dict()
    data = controller.state_dict()
    data.update(memory=memory_for(controller, state), epoch=1)
    if mutation == 'identity':
        data['memory'] = memory_for(controller, state, identity={'source': 'other'})
    elif mutation == 'epoch': data['epoch'] = 2
    elif mutation == 'negative': data['epoch'] = -1
    elif mutation == 'exit': data['exit_passes'] = 2
    elif mutation == 'without_plan': data.update(memory=None, exit_passes=1)
    with pytest.raises(ValueError):
        controller.load_state_dict(data)
    assert controller.state_dict() == original
