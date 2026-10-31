"""Recovery memory is shifted controls, never a reusable safety certificate."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from continuous_recovery.plan import (RecoveryPlan, certify_tail, compatible,
                                      create_recovery_plan, shift_plan, validate)


@pytest.fixture
def case():
    c = dict(x=[40000., 2400., -.01, 0.], alpha_prev=.05, t=0., source=0, target=0,
             fraction=1., rate=0., CL_des=.025, CL_raw=.025, recovery_epoch=2)
    identity = dict(source_sha256='abc', model_sha256='def', config_sha256='ghi')
    times = np.r_[np.arange(19)*.5, np.arange(12., 61., 3.)]
    plan = SimpleNamespace(success=True, status='SUCCESS', times=times, alpha_nodes=.05+.0001*times)
    memory = create_recovery_plan(plan, c, identity, epoch=2)
    calls, predictions = [], []
    reject = dict(horizon=None)

    def audit(context, nodes, times):
        h = float(times[-1])
        calls.append(dict(context=deepcopy(context), nodes=nodes.copy(), times=times.copy()))
        valid = h != reject['horizon']
        proof = dict(valid=valid, control_valid=valid, continuous_at_entry=nodes[0] == context['alpha_prev'],
                     CL_target=context['CL_des'], maximum_alpha_rate_degps=1.,
                     grids={key: dict(valid=valid, CL_mean_first3=context['CL_des'],
                                     CL_rmse_first3=.001, CL_peak_error_first3=.002)
                            for key in ('0.1', '0.5')})
        arrays = {}
        for key in ('0.1', '0.5'):
            tt = np.arange(1, int(round(h/float(key)))+1)*float(key)
            arrays[key] = dict(x=np.asarray(context['x'])+np.column_stack([
                -tt, -tt, np.zeros(len(tt)), 2300.*tt]))
        return proof, arrays

    guard = SimpleNamespace(duration=1200., horizon=30., budget=30., cfg={}, audit=audit,
        _quality=lambda audit: {'passed': True},
        db=SimpleNamespace(generate=lambda mach, cl, current: {'geometry_index': current}),
        reachability=SimpleNamespace(check=lambda x, alpha, z, t, raw, enabled: {'CL_cmd': raw}))

    def pre(context):
        predictions.append(deepcopy(context))
        return 6.25, {'source': 'LIVE_PRE'}
    return c, identity, plan, memory, guard, calls, predictions, pre, reject


def advance(c, memory, t):
    c = deepcopy(c)
    c['t'] = t
    c['alpha_prev'] = float(np.interp(t-memory['started_at_s'], memory['controls']['times_s'],
                                    memory['controls']['alpha_nodes_rad']))
    return c


def test_json_memory_roundtrip_and_namespace(case):
    c, identity, plan, memory, *_ = case
    restored = json.loads(json.dumps(memory, allow_nan=False))
    assert restored == memory and RecoveryPlan.validate(restored)
    assert memory['executable_certificate_stored'] is False
    assert compatible(c, restored, identity, 1200.)[0]


def test_tampered_control_or_provenance_rejected(case):
    c, identity, plan, memory, *_ = case
    memory['controls']['alpha_nodes_rad'][-1] += .001
    with pytest.raises(ValueError, match='hash'):
        validate(memory)
    assert compatible(c, memory, identity, 1200.)[1] == 'RECOVERY_MEMORY_INVALID'


def test_shift_retains_controls_and_resamples_all_halfseconds(case):
    c, identity, plan, memory, *_ = case
    c = advance(c, memory, 3.)
    nodes, times, detail = shift_plan(c, memory)
    assert np.array_equal(times, np.arange(115)*.5)
    assert np.array_equal(nodes, np.interp(times+3., plan.times, plan.alpha_nodes))
    assert np.array_equal(times[:7], np.arange(7)*.5)
    assert detail['re_integration_required'] and detail['entry_reset_difference_rad'] == 0.


@pytest.mark.parametrize('kind,status', [
    ('cl', 'RECOVERY_CL_REQUEST_CHANGED'), ('identity', 'RECOVERY_IDENTITY_CHANGED'),
    ('epoch', 'RECOVERY_EPOCH_CHANGED'), ('alpha', 'RECOVERY_ALPHA_ENTRY_CHANGED'),
    ('geometry', 'RECOVERY_GEOMETRY_CHANGED'), ('time', 'RECOVERY_TIME_OUTSIDE_TASK')])
def test_incompatible_changes_have_distinct_unknown_reasons(case, kind, status):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    if kind == 'cl': c['CL_raw'] = .03
    if kind == 'identity': identity['source_sha256'] = 'changed'
    if kind == 'epoch': c['recovery_epoch'] = 3
    if kind == 'alpha': c['alpha_prev'] = np.nextafter(c['alpha_prev'], np.inf)
    if kind == 'geometry': c.update(source=1, target=1)
    if kind == 'time': c['t'] = -3.
    result = certify_tail(guard, c, memory, identity, pre)
    assert not result.accepted and result.status == status
    assert result.alpha_nodes is None and calls == []
    assert result.diagnostics['proof_status'] == 'UNKNOWN_NOT_CERTIFIED'


def test_shrinking_plan_never_shrinks_h30_requirement(case):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    at27 = advance(c, memory, 27.)
    assert certify_tail(guard, at27, memory, identity, pre).accepted
    calls.clear()
    at30 = advance(c, memory, 30.)
    result = certify_tail(guard, at30, memory, identity, pre)
    assert not result.accepted and result.status == 'RECOVERY_HORIZON_INSUFFICIENT'
    assert result.diagnostics['compatibility']['required_witness_s'] == 33.
    assert result.diagnostics['compatibility']['remaining_plan_s'] == 30.
    assert not calls


def test_current_state_reintegrated_and_certificate_not_reused(case):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    first = certify_tail(guard, c, memory, identity, pre)
    c['x'][0] += 10.
    second = certify_tail(guard, c, memory, identity, pre)
    assert first.accepted and second.accepted and len(calls) == 6
    assert calls[3]['context']['x'][0] == 40010.
    assert first.diagnostics['plan_sha256'] != second.diagnostics['plan_sha256']
    assert second.diagnostics['certificate_reused'] is False
    reject['horizon'] = 60.
    third = certify_tail(guard, c, memory, identity, pre)
    assert not third.accepted and third.status == 'RECOVERY_FULL_TAIL_AUDIT_REJECTED'


@pytest.mark.parametrize('reject_h,status', [(60., 'RECOVERY_FULL_TAIL_AUDIT_REJECTED'),
    (3., 'RECOVERY_CURRENT_AUDIT_REJECTED'), (33., 'RECOVERY_WITNESS_AUDIT_REJECTED')])
def test_three_independent_audits_required(case, reject_h, status):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    reject['horizon'] = reject_h
    result = certify_tail(guard, c, memory, identity, pre)
    assert not result.accepted and result.status == status
    assert result.arrays is None


@pytest.mark.parametrize('remaining,future_checks', [(3., 0), (6., 1), (15., 4)])
def test_only_actual_mission_end_shortens_required_horizon(case, remaining, future_checks):
    c, identity, plan, _, guard, calls, predictions, pre, reject = case
    c['t'] = 1200.-remaining
    plan.times = np.arange(int(remaining/.5)+1)*.5
    plan.alpha_nodes = .05+.0001*plan.times
    memory = create_recovery_plan(plan, c, identity, epoch=2)
    result = certify_tail(guard, c, memory, identity, pre)
    assert result.accepted and len(predictions) == future_checks
    assert [v['times'][-1] for v in calls] == [remaining, 3., remaining]
    assert result.diagnostics['future_horizon_s'] == remaining-3.


def test_future_mach_uses_live_pre_callback_not_tail_mean(case):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    values = []
    def generate(mach, cl, current):
        values.append((mach, cl, current))
        return dict(geometry_index=current)
    guard.db.generate = generate
    result = certify_tail(guard, c, memory, identity, pre)
    assert result.accepted and len(predictions) == 10
    assert [v['t'] for v in predictions] == list(np.arange(3., 33., 3.))
    assert all(v[0] == 6.25 for v in values)
    assert all(v['CL_raw'] == .025 for v in predictions)
    assert all(v['pre_mach_definition'] == 'same_callback_as_live_pre'
               for v in result.diagnostics['generator_checks'])


def test_generator_none_preserves_live_hold_semantics(case):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    guard.db.generate = lambda mach, cl, current: dict(geometry_index=None)
    result = certify_tail(guard, c, memory, identity, pre)
    assert result.accepted
    assert not any(v['equal_lift_geometry_found'] for v in result.diagnostics['generator_checks'])


def test_geometry_transition_settles_without_extending_lock(case):
    c, identity, plan, _, guard, calls, predictions, pre, reject = case
    c.update(source=0, target=1, fraction=0., rate=.125)
    memory = create_recovery_plan(plan, c, identity, epoch=2)
    guard.db.generate = lambda mach, cl, current: dict(geometry_index=2)
    result = certify_tail(guard, c, memory, identity, pre)
    assert result.status == 'RECOVERY_GENERATOR_INCOMPATIBLE'
    assert [(v['relative_time_s'], v['transition_locked'], v['compatible'])
            for v in result.diagnostics['generator_checks']] == [
                (3., True, True), (6., True, True), (9., False, False)]
    c = advance(c, memory, 9.)
    c.update(source=1, target=1, fraction=1., rate=0.)
    assert compatible(c, memory, identity, 1200.)[0]
    c.update(source=0, target=1, fraction=0., rate=.125)
    assert compatible(c, memory, identity, 1200.)[1] == 'RECOVERY_GEOMETRY_CHANGED'


def test_deadline_covers_audit_and_pre_callback(case, monkeypatch):
    c, identity, plan, memory, guard, calls, predictions, pre, reject = case
    clock = [0.]
    monkeypatch.setattr('continuous_recovery.plan.time.perf_counter', lambda: clock[0])
    def slow_pre(context):
        clock[0] = 31.
        return 6.
    result = certify_tail(guard, c, memory, identity, slow_pre)
    assert result.status == 'SEARCH_BUDGET_EXHAUSTED' and not result.accepted
    assert result.diagnostics['budget_exceeded']
    assert result.alpha_nodes is None


def test_short_or_failed_plan_cannot_manufacture_pass(case):
    c, identity, plan, _, guard, calls, predictions, pre, reject = case
    plan.success = False
    with pytest.raises(ValueError, match='unsuccessful'):
        create_recovery_plan(plan, c, identity)
    plan.success = True
    plan.times, plan.alpha_nodes = plan.times[:10], plan.alpha_nodes[:10]
    memory = create_recovery_plan(plan, c, identity, epoch=2)
    result = certify_tail(guard, c, memory, identity, pre)
    assert not result.accepted and result.status == 'RECOVERY_HORIZON_INSUFFICIENT'

