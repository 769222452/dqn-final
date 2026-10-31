"""An OCP tail witnesses only its unchanged prefix and compatible geometry policy."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from continuous_surrogate.safety import ContinuousSafety, OCP_TAIL_PROTOCOL


@pytest.fixture
def setup():
    safety = ContinuousSafety.__new__(ContinuousSafety)
    safety.duration, safety.horizon, safety.budget = 1200., 30., 30.
    safety.cfg = {}
    safety.quality_tolerance, safety.mean_tolerance, safety.peak_tolerance = .005, .0025, .01
    safety.rate = .0872665
    safety.plant = SimpleNamespace(atmosphere=SimpleNamespace(speed_of_sound=lambda h: 300.))
    safety.reachability = SimpleNamespace(check=lambda x, alpha, z, t, raw, enabled: {'CL_cmd': raw})
    safety.db = SimpleNamespace(generate=lambda mach, cl, current: dict(geometry_index=current, CL_des=cl))
    calls = []
    switches = dict(reject_horizon=None)

    def audit(context, nodes, times):
        horizon = float(times[-1])
        calls.append((np.asarray(nodes).copy(), np.asarray(times).copy()))
        valid = horizon != switches['reject_horizon']
        proof = dict(valid=valid, control_valid=valid, continuous_at_entry=True,
                     CL_target=context['CL_des'], maximum_alpha_rate_degps=1.,
                     grids={dt: dict(valid=valid, CL_mean_first3=context['CL_des'],
                                      CL_rmse_first3=.001, CL_peak_error_first3=.002)
                            for dt in ('0.1', '0.5')})
        arrays = {}
        for key in ('0.1', '0.5'):
            tt = np.arange(1, int(round(horizon/float(key)))+1)*float(key)
            arrays[key] = dict(x=np.column_stack([40000.-tt, 2400.-tt, np.full(len(tt), -.01), 2300.*tt]))
        return proof, arrays

    safety.audit = audit
    safety.correct = lambda *args: pytest.fail('P1 must never change an OCP tail prefix')
    c = dict(x=[40000., 2400., -.01, 0.], alpha_prev=.05, t=0., source=0, target=0,
             fraction=1., rate=0., CL_des=.025, CL_raw=.025)
    times = np.arange(121)*.5
    plan = SimpleNamespace(success=True, status='SUCCESS', times=times, alpha_nodes=.05+times*.001)
    return safety, c, plan, calls, switches


def test_exact_prefix_uses_full_tail_without_p1(setup):
    safety, c, plan, calls, _ = setup
    before = deepcopy(c)
    result = safety.assess_ocp_plan(c, plan)
    assert result.accepted and result.status == 'OCP_TAIL_WITNESS_FOUND'
    assert np.array_equal(result.alpha_nodes, plan.alpha_nodes[:7])
    assert np.array_equal(result.times, plan.times[:7])
    assert [t[-1] for _, t in calls] == [60., 3., 33.]
    assert result.diagnostics['witness_protocol'] == OCP_TAIL_PROTOCOL
    assert result.diagnostics['p1_applied'] is False
    assert result.diagnostics['future_horizon_s'] == 30.
    assert len(result.diagnostics['generator_checks']) == 10
    assert c == before


@pytest.mark.parametrize('horizon,status', [(60., 'OCP_FULL_PLAN_AUDIT_REJECTED'),
                                           (3., 'OCP_PREFIX_AUDIT_REJECTED'),
                                           (33., 'OCP_TAIL_AUDIT_REJECTED')])
def test_each_independent_audit_can_refuse_execution(setup, horizon, status):
    safety, c, plan, _, switches = setup
    switches['reject_horizon'] = horizon
    result = safety.assess_ocp_plan(c, plan)
    assert not result.accepted and result.status == status
    assert result.alpha_nodes is None and result.arrays is None
    assert result.diagnostics['proof_status'] == 'UNKNOWN_NOT_CERTIFIED'


@pytest.mark.parametrize('remaining,future_checks', [(3., 0), (6., 1), (15., 4)])
def test_mission_end_truncates_entire_witness(setup, remaining, future_checks):
    safety, c, plan, calls, _ = setup
    c['t'] = 1200.-remaining
    plan.times = np.arange(int(remaining/.5)+1)*.5
    plan.alpha_nodes = .05+plan.times*.001
    result = safety.assess_ocp_plan(c, plan)
    assert result.accepted
    assert [t[-1] for _, t in calls] == [remaining, 3., remaining]
    assert result.diagnostics['future_horizon_s'] == remaining-3.
    assert len(result.diagnostics['generator_checks']) == future_checks


@pytest.mark.parametrize('change,status', [
    ('failed', 'OCP_SOLVE_NOT_SUCCESSFUL'),
    ('short', 'OCP_PLAN_HORIZON_MISMATCH'),
    ('discontinuous', 'OCP_PLAN_DISCONTINUOUS_ENTRY'),
    ('nonfinite', 'OCP_PLAN_NUMERICAL_OR_DOMAIN_FAILURE'),
    ('subhalfsecond', 'OCP_PLAN_UNSUPPORTED_EXECUTION_GRID')])
def test_bad_or_incompatible_plan_has_no_executable_prefix(setup, change, status):
    safety, c, plan, calls, _ = setup
    if change == 'failed':
        plan.success = False
    elif change == 'short':
        plan.times, plan.alpha_nodes = plan.times[:61], plan.alpha_nodes[:61]
    elif change == 'discontinuous':
        plan.alpha_nodes[0] += .001
    elif change == 'nonfinite':
        plan.alpha_nodes[-1] = np.nan
    else:
        plan.times = np.insert(plan.times, 1, .25)
        plan.alpha_nodes = np.insert(plan.alpha_nodes, 1, .05025)
    result = safety.assess_ocp_plan(c, plan)
    assert not result.accepted and result.status == status
    assert result.alpha_nodes is None and calls == []


def test_generator_change_rejects_static_geometry_tail_without_calling_it_unsafe(setup):
    safety, c, plan, _, _ = setup
    safety.db.generate = lambda mach, cl, current: dict(geometry_index=current+1)
    result = safety.assess_ocp_plan(c, plan)
    assert not result.accepted and result.status == 'OCP_TAIL_GENERATOR_INCOMPATIBLE'
    assert result.diagnostics['physical_plan_audit_passed'] is True
    assert result.diagnostics['universal_infeasibility_proven'] is False
    assert result.diagnostics['generator_checks'][0]['relative_time_s'] == 3.


def test_transition_lock_is_honored_until_original_eight_second_completion(setup):
    safety, c, plan, _, _ = setup
    c.update(source=0, target=1, fraction=0., rate=.125)
    safety.db.generate = lambda mach, cl, current: dict(geometry_index=2)
    result = safety.assess_ocp_plan(c, plan)
    checks = result.diagnostics['generator_checks']
    assert result.status == 'OCP_TAIL_GENERATOR_INCOMPATIBLE'
    assert [(v['relative_time_s'], v['transition_locked'], v['compatible']) for v in checks] == [
        (3., True, True), (6., True, True), (9., False, False)]


def test_hash_binds_tail_even_when_current_prefix_is_identical(setup):
    safety, c, plan, _, _ = setup
    first = safety.assess_ocp_plan(c, plan)
    plan.alpha_nodes[-1] += .001
    second = safety.assess_ocp_plan(c, plan)
    assert first.accepted and second.accepted
    assert np.array_equal(first.alpha_nodes, second.alpha_nodes)
    assert first.diagnostics['plan_sha256'] != second.diagnostics['plan_sha256']


def test_hash_binds_current_cl_and_actual_entry_context(setup):
    safety, c, plan, _, _ = setup
    first = safety.assess_ocp_plan(c, plan)
    c['CL_raw'] = .03
    second = safety.assess_ocp_plan(c, plan)
    assert first.accepted and second.accepted
    assert first.diagnostics['plan_sha256'] != second.diagnostics['plan_sha256']


def test_budget_exhaustion_never_returns_a_candidate(setup, monkeypatch):
    safety, c, plan, _, _ = setup
    clock = [0.]
    original = safety.audit

    def audit(*args):
        clock[0] = 31.
        return original(*args)

    monkeypatch.setattr('continuous_surrogate.safety.time.perf_counter', lambda: clock[0])
    safety.audit = audit
    result = safety.assess_ocp_plan(c, plan)
    assert not result.accepted and result.status == 'SEARCH_BUDGET_EXHAUSTED'
    assert result.alpha_nodes is None and result.diagnostics['budget_exceeded']
