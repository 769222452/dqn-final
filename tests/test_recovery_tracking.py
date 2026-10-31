"""Orthogonal tests for tracking-priority P1 selection and fail-closed timing."""
from types import SimpleNamespace

import numpy as np
import pytest
import casadi as ca

import continuous_recovery.tracking as module
from continuous_recovery.tracking import TrackingPrioritySafety
from continuous_surrogate.safety import ContinuousSafety


@pytest.fixture
def guard():
    obj = TrackingPrioritySafety.__new__(TrackingPrioritySafety)
    obj.correction_budget = 2.
    obj.lo, obj.hi, obj.scale, obj.rate = -.1, .3, 1., .1
    obj.cfg = dict(max_iterations=60, ftol=1e-12)
    obj.db = SimpleNamespace(parameters=np.zeros((1, 2)))
    obj.quality_tolerance, obj.mean_tolerance, obj.peak_tolerance = .005, .0025, .01
    obj.primary_floor = 0.
    obj.baseline_shift = 0.
    obj.reject_grid = None
    obj.audit_calls = []

    def candidate(context, return_diagnostics=False):
        a = np.r_[context['alpha_prev'], np.full(6, .1+obj.baseline_shift)]
        return (a, {}) if return_diagnostics else a

    def evaluate(z, p):
        error = z-.1
        primary = obj.primary_floor+float(np.mean(error**2))
        dj = error/3
        reference = p[6:12]
        diff = z-reference
        return primary, dj, float(np.sum(diff**2)), 2*diff, np.ones(1), np.zeros((1, 6))

    def audit(context, a):
        obj.audit_calls.append(a.copy())
        error = float(np.max(np.abs(a[1:]-.1)))
        # A deterministic stand-in for two independently integrated grids:
        # even a perfect tracking candidate cannot override a grid rejection.
        rate_ok = bool(np.max(np.abs(np.diff(a))/.5) <= obj.rate+1e-12)
        physical_ok = bool(np.min(a) >= obj.lo and np.max(a) <= obj.hi and rate_ok)
        grids = {name: dict(valid=physical_ok and name != obj.reject_grid,
                    CL_mean_first3=.025+error, CL_rmse_first3=error,
                    CL_peak_error_first3=error)
                 for name in ('0.1', '0.5')}
        proof = dict(valid=all(v['valid'] for v in grids.values()), grids=grids,
                     CL_target=.025, maximum_alpha_rate_degps=np.rad2deg(np.max(np.abs(np.diff(a))/.5)))
        return proof, {name: dict(x=np.zeros((6, 4))) for name in grids}

    obj.candidate, obj.priority_function, obj.audit = candidate, evaluate, audit
    return obj


@pytest.fixture
def context():
    return dict(x=[40000., 2400., -.01, 0.], alpha_prev=.1, CL_des=.025,
                fraction=1., rate=0., source=0, target=0)


def proposal(error):
    return np.r_[.1, np.full(6, .1+error)]


def test_tracking_diagnostic_threshold_has_no_selection_jump(guard, context):
    below = guard.correct(context, proposal(.004999))
    above = guard.correct(context, proposal(.005001))
    assert below.accepted and above.accepted
    assert below.diagnostics['path'] == above.diagnostics['path'] == 'NEAR_ZERO_LINE_CANDIDATE_PROJECTION'
    assert not below.diagnostics['quality_gate_used_for_selection']
    # The old .005 error threshold is crossed by the raw proposal. Both
    # outputs instead obey the same continuous primary-optimality allowance.
    assert np.max(abs(below.alpha_nodes-above.alpha_nodes)) < 1e-8
    assert below.alpha_nodes[0] == above.alpha_nodes[0] == context['alpha_prev']


@pytest.mark.parametrize('error', [.00001, .002499, .002501, .009999, .010001, -.03])
def test_primary_budget_is_enforced_on_every_accepted_candidate(guard, context, error):
    result = guard.correct(context, proposal(error))
    assert result.accepted
    diag = result.diagnostics
    assert diag['primary_selected'] <= diag['primary_limit']+1e-14
    assert diag['primary_limit'] == pytest.approx(diag['primary_best']*1.01+1e-8)
    assert result.alpha_nodes[0] == context['alpha_prev']
    assert result.audit['valid']
    assert not diag['global_optimality_claimed']
    assert len(guard.audit_calls) >= 2


def test_immutable_measured_alpha_cannot_be_projected_away(guard, context):
    a = proposal(.01)
    a[0] += .0001
    original = a.copy()
    result = guard.correct(context, a)
    assert not result.accepted
    assert result.status == 'TRACKING_NUMERICAL_OR_DOMAIN_FAILURE'
    assert not guard.audit_calls
    np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize('grid', ['0.1', '0.5'])
def test_either_existing_physical_audit_grid_can_reject(guard, context, grid):
    guard.reject_grid = grid
    result = guard.correct(context, proposal(.01))
    assert not result.accepted and result.alpha_nodes is None
    assert result.diagnostics['proof_status'] == 'UNKNOWN_NOT_CERTIFIED'
    assert guard.audit_calls


def test_numerical_failure_does_not_release_earlier_safe_candidate(guard, context):
    original = guard.priority_function
    count = 0

    def fail_after_baseline(z, p):
        nonlocal count
        count += 1
        output = list(original(z, p))
        if count >= 2:
            output[0] = np.nan
        return output

    guard.priority_function = fail_after_baseline
    result = guard.correct(context, proposal(.01))
    assert not result.accepted
    assert result.status == 'TRACKING_NUMERICAL_OR_DOMAIN_FAILURE'
    assert result.diagnostics['attempts'][0]['audit']['valid']


def test_late_independent_audit_cannot_release_a_candidate(guard, context, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    original = guard.audit

    def delayed_audit(c, a):
        proof = original(c, a)
        clock[0] += 3.
        return proof

    guard.audit = delayed_audit
    result = guard.correct(context, proposal(.01))
    assert not result.accepted
    assert result.status == 'TRACKING_CORRECTION_BUDGET_EXHAUSTED'
    assert result.diagnostics['budget_exceeded']


def test_late_candidate_generation_counts_against_budget(guard, context, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    original = guard.candidate

    def delayed_candidate(*args, **kwargs):
        clock[0] = 2.01
        return original(*args, **kwargs)

    guard.candidate = delayed_candidate
    result = guard.correct(context, proposal(.01))
    assert not result.accepted and not guard.audit_calls
    assert result.status == 'TRACKING_CORRECTION_BUDGET_EXHAUSTED'


def test_nonzero_unavoidable_error_uses_bounded_two_stage_optimization(guard, context):
    guard.primary_floor = .001
    guard.baseline_shift = .03
    result = guard.correct(context, proposal(.04))
    assert result.accepted
    assert result.diagnostics['path'] == 'TWO_STAGE_CONSTRAINED_OPTIMIZATION'
    assert [v['stage'] for v in result.diagnostics['optimizer']] == [1, 2]
    assert result.diagnostics['primary_selected'] <= result.diagnostics['primary_limit']+1e-14
    assert result.diagnostics['cache_hits'] > 0


def test_solver_failure_is_unknown_even_if_baseline_is_physically_safe(guard, context, monkeypatch):
    guard.primary_floor = .001
    monkeypatch.setattr(module, 'minimize', lambda *args, **kwargs:
                        SimpleNamespace(success=False, status=9, message='Iteration limit', nit=60))
    result = guard.correct(context, proposal(.01))
    assert not result.accepted
    assert result.status == 'TRACKING_PRIMARY_OPTIMIZER_NOT_CONVERGED'
    assert result.diagnostics['attempts'][0]['audit']['valid']


def test_actual_physical_limits_remain_shared_and_not_relaxed(guard, context):
    # Deliberately invalid alpha/rate cannot be accepted from the proposal.
    # A corrected valid witness may be returned, but no failed audit is reused.
    result = guard.correct(context, proposal(.25))
    assert result.accepted
    assert np.max(result.alpha_nodes) <= guard.hi
    assert np.max(abs(np.diff(result.alpha_nodes))/.5) <= guard.rate
    assert not result.diagnostics['attempts'][1]['audit']['valid']
    assert result.audit['valid']


def test_symbolic_physical_constraints_equal_old_p1_and_tracking_is_pointwise(monkeypatch):
    # An exact toy RK signature lets us compare symbolic constraints without
    # database loading, generated C compilation, an OCP solve or a flight.
    obj = TrackingPrioritySafety.__new__(TrackingPrioritySafety)
    obj.scale, obj.clscale, obj.rate = .05, .1, .1
    obj.wn, obj.ws, obj.wcl = np.ones(6), .01, .05
    obj.db = SimpleNamespace(ncoef=1, M=np.array([3.5, 8.25]))
    obj.limits = dict(altitude_m=[20000., 80000.], velocity_mps=[800., 6000.],
                      gamma_rad=[-.35, .35], dynamic_pressure_pa_max=80000.,
                      heat_proxy_wm2_max=2e6, load_factor_g_max=3.)

    def rk(x, a, slew, fraction, morph_rate, c0, c1, dt, tau):
        stages = ca.DM([[0., .5, .5, 1., 1.]])
        angles = a+slew*dt*stages
        states = ca.repmat(x, 1, 5)
        ss = ca.vertcat(states, ca.DM.ones(3, 5), 5.*ca.DM.ones(1, 5),
                        angles, ca.DM.zeros(3, 5))
        return x, ss

    obj.plant = SimpleNamespace(rk=rk)
    monkeypatch.setattr(module.codegen, 'compile_function', lambda fn: fn)
    fresh = obj._build_priority_function()
    old = ContinuousSafety._build_p1(obj)
    nodes = np.array([.05, -.05, .05, -.05, .05, -.05, .05])
    p = np.r_[[40000., 2400., 0., 0.], nodes[0], 0., nodes[1:], 1., 0., 0., 0.]
    fv, ov = fresh(nodes[1:]/obj.scale, p), old(nodes[1:]/obj.scale, p)
    np.testing.assert_array_equal(np.asarray(fv[4]), np.asarray(ov[2]))
    np.testing.assert_array_equal(np.asarray(fv[5]), np.asarray(ov[3]))
    # Every linear segment has zero mean CL, yet its pointwise error is >0.
    assert float(fv[0]) == pytest.approx(1/12)
    assert float(fv[2]) == pytest.approx(float(ov[0]))
