"""Serializable recovery controls, re-certified from the current measured state.

Memory stores controls and provenance only.  It never stores an executable PASS:
each use re-integrates the complete remaining plan on both independent grids and
checks the full required future horizon, including the original Generator policy.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from certified_flight.transition import Transition
from continuous_surrogate.safety import SafetyResult
from continuous_tracking.plant import prefix, validate_plan

PROTOCOL = 'continuous-recovery-plan/1'
WITNESS_PROTOCOL = 'continuous-recovery-tail-witness/1'


def _json(value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(f'Not JSON serializable: {type(v).__name__}')
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _sha(value):
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _physical_context(context):
    keys = ('x', 'alpha_prev', 't', 'source', 'target', 'fraction', 'rate', 'CL_des')
    out = {key: deepcopy(context[key]) for key in keys}
    out['CL_raw'] = float(context.get('CL_raw', context['CL_des']))
    return _json(out)


def _transition(context):
    fraction, rate = float(context['fraction']), float(context['rate'])
    source, target = int(context['source']), int(context['target'])
    if not 0. <= fraction <= 1. or rate not in (0., .125):
        raise ValueError('Invalid eight-second transition')
    active = source != target and fraction < 1.
    if rate != (.125 if active else 0.):
        raise ValueError('Transition fraction and rate disagree')
    if source == target and fraction != 1.:
        raise ValueError('Settled geometry requires unit blend fraction')
    z = Transition(source, target, float(context['t'])-8.*fraction, 8.)
    z.settle(float(context['t']))
    return z


def create_recovery_plan(plan, context, identity, epoch=0):
    """Create JSON memory from successful teacher controls, not stored states.

    Solver success is only provenance.  ``certify_tail`` must subsequently pass
    before any prefix may execute, including on the plan's first use.
    """
    if not bool(plan.success):
        raise ValueError('Cannot create recovery memory from unsuccessful solver')
    nodes, times = validate_plan(plan.alpha_nodes, plan.times)
    if nodes[0] != float(context['alpha_prev']):
        raise ValueError('Recovery plan does not start at measured alpha')
    if np.max(abs(times/.5-np.round(times/.5))) > 1e-8:
        raise ValueError('Recovery plan breakpoints must align with the audit grid')
    z = _transition(context)
    origin = _physical_context(context)
    controls = dict(times_s=times.tolist(), alpha_nodes_rad=nodes.tolist())
    memory = dict(protocol=PROTOCOL, epoch=int(epoch), identity=_json(identity),
                  started_at_s=float(context['t']), ends_at_s=float(context['t'])+float(times[-1]),
                  request_CL=origin['CL_raw'], origin_context=origin,
                  origin_transition=_json(vars(z)), controls=controls,
                  source=dict(kind='REAL_CONTINUOUS_OCP', solver_status=str(getattr(plan, 'status', 'SUCCESS')),
                              controls_sha256=_sha(controls), context_sha256=_sha(origin)),
                  executable_certificate_stored=False)
    memory['payload_sha256'] = _sha(memory)
    validate(memory)
    return memory


def validate(memory):
    """Validate a checkpoint-restored memory and its provenance hash."""
    if not isinstance(memory, dict) or memory.get('protocol') != PROTOCOL:
        raise ValueError('Incompatible recovery memory protocol')
    payload = deepcopy(memory)
    expected = payload.pop('payload_sha256', None)
    if expected != _sha(payload):
        raise ValueError('Recovery memory payload hash mismatch')
    if memory.get('executable_certificate_stored') is not False:
        raise ValueError('Recovery memory cannot carry an executable certificate')
    nodes, times = validate_plan(memory['controls']['alpha_nodes_rad'], memory['controls']['times_s'])
    if np.max(abs(times/.5-np.round(times/.5))) > 1e-8:
        raise ValueError('Unsupported recovery control grid')
    if memory['ends_at_s'] != memory['started_at_s']+float(times[-1]):
        raise ValueError('Inconsistent absolute recovery horizon')
    if memory['source']['controls_sha256'] != _sha(memory['controls']):
        raise ValueError('Recovery control provenance mismatch')
    if memory['source']['context_sha256'] != _sha(memory['origin_context']):
        raise ValueError('Recovery origin provenance mismatch')
    if memory['started_at_s'] != memory['origin_context']['t']:
        raise ValueError('Recovery origin time mismatch')
    if memory['request_CL'] != memory['origin_context']['CL_raw']:
        raise ValueError('Recovery request provenance mismatch')
    if nodes[0] != memory['origin_context']['alpha_prev']:
        raise ValueError('Recovery initial control provenance mismatch')
    if _json(vars(_transition(memory['origin_context']))) != memory['origin_transition']:
        raise ValueError('Recovery transition provenance mismatch')
    return True


def compatible(context, memory, identity, mission_duration, horizon=30.):
    """Check reuse eligibility; success here is explicitly not a certificate."""
    detail = dict(physical_infeasibility_proven=False, certificate_reused=False)
    try:
        validate(memory)
        if _sha(memory['identity']) != _sha(identity):
            return False, 'RECOVERY_IDENTITY_CHANGED', detail
        if 'recovery_epoch' in context and int(context['recovery_epoch']) != memory['epoch']:
            return False, 'RECOVERY_EPOCH_CHANGED', detail
        raw = float(context.get('CL_raw', context['CL_des']))
        if raw != memory['request_CL']:
            return False, 'RECOVERY_CL_REQUEST_CHANGED', detail
        now = float(context['t'])
        elapsed = now-memory['started_at_s']
        remaining = float(mission_duration)-now
        tail = memory['ends_at_s']-now
        required = min(3.+float(horizon), remaining)
        detail.update(elapsed_s=elapsed, remaining_mission_s=remaining,
                      remaining_plan_s=tail, required_witness_s=required)
        if elapsed < 0. or remaining <= 0.:
            return False, 'RECOVERY_TIME_OUTSIDE_TASK', detail
        if tail+1e-9 < required:
            return False, 'RECOVERY_HORIZON_INSUFFICIENT', detail
        if abs(elapsed/.5-round(elapsed/.5)) > 1e-8:
            return False, 'RECOVERY_UNSUPPORTED_EXECUTION_GRID', detail
        expected = Transition(**memory['origin_transition'])
        expected.settle(now)
        actual = _transition(context)
        if (actual.source != expected.source or actual.target != expected.target
                or abs(actual.fraction(now)-expected.fraction(now)) > 1e-10
                or actual.active(now) != expected.active(now)):
            detail.update(expected_transition=vars(expected), actual_transition=vars(actual))
            return False, 'RECOVERY_GEOMETRY_CHANGED', detail
        a = np.interp(elapsed, memory['controls']['times_s'], memory['controls']['alpha_nodes_rad'])
        detail['entry_alpha_difference_rad'] = float(context['alpha_prev'])-float(a)
        # Exact matching is intentional: even an apparently harmless snap can
        # hide a different executed control. No cached certificate is sufficient.
        if float(a) != float(context['alpha_prev']):
            return False, 'RECOVERY_ALPHA_ENTRY_CHANGED', detail
        return True, 'RECOVERY_CONTEXT_COMPATIBLE', detail
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        detail['error'] = repr(exc)
        return False, 'RECOVERY_MEMORY_INVALID', detail


def shift_plan(context, memory, duration=None):
    """Shift controls, retaining all breakpoints and adding half-second nodes."""
    validate(memory)
    elapsed = float(context['t'])-memory['started_at_s']
    old_t = np.asarray(memory['controls']['times_s'], float)
    old_a = np.asarray(memory['controls']['alpha_nodes_rad'], float)
    last = float(old_t[-1]-elapsed)
    if duration is not None:
        last = min(last, float(duration))
    if elapsed < 0. or last <= 0.:
        raise ValueError('Current time outside recovery controls')
    shifted = old_t[(old_t > elapsed) & (old_t < elapsed+last)]-elapsed
    grid = np.arange(0., last+1e-9, .5)
    times = np.unique(np.r_[0., shifted, grid, last])
    if np.max(abs(times/.5-np.round(times/.5))) > 1e-8:
        raise ValueError('Shifted recovery breakpoints do not align with audit grid')
    nodes = np.interp(times+elapsed, old_t, old_a)
    if nodes[0] != float(context['alpha_prev']):
        raise ValueError('Cannot snap recovery plan to different measured alpha')
    nodes[0] = float(context['alpha_prev'])
    return nodes, times, dict(elapsed_s=elapsed, shifted_duration_s=last,
                             entry_reset_difference_rad=0., re_integration_required=True)


def certify_tail(guard, context, memory, identity, pre_mach_callback):
    """Re-audit recovery controls at this state; only the current prefix executes.

    ``pre_mach_callback(future_context)`` is the same Pre path as the live loop.
    It returns a Mach scalar or ``(Mach, diagnostics)``. The OCP's stored states
    and old certificate are never used to establish the new witness.
    """
    started = time.perf_counter()
    deadline = started+float(guard.budget)
    diag = dict(witness_protocol=WITNESS_PROTOCOL, source='REUSED_AUDITED_OCP_TAIL',
                memory_sha256=memory.get('payload_sha256') if isinstance(memory, dict) else None,
                budget_s=float(guard.budget), horizon_s=float(guard.horizon),
                certificate_reused=False, re_integrated_from_current_state=True,
                universal_infeasibility_proven=False, generator_checks=[], p1_applied=False)

    def expired():
        return time.perf_counter() > deadline

    def finish(status, nodes=None, times=None, audit=None, arrays=None):
        accepted = nodes is not None and not expired()
        if nodes is not None and not accepted:
            status = 'SEARCH_BUDGET_EXHAUSTED'
        diag.update(status=status, proof_status='VERIFIED_PASS' if accepted else 'UNKNOWN_NOT_CERTIFIED',
                    wall_s=time.perf_counter()-started, budget_exceeded=expired(),
                    future_candidate_witness=accepted)
        if accepted:
            diag.update(quality=guard._quality(audit), safety_intervention=True,
                        selected_source='REUSED_AUDITED_OCP_TAIL', selected_policy='EXACT_OCP_TAIL',
                        selected_alpha_nodes_rad=nodes.tolist())
        return SafetyResult(accepted, status, nodes.copy() if accepted else None,
                            times.copy() if accepted else None, audit if accepted else None,
                            arrays if accepted else None, diag)

    try:
        ok, reason, detail = compatible(context, memory, identity, guard.duration, guard.horizon)
        diag['compatibility'] = detail
        if not ok:
            return finish(reason)
        remaining = float(guard.duration)-float(context['t'])
        nodes, times, shifted = shift_plan(context, memory, remaining)
        diag['shift'] = shifted
        binding = dict(protocol=WITNESS_PROTOCOL, memory_sha256=memory['payload_sha256'],
                       identity=_json(identity), context=_physical_context(context),
                       alpha_nodes_rad=nodes.tolist(), times_s=times.tolist())
        diag.update(plan_binding=binding, plan_sha256=_sha(binding))
        full_audit, _ = guard.audit(context, nodes, times)
        diag['full_remaining_plan_audit'] = full_audit
        if expired():
            return finish('SEARCH_BUDGET_EXHAUSTED')
        if not full_audit['valid']:
            return finish('RECOVERY_FULL_TAIL_AUDIT_REJECTED')
        current_duration = min(3., remaining)
        current_nodes, current_times = prefix(nodes, times, current_duration)
        current_audit, current_arrays = guard.audit(context, current_nodes, current_times)
        diag['current_3s_audit'] = current_audit
        if expired():
            return finish('SEARCH_BUDGET_EXHAUSTED')
        if not current_audit['valid']:
            return finish('RECOVERY_CURRENT_AUDIT_REJECTED')
        duration = min(3.+float(guard.horizon), remaining)
        witness_nodes, witness_times = prefix(nodes, times, duration)
        witness_audit, witness_arrays = guard.audit(context, witness_nodes, witness_times)
        diag.update(full_witness_audit=witness_audit, witness_duration_s=duration,
                    future_horizon_s=max(0., duration-current_duration),
                    witness_alpha_nodes_rad=witness_nodes.tolist(), witness_times_s=witness_times.tolist())
        if expired():
            return finish('SEARCH_BUDGET_EXHAUSTED')
        if not witness_audit['valid']:
            return finish('RECOVERY_WITNESS_AUDIT_REJECTED')
        actual = witness_arrays['0.5']
        z = _transition(context)
        raw = float(memory['request_CL'])
        for tau in np.arange(3., duration-1e-9, 3.):
            if expired():
                return finish('SEARCH_BUDGET_EXHAUSTED')
            now = float(context['t'])+float(tau)
            z.settle(now)
            x = np.asarray(actual['x'][int(round(tau/.5))-1], float)
            alpha = float(np.interp(tau, times, nodes))
            reachable = guard.reachability.check(x, alpha, z, now, raw,
                                                enabled=guard.cfg.get('reachability_enabled', True))
            future = deepcopy(context)
            future.update(x=x.tolist(), alpha_prev=alpha, t=now, source=z.source, target=z.target,
                          fraction=z.fraction(now), rate=.125 if z.active(now) else 0.,
                          CL_raw=raw, CL_des=float(reachable['CL_cmd']), CD_des=None)
            prediction = pre_mach_callback(future)
            if isinstance(prediction, tuple):
                mach, pre_info = prediction
            else:
                mach, pre_info = prediction, {}
            mach = float(np.asarray(mach).reshape(-1)[0])
            if not np.isfinite(mach):
                return finish('RECOVERY_PRE_MACH_INVALID')
            if expired():
                return finish('SEARCH_BUDGET_EXHAUSTED')
            choice = guard.db.generate(mach, future['CL_des'], current=z.target)
            geometry = choice['geometry_index']
            locked = z.active(now)
            # The live Generator path holds the current geometry when equal-
            # lift selection returns None. Preserve that control semantics;
            # actual CL tracking remains a separate diagnostic from safety.
            same = bool(locked or geometry is None or int(geometry) == z.target)
            diag['generator_checks'].append(_json(dict(time_s=now, relative_time_s=float(tau),
                context=_physical_context(future), pre_mach=mach, pre_diagnostics=pre_info,
                pre_mach_definition='same_callback_as_live_pre', generator=choice,
                CL_reachability=reachable, transition_locked=locked, compatible=same,
                equal_lift_geometry_found=geometry is not None)))
            if not same:
                diag.update(physical_plan_audit_passed=True, generator_compatible=False,
                            rejection_meaning='Future Generator differs from planned geometry; no infeasibility claim')
                return finish('RECOVERY_GENERATOR_INCOMPATIBLE')
        if expired():
            return finish('SEARCH_BUDGET_EXHAUSTED')
        diag.update(physical_plan_audit_passed=True, generator_compatible=True,
                    accepted_witness=True, safe_duration_s=max(0., duration-current_duration))
        return finish('RECOVERY_TAIL_WITNESS_FOUND', current_nodes, current_times,
                      current_audit, current_arrays)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, RuntimeError,
            FloatingPointError, OverflowError) as exc:
        diag['error'] = repr(exc)
        return finish('RECOVERY_NUMERICAL_OR_DOMAIN_FAILURE')


class RecoveryPlan:
    """Convenience namespace; persisted instances remain plain JSON dictionaries."""
    create = staticmethod(create_recovery_plan)
    validate = staticmethod(validate)
    compatible = staticmethod(compatible)
    shift = staticmethod(shift_plan)

