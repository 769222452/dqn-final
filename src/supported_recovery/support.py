"""Exact, version-bound evidence for proposed recovery-policy actions.

This gate is an evidence lookup, never a safety certificate. A supported request
still goes through the existing fresh certification and transactional execution.
There is deliberately no uncalibrated nearest-neighbour fallback.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from certified_flight import CL_ACTIONS
from certified_flight.transition import Transition

CONTEXT_PROTOCOL = 'supported-recovery-context/1'
LABEL_PROTOCOL = 'supported-recovery-label-support/1'
CONTINUATION_PROTOCOL = 'supported-recovery-continuation/1'
ACTION_SEMANTICS = 'certified_policy_request'
_STATE_FIELDS = frozenset(('x', 'alpha', 't', 'transition', 'reward', 'switches',
                          'history_valid', 'previous_cl_raw', 'previous_cl_cmd', 'previous_cl_actual'))


def _json(value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(f'Unsupported support value: {type(v).__name__}')
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def canonical_hash(value):
    """Same finite-JSON encoding as the existing digest; reject non-finite data."""
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prefix_chain_hash(chain):
    """Hash exactly the supplied predecessor chain, excluding the proposed stage."""
    return canonical_hash(chain)


def _action(action):
    if type(action) is not int or not 0 <= action < len(CL_ACTIONS):
        raise ValueError('An absolute nineteen-action index is required')
    return action


def continuation_identity(reference_chain, mode='closed-loop', schedule=None):
    """Return a self-checking continuation contract shared by labels and stages."""
    if mode not in ('closed-loop', 'open-loop'):
        raise ValueError('Unsupported continuation mode')
    if mode == 'closed-loop' and schedule is not None:
        raise ValueError('Closed-loop continuation cannot carry a replay schedule')
    if mode == 'open-loop' and schedule is None:
        raise ValueError('Open-loop continuation requires its actual replay schedule')
    normalized = None
    if schedule is not None:
        pairs = schedule.items() if isinstance(schedule, dict) else schedule
        normalized = []
        seen = set()
        for time, action in pairs:
            time = float(time)
            if not np.isfinite(time) or time < 0 or abs(time/12.-round(time/12.)) > 1e-9:
                raise ValueError('Continuation schedule must use twelve-second boundaries')
            if time in seen:
                raise ValueError('Duplicate continuation decision time')
            seen.add(time)
            normalized.append([time, _action(action)])
        normalized.sort()
    payload = dict(protocol=CONTINUATION_PROTOCOL,
                   reference_prefix_sha256=prefix_chain_hash(reference_chain),
                   mode=mode, schedule=normalized)
    return dict(payload, sha256=canonical_hash(payload))


def _continuation_errors(stage, actual_prefix):
    identity = stage.get('continuation')
    if not isinstance(identity, dict):
        return ['CONTINUATION_METADATA_MISSING']
    payload = {k: v for k, v in identity.items() if k != 'sha256'}
    if (set(payload) != {'protocol', 'reference_prefix_sha256', 'mode', 'schedule'}
            or payload.get('protocol') != CONTINUATION_PROTOCOL
            or payload.get('mode') not in ('closed-loop', 'open-loop')):
        return ['CONTINUATION_METADATA_INVALID']
    if (payload['mode'] == 'closed-loop' and payload['schedule'] is not None
            or payload['mode'] == 'open-loop' and payload['schedule'] is None):
        return ['CONTINUATION_METADATA_INVALID']
    if (identity.get('sha256') != canonical_hash(payload)
            or stage.get('continuation_sha256') != identity.get('sha256')):
        return ['CONTINUATION_HASH_MISMATCH']
    if payload['reference_prefix_sha256'] != actual_prefix:
        return ['CONTINUATION_PREFIX_MISMATCH']
    return []


def support_context(snapshot_or_env):
    """Extract all state that the current controller may use for future control.

    ``TrainingFlight.snapshot`` is the sole source. RecoveryController.prepare /
    decide consume physical state and controller memory. TrainingCertifier.commit
    and execution.commit_certified only accumulate reward and switch counts;
    these two totals do not affect control, mission reward increments or outcome.
    Everything else, including recovery-plan provenance/epoch/exit confirmations,
    identity and live certification conditions, is retained. Unknown state fields
    fail closed so a future controller field cannot silently be omitted.
    """
    snapshot = snapshot_or_env.snapshot() if hasattr(snapshot_or_env, 'snapshot') else snapshot_or_env
    snapshot = _json(snapshot)
    if snapshot.get('protocol') != 'continuous-recovery-training-flight/1':
        raise ValueError('A TrainingFlight snapshot is required for exact support')
    state = snapshot['state']
    if set(state) != _STATE_FIELDS:
        raise ValueError('Unknown or missing physical support-context fields')
    identity = snapshot['identity']
    memory = snapshot['controller_memory']
    if not identity or not isinstance(memory, dict) or memory.get('identity') != identity:
        raise ValueError('Runtime and recovery-controller identities must agree')
    if not {'memory', 'epoch', 'exit_passes'} <= memory.keys():
        raise ValueError('Incomplete recovery controller memory')
    conditions = snapshot['certification_conditions']
    if not isinstance(conditions, dict):
        raise ValueError('Certification conditions must be explicit')
    x = np.asarray(state['x'], float)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError('A finite complete physical state is required')
    if type(state['history_valid']) is not bool:
        raise ValueError('History validity must be explicit')
    t = float(state['t'])
    z = Transition(**state['transition'])
    if (type(z.source) is not int or type(z.target) is not int
            or z.source < 0 or z.target < 0 or z.duration != 8.
            or not np.isfinite(z.started) or z.started > t):
        raise ValueError('Invalid eight-second geometry transition')
    z.settle(t)
    physical = dict(x=x.tolist(), alpha=float(state['alpha']), t=t,
                    transition=dict(source=z.source, target=z.target,
                                    fraction=float(z.fraction(t)),
                                    rate=.125 if z.active(t) else 0., duration_s=8.),
                    history_valid=state['history_valid'])
    for key in ('previous_cl_raw', 'previous_cl_cmd', 'previous_cl_actual'):
        physical[key] = None if state[key] is None else float(state[key])
    context = dict(protocol=CONTEXT_PROTOCOL, runtime_identity=identity,
                   state=physical, controller_memory=memory,
                   certification_conditions=conditions)
    return _json(context)


def context_hash(snapshot_or_env):
    return canonical_hash(support_context(snapshot_or_env))


def build_support_label(snapshot_or_env, *, reference_prefix_sha256, continuation_sha256,
                        base_action, candidate_action, valid=True, label_kind='branch', advantage=None):
    """Bind one measured action label; a synthetic base zero supports only itself."""
    _action(base_action)
    _action(candidate_action)
    if type(valid) is not bool or label_kind not in ('branch', 'baseline_zero'):
        raise ValueError('Explicit label validity and known label kind required')
    if label_kind == 'baseline_zero' and candidate_action != base_action:
        raise ValueError('A baseline-zero label cannot support another action')
    if not reference_prefix_sha256 or not continuation_sha256:
        raise ValueError('Reference prefix and continuation identities are required')
    if advantage is not None and not np.isfinite(float(advantage)):
        raise ValueError('Measured advantage must be finite when supplied')
    if label_kind == 'baseline_zero' and advantage not in (None, 0.):
        raise ValueError('A baseline-zero label must have zero advantage')
    context = support_context(snapshot_or_env)
    label = dict(protocol=LABEL_PROTOCOL, action_semantics=ACTION_SEMANTICS,
                context=context, context_sha256=canonical_hash(context),
                reference_prefix_sha256=reference_prefix_sha256,
                runtime_identity=deepcopy(context['runtime_identity']),
                continuation_sha256=continuation_sha256,
                base_action=base_action, candidate_action=candidate_action,
                valid=valid, label_kind=label_kind)
    if advantage is not None:
        label['advantage'] = float(advantage)
    return label


def validate_support_mode(mode):
    if mode != 'exact':
        raise ValueError('Only exact support is enabled; local support has no validated calibration')


def support_decision(stage, context, *, reference_prefix_sha256, base_action, candidate_action):
    """Return an auditable candidate-specific decision without modifying evidence."""
    validate_support_mode(stage.get('support_mode', 'exact'))
    _action(base_action)
    _action(candidate_action)
    stage_errors = []
    if stage.get('action_semantics') != ACTION_SEMANTICS:
        stage_errors.append('ACTION_SEMANTICS_MISMATCH')
    if stage.get('reference_prefix_sha256') != reference_prefix_sha256:
        stage_errors.append('REFERENCE_PREFIX_MISMATCH')
    if stage.get('runtime_identity') != context['runtime_identity']:
        stage_errors.append('RUNTIME_IDENTITY_MISMATCH')
    stage_errors.extend(_continuation_errors(stage, reference_prefix_sha256))
    common = dict(action=candidate_action, base_action=base_action, exact_support=False,
                  context_sha256=canonical_hash(context), matched_label_indices=[])
    if stage_errors:
        return dict(common, supported=False, reasons=stage_errors, reason=stage_errors[0])
    rejected, matches, action_records = set(), [], 0
    for index, label in enumerate(stage.get('support_records', [])):
        if label.get('candidate_action') != candidate_action:
            continue
        action_records += 1
        reasons = []
        if label.get('protocol') != LABEL_PROTOCOL or label.get('valid') is not True:
            reasons.append('LABEL_NOT_VALID')
        if label.get('label_kind') not in ('branch', 'baseline_zero'):
            reasons.append('LABEL_KIND_INVALID')
        if label.get('label_kind') == 'baseline_zero' and label.get('base_action') != candidate_action:
            reasons.append('BASELINE_ZERO_OTHER_ACTION')
        for key, expected, reason in (
            ('action_semantics', ACTION_SEMANTICS, 'LABEL_ACTION_SEMANTICS_MISMATCH'),
            ('reference_prefix_sha256', reference_prefix_sha256, 'LABEL_REFERENCE_PREFIX_MISMATCH'),
            ('runtime_identity', context['runtime_identity'], 'LABEL_RUNTIME_IDENTITY_MISMATCH'),
            ('continuation_sha256', stage['continuation_sha256'], 'LABEL_CONTINUATION_MISMATCH'),
            ('base_action', base_action, 'LABEL_BASE_ACTION_MISMATCH'),
        ):
            if label.get(key) != expected:
                reasons.append(reason)
        if not isinstance(label.get('context'), dict) or label.get('context_sha256') != canonical_hash(label['context']):
            reasons.append('LABEL_CONTEXT_HASH_INVALID')
        if label.get('context_sha256') != common['context_sha256'] or label.get('context') != context:
            reasons.append('EXACT_CONTEXT_MISMATCH')
        if reasons:
            rejected.update(reasons)
        else:
            matches.append(index)
    if matches:
        advantages = [stage['support_records'][i]['advantage'] for i in matches
                      if 'advantage' in stage['support_records'][i]]
        result = dict(common, supported=True, exact_support=True, reasons=['EXACT_SUPPORT'],
                      reason='EXACT_SUPPORT', matched_label_indices=matches)
        if advantages:
            result['measured_advantages'] = advantages
            if len(set(advantages)) == 1:
                result['measured_advantage'] = advantages[0]
            else:
                result['measured_advantage_conflict'] = True
        return result
    reasons = sorted(rejected) if action_records else ['NO_CANDIDATE_ACTION_LABEL']
    return dict(common, supported=False, reasons=reasons, reason=reasons[0])
