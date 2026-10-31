"""Frozen support-gated candidates and paired, freshly certified continuations.

Both arms start from the same full flight/controller snapshot.  In particular,
an off-nominal state never borrows a nominal trajectory's cached remaining return.
The production backend still owns all certification, auditing and PASS commits.
"""
from copy import deepcopy
from pathlib import Path
import time
import traceback

import numpy as np

from certified_flight import CL_ACTIONS, OracleRequired
from continuous_surrogate.experiment import digest
from continuous_tracking.experiment import clean, write_json
from continuous_recovery.training_rollouts import candidate_diagnostics
from rollout_improvement.features import features
from rollout_improvement.policy import DECISION_S, INITIAL_ACTION
from rollout_improvement.rollouts import MISSION_S, mission_completed, _terminal_stop

PROTOCOL = 'supported-recovery-rollouts/1'
ACTION_SEMANTICS = 'certified_policy_request'


def build_training_flight(*args, **kwargs):
    from continuous_recovery.training_backend import build_training_flight as build
    return build(*args, **kwargs)


def make_policy(*args, **kwargs):
    from .policy import make_policy as make
    return make(*args, **kwargs)


def policy_action(*args, **kwargs):
    from .policy import policy_action as select
    return select(*args, **kwargs)


def snapshot_policy(policy):
    from continuous_recovery.training_backend import snapshot_policy as snapshot
    return snapshot(policy)


def _boundary(t):
    return bool(np.isfinite(t) and abs(t/DECISION_S-round(t/DECISION_S)) < 1e-9)


def _action(value, name):
    if type(value) is not int or not 0 <= value < len(CL_ACTIONS):
        raise ValueError(name+' must be an absolute action index')
    return value


def _limit(value):
    if value is not None and (type(value) is not int or not 1 <= value <= 400):
        raise ValueError('max_decisions must be an integer from 1 to 400')
    return value


def decision_binding(certifier, policy, runtime_spec):
    """Read-only selection and full origin evidence before a new 12 s decision."""
    env = certifier.env
    if not _boundary(float(env.t)):
        raise ValueError('Bindings require an exact twelve-second decision boundary')
    before = env.snapshot_hash()
    scheduler_before = snapshot_policy(policy)
    logs_before = deepcopy(policy.logs)
    action, selection = policy_action(policy.chain, env, record=True)
    if (env.snapshot_hash() != before or snapshot_policy(policy) != scheduler_before
            or policy.logs != logs_before):
        raise ValueError('Read-only policy selection changed flight, scheduler or support logs')
    state = clean(env.snapshot())
    return dict(protocol=PROTOCOL, t=float(env.t), G_prefix=float(env.total_reward),
        action=int(action), features=features(env).tolist(), state=state,
        state_sha256=before, policy_state=scheduler_before,
        policy_state_sha256=digest(scheduler_before), chain_sha256=digest(policy.chain),
        runtime_spec_sha256=digest(runtime_spec), selection=clean(selection))


def certified_fly(certifier, policy, *, max_decisions=None, boundary_callback=None):
    """Run the existing fresh-PASS transaction, retaining requests and executions."""
    maximum = _limit(max_decisions)
    env, total, stop = certifier.env, 0., None
    executed, requested, trace = [], [], []
    if env.done:
        return dict(G_sum=total, stop=_terminal_stop(env), executed=executed,
                    requested=requested, trace=trace)
    while True:
        if maximum is not None and len(executed) >= maximum:
            stop = 'MAX_DECISIONS'
            break
        t = float(env.t)
        if boundary_callback is not None and _boundary(t):
            boundary_callback()
        pref = policy.preference()
        request = int(np.argmax(pref))
        certifier.bind_policy_state(snapshot_policy(policy))
        pool = certifier.pool(pref)
        mask = np.asarray(pool['mask'], bool)
        if mask.shape != (len(CL_ACTIONS),):
            raise ValueError('Certification must retain all nineteen absolute actions')
        if not mask.any():
            stop = 'NEXT_STATE_NOT_CERTIFIED'
            trace.append(dict(t=t, requested_action=request, executed_action=None, committed=False))
            break
        action = int(np.argmax(np.where(mask, pref, -np.inf)))
        try:
            reward, done, _ = certifier.commit(pool, action)
        except OracleRequired as exc:
            stop = 'EXECUTION_REFUSED:'+exc.reason
            trace.append(dict(t=t, requested_action=request, executed_action=None, committed=False))
            break
        policy.record_execution(action, t=t)
        total += float(reward)
        requested.append(request)
        executed.append(action)
        trace.append(dict(t=t, requested_action=request, executed_action=action, committed=True))
        if done:
            stop = _terminal_stop(env)
            break
    return dict(G_sum=total, stop=stop, executed=executed, requested=requested, trace=trace)


def _result(folder, value):
    write_json(Path(folder)/'job_result.json', value)
    return value


def _identity(chain, runtime_spec):
    return dict(protocol=PROTOCOL, chain_sha256=digest(chain), runtime_spec_sha256=digest(runtime_spec))


def _validate_binding(binding, runtime_spec):
    if binding.get('runtime_spec_sha256') != digest(runtime_spec):
        raise ValueError('Origin runtime specification differs from paired runtime')
    state, scheduler = binding['state'], binding['policy_state']
    if binding.get('state_sha256') != digest(state):
        raise ValueError('Origin physical/controller snapshot hash mismatch')
    if binding.get('policy_state_sha256') != digest(scheduler):
        raise ValueError('Origin policy snapshot hash mismatch')
    if (scheduler.get('protocol') != 'recovery-training-policy-state/1'
            or not scheduler.get('chain_sha256')
            or binding.get('chain_sha256') != scheduler.get('chain_sha256')):
        raise ValueError('Origin policy identity mismatch')
    _action(binding.get('action'), 'origin proposed action')
    t = float(binding['t'])
    if not _boundary(t) or not 0 <= t < MISSION_S:
        raise ValueError('Paired origins require a twelve-second boundary before mission end')
    # Source scheduler belongs to the observed candidate, not necessarily the
    # reference.  Its held action ends at this boundary; it is provenance only.
    tick = scheduler.get('decision_t')
    history = scheduler.get('decisions')
    if not isinstance(history, list):
        raise ValueError('Origin policy decision history is missing')
    if scheduler.get('action') is None:
        if tick is not None or history:
            raise ValueError('Inconsistent undecided origin policy')
    else:
        _action(scheduler['action'], 'origin held action')
        if (tick is None or not _boundary(float(tick)) or not 0 <= float(tick) < t
                or not history or list(history[-1]) != [tick, scheduler['action']]):
            raise ValueError('Origin must precede its new boundary decision')
    previous = -1.
    for old_t, old_a in history:
        _action(old_a, 'origin historical action')
        if not _boundary(float(old_t)) or not previous < float(old_t) < t:
            raise ValueError('Invalid origin decision history')
        previous = float(old_t)
    return t


def _arm_result(env, policy, flown, start_reward, start_hash):
    remaining = float(env.total_reward)-start_reward
    if not np.isfinite(remaining) or abs(remaining-flown['G_sum']) > 1e-12:
        raise ValueError('Paired arm rewards disagree with committed flight state')
    completed = mission_completed(env, flown['stop'])
    first = [row for row in flown['trace'] if row['t'] < policy.override[0]+DECISION_S]
    return dict(G_rest=remaining, G_sum=flown['G_sum'], completed=completed,
        completed_1200s=completed, success=completed, time_s=float(env.t), stop=flown['stop'],
        outcome=env.last_outcome, initial_snapshot_sha256=start_hash,
        final_snapshot_sha256=env.snapshot_hash(), final_state=env.x.tolist(),
        final_policy_state=snapshot_policy(policy), requested_actions=flown['requested'],
        actual_executed_actions=flown['executed'], execution_trace=flown['trace'],
        action_semantics=ACTION_SEMANTICS, first_12s_trace=first,
        first_12s_requested_actions=[row['requested_action'] for row in first],
        first_12s_actual_executed_actions=[row['executed_action'] for row in first if row['committed']],
        first_12s_substitution_count=sum(row['requested_action'] != row['executed_action']
                                        for row in first if row['committed']),
        substitution_count=sum(row['requested_action'] != row['executed_action']
                               for row in flown['trace'] if row['committed']),
        decisions=clean(policy.decisions), support_logs=clean(policy.logs),
        steps=len(flown['executed']), failure_semantics=None if completed else 'UNKNOWN_NOT_CERTIFIED',
        universal_infeasibility_proven=False)


def paired_job(args):
    """Recompute both returns at an observed state under one frozen reference tail.

    Required: folder, runtime_spec, reference_chain, binding, base_action, action.
    Optional: continuation (closed-loop default), schedule, max_decisions.  An
    explicit bounded check never creates a valid training label, even if its
    origin happens to be close enough to the end to finish within that bound.
    Cached G/G_prefix/G_rest supplied by callers have no role in either return.
    """
    folder = Path(args['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = dict(protocol=PROTOCOL, valid=False, advantage=None, completed=False,
                  failure_semantics='UNKNOWN_NOT_CERTIFIED', universal_infeasibility_proven=False,
                  baseline_recomputed=True, certificates_reused=False, action_semantics=ACTION_SEMANTICS,
                  evidence_path=str(folder), query_origin=args.get('query_origin'))
    try:
        from .support import continuation_identity, build_support_label
        runtime_spec, reference = deepcopy(args['runtime_spec']), deepcopy(args['reference_chain'])
        binding = deepcopy(args['binding'])
        result.update(reference_prefix_sha256=digest(reference), runtime_spec_sha256=digest(runtime_spec),
                      binding=binding, features=deepcopy(binding.get('features')))
        origin_t = _validate_binding(binding, runtime_spec)
        base = _action(args['base_action'], 'base_action')
        candidate = _action(args.get('action', args.get('candidate_action')), 'candidate action')
        maximum = _limit(args.get('max_decisions'))
        continuation = continuation_identity(reference, mode=args.get('continuation', 'closed-loop'),
                                             schedule=args.get('schedule'))
        schedule = continuation['schedule']
        if schedule is not None:
            mapping = dict(schedule)
            required = [float(t) for t in np.arange(origin_t, MISSION_S, DECISION_S)]
            if any(t not in mapping for t in required):
                raise ValueError('Open-loop continuation must cover every remaining decision')
        result.update(t=origin_t, action=candidate, candidate_action=candidate, base_action=base,
            continuation=continuation, continuation_sha256=continuation['sha256'],
            prefix_state_sha256=binding['state_sha256'], origin_policy_state_sha256=binding['policy_state_sha256'],
            bounded_smoke=maximum is not None)
        write_json(folder/'origin.json', dict(binding=binding, reference_chain=reference,
            reference_prefix_sha256=digest(reference), continuation=continuation,
            base_action=base, candidate_action=candidate, runtime_spec_sha256=digest(runtime_spec),
            cached_returns_used=False, certificates_reused=False))
        arms = []
        for name, action in (('base', base), ('candidate', candidate)):
            env, certifier = build_training_flight(folder/name, runtime_spec)
            env.load_state_dict(deepcopy(binding['state']))
            if env.snapshot_hash() != binding['state_sha256'] or float(env.t) != origin_t:
                raise ValueError('Restored paired origin differs in physical/controller state')
            if not np.array_equal(np.asarray(binding.get('features'), float), features(env)):
                raise ValueError('Origin network features differ from restored physical state')
            reference_policy = make_policy(reference, env)
            if reference_policy.decide() != base:
                raise ValueError('base_action does not match frozen reference at observed state')
            policy = make_policy(reference, env, override=(origin_t, action), schedule=schedule)
            origin = dict(state=env.snapshot(), state_sha256=env.snapshot_hash(),
                policy_state=snapshot_policy(policy), reference_prefix_sha256=digest(reference),
                action=action, requested_CL=CL_ACTIONS[action], continuation_sha256=continuation['sha256'],
                source_policy_state=deepcopy(binding['policy_state']), certificates_reused=False,
                action_semantics=ACTION_SEMANTICS)
            write_json(folder/name/'restored_origin.json', origin)
            arms.append((name, env, certifier, policy, float(env.total_reward), env.snapshot_hash()))
        if arms[0][-1] != arms[1][-1] or arms[0][1] is arms[1][1] or arms[0][2] is arms[1][2]:
            raise ValueError('Paired arms must own independent identical starting states')
        for name, env, certifier, policy, start_reward, start_hash in arms:
            flown = certified_fly(certifier, policy, max_decisions=maximum)
            arm = _arm_result(env, policy, flown, start_reward, start_hash)
            write_json(folder/name/'result.json', arm)
            result[name] = arm
            if name == 'base':
                result['baseline'] = arm
            if digest(reference) != result['reference_prefix_sha256']:
                raise ValueError('Frozen reference changed during paired continuation')
        completed = result['base']['completed'] and result['candidate']['completed']
        valid = bool(completed and maximum is None)
        result.update(completed=bool(completed), valid=valid,
            G_base_rest=result['base']['G_rest'], G_rest=result['candidate']['G_rest'],
            G_candidate_rest=result['candidate']['G_rest'],
            advantage=result['candidate']['G_rest']-result['base']['G_rest'] if valid else None,
            failure_semantics=None if valid else ('BOUNDED_CHECK_NOT_TRAINING_LABEL'
                                                 if maximum is not None else 'UNKNOWN_NOT_CERTIFIED'),
            steps=result['base']['steps']+result['candidate']['steps'])
        result['support_label'] = build_support_label(binding['state'],
            reference_prefix_sha256=result['reference_prefix_sha256'],
            continuation_sha256=continuation['sha256'], base_action=base,
            candidate_action=candidate, valid=valid, label_kind='branch', advantage=result['advantage'])
        result['support_label'].update(advantage=result['advantage'],
            evidence_path=str(folder), origin_state_sha256=binding['state_sha256'],
            action_semantics=ACTION_SEMANTICS)
    except Exception:
        result.update(valid=False, advantage=None, error=traceback.format_exc(),
                      failure_semantics='UNKNOWN_NOT_CERTIFIED')
    result['wall_s'] = time.perf_counter()-started
    result['total_wall_s'] = result['wall_s']
    return _result(folder, result)


def candidate_job(args):
    """Evaluate a frozen gated chain; capture every actual pre-decision state."""
    folder = Path(args['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    chain, runtime_spec = deepcopy(args['chain']), deepcopy(args['runtime_spec'])
    identity, bindings = _identity(chain, runtime_spec), {}
    started = time.perf_counter()
    try:
        maximum = _limit(args.get('max_decisions'))
        env, certifier = build_training_flight(folder/'run', runtime_spec,
                                               initial_offset=args.get('initial_offset'))
        policy = make_policy(chain, env)

        def record():
            binding = decision_binding(certifier, policy, runtime_spec)
            key = str(int(round(binding['t'])))
            bindings[key] = binding
            (folder/'decision_prefix').mkdir(exist_ok=True)
            write_json(folder/'decision_prefix'/f't{int(binding["t"]):04d}.json', binding)

        start_reward = float(env.total_reward)
        flown = certified_fly(certifier, policy, max_decisions=maximum, boundary_callback=record)
        G = float(env.total_reward)-start_reward
        if not np.isfinite(G) or abs(G-flown['G_sum']) > 1e-12:
            raise ValueError('Candidate reward disagrees with committed flight state')
        if digest(chain) != identity['chain_sha256']:
            raise ValueError('Frozen candidate changed during evaluation')
        completed = mission_completed(env, flown['stop'])
        counts = np.bincount(flown['executed'], minlength=len(CL_ACTIONS))
        requested = np.bincount([int(a) for _, a in policy.decisions], minlength=len(CL_ACTIONS))
        value = dict(identity, initial_offset=clean(args.get('initial_offset')),
            G=G, steps=len(flown['executed']), stop=flown['stop'],
            success=completed, completed=completed, completed_1200s=completed,
            bounded_smoke=maximum is not None and maximum < 400, time_s=float(env.t), outcome=env.last_outcome,
            final_state=env.x.tolist(), range_km=float(env.x[3]/1000),
            decisions_changed=int(sum(a != INITIAL_ACTION for _, a in policy.decisions)),
            decisions=len(policy.decisions), switches=int(env.switches), geometry_switches=int(env.switches),
            cl_action_changes=sum(a != b for a, b in zip(flown['executed'], flown['executed'][1:])),
            action_counts=counts.tolist(), requested_action_counts=requested.tolist(),
            mean_CL=float(counts @ np.asarray(CL_ACTIONS)/max(counts.sum(), 1)),
            schedule=clean(policy.decisions), decision_prefix=bindings, visited_bindings=bindings,
            requested_actions=flown['requested'], actual_executed_actions=flown['executed'],
            execution_trace=flown['trace'], support_logs=clean(policy.logs), action_semantics=ACTION_SEMANTICS,
            final_snapshot_sha256=env.snapshot_hash(), final_policy_state=snapshot_policy(policy),
            failure_semantics=None if completed else 'UNKNOWN_NOT_CERTIFIED', universal_infeasibility_proven=False,
            wall_s=time.perf_counter()-started)
        value.update(candidate_diagnostics(env, certifier))
        write_json(folder/'support_logs.json', value['support_logs'])
        return _result(folder, value)
    except Exception:
        return _result(folder, dict(identity, error=traceback.format_exc(), decision_prefix=bindings,
            visited_bindings=bindings, success=False, completed=False, completed_1200s=False,
            failure_semantics='UNKNOWN_NOT_CERTIFIED', universal_infeasibility_proven=False,
            wall_s=time.perf_counter()-started))


def dispatch(job):
    kind, args = job
    if kind in ('paired', 'pair'):
        return paired_job(args)
    if kind == 'candidate':
        return candidate_job(args)
    raise ValueError('Unknown supported recovery job: '+str(kind))
