"""Fresh certified branch labels for the continuous recovery execution protocol.

The upper policy, nineteen absolute actions and twelve-second branching rule
are the original policy-iteration algorithm. A label is admitted only after a
complete audited continuation and an exact check against its nominal prefix.
Old controller rewards, labels and certificates are never imported here.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
import traceback

import numpy as np

from certified_flight import CL_ACTIONS, OracleRequired
from continuous_surrogate.experiment import digest
from continuous_tracking.experiment import clean, write_json
from rollout_improvement.features import features
from rollout_improvement.policy import (ChainPolicy, DECISION_S, INITIAL_ACTION,
                                        chain_action)
from rollout_improvement.rollouts import (MISSION_S, alternatives, mission_completed,
                                         _terminal_stop)

PROTOCOL = 'continuous-recovery-branch-rollouts/1'


def build_training_flight(*args, **kwargs):
    # Lazy import keeps lightweight policy tests independent of OCP construction.
    from .training_backend import build_training_flight as build
    return build(*args, **kwargs)


def snapshot_policy(policy):
    """All scheduler memory, including the held request and previous decisions."""
    from .training_backend import snapshot_policy as snapshot
    return snapshot(policy)


def clone_policy(policy, env, *, override=None, schedule=None):
    """Own scheduler memory on a fork; the next decision can impose an action."""
    from .training_backend import restore_policy
    return restore_policy(deepcopy(policy.chain), env, snapshot_policy(policy),
                          override=override, schedule=schedule)


def _at_decision_boundary(env):
    return abs(float(env.t)/DECISION_S-round(float(env.t)/DECISION_S)) < 1e-9


def decision_binding(certifier, policy):
    """Bind a nominal return prefix to its actual physical and controller state."""
    if not _at_decision_boundary(certifier.env):
        raise ValueError('Branches require an exact twelve-second decision boundary')
    f = features(certifier.env).tolist()
    state = clean(certifier.env.snapshot())
    policy_state = snapshot_policy(policy)
    return dict(t=float(certifier.env.t), G_prefix=float(certifier.env.total_reward),
        action=int(chain_action(policy.chain, np.asarray(f))), features=f,
        state=state, state_sha256=certifier.env.snapshot_hash(),
        policy_state=policy_state, policy_state_sha256=digest(policy_state))


def certified_fly(certifier, policy, until=None, *, boundary_callback=None, max_decisions=None):
    """Only execute PASS actions; preserve UNKNOWN and record scheduler bindings."""
    env, total, stop, actions = certifier.env, 0., None, []
    if env.done:
        return total, _terminal_stop(env), actions
    while until is None or env.t < until-1e-9:
        if max_decisions is not None and len(actions) >= max_decisions:
            stop = 'MAX_DECISIONS'
            break
        if boundary_callback is not None and _at_decision_boundary(env):
            boundary_callback(decision_binding(certifier, policy))
        pref = policy.preference()
        certifier.bind_policy_state(snapshot_policy(policy))
        pool = certifier.pool(pref)
        mask = np.asarray(pool['mask'], bool)
        if mask.shape != (len(CL_ACTIONS),):
            raise ValueError('Certification mask must retain all nineteen absolute actions')
        if not mask.any():
            stop = 'NEXT_STATE_NOT_CERTIFIED'
            break
        action = int(np.argmax(np.where(mask, pref, -np.inf)))
        try:
            reward, done, _ = certifier.commit(pool, action)
        except OracleRequired as exc:
            stop = 'EXECUTION_REFUSED:'+exc.reason
            break
        total += float(reward)
        actions.append(action)
        if done:
            stop = _terminal_stop(env)
            break
    return total, stop, actions


def _result(path, value):
    write_json(Path(path)/'job_result.json', value)
    return value


def _identity(chain, runtime_spec):
    return dict(protocol=PROTOCOL, chain_sha256=digest(chain),
                runtime_spec_sha256=digest(runtime_spec))


def candidate_diagnostics(env, certifier):
    """Aggregate committed execution separately from all certification trials.

    CL errors and margins come from the backend's independent execution audits.
    Refused/retried candidates contribute only to attempted-controller counters;
    they cannot be counted as executed OOD or OCP steps.
    """
    rows, attempts = env.rows, certifier.records
    margins = {}
    if rows:
        keys = set(rows[0]['margins'])
        if any(set(row['margins']) != keys for row in rows):
            raise ValueError('Execution audits have inconsistent constraint margin fields')
        margins = {key:min(float(row['margins'][key]) for row in rows) for key in sorted(keys)}
        durations = np.array([float(row['t1'])-float(row['t0']) for row in rows])
        squared = np.array([float(row['CL_rmse'])**2 for row in rows])
        if (not np.isfinite(durations).all() or np.any(durations <= 0.)
                or not np.isfinite(squared).all() or not np.isfinite(list(margins.values())).all()):
            raise ValueError('Execution audit diagnostics must be finite and have positive durations')
        rmse = float(np.sqrt(np.average(squared, weights=durations)))
    else:
        rmse = None
    result = dict(CL_tracking_RMSE=rmse, CL_tracking_target='projected_CL_cmd',
        max_boundary_jump_deg=max((abs(float(row['boundary_jump_deg'])) for row in rows), default=0.),
        max_alpha_rate_degps=max((float(row['max_alpha_rate_degps']) for row in rows), default=0.),
        minimum_normalized_margins=margins, diagnostic_execution_steps=len(rows),
        certification_attempts=len(attempts),
        certification_status_counts=dict(Counter(str(attempt['status']) for attempt in attempts)),
        certification_refusal_reasons=dict(Counter(str(attempt['reason']) for attempt in attempts
                                                   if not attempt['accepted'])),
        safety_substitution_steps=sum(int(row['action']) != int(row['requested_action']) for row in rows),
        recovery_tail_steps=sum(row['selected_source'] == 'REUSED_AUDITED_OCP_TAIL' for row in rows),
        nominal_controller_steps=sum(row['selected_source'] == 'TRACKING_PRIORITY_NOMINAL' for row in rows))
    for name, evidence in (('executed', rows), ('attempted', attempts)):
        details = [item['details'] for item in evidence]
        for head in ('pre', 'post'):
            count = sum((detail.get(head) or {}).get('supported') is False for detail in details)
            result[name+'_'+head+'_ood_count'] = count
            result[name+'_'+head+'_ood_ratio'] = count/len(evidence) if evidence else None
        calls = sum(int(detail.get('ocp_called', 0)) for detail in details)
        result[name+'_ocp_calls'] = calls
        result[name+'_ocp_ratio'] = calls/len(evidence) if evidence else None
    result['diagnostic_ratio_denominators'] = dict(executed='committed 3 s execution steps',
                                                  attempted='all certification attempts, including retries')
    return result


def _assert_nominal(nominal, chain, runtime_spec):
    expected = _identity(chain, runtime_spec)
    if any(nominal.get(k) != value for k, value in expected.items()):
        raise ValueError('Nominal return belongs to another controller, model or policy')
    if not (nominal.get('success') and nominal.get('completed')
            and nominal.get('stop') == 'DONE' and nominal.get('outcome') == 'SUCCESS'
            and abs(float(nominal.get('time_s', -1))-MISSION_S) <= 1e-6):
        raise ValueError('Branch advantages require a complete successful nominal flight')
    if nominal.get('initial_offset') is not None:
        raise ValueError('Nominal prefix for training must use the same nominal initial state')
    if not nominal.get('decision_prefix'):
        raise ValueError('Nominal flight lacks exact state/controller/policy prefix bindings')


def _verify_prefix(expected, actual):
    # Exact equality is intentional. A different search-budget outcome is not
    # deterministic evidence for the nominal trajectory and cannot form a label.
    keys = ('t', 'G_prefix', 'action', 'features', 'state_sha256', 'policy_state_sha256')
    different = [key for key in keys if expected.get(key) != actual.get(key)]
    if different:
        raise ValueError('NOMINAL_PREFIX_MISMATCH: '+', '.join(different))


def branch_job(args):
    """Regenerate verified nominal prefixes; perturb once, then continue to 1200 s.

    args contains folder, chain, times, sweep, schedule, runtime_spec and the
    *actual complete* nominal candidate result. Schedule is None for feedback
    continuation; otherwise the original nominal sequence is held open loop.
    """
    folder = Path(args['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    chain, runtime_spec = args['chain'], args['runtime_spec']
    nominal = args['nominal']
    rows, prefix, bindings, steps = [], {}, {}, 0
    started = time.perf_counter()
    identity = _identity(chain, runtime_spec)
    try:
        _assert_nominal(nominal, chain, runtime_spec)
        times = [float(t) for t in args['times']]
        if (times != sorted(set(times)) or any(t < 0 or t >= MISSION_S
                or abs(t/DECISION_S-round(t/DECISION_S)) > 1e-9 for t in times)):
            raise ValueError('Branch times must be unique increasing twelve-second boundaries before 1200 s')
        schedule = args.get('schedule')
        sched = None if schedule is None else {int(round(float(t))): int(a) for t, a in schedule}
        if schedule is not None and schedule != nominal['schedule']:
            raise ValueError('Continuation schedule must come from this complete nominal flight')
        env, certifier = build_training_flight(folder/'base', runtime_spec)
        pi = ChainPolicy(chain, env)
        for t_b in times:
            _, stop, pre_actions = certified_fly(certifier, pi, until=t_b)
            steps += len(pre_actions)
            if stop is not None:
                return _result(folder, dict(identity, error=f'nominal stopped ({stop}) before {t_b}',
                    rows=rows, prefix=prefix, steps=steps, stop=stop, outcome=env.last_outcome,
                    time_s=float(env.t), wall_s=time.perf_counter()-started))
            actual = decision_binding(certifier, pi)
            key = str(int(round(t_b)))
            expected = nominal['decision_prefix'].get(key)
            if expected is None:
                raise ValueError('Nominal flight has no prefix at '+key)
            try:
                _verify_prefix(expected, actual)
            except ValueError:
                write_json(folder/f'prefix_mismatch_t{int(t_b):04d}.json',
                           dict(expected=expected, actual=actual))
                raise
            bindings[key] = actual
            G_prefix, a0, f = actual['G_prefix'], actual['action'], actual['features']
            prefix[str(t_b)] = [G_prefix, a0, f]
            if sched is not None and sched.get(int(round(t_b))) != a0:
                raise ValueError('Nominal policy action differs from the bound schedule')
            nominal_rest = float(nominal['G'])-G_prefix
            for action in alternatives(a0, t_b, args.get('sweep', 0)):
                tick = time.perf_counter()
                branch_folder = folder/f't{int(t_b):04d}_a{action:02d}'
                branch = certifier.fork(branch_folder)
                if branch.env.snapshot_hash() != actual['state_sha256']:
                    raise ValueError('Branch fork changed physical/history/controller memory')
                pol = clone_policy(pi, branch.env, override=(t_b, action), schedule=sched)
                origin = dict(binding=actual, requested_action=int(action), requested_CL=CL_ACTIONS[action],
                              branch_policy_state=snapshot_policy(pol))
                write_json(branch_folder/'branch_policy_origin.json', origin)
                g1, s1, actions = certified_fly(branch, pol, until=t_b+DECISION_S)
                g2, s2, tail = (0., s1, []) if s1 is not None else certified_fly(branch, pol)
                steps += len(actions)+len(tail)
                completed = mission_completed(branch.env, s2)
                g_rest = float(branch.env.total_reward)-G_prefix
                if abs(g_rest-(g1+g2)) > 1e-12:
                    raise ValueError('Branch accumulated rewards disagree with committed flight state')
                row = dict(t=t_b, action=int(action), requested_action=int(action),
                    requested_CL=CL_ACTIONS[action], base_action=a0, features=f,
                    executed=actions[:4], actual_executed_actions=actions+tail,
                    G_rest=g_rest, G_base_rest=nominal_rest,
                    advantage=g_rest-nominal_rest if completed else None,
                    prefix_verified=True, prefix_state_sha256=actual['state_sha256'],
                    prefix_policy_state_sha256=actual['policy_state_sha256'],
                    valid=completed, completed=completed, stop=s2,
                    failure_semantics=None if completed else 'UNKNOWN_NOT_CERTIFIED',
                    universal_infeasibility_proven=False,
                    outcome=branch.env.last_outcome, time_s=float(branch.env.t),
                    steps=len(actions)+len(tail), final_state=branch.env.x.tolist(),
                    final_snapshot_sha256=branch.env.snapshot_hash(),
                    final_policy_state=snapshot_policy(pol),
                    evidence_path=str(branch_folder), wall_s=time.perf_counter()-tick)
                rows.append(row)
                write_json(branch_folder/'branch_result.json', row)
        return _result(folder, dict(identity, rows=rows, prefix=prefix, decision_prefix=bindings,
            steps=steps, invalid_branches=sum(not row['valid'] for row in rows),
            wall_s=time.perf_counter()-started))
    except Exception:
        return _result(folder, dict(identity, error=traceback.format_exc(), rows=rows,
            prefix=prefix, decision_prefix=bindings, steps=steps,
            wall_s=time.perf_counter()-started))


def candidate_job(args):
    """Evaluate one frozen candidate with fresh certification; short runs never pass."""
    folder = Path(args['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    chain, runtime_spec = args['chain'], args['runtime_spec']
    initial_offset = args.get('initial_offset')
    identity = _identity(chain, runtime_spec)
    bindings = {}
    try:
        maximum = args.get('max_decisions', 400)
        if type(maximum) is not int or not 1 <= maximum <= 400:
            raise ValueError('max_decisions must be an integer from 1 to 400')
        env, certifier = build_training_flight(folder/'run', runtime_spec, initial_offset=initial_offset)
        pi = ChainPolicy(chain, env)
        tick = time.perf_counter()

        def record(binding):
            key = str(int(round(binding['t'])))
            bindings[key] = binding
            (folder/'decision_prefix').mkdir(exist_ok=True)
            write_json(folder/'decision_prefix'/f't{int(binding["t"]):04d}.json', binding)

        G_sum, stop, actions = certified_fly(certifier, pi, boundary_callback=record, max_decisions=maximum)
        G = float(env.total_reward)
        if abs(G-G_sum) > 1e-12:
            raise ValueError('Candidate rewards differ from committed environment total')
        counts = np.bincount(actions, minlength=len(CL_ACTIONS))
        requested = np.bincount([int(a) for _, a in pi.decisions], minlength=len(CL_ACTIONS))
        completed = mission_completed(env, stop)
        value = dict(identity, initial_offset=clean(initial_offset), steps=len(actions), G=G,
            stop=stop, success=completed, completed=completed, completed_1200s=completed, outcome=env.last_outcome,
            time_s=float(env.t), final_state=env.x.tolist(), range_km=float(env.x[3]/1000),
            decisions_changed=int(sum(a != INITIAL_ACTION for _, a in pi.decisions)),
            decisions=len(pi.decisions), switches=int(env.switches), geometry_switches=int(env.switches),
            cl_action_changes=int(sum(a != b for a, b in zip(actions, actions[1:]))),
            action_counts=counts.tolist(), requested_action_counts=requested.tolist(),
            mean_CL=float(counts @ np.asarray(CL_ACTIONS)/max(counts.sum(), 1)),
            wall_s=time.perf_counter()-tick,
            schedule=[[float(t), int(a)] for t, a in pi.decisions],
            decision_prefix=bindings, final_snapshot_sha256=env.snapshot_hash(),
            final_policy_state=snapshot_policy(pi),
            failure_semantics=None if completed else 'UNKNOWN_NOT_CERTIFIED',
            universal_infeasibility_proven=False)
        value.update(candidate_diagnostics(env, certifier))
        return _result(folder, value)
    except Exception:
        return _result(folder, dict(identity, error=traceback.format_exc(), decision_prefix=bindings,
                                   success=False, completed=False, completed_1200s=False))


def dispatch(job):
    kind, args = job
    if kind == 'branch':
        return branch_job(args)
    if kind == 'candidate':
        return candidate_job(args)
    raise ValueError('Unknown recovery training job: '+str(kind))
