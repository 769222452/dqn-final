"""Bounded real-dynamics branch check; does not fit a network or train an iteration."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from certified_flight import CL_ACTIONS
    from continuous_recovery.preparation import DEFAULT_CONFIG
    from continuous_recovery.training import prepare_runtime
    from continuous_recovery.training_backend import (build_training_flight,
        snapshot_policy, restore_policy)
    from continuous_recovery.training_rollouts import certified_fly
    from continuous_tracking.experiment import write_json
    from continuous_surrogate.experiment import digest
    from rollout_improvement.policy import ChainPolicy, chain_action
    from rollout_improvement.features import features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--initial-policy', required=True, type=Path)
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error('--out already exists; refusing overwrite')
    start = time.perf_counter()
    spec, chain, binding = prepare_runtime(out, args.model, config=args.config,
        initial_policy=args.initial_policy,
        settings=dict(purpose='bounded_real_branch_validation', seed=17))
    result = dict(protocol='recovery-training-real-branch-check/1', passed=False,
        training_performed=False, network_updates=0, full_mission_completed=False,
        runtime_binding=binding['sha256'], checks={}, errors=[])

    def record():
        result['wall_s'] = time.perf_counter()-start
        write_json(out/'validation.json', result)

    try:
        env, certifier = build_training_flight(out/'trunk', spec)
        policy = ChainPolicy(chain, env)
        prefix_reward, stop, actions = certified_fly(certifier, policy, until=48.)
        if stop is not None or env.t != 48.:
            raise RuntimeError(f'Prefix stopped at {env.t}: {stop}')
        origin = env.snapshot()
        scheduler = snapshot_policy(policy)
        result.update(prefix_time_s=env.t, prefix_steps=len(actions), prefix_reward=prefix_reward,
                      recovery_active=origin['controller_memory']['memory'] is not None,
                      prefix_snapshot_sha256=env.snapshot_hash())
        if not result['recovery_active']:
            raise RuntimeError('This check requires an actual active recovery plan at t=48 s')
        same = certifier.fork(out/'same_action')
        alternate = certifier.fork(out/'changed_action')
        same_policy = restore_policy(chain, same.env, scheduler)
        a0 = int(chain_action(policy.chain, features(env)))
        alternative_action = a0-1 if a0 > 0 else 1
        alternate_policy = restore_policy(chain, alternate.env, scheduler,
                                           override=(48., alternative_action))
        result['checks']['forks_start_identically'] = (
            same.env.snapshot_hash() == alternate.env.snapshot_hash() == env.snapshot_hash())
        branch_origin_hash = alternate.env.snapshot_hash()
        same_origin_hash = same.env.snapshot_hash()
        g0, s0, a_trunk = certified_fly(certifier, policy, until=60.)
        result['checks']['nominal_does_not_mutate_siblings'] = (
            alternate.env.snapshot_hash() == branch_origin_hash
            and same.env.snapshot_hash() == same_origin_hash)
        g1, s1, a_same = certified_fly(same, same_policy, until=60.)
        same_after = same.env.snapshot_hash()
        trunk_after = env.snapshot_hash()
        result['checks']['same_action_exact_full_state'] = trunk_after == same_after
        result['checks']['same_action_exact_reward'] = g0 == g1
        result['checks']['same_action_exact_policy_scheduler'] = (
            snapshot_policy(policy) == snapshot_policy(same_policy))
        result['checks']['same_action_exact_actions'] = a_trunk == a_same
        result['checks']['same_action_exact_controls'] = (
            [r['alpha_nodes_deg'] for r in env.rows[-4:]]
            == [r['alpha_nodes_deg'] for r in same.env.rows])
        result['checks']['same_action_12s_continuations_completed'] = (
            s0 is None and s1 is None and env.t == same.env.t == 60.)
        result['checks']['same_branch_does_not_mutate_other_sibling'] = (
            alternate.env.snapshot_hash() == branch_origin_hash)
        g2, s2, a_changed = certified_fly(alternate, alternate_policy, until=51.)
        result['checks']['changed_branch_does_not_mutate_nominal_or_sibling'] = (
            env.snapshot_hash() == trunk_after and same.env.snapshot_hash() == same_after)
        attempts = alternate.records
        result['checks']['changed_request_was_proposed'] = bool(
            attempts and attempts[0]['action'] == alternative_action)
        events = (attempts[0].get('details') or {}).get('events', []) if attempts else []
        result['checks']['changed_request_invalidates_old_recovery'] = any(
            isinstance(e, dict) and e.get('event') == 'RECOVERY_INVALIDATED_RAW_CL_CHANGE'
            and e.get('new_CL_raw') == CL_ACTIONS[alternative_action] for e in events)
        result['checks']['all_executed_actions_have_PASS'] = all(
            r.get('old_certificates_reused') is False
            and r.get('certified_trajectory_reused') is True
            and r.get('actual_execution_reintegrated') is False
            and r.get('execution_integration_calls') == 0
            for flight in (env, same.env, alternate.env) for r in flight.rows)
        result.update(nominal_end_s=env.t, same_action_end_s=same.env.t,
            changed_action_end_s=alternate.env.t, nominal_continuation_reward=g0,
            same_action_continuation_reward=g1, changed_action_partial_reward=g2,
            changed_action_stop=s2, baseline_action=a0, changed_requested_action=alternative_action,
            changed_actual_actions=a_changed, new_certified_steps=len(env.rows)+len(same.env.rows)+len(alternate.env.rows),
            nominal_final_snapshot_sha256=trunk_after, same_final_snapshot_sha256=same_after,
            changed_final_snapshot_sha256=alternate.env.snapshot_hash(),
            no_training_labels_created=True)
        result['passed'] = all(result['checks'].values())
        if not result['passed']:
            result['errors'] = [key for key, passed in result['checks'].items() if not passed]
    except Exception as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc))
        record()
        raise
    record()
    import json
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
