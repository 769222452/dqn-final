"""Bounded real-flight check of nominal and off-nominal paired continuations.

This explicit CLI does not train, fit a network, or create valid return labels.
It runs only when main/run_check is invoked; importing the module starts no work.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))

PROTOCOL = 'supported-recovery-bounded-real-check/1'
CACHE_SENTINEL = 1.e50


def _pair_checks(result, origin_hash, base_action, candidate_action):
    """Check actual short execution, never infer a full-return training label."""
    base, candidate = result.get('base', {}), result.get('candidate', {})
    checks = dict(no_rollout_error=not result.get('error'),
        identical_restored_origins=base.get('initial_snapshot_sha256') == origin_hash
                                  == candidate.get('initial_snapshot_sha256'),
        both_four_steps=base.get('steps') == candidate.get('steps') == 4,
        both_reach_60s=base.get('time_s') == candidate.get('time_s') == 60.,
        base_request_held_12s=base.get('first_12s_requested_actions') == [base_action]*4,
        candidate_request_held_12s=candidate.get('first_12s_requested_actions') == [candidate_action]*4,
        base_execution_trace_complete=len(base.get('first_12s_actual_executed_actions', [])) == 4,
        candidate_execution_trace_complete=len(candidate.get('first_12s_actual_executed_actions', [])) == 4,
        no_valid_training_label=result.get('valid') is False and result.get('advantage') is None
                               and result.get('support_label', {}).get('valid') is False,
        incomplete_full_horizon=not result.get('completed'),
        request_semantics=result.get('action_semantics') == 'certified_policy_request',
        fresh_certificates=result.get('certificates_reused') is False,
        cached_returns_not_used=result.get('baseline_recomputed') is True
            and result.get('G_base_rest') == base.get('G_rest')
            and result.get('G_candidate_rest') == candidate.get('G_rest'))
    for name, arm in (('base', base), ('candidate', candidate)):
        reward, reward_sum = arm.get('G_rest'), arm.get('G_sum')
        checks['cached_returns_not_used'] = bool(checks['cached_returns_not_used']
            and isinstance(reward, (float, int)) and isinstance(reward_sum, (float, int))
            and abs(reward-reward_sum) <= 1e-12 and abs(reward) < CACHE_SENTINEL/2)
        checks[name+'_committed_trace'] = (len(arm.get('execution_trace', [])) == 4
            and all(row.get('committed') is True for row in arm.get('execution_trace', [])))
    return checks


def run_check(out, model, initial_policy, config=None):
    from certified_flight import CL_ACTIONS
    from continuous_recovery.preparation import DEFAULT_CONFIG
    from continuous_recovery.training import prepare_runtime
    from continuous_tracking.experiment import write_json
    from rollout_improvement.policy import chain_action, constant_stage
    from supported_recovery.rollouts import candidate_job, paired_job
    import numpy as np

    out = Path(out).resolve()
    if out.exists():
        raise ValueError('--out must be a new directory; existing evidence is preserved')
    started = time.perf_counter()
    spec, reference, runtime_binding = prepare_runtime(out, model,
        config=DEFAULT_CONFIG if config is None else config, initial_policy=initial_policy,
        settings=dict(check_protocol=PROTOCOL, purpose='bounded_nominal_and_off_nominal_pair_check',
                      source_execution_steps=20, paired_execution_steps=4, training_performed=False))
    result = dict(protocol=PROTOCOL, passed=False, training_performed=False, network_updates=0,
        valid_training_labels=0, full_mission_completed=False, runtime_binding=runtime_binding['sha256'],
        checks={}, pairs={}, sources={}, errors=[], evidence_path=str(out))
    try:
        nominal = candidate_job(dict(folder=str(out/'reference_source'), runtime_spec=spec,
                                     chain=reference, max_decisions=20))
        result['sources']['reference'] = dict(evidence_path=str(out/'reference_source'),
            time_s=nominal.get('time_s'), steps=nominal.get('steps'), stop=nominal.get('stop'))
        if nominal.get('error') or nominal.get('time_s') != 60. or nominal.get('steps') != 20:
            raise RuntimeError('Reference source did not complete its bounded 60 s flight: '
                               +str(nominal.get('error') or nominal.get('stop')))
        nominal_binding = nominal['decision_prefix']['48']
        initial_action = nominal['decision_prefix']['0']['action']
        off_nominal_cl = .035 if initial_action == CL_ACTIONS.index(.03) else .03
        off_chain = deepcopy(reference)+[constant_stage(off_nominal_cl)]
        off_nominal = candidate_job(dict(folder=str(out/'off_nominal_source'), runtime_spec=spec,
                                         chain=off_chain, max_decisions=20))
        result['sources']['off_nominal'] = dict(evidence_path=str(out/'off_nominal_source'),
            time_s=off_nominal.get('time_s'), steps=off_nominal.get('steps'), stop=off_nominal.get('stop'),
            appended_constant_CL=off_nominal_cl)
        if off_nominal.get('error') or off_nominal.get('time_s') != 60. or off_nominal.get('steps') != 20:
            raise RuntimeError('Off-nominal source did not complete its bounded 60 s flight: '
                               +str(off_nominal.get('error') or off_nominal.get('stop')))
        off_binding = off_nominal['decision_prefix']['48']
        result['checks']['off_nominal_is_a_different_actual_state'] = (
            nominal_binding['state_sha256'] != off_binding['state_sha256'])
        result['checks']['both_sources_are_bounded'] = (nominal.get('completed_1200s') is False
                                                        and off_nominal.get('completed_1200s') is False)
        for name, binding in (('reference', nominal_binding), ('off_nominal', off_binding)):
            base = int(chain_action(reference, np.asarray(binding['features'], float)))
            alternative = base-1 if base > 0 else base+1
            observed = deepcopy(binding)
            # Deliberately inconsistent cached scores must not enter either arm.
            # The verified physical snapshot and its accumulated reward are intact.
            observed['G_prefix'] = -CACHE_SENTINEL
            pair = paired_job(dict(folder=str(out/(name+'_pair')), runtime_spec=spec,
                reference_chain=reference, binding=observed, base_action=base, action=alternative,
                continuation='closed-loop', max_decisions=4, query_origin=name+'_t48_bounded_check',
                G=CACHE_SENTINEL, G_base_rest=CACHE_SENTINEL, G_candidate_rest=-CACHE_SENTINEL))
            checks = _pair_checks(pair, binding['state_sha256'], base, alternative)
            result['pairs'][name] = dict(evidence_path=str(out/(name+'_pair')), checks=checks,
                origin_snapshot_sha256=binding['state_sha256'], base_action=base,
                candidate_action=alternative, action_semantics='certified_policy_request',
                G_base_partial=pair.get('G_base_rest'), G_candidate_partial=pair.get('G_candidate_rest'),
                base_actual_actions=pair.get('base', {}).get('first_12s_actual_executed_actions'),
                candidate_actual_actions=pair.get('candidate', {}).get('first_12s_actual_executed_actions'),
                valid=pair.get('valid'), error=pair.get('error'))
            result['checks'].update({name+'_'+key: value for key, value in checks.items()})
        result['passed'] = bool(result['checks']) and all(result['checks'].values())
        result['errors'] = [key for key, passed in result['checks'].items() if not passed]
    except Exception as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc))
    result['wall_s'] = time.perf_counter()-started
    write_json(out/'validation.json', result)
    return result


def main(argv=None):
    from continuous_recovery.preparation import DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--initial-policy', required=True, type=Path)
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out already exists; refusing overwrite')
    result = run_check(args.out, args.model, args.initial_policy, args.config)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    # Match the serial production entry points only when explicitly invoked.
    import os
    for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[_key] = '1'
    raise SystemExit(main())
