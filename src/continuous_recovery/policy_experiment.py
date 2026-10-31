"""Frozen learned-policy transfer to the continuous recovery controller.

Every 12 seconds the unchanged policy reads the actual closed-loop state. Every
three seconds its preferred CL is certified, with the original nearest-CL
fallback order. A refused candidate never changes controller memory or the
flight. This is an evaluation, not a continuation of the old learning protocol.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from certified_flight import CL_ACTIONS
from continuous_surrogate.experiment import digest
from continuous_tracking.experiment import clean, write_json
from .experiment import initial_state, _event_name, _memory_summary
from .execution import seal_execution, commit_certified

PROTOCOL = 'continuous-recovery-frozen-chain-flight/1'


def _pool_record(pool):
    """Persist certificate-search evidence without live controller objects."""
    keys = ('action', 'cmd', 'checked', 'statuses', 'reasons', 'mask', 'attempts',
            'retry', 'wall_s', 'controller_state', 'details')
    return clean({key: pool.get(key) for key in keys})


def run_policy_flight(system, model, policy, out, max_decisions=400):
    from .controller import RecoveryController
    from .learned_policy import certify_ranked

    if not isinstance(max_decisions, int) or not 1 <= max_decisions <= 400:
        raise ValueError('max_decisions must be an integer in [1, 400]')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    mission, db = system['mission'], system['db']
    state, _ = initial_state(system)
    state.update(history_valid=False, previous_cl_raw=None, previous_cl_cmd=None,
                 previous_cl_actual=None)
    initial = deepcopy(state)
    policy_identity = {key:deepcopy(value) for key, value in policy.metadata.items() if key != 'chain'}
    policy_identity['metadata_sha256'] = digest(policy.metadata)
    # One immutable complete copy; per-decision proofs bind its hashes, rather
    # than duplicating all ensemble weights hundreds of times.
    write_json(out/'frozen_policy.json', policy.metadata)
    retry_options = deepcopy(system.get('original', {}).get('dead_end_retry'))
    identity_payload = dict(protocol=PROTOCOL, source=system['identity']['sha256'],
        config=system['config'], weights_hash=system.get('model_hash'),
        model=getattr(model, 'metadata', {}), policy=policy_identity,
        retry_options=retry_options,
        policy_driven=True, fixed_config_flight_CL_used=False,
        decision_s=12., execution_s=3., CL_actions=list(CL_ACTIONS))
    identity = digest(identity_payload)
    controller = RecoveryController(system, model, identity)
    write_json(out/'evaluation_identity.json', dict(sha256=identity, payload=identity_payload))
    write_json(out/'initial_state.json', dict(state=state, controller_memory=controller.state_dict(),
        policy_state=policy.state_dict(), evaluation='Frozen external policy transferred to new controller'))
    rows, counts, executed_counts, requested_counts = [], Counter(), Counter(), Counter()
    event_counts, decision_times = Counter(), []
    action_stats = [dict(action=a, CL=float(cl), certification_attempts=0, PASS=0,
        unknown=0, rejected=0, ocp_no_solution=0, other_refusal=0,
        executed=0, pre_ood=0, post_ood=0, ocp_calls=0,
        statuses={}, reasons={}, nominal_statuses={}) for a, cl in enumerate(CL_ACTIONS)]
    reason, last_certification, last_details = 'NOT_STARTED', None, None
    start, cpu_start = time.perf_counter(), time.process_time()

    def result():
        margins = {key:min(row['margins'][key] for row in rows)
                   for key in rows[0]['margins']} if rows else {}
        choices = policy.state_dict().get('decisions', [])
        return dict(protocol=PROTOCOL, evaluation='frozen_policy_controller_transfer',
            completed_1200s=reason == 'SUCCESS', stop_reason=reason,
            range_km=(state['x'][3]-initial['x'][3])/1000,
            reward=state['reward'], time_s=state['t'], steps=len(rows),
            attempted_decisions=len(decision_times), max_decisions=max_decisions,
            policy_decisions=len(choices), policy_decision_s=12., execution_s=3.,
            switches=state['switches'], CL_action_changes=sum(
                rows[i]['action'] != rows[i-1]['action'] for i in range(1, len(rows))),
            requested_action_counts=[requested_counts[a] for a in range(len(CL_ACTIONS))],
            executed_action_counts=[executed_counts[a] for a in range(len(CL_ACTIONS))],
            action_CL=list(CL_ACTIONS), action_certification=action_stats,
            safety_substitution_steps=sum(row['action'] != row['requested_action'] for row in rows),
            CL_rmse=float(np.sqrt(np.mean([row['CL_rmse']**2 for row in rows]))) if rows else None,
            CL_raw_tracking_rmse=float(np.sqrt(np.mean([row['CL_raw_rmse']**2 for row in rows]))) if rows else None,
            max_boundary_jump_deg=max([abs(row['boundary_jump_deg']) for row in rows], default=0),
            max_alpha_rate_degps=max([row['max_alpha_rate_degps'] for row in rows], default=0),
            minimum_normalized_margins=margins,
            tracking_deviation_steps=sum(not row['quality']['passed'] for row in rows if row['quality']),
            safety_intervention_steps=sum(row['safety_intervention'] for row in rows),
            total_drag_work_J=sum(row['drag_work_J'] for row in rows),
            wall_s=time.perf_counter()-start, cpu_s=time.process_time()-cpu_start,
            decision_wall_p50_s=float(np.median(decision_times)) if decision_times else None,
            decision_wall_p95_s=float(np.percentile(decision_times, 95)) if decision_times else None,
            decision_wall_max_s=max(decision_times, default=None),
            decision_wall_s=list(decision_times), counts=dict(counts), events=dict(event_counts),
            selected_source_counts=dict(Counter(row['selected_source'] for row in rows)),
            final_state=state, initial_state=initial,
            controller_memory_summary=_memory_summary(controller.state_dict()),
            policy_metadata=policy_identity, policy_state=policy.state_dict(),
            rows=rows, last_certification=last_certification,
            last_reason=(last_details or {}).get('reason'),
            training_performed=False, network_updates=0, surrogate_refit_performed=False,
            old_certificates_reused=False, old_training_labels_reused=False,
            fixed_config_flight_CL_used=False,
            safety_scope='Fresh current3s/H30 and complete remaining recovery-tail dual-grid audits; no recursive guarantee')

    def checkpoint():
        write_json(out/'checkpoint.json', dict(protocol=PROTOCOL, identity=identity,
            state=state, controller_memory=controller.state_dict(), policy_state=policy.state_dict(),
            policy_metadata=policy_identity, policy_weights_file=str(out/'frozen_policy.json'),
            policy_weights_metadata_sha256=policy_identity['metadata_sha256'], executed_steps=len(rows),
            source_identity_sha256=system['identity']['sha256'], model_sha256=system.get('model_hash'),
            training_performed=False))

    def count_attempts(pool):
        for attempt in pool.get('attempts', []):
            action = int(attempt['action'])
            stat = action_stats[action]
            status = str(attempt.get('status', 'UNKNOWN_NOT_CERTIFIED'))
            detail = attempt.get('details') or {}
            passed = bool(attempt.get('accepted', status == 'VERIFIED_PASS'))
            stat['certification_attempts'] += 1
            bucket = {'VERIFIED_PASS': 'PASS', 'UNKNOWN_NOT_CERTIFIED': 'unknown',
                      'VERIFIED_REJECT': 'rejected', 'OCP_NO_SOLUTION_FOUND': 'ocp_no_solution'}.get(status,
                      'other_refusal')
            stat[bucket] += 1
            stat['statuses'][status] = stat['statuses'].get(status, 0)+1
            refusal = str(attempt.get('reason') or detail.get('reason') or status)
            if not passed:
                stat['reasons'][refusal] = stat['reasons'].get(refusal, 0)+1
            nominal = str(detail.get('nominal_status', 'NOT_RECORDED'))
            stat['nominal_statuses'][nominal] = stat['nominal_statuses'].get(nominal, 0)+1
            counts['certification_attempts'] += 1
            counts['candidate_'+status] += 1
            counts['true_ocp_fallback'] += int(detail.get('ocp_called', 0))
            stat['ocp_calls'] += int(detail.get('ocp_called', 0))
            for head in ('pre', 'post'):
                if (detail.get(head) or {}).get('supported') is False:
                    stat[head+'_ood'] += 1
                    counts[head+'_ood'] += 1
            events = detail.get('events', [])
            if isinstance(events, dict):
                event_counts.update(events)
            else:
                event_counts.update(_event_name(event) for event in events)

    checkpoint()
    try:
        for step in range(max_decisions):
            tick, cpu_tick = time.perf_counter(), time.process_time()
            folder = out/f'decision_{step:04d}'
            folder.mkdir()
            write_json(folder/'before.json', state)
            manager_before = deepcopy(controller.state_dict())
            write_json(folder/'manager_before.json', manager_before)
            request = policy.choose(state, history_valid=state['history_valid'],
                                    previous_cl_raw=state['previous_cl_raw'])
            if request['new_policy_decision']:
                requested_counts[int(request['action'])] += 1
            policy_state = deepcopy(policy.state_dict())
            write_json(folder/'policy_request.json', dict(request=request, policy_state=policy_state,
                policy_identity=policy_identity))
            pool = certify_ranked(controller, state, request['preference'], folder/'candidates',
                                  retry_options=retry_options)
            count_attempts(pool)
            write_json(folder/'action_pool.json', _pool_record(pool))
            if digest(controller.state_dict()) != digest(manager_before):
                raise ValueError('Candidate certification changed the live controller memory')
            last_details = clean(pool.get('details') or {})
            checked = pool.get('selected')
            if pool['action'] is None or checked is None or not checked.accepted:
                decision_times.append(time.perf_counter()-tick)
                reason = 'NO_CERTIFIED_ACTION'
                counts[reason] += 1
                last_certification = str(folder/'action_pool.json')
                write_json(out/'stop_state.json', dict(reason=reason, state=state,
                    last_certification=last_certification, details=_pool_record(pool),
                    policy_request=request, policy_state=policy_state,
                    controller_memory=manager_before,
                    meaning='No checked action received PASS; unchecked and timed-out actions remain UNKNOWN'))
                break
            action = int(pool['action'])
            if not pool['mask'][action] or pool['statuses'][action] != 'VERIFIED_PASS':
                raise ValueError('Selected action is absent from the certified PASS set')
            details = pool['details']
            last_details = clean(details)
            z, cmd, manager_after = pool['z'], float(pool['cmd']), deepcopy(pool['controller_state'])
            chosen_controller = pool['manager_after']
            if digest(chosen_controller.state_dict()) != digest(manager_after):
                raise ValueError('Selected candidate manager differs from its certified state')
            nodes = np.asarray(checked.alpha_nodes, float)
            if not np.array_equal(checked.times, np.arange(7)*.5):
                raise ValueError('Certified execution must retain the original three-second grid')
            witness = dict(guard=checked.diagnostics, manager_before=manager_before,
                manager_after=manager_after, policy_identity=policy_identity,
                policy_state=policy_state, policy_request=request, requested_action=int(request['action']),
                selected_action=action, selected_raw_CL=float(CL_ACTIONS[action]),
                certified_action_mask=pool['mask'], statuses=pool['statuses'],
                candidate_evidence_hash=digest(_pool_record(pool)))
            seal_started = time.perf_counter()
            evidence = seal_execution(system, state, z, identity, cmd, witness, checked)
            seal_wall = time.perf_counter()-seal_started
            cert = evidence.proof
            last_certification = str(folder/'certificate.json')
            write_json(last_certification, dict(cert, status='VERIFIED_PASS',
                proposed_transition=vars(z), future_witness=witness,
                reachability=details.get('reachable')))
            commit_started = time.perf_counter()
            next_state, audit, arrays = commit_certified(
                system, state, z, nodes, cert, identity, cmd, witness, evidence)
            commit_wall = time.perf_counter()-commit_started
            dense = audit['grids']['0.1']
            margins = {key:min(dense['margins'][key], audit['grids']['0.5']['margins'][key])
                       for key in dense['margins']}
            samples = arrays['0.1']['samples']
            weights = np.array([1, 2, 2, 1])/6
            energy = (samples[10]*samples[1]).reshape(-1, 5)[:, :4] @ weights
            raw_error = samples[8].reshape(-1, 5)[:, :4]-CL_ACTIONS[action]
            raw_rmse = np.sqrt(np.mean(raw_error**2 @ weights))
            elapsed = time.perf_counter()-tick
            decision_times.append(elapsed)
            selected_source = str(details.get('selected_source', checked.diagnostics.get('selected_source', 'UNKNOWN')))
            row = dict(step=step, t0=state['t'], t1=next_state['t'], action=action,
                requested_action=int(request['action']), requested_CL=float(request['CL_raw']),
                policy_decision_time_s=request['policy_decision_time_s'],
                new_policy_decision=bool(request['new_policy_decision']),
                policy_features=request['features'], CL_raw=float(CL_ACTIONS[action]), CL_cmd=cmd,
                CL_mean=dense['CL_mean_first3'], CL_rmse=dense['CL_rmse_first3'], CL_raw_rmse=float(raw_rmse),
                alpha_nodes_deg=np.rad2deg(nodes).tolist(),
                boundary_jump_deg=float(np.rad2deg(nodes[0]-state['alpha'])),
                max_alpha_rate_degps=audit['maximum_alpha_rate_degps'],
                predicted_flight_mach=float(details['mach']), generator=details.get('generator'),
                switched=bool(details.get('switched', False)), source=db.ids[z.source], target=db.ids[z.target],
                fraction=z.fraction(state['t']), margins=margins, pre=details.get('pre'), post=details.get('post'),
                fallbacks=details.get('fallbacks', []), safety_status=checked.status, wall_s=elapsed,
                cpu_s=time.process_time()-cpu_tick, quality=checked.diagnostics.get('quality'),
                safety_intervention=checked.diagnostics.get('safety_intervention', False),
                drag_work_J=float(np.sum(energy)*.1), dynamic_pressure_max_pa=float(samples[4].max()),
                heat_max_wm2=float(samples[5].max()), load_max_g=float(samples[6].max()),
                state_end=next_state['x'], selected_source=selected_source, events=details.get('events', []),
                checked_actions=pool['checked'], certification_statuses=pool['statuses'],
                controller_memory_summary=_memory_summary(manager_after),
                execution_protocol=cert['protocol'], certified_trajectory_reused=True,
                actual_execution_reintegrated=False, execution_integration_calls=0,
                execution_seal_wall_s=seal_wall, execution_commit_wall_s=commit_wall)
            np.savez_compressed(folder/'executed.npz', alpha_nodes=nodes, times=np.arange(7)*.5,
                **{f'{key}_{field}': arr[field] for key, arr in arrays.items() for field in ('x', 'samples')})
            next_state['switches'] += int(row['switched'])
            next_state.update(history_valid=True, previous_cl_raw=float(CL_ACTIONS[action]),
                              previous_cl_cmd=cmd, previous_cl_actual=float(row['CL_mean']))
            controller, state = chosen_controller, next_state
            rows.append(row)
            executed_counts[action] += 1
            action_stats[action]['executed'] += 1
            counts['VERIFIED_PASS'] += 1
            counts[checked.status] += 1
            for head in ('pre', 'post'):
                counts['executed_'+head+'_ood'] += int((details.get(head) or {}).get('supported') is False)
            counts['executed_true_ocp_fallback'] += int(details.get('ocp_called', 0))
            reason = mission.outcome(state['t'], np.asarray(state['x']))
            checkpoint()
            write_json(out/'result.json', result())
            print(json.dumps(dict(stage='frozen_policy_recovery_flight', t=state['t'],
                range_km=state['x'][3]/1000, requested_CL=request['CL_raw'], executed_CL=CL_ACTIONS[action],
                CL_cmd=cmd, CL_rmse=row['CL_rmse'], selected_source=selected_source,
                status=checked.status, checked_actions=pool['checked'], wall_s=elapsed)), flush=True)
            if reason != 'INCOMPLETE':
                break
        if reason == 'INCOMPLETE':
            reason = 'MAX_DECISIONS'
            write_json(out/'stop_state.json', dict(reason=reason, state=state,
                last_certification=last_certification, controller_memory=controller.state_dict(),
                policy_state=policy.state_dict(), meaning='Requested decision limit reached; not mission success'))
    except Exception as exc:
        reason = 'ERROR:'+type(exc).__name__+':'+str(exc)
        write_json(out/'stop_state.json', dict(reason=reason, state=state,
            last_certification=last_certification, controller_memory=controller.state_dict(),
            policy_state=policy.state_dict(), details=last_details))
        raise
    finally:
        write_json(out/'result.json', result())
    return result()
