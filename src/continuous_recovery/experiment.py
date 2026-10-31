"""Transactional fixed-CL flight using the isolated recovery controller.

Control-manager memory is part of each certificate witness. A successful current
guard audit is sealed once; execution verifies its binding and consumes its exact
trajectory without integrating the same controls again.
"""
from copy import copy, deepcopy
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np

from certified_flight import CL_ACTIONS
from certified_flight.transition import Transition
from continuous_tracking.experiment import clean, write_json
from continuous_surrogate.experiment import context_from_state, digest
from .execution import seal_execution, commit_certified

PROTOCOL = 'continuous-recovery-flight/1'


def bind_proof(state, nodes, identity, transition, cl, endpoint, witness):
    return dict(protocol=PROTOCOL, state_hash=digest(state), nodes_hash=digest(np.asarray(nodes)),
                identity_hash=identity, transition_hash=digest(vars(transition)), CL_cmd=float(cl),
                times_s=(np.arange(7)*.5).tolist(), endpoint=list(np.asarray(endpoint)),
                witness_hash=digest(witness))


def commit_audited(system, state, proposed_transition, nodes, proof, identity, cl, witness):
    """Reintegration reference for regression checks; not used by flight runners."""
    expected = bind_proof(state, nodes, identity, proposed_transition, cl,
                          proof.get('endpoint', []), witness)
    if proof != expected:
        raise ValueError('Stale or incompatible recovery-controller certificate')
    c = context_from_state(dict(state, transition=vars(proposed_transition)), cl)
    audit, arrays = system['plant'].audit(c, nodes, np.arange(7)*.5, cl)
    if not audit['valid']:
        raise ValueError('PRECOMMIT_CONTINUOUS_AUDIT_REJECTED')
    x = arrays['0.5']['x'][-1].copy()
    if not np.array_equal(x, proof['endpoint']):
        raise ValueError('Certified versus committed endpoint mismatch')
    now = state['t']+3.
    z = copy(proposed_transition)
    z.settle(now)
    outcome = system['mission'].outcome(now, x)
    reward = system['mission'].reward(float(x[3]-state['x'][3]), outcome)['reward']
    return dict(state, x=x.tolist(), alpha=float(nodes[-1]), t=now,
                transition=vars(z).copy(), reward=state['reward']+reward), audit, arrays


def _event_name(event):
    if isinstance(event, dict):
        return str(event.get('event', event.get('type', event.get('name', 'UNNAMED_EVENT'))))
    return str(event)


def _memory_summary(memory):
    """Small per-step index; complete JSON memory stays in the checkpoint/witness."""
    if not isinstance(memory, dict):
        return dict(available=False)
    summary = dict(available=True, state_sha256=digest(memory), keys=sorted(memory))
    for key in ('mode', 'epoch', 'recovery_epoch', 'exit_confirmations', 'nominal_pass_streak',
                'exit_streak', 'consecutive_nominal_passes', 'entered_at_s'):
        if key in memory and not isinstance(memory[key], (dict, list)):
            summary[key] = memory[key]
    for key, value in memory.items():
        if isinstance(value, dict) and value.get('protocol') == 'continuous-recovery-plan/1':
            summary['plan'] = {k:value.get(k) for k in
                ('epoch', 'started_at_s', 'ends_at_s', 'request_CL', 'payload_sha256')}
    return summary


def initial_state(system, start_state=None):
    """Accept a before.json state or a checkpoint wrapper, without changing it."""
    if start_state is None:
        mission, db = system['mission'], system['db']
        state = dict(x=list(mission.data['initial_state']['x_h_V_gamma_s']),
                     alpha=mission.data['initial_state']['alpha_rad'], t=0.,
                     transition=vars(Transition(db.index['G42'], db.index['G42'])), reward=0., switches=0)
        memory = None
    else:
        source = (json.loads(Path(start_state).read_text(encoding='utf-8'))
                  if isinstance(start_state, (str, Path)) else deepcopy(start_state))
        memory = source.get('controller_memory') if 'state' in source else None
        state = deepcopy(source.get('state', source))
        state.setdefault('reward', 0.)
        state.setdefault('switches', 0)
    x = np.asarray(state['x'], float)
    if x.shape != (4,) or not np.isfinite(np.r_[x, state['alpha'], state['t'], state['reward']]).all():
        raise ValueError('Start state must contain finite x, alpha, t and reward')
    if not 0. <= state['t'] < float(system['mission'].data['duration_s']):
        raise ValueError('Start time must precede the mission end')
    if abs(float(state['t'])/3.-round(float(state['t'])/3.)) > 1e-9:
        raise ValueError('Start state must be on the original three-second decision grid')
    Transition(**state['transition'])
    return state, deepcopy(memory)


def run_flight(system, model, out, start_state=None, max_decisions=400):
    """Run at most the requested decisions; setup/fit time is excluded."""
    from .controller import RecoveryController

    if not isinstance(max_decisions, int) or not 1 <= max_decisions <= 400:
        raise ValueError('max_decisions must be an integer in [1, 400]')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    mission, db = system['mission'], system['db']
    cfg = system['config']
    cl = float(cfg['flight']['CL'])
    if cl not in CL_ACTIONS:
        raise ValueError('Unchanged absolute CL action required')
    state, restored_memory = initial_state(system, start_state)
    initial = deepcopy(state)
    identity = digest(dict(protocol=PROTOCOL, source=system['identity']['sha256'], config=cfg,
                           weights_hash=system.get('model_hash'), model=getattr(model, 'metadata', {})))
    controller = RecoveryController(system, model, identity)
    if restored_memory is not None:
        controller.load_state_dict(restored_memory)
    write_json(out/'initial_state.json', dict(state=state, controller_memory=controller.state_dict(),
                                            source=str(start_state) if isinstance(start_state, (str, Path)) else None))
    rows, counts, event_counts, decision_times = [], Counter(), Counter(), []
    reason = 'NOT_STARTED'
    last_certification, last_details = None, None
    # The independently reported flight interval starts after build/load and
    # includes every safety query, real OCP fallback, commit and evidence write.
    start, cpu_start = time.perf_counter(), time.process_time()

    def result():
        margins = {k:min(r['margins'][k] for r in rows) for k in rows[0]['margins']} if rows else {}
        return dict(protocol=PROTOCOL, completed_1200s=reason == 'SUCCESS', stop_reason=reason,
            range_km=(state['x'][3]-mission.data['initial_state']['x_h_V_gamma_s'][3])/1000,
            segment_range_km=(state['x'][3]-initial['x'][3])/1000,
            time_s=state['t'], initial_time_s=initial['t'], simulated_duration_s=state['t']-initial['t'],
            steps=len(rows), attempted_decisions=len(decision_times), max_decisions=max_decisions,
            switches=state['switches'], segment_switches=state['switches']-initial['switches'],
            reward=state['reward'], segment_reward=state['reward']-initial['reward'],
            wall_s=time.perf_counter()-start, cpu_s=time.process_time()-cpu_start, CL=cl,
            CL_rmse=float(np.sqrt(np.mean([r['CL_rmse']**2 for r in rows]))) if rows else None,
            max_boundary_jump_deg=max([abs(r['boundary_jump_deg']) for r in rows], default=0),
            max_alpha_rate_degps=max([r['max_alpha_rate_degps'] for r in rows], default=0),
            minimum_normalized_margins=margins,
            tracking_deviation_steps=sum(not r['quality']['passed'] for r in rows if r['quality']),
            safety_intervention_steps=sum(r['safety_intervention'] for r in rows),
            total_drag_work_J=sum(r['drag_work_J'] for r in rows),
            decision_wall_p50_s=float(np.median(decision_times)) if decision_times else None,
            decision_wall_p95_s=float(np.percentile(decision_times, 95)) if decision_times else None,
            decision_wall_max_s=max(decision_times, default=None),
            decision_wall_s=list(decision_times), counts=dict(counts), events=dict(event_counts),
            selected_source_counts=dict(Counter(r['selected_source'] for r in rows)),
            final_state=state, controller_memory_summary=_memory_summary(controller.state_dict()),
            initial_state=initial, rows=rows, last_certification=last_certification,
            last_reason=(last_details or {}).get('reason'),
            safety_scope='fresh current3s/H30 and complete remaining recovery-tail dual-grid audits; no recursive guarantee',
            training_performed=False, surrogate_refit_performed=False,
            pilot_not_production_qualification=True)

    def checkpoint():
        write_json(out/'checkpoint.json', dict(protocol=PROTOCOL, identity=identity, state=state,
                   controller_memory=controller.state_dict(), executed_steps=len(rows)))

    checkpoint()
    try:
        for step in range(max_decisions):
            tick, cpu_tick = time.perf_counter(), time.process_time()
            folder = out/f'decision_{step:04d}'
            folder.mkdir()
            write_json(folder/'before.json', state)
            manager_before = deepcopy(controller.state_dict())
            write_json(folder/'manager_before.json', manager_before)
            checked, z, cmd, details, memory = controller.decide(state, cl, folder)
            manager_after = deepcopy(controller.state_dict())
            if digest(memory) != digest(manager_after):
                raise ValueError('Controller return memory disagrees with persisted manager state')
            last_details = clean(details)
            write_json(folder/'decision.json', dict(details=details, controller_memory=manager_after))
            last_certification = str(folder/'safety.json')
            write_json(last_certification, dict(accepted=checked.accepted, status=checked.status,
                       audit=checked.audit, diagnostics=checked.diagnostics, details=details))
            counts['true_ocp_fallback'] += int(details.get('ocp_called', 0))
            for head in ('pre', 'post'):
                record = details.get(head) or {}
                if record.get('supported') is False:
                    counts[head+'_ood'] += 1
            if (details.get('post') or {}).get('supported') is True:
                counts['post_surrogate_proposed'] += 1
            events = details.get('events', [])
            if isinstance(events, dict):
                event_counts.update(events)
            else:
                event_counts.update(_event_name(e) for e in events)
            if not checked.accepted:
                decision_times.append(time.perf_counter()-tick)
                reason = 'UNKNOWN_NOT_CERTIFIED'
                counts[reason] += 1
                counts[checked.status] += 1
                write_json(out/'stop_state.json', dict(reason=reason, state=state,
                    last_certification=last_certification, safety_status=checked.status,
                    last_reason=details.get('reason', checked.diagnostics.get('reason', checked.status)),
                    details=details, controller_memory=manager_after,
                    last_committed_controller_memory=manager_before))
                break
            nodes = np.asarray(checked.alpha_nodes, float)
            if not np.array_equal(checked.times, np.arange(7)*.5):
                raise ValueError('Certified recovery execution must retain original three-second grid')
            witness = dict(guard=checked.diagnostics, manager_before=manager_before,
                           manager_after=manager_after)
            seal_started = time.perf_counter()
            evidence = seal_execution(system, state, z, identity, cmd, witness, checked)
            seal_wall = time.perf_counter()-seal_started
            cert = evidence.proof
            write_json(folder/'certificate.json', dict(cert, status='VERIFIED_PASS',
                proposed_transition=vars(z), future_witness=witness, reachability=details.get('reachable')))
            commit_started = time.perf_counter()
            next_state, audit, arrays = commit_certified(
                system, state, z, nodes, cert, identity, cmd, witness, evidence)
            commit_wall = time.perf_counter()-commit_started
            dense = audit['grids']['0.1']
            margins = {k:min(dense['margins'][k], audit['grids']['0.5']['margins'][k])
                       for k in dense['margins']}
            samples = arrays['0.1']['samples']
            energy = (samples[10]*samples[1]).reshape(-1, 5)[:, :4] @ (np.array([1, 2, 2, 1])/6)
            elapsed = time.perf_counter()-tick
            decision_times.append(elapsed)
            selected = str(details.get('selected_source', checked.diagnostics.get('selected_source', 'UNKNOWN')))
            row = dict(step=step, t0=state['t'], t1=next_state['t'], CL_raw=cl, CL_cmd=float(cmd),
                CL_mean=dense['CL_mean_first3'], CL_rmse=dense['CL_rmse_first3'],
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
                state_end=next_state['x'], selected_source=selected, events=events,
                controller_memory_summary=_memory_summary(manager_after),
                execution_protocol=cert['protocol'], certified_trajectory_reused=True,
                actual_execution_reintegrated=False, execution_integration_calls=0,
                execution_seal_wall_s=seal_wall, execution_commit_wall_s=commit_wall)
            np.savez_compressed(folder/'executed.npz', alpha_nodes=nodes, times=np.arange(7)*.5,
                **{f'{key}_{field}': arr[field] for key, arr in arrays.items() for field in ('x', 'samples')})
            next_state['switches'] += int(row['switched'])
            state = next_state
            rows.append(row)
            counts['VERIFIED_PASS'] += 1
            counts[checked.status] += 1
            reason = mission.outcome(state['t'], np.asarray(state['x']))
            checkpoint()
            write_json(out/'result.json', result())
            print(json.dumps(dict(stage='recovery_flight', t=state['t'], range_km=state['x'][3]/1000,
                CL_rmse=row['CL_rmse'], selected_source=selected, status=checked.status,
                wall_s=elapsed)), flush=True)
            if reason != 'INCOMPLETE':
                break
        if reason == 'INCOMPLETE':
            reason = 'MAX_DECISIONS'
            write_json(out/'stop_state.json', dict(reason=reason, state=state,
                last_certification=last_certification, controller_memory=controller.state_dict(),
                meaning='Requested decision limit reached; not a completed 1200-second mission'))
    except Exception as exc:
        reason = 'ERROR:'+type(exc).__name__+':'+str(exc)
        write_json(out/'stop_state.json', dict(reason=reason, state=state,
            last_certification=last_certification, controller_memory=controller.state_dict(),
            details=last_details))
        raise
    finally:
        write_json(out/'result.json', result())
    return result()

