"""Replay saved current-3s controls; compare fresh and sealed commits exactly.

No policy inference, future-witness search, OCP solve, or training is performed.
Saved controls and witnesses select the cases; each sealed payload comes from a
new real dual-grid audit, never from the saved executed.npz or an old PASS flag.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
DEFAULT_FLIGHT = ROOT/'runs/continuous_recovery_policy_20261031_01/nominal/flight'
REPRESENTATIVE_STEPS = (0, 1, 3, 14, 16, 20, 21, 399)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(left, right):
    """Compare float bits and array bytes, including signed zero and dtype."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and left.tobytes(order='C') == right.tobytes(order='C'))
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and left.keys() == right.keys()
                and all(exact(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(exact(a, b) for a, b in zip(left, right)))
    if isinstance(left, float) or isinstance(right, float):
        return (type(left) is type(right)
                and struct.pack('!d', left) == struct.pack('!d', right))
    return type(left) is type(right) and left == right


class CallCounter:
    """Wrap the actual runtime methods and retain counts for each timed phase."""

    def __init__(self, system):
        self.phase = 'outside_measured_phases'
        self.counts = {}
        self.originals = []
        self.wrap(system['plant'], 'audit', 'audit')
        self.wrap(system['plant'], 'integrate', 'integrate')
        for name in ('solve_audited', 'solve_context', 'solve'):
            if callable(getattr(system['teacher'], name, None)):
                self.wrap(system['teacher'], name, 'ocp_'+name)

    def wrap(self, obj, name, metric):
        original = getattr(obj, name)
        self.originals.append((obj, name, original))

        def counted(*args, **kwargs):
            count = self.counts.setdefault(self.phase, Counter())
            count[metric] += 1
            return original(*args, **kwargs)

        setattr(obj, name, counted)

    def call(self, phase, function, *args, **kwargs):
        before = self.counts.get(phase, Counter()).copy()
        previous, self.phase = self.phase, phase
        started = time.perf_counter()
        try:
            value = function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter()-started
            self.phase = previous
        return value, elapsed, dict(self.counts.get(phase, Counter())-before)

    def restore(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def finalize(state, pool, audit):
    """Apply the policy-flight bookkeeping that follows either commit method."""
    from certified_flight import CL_ACTIONS
    state = deepcopy(state)
    state['switches'] += int(bool(pool['details'].get('switched', False)))
    state.update(history_valid=True, previous_cl_raw=float(CL_ACTIONS[pool['action']]),
                 previous_cl_cmd=float(pool['cmd']),
                 previous_cl_actual=float(audit['grids']['0.1']['CL_mean_first3']))
    return state


def timing_summary(values):
    return dict(count=len(values), total_s=float(sum(values)),
                p50_s=float(np.percentile(values, 50)),
                p95_s=float(np.percentile(values, 95)), max_s=float(max(values))) if values else {}


def replay_one(system, counter, folder, saved_rows, checkpoint):
    from certified_flight.transition import Transition
    from continuous_recovery.execution import seal_execution, commit_certified
    from continuous_recovery.experiment import bind_proof, commit_audited
    from continuous_surrogate.experiment import context_from_state, digest
    from continuous_surrogate.safety import SafetyResult

    state = read_json(folder/'before.json')
    certificate = read_json(folder/'certificate.json')
    pool = read_json(folder/'action_pool.json')
    witness = certificate['future_witness']
    z = Transition(**certificate['proposed_transition'])
    identity, cl = certificate['identity_hash'], float(certificate['CL_cmd'])
    with np.load(folder/'executed.npz', allow_pickle=False) as saved:
        recorded = {key: saved[key].copy() for key in saved.files}
    nodes, times = recorded['alpha_nodes'], recorded['times']
    original_inputs = deepcopy((state, vars(z), nodes, witness))
    expected = bind_proof(state, nodes, identity, z, cl, certificate['endpoint'], witness)
    old_proof = {key: certificate[key] for key in expected}
    if certificate['status'] != 'VERIFIED_PASS' or not exact(old_proof, expected):
        raise ValueError('Saved certificate binding is invalid: '+str(folder))
    context = context_from_state(dict(state, transition=vars(z)), cl)
    (audit, arrays), audit_s, audit_calls = counter.call(
        'current_audit', system['plant'].audit, context, nodes, times, cl)
    if not audit['valid']:
        raise ValueError('Saved current controls failed a fresh audit: '+str(folder))
    # The old production path also binds the current audit before committing.
    # Keep saved-proof validation separate and time this necessary live work.
    fresh_proof, bind_s, bind_calls = counter.call('fresh_bind', bind_proof,
        state, nodes, identity, z, cl, arrays['0.5']['x'][-1], witness)
    # The live guard adds CL_target. The execution payload intentionally returns
    # the original plant.audit schema, which does not include this extra field.
    checked = SafetyResult(True, 'VERIFIED_PASS', nodes.copy(), times.copy(),
                          dict(deepcopy(audit), CL_target=cl), arrays, deepcopy(witness['guard']))
    checked_before = deepcopy(vars(checked))
    evidence, seal_s, seal_calls = counter.call(
        'seal', seal_execution, system, state, z, identity, cl, witness, checked)
    old, old_s, old_calls = counter.call('fresh_commit', commit_audited,
        system, state, z, nodes, fresh_proof, identity, cl, witness)
    new, new_s, new_calls = counter.call('reused_commit', commit_certified,
        system, state, z, nodes, evidence.proof, identity, cl, witness, evidence)
    old_state, old_audit, old_arrays = old
    new_state, new_audit, new_arrays = new
    finalized_old, finalized_new = finalize(old_state, pool, old_audit), finalize(new_state, pool, new_audit)
    step = int(folder.name.removeprefix('decision_'))
    next_folder = folder.parent/f'decision_{step+1:04d}'
    saved_next = (read_json(next_folder/'before.json') if (next_folder/'before.json').is_file()
                  else checkpoint['state'])
    saved_row = saved_rows[step]
    margins = {key: min(new_audit['grids']['0.1']['margins'][key],
                        new_audit['grids']['0.5']['margins'][key])
               for key in new_audit['grids']['0.1']['margins']}
    saved_arrays_exact = all(exact(new_arrays[grid][field], recorded[grid+'_'+field])
                             for grid in ('0.1', '0.5') for field in ('x', 'samples'))

    def second_commit():
        try:
            commit_certified(system, state, z, nodes, evidence.proof, identity, cl, witness, evidence)
        except ValueError as exc:
            return dict(rejected=True, error=str(exc))
        return dict(rejected=False)

    second, second_s, second_calls = counter.call('second_commit_probe', second_commit)
    checks = dict(
        original_grid=exact(times, np.arange(7)*.5),
        current_audit_calls=audit_calls == {'audit': 1, 'integrate': 2},
        fresh_bind_has_no_integration=not bind_calls,
        exact_fresh_and_saved_proof=exact(fresh_proof, old_proof),
        seal_has_no_integration=not seal_calls,
        fresh_commit_calls=old_calls == {'audit': 1, 'integrate': 2},
        reused_commit_has_no_integration=not new_calls,
        exact_state=exact(old_state, new_state),
        exact_reward=exact(old_state['reward'], new_state['reward']),
        exact_audit_and_constraints=exact(old_audit, new_audit),
        exact_all_array_fields=exact(old_arrays, new_arrays),
        exact_current_audit=exact(audit, new_audit),
        exact_current_arrays=exact(arrays, new_arrays),
        exact_saved_trajectories=saved_arrays_exact,
        exact_finalized_state=exact(finalized_old, finalized_new),
        exact_saved_next_state=exact(finalized_new, saved_next),
        exact_saved_constraint_margins=exact(margins, saved_row['margins']),
        inputs_unchanged=exact(original_inputs, (state, vars(z), nodes, witness)),
        checked_payload_unchanged=exact(checked_before, vars(checked)),
        single_use_enforced=second['rejected'],
        second_commit_has_no_integration=not second_calls)
    return dict(step=step, time_s=state['t'], selected_source=pool['details']['selected_source'],
        morph_active=z.active(state['t']), switched=bool(pool['details'].get('switched', False)),
        historical_ocp_called=bool(pool['details'].get('ocp_called', False)),
        source_hashes={name: file_hash(folder/name) for name in
                       ('before.json', 'certificate.json', 'executed.npz')},
        checks=checks, passed=all(checks.values()),
        failed_checks=[name for name, passed in checks.items() if not passed],
        wall_s=dict(current_audit=audit_s, fresh_bind=bind_s, seal=seal_s, fresh_commit=old_s,
                    reused_commit=new_s, second_commit_probe=second_s),
        calls=dict(current_audit=audit_calls, fresh_bind=bind_calls, seal=seal_calls, fresh_commit=old_calls,
                   reused_commit=new_calls, second_commit_probe=second_calls),
        final_state_sha256=digest(finalized_new), reward=finalized_new['reward'],
        single_use_probe=second)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; overwrite is refused')
    parser.add_argument('--flight', type=Path, default=DEFAULT_FLIGHT,
                        help='Completed policy flight with before/certificate/executed evidence')
    parser.add_argument('--config', type=Path, help='Defaults to the saved flight parent config.json')
    parser.add_argument('--all', action='store_true', help='Replay every saved executed step (normally 400)')
    args = parser.parse_args(argv)
    out, flight = args.out.resolve(), args.flight.resolve()
    if out.exists():
        parser.error('--out already exists; refusing overwrite')
    config = (args.config or flight.parent/'config.json').resolve()
    if not config.is_file() or not (flight/'result.json').is_file():
        parser.error('Saved config and completed flight result.json are required')
    available = {int(path.name.removeprefix('decision_')): path for path in sorted(flight.glob('decision_*'))
                 if (path/'executed.npz').is_file()}
    if not available:
        parser.error('No saved executed steps found')
    selected = (sorted(available) if args.all else sorted(
        set(REPRESENTATIVE_STEPS).intersection(available) | {min(available), max(available)}))
    out.mkdir(parents=True, exist_ok=False)
    from continuous_recovery.preparation import build
    from continuous_surrogate.experiment import digest
    from continuous_tracking.experiment import write_json

    started = time.perf_counter()
    result = dict(protocol='continuous-recovery-execution-replay/1', passed=False,
        scope='Replay saved current 3s choices; future witnesses and policy choices are not recomputed',
        bitwise_comparison='Float IEEE-754 bytes and ndarray dtype/shape/C-order bytes; no tolerance',
        saved_flight_path=str(flight), config_path=str(config), config_sha256=file_hash(config),
        selection='all_saved_executions' if args.all else 'representative_saved_executions',
        available_steps=len(available), requested_steps=selected, replayed_steps=0,
        training_performed=False, network_updates=0, policy_inference_performed=False,
        future_witness_search_performed=False, saved_PASS_used_as_new_payload=False,
        rows=[], checks={}, errors=[])
    counter = None

    def persist():
        result['wall_s'] = time.perf_counter()-started
        write_json(out/'validation.json', result)

    try:
        setup_started = time.perf_counter()
        system = build(config)
        result['build_wall_s'] = time.perf_counter()-setup_started
        result['source_identity_sha256'] = system['identity']['sha256']
        result['implementation_hashes'] = {str(path.relative_to(ROOT)): file_hash(path) for path in
            (Path(__file__), ROOT/'src/continuous_recovery/execution.py',
             ROOT/'src/continuous_recovery/experiment.py', ROOT/'src/continuous_tracking/plant.py')}
        saved_result = read_json(flight/'result.json')
        checkpoint = read_json(flight/'checkpoint.json')
        saved_rows = {int(row['step']): row for row in saved_result['rows']}
        counter = CallCounter(system)
        persist()
        for index, step in enumerate(selected):
            row = replay_one(system, counter, available[step], saved_rows, checkpoint)
            result['rows'].append(row)
            result['replayed_steps'] = len(result['rows'])
            if not row['passed']:
                result['errors'].append(dict(step=step, failed_checks=row['failed_checks']))
            if (index+1) % 25 == 0 or index+1 == len(selected):
                persist()
                print(json.dumps(dict(stage='execution_replay', completed=index+1,
                    requested=len(selected), passed_so_far=not result['errors'])), flush=True)
        totals = Counter()
        for calls in counter.counts.values():
            totals.update(calls)
        result['call_counts_by_phase'] = {key: dict(value) for key, value in counter.counts.items()}
        for phase in ('fresh_bind', 'seal', 'reused_commit', 'second_commit_probe'):
            result['call_counts_by_phase'].setdefault(phase, {})
        result['total_call_counts'] = dict(totals)
        result['ocp_solve_calls'] = sum(value for key, value in totals.items() if key.startswith('ocp_'))
        result['timing'] = {phase: timing_summary([row['wall_s'][phase] for row in result['rows']])
                           for phase in ('current_audit', 'fresh_bind', 'seal', 'fresh_commit', 'reused_commit')}
        bind = result['timing']['fresh_bind']['total_s']
        fresh = result['timing']['fresh_commit']['total_s']
        reused = result['timing']['reused_commit']['total_s']
        seal = result['timing']['seal']['total_s']
        result['timing']['fresh_over_reused_commit_ratio'] = fresh/reused
        result['timing']['fresh_over_seal_plus_reused_ratio'] = fresh/(seal+reused)
        result['timing']['fresh_bind_plus_commit_total_s'] = bind+fresh
        result['timing']['seal_plus_reused_total_s'] = seal+reused
        result['timing']['fresh_bind_plus_commit_over_seal_plus_reused_ratio'] = (bind+fresh)/(seal+reused)
        result['coverage'] = dict(selected_sources=dict(Counter(row['selected_source'] for row in result['rows'])),
            morph_steps=sum(row['morph_active'] for row in result['rows']),
            switched_steps=sum(row['switched'] for row in result['rows']),
            historical_ocp_steps=sum(row['historical_ocp_called'] for row in result['rows']),
            final_saved_state_sha256=digest(checkpoint['state']),
            saved_mission_completed=bool(saved_result.get('completed_1200s')),
            all_saved_steps_replayed=selected == sorted(available))
        result['checks'] = dict(all_selected_steps_replayed=len(result['rows']) == len(selected),
            every_step_exact=all(row['passed'] for row in result['rows']),
            no_ocp_solves=result['ocp_solve_calls'] == 0,
            expected_real_audit_calls=totals['audit'] == 2*len(selected),
            expected_real_integration_calls=totals['integrate'] == 4*len(selected),
            reused_commit_zero_calls=not result['call_counts_by_phase']['reused_commit'])
        result['passed'] = all(result['checks'].values()) and not result['errors']
    except Exception as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc))
        if counter is not None:
            result['call_counts_by_phase'] = {key: dict(value) for key, value in counter.counts.items()}
    finally:
        if counter is not None:
            counter.restore()
        persist()
    print(json.dumps({key: result[key] for key in ('passed', 'replayed_steps', 'wall_s', 'errors')},
                     ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
