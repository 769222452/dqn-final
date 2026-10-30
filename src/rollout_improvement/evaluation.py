"""Generalisation tests of a trained policy.

1. Initial conditions: certified flights from the nominal and perturbed initial states (altitude and speed
   offsets), compared with a constant-CL reference policy flown from the same states.
2. Within the mission: at given decision times the action is changed by one class for one decision; the flight
   is continued once by the policy re-deciding (closed loop) and once by replaying its nominal decisions (open
   loop). A_closed - A_open measures how the policy reacts to states slightly off its own trajectory.

Test sets:
- development: 12 perturbed initial states on a grid and 7 probe times, used while the method was developed;
- held-out-1, held-out-2, held-out-3: 40 initial states each, drawn uniformly from the same offset range, and
  7 probe times each, drawn from the decision times not used by the earlier sets; generated from a fixed seed
  with separate random streams. held-out-3 is the test set of the proposed method.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from certified_flight import build_flight
from certified_flight.serialization import write_json

from .ensemble import LABEL_SCALE
from .features import features
from .identity import digest_json, experiment_identity
from .policy import MAX_ACTION, ChainPolicy, chain_action, constant_stage
from .rollouts import certified_fly, mission_completed

INITIAL_OFFSETS = [(dh, dv) for dh in (-500., 0., 500.) for dv in (-20., 0., 20.)] + \
                  [(-2000., 0.), (2000., 0.), (0., -100.), (0., 100.)]
PROBE_TIMES = (36., 96., 168., 240., 360., 480., 600.)

HELD_OUT_SEED = 20261023
HELD_OUT_STATES = 40
MAX_OFFSET = (2000., 100.)   # largest altitude (m) and speed (m/s) offsets of the development set


def held_out_offsets(n=HELD_OUT_STATES, seed=HELD_OUT_SEED, stream=0):
    """n initial-state offsets drawn uniformly from |dh| <= 2000 m, |dV| <= 100 m/s (rounded to 1 m, 0.1 m/s)."""
    rng = np.random.default_rng([seed, stream])
    dh = np.round(rng.uniform(-MAX_OFFSET[0], MAX_OFFSET[0], n), 0)
    dv = np.round(rng.uniform(-MAX_OFFSET[1], MAX_OFFSET[1], n), 1)
    return [(float(a), float(b)) for a, b in zip(dh, dv)]


def held_out_probe_times(n=len(PROBE_TIMES), seed=HELD_OUT_SEED, stream=1, exclude=PROBE_TIMES):
    """n decision times (12 s grid, 12-1176 s) drawn without replacement from those not in `exclude`."""
    candidates = [12.*k for k in range(1, 99) if 12.*k not in exclude]
    rng = np.random.default_rng([seed, stream])
    return tuple(sorted(float(t) for t in rng.choice(candidates, n, replace=False)))


def test_set_points(name):
    """(initial offsets including the nominal state, probe times) of a test set."""
    if name == 'development':
        return list(INITIAL_OFFSETS), PROBE_TIMES
    first_times = held_out_probe_times()
    if name == 'held-out-1':
        return [(0., 0.)]+held_out_offsets(), first_times
    second_times = held_out_probe_times(stream=3, exclude=PROBE_TIMES+first_times)
    if name == 'held-out-2':
        return [(0., 0.)]+held_out_offsets(stream=2), second_times
    if name == 'held-out-3':
        return [(0., 0.)]+held_out_offsets(stream=4), \
            held_out_probe_times(stream=5, exclude=PROBE_TIMES+first_times+second_times)
    raise ValueError(f'unknown test set {name!r}')


TEST_SETS = ('held-out-3', 'held-out-2', 'held-out-1', 'development')


def paired_bootstrap_ci(values, level=.95, resamples=10000, seed=0):
    """Percentile bootstrap confidence interval of the mean of paired differences."""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    return [float(np.quantile(means, (1-level)/2)), float(np.quantile(means, (1+level)/2))]


def _name(d):
    return f'dh{d[0]:g}_dv{d[1]:g}'


EVALUATION_SCHEMA = 'certified-policy-evaluation/2'


class EvaluationCacheMismatch(ValueError):
    """An existing result belongs to another experiment and must remain untouched."""


def _request_identity(kind, chain, p, schedule, runtime):
    request = dict(schema=EVALUATION_SCHEMA, runtime=runtime, kind=kind, chain=chain,
                   parameters=list(p), schedule=None if schedule is None else
                   sorted((int(round(float(t))), int(a)) for t, a in schedule.items()))
    return dict(schema=EVALUATION_SCHEMA, sha256=digest_json(request), runtime=runtime)


def _counts(results, requested):
    return dict(requested=requested, returned=len(results),
                complete=sum(bool(r.get('completed')) for r in results),
                failed=sum(not r.get('completed', False) and not r.get('error') and not r.get('skipped')
                           for r in results),
                skipped=sum(bool(r.get('skipped')) for r in results),
                error=sum(bool(r.get('error')) for r in results), missing=requested-len(results))


def _mean(values):
    return float(np.mean(values)) if values else None


def _finish_result(out, res, identity, run_dir, discard_evidence):
    res.update(identity=identity, schema=EVALUATION_SCHEMA, evidence_path=str(run_dir),
               evidence_retained=not discard_evidence)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'result.json', res)
    if discard_evidence and run_dir.exists():
        # run_dir is always an immediate child created for this exact request.
        if run_dir.resolve().parent != out.resolve():
            raise ValueError('refusing to discard evidence outside the evaluation directory')
        shutil.rmtree(run_dir)
    return res


def _job(args):
    """Execute/cache one exact request; failures can retry without losing previous evidence."""
    kind, chain, p, schedule, folder = args[:5]
    runtime = args[5] if len(args) > 5 else experiment_identity()
    discard_evidence = bool(args[6]) if len(args) > 6 else False
    identity = _request_identity(kind, chain, p, schedule, runtime)
    out = Path(folder)
    done = out/'result.json'
    attempt = 1
    if done.exists():
        previous = json.loads(done.read_text(encoding='utf-8'))
        if previous.get('identity') != identity or previous.get('schema') != EVALUATION_SCHEMA:
            raise EvaluationCacheMismatch(f'{done} belongs to another or unversioned evaluation; '
                                          'choose a new output directory. Existing evidence was preserved.')
        if not previous.get('error'):
            return previous
        # A computation/I/O failure is not a permanently cached scientific result.
        attempt = int(previous.get('attempt', 1)) + 1
        archived = out/f'result.attempt{attempt-1:04d}.json'
        if archived.exists():
            raise EvaluationCacheMismatch(f'previous-attempt archive already exists: {archived}')
        write_json(archived, previous)
    run_dir = out/('run' if attempt == 1 else f'run_attempt{attempt:04d}')
    if run_dir.exists():
        raise EvaluationCacheMismatch(f'{run_dir} already contains incomplete evidence; '
                                      'choose a new output directory instead of overwriting it')
    try:
        tick = time.perf_counter()
        if kind == 'initial':
            env, certifier = build_flight(run_dir, 'evaluation', initial_offset=p)
            pi = ChainPolicy(chain, env)
            g, stop, acts = certified_fly(certifier, pi)
            res = dict(kind=kind, dh=p[0], dV=p[1], G=g, stop=stop,
                       outcome=env.last_outcome, completed=mission_completed(env, stop),
                       stop_time_s=float(env.t), range_km=float(env.x[3]/1000), switches=int(env.switches),
                       cl_action_changes=int(sum(a != b for a, b in zip(acts, acts[1:]))),
                       schedule=[[float(t), int(a)] for t, a in pi.decisions])
            executed = acts
        elif kind == 'probe':
            t_b, delta, loop = p
            if loop not in ('closed', 'open'):
                raise ValueError(f'unknown probe loop {loop!r}')
            env, certifier = build_flight(run_dir, 'evaluation')
            g0, stop, acts0 = certified_fly(certifier, ChainPolicy(chain, env), until=t_b)
            executed = acts0
            res = dict(kind=kind, t=t_b, delta=delta, loop=loop, completed=False,
                       stop=stop, stop_time_s=float(env.t), outcome=env.last_outcome)
            if stop is not None or abs(float(env.t)-float(t_b)) > 1e-8:
                res.update(valid=False, failure_reason='PROBE_PREFIX_NOT_REACHED', G=g0,
                           range_km=float(env.x[3]/1000), switches=int(env.switches))
            else:
                a = chain_action(chain, features(env))+delta
                if not 0 <= a <= MAX_ACTION:
                    res.update(skipped=True, valid=False, skip_reason='ACTION_OUT_OF_RANGE')
                else:
                    pol = ChainPolicy(chain, env, override=(t_b, a),
                                      schedule=schedule if loop == 'open' else None)
                    g1, stop, acts1 = certified_fly(certifier, pol)
                    changed = sum(1 for t, x in pol.decisions if t > t_b+1e-9 and
                                  x != schedule[int(round(t))]) if loop == 'closed' else 0
                    completed = mission_completed(env, stop)
                    acts = acts0+acts1
                    executed = acts
                    res.update(G=g0+g1, stop=stop, completed=completed, valid=completed,
                               outcome=env.last_outcome, stop_time_s=float(env.t),
                               range_km=float(env.x[3]/1000), switches=int(env.switches),
                               cl_action_changes=int(sum(a != b for a, b in zip(acts, acts[1:]))),
                               redecisions_changed=changed)
        else:
            raise ValueError(f'unknown evaluation job kind {kind!r}')
        res.update(geometry_switches=int(env.switches), steps=len(executed), time_s=float(env.t),
                   final_state=env.x.tolist())
        res['wall_s'] = time.perf_counter()-tick
    except Exception:
        res = dict(kind=kind, p=list(p), completed=False, error=traceback.format_exc(),
                   wall_s=time.perf_counter()-tick)
    res['attempt'] = attempt
    return _finish_result(out, res, identity, run_dir, discard_evidence)


def evaluate(chain, out, reference_cl=.025, workers=8, log=print, test_set='held-out-2', reference_out=None,
             discard_evidence=False):
    """Run paired tests. Missing/error cases withhold the main aggregate; evidence is kept by default."""
    offsets, probe_times = test_set_points(test_set)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(reference_out) if reference_out else out
    ref_chain = [constant_stage(reference_cl)]
    runtime = experiment_identity()
    comparison_identity = dict(schema=EVALUATION_SCHEMA, runtime=runtime, test_set=test_set,
                               offsets=offsets, probe_times=probe_times, reference_chain=ref_chain)
    summary_identity = dict(comparison=comparison_identity, chain_sha256=digest_json(chain))
    old_summary = out/'summary.json'
    if old_summary.exists():
        old = json.loads(old_summary.read_text(encoding='utf-8'))
        # JSON round-tripping normalises tuple/list representations in test points.
        if digest_json(old.get('identity')) != digest_json(summary_identity):
            raise EvaluationCacheMismatch(f'{old_summary} belongs to another or unversioned evaluation; '
                                          'choose a new output directory')
    def job(kind, policy, p, schedule, folder):
        return kind, policy, p, schedule, str(folder), runtime, discard_evidence
    with mp.get_context('spawn').Pool(workers) as pool:
        jobs = [job('initial', chain, d, None, out/f'initial_{_name(d)}') for d in offsets]
        jobs += [job('initial', ref_chain, d, None, ref_dir/f'reference_cl{reference_cl:g}_{_name(d)}')
                 for d in offsets]
        res = pool.map(_job, jobs, chunksize=1)
        inits, refs = res[:len(offsets)], res[len(offsets):]
        nominal = next((r for r in inits if r.get('dh') == 0 and r.get('dV') == 0 and not r.get('error')), None)
        nominal_ok = bool(nominal and nominal.get('completed'))
        if nominal_ok:
            schedule = {int(round(t)): a for t, a in nominal['schedule']}
            probes = pool.map(_job, [job('probe', chain, (t, d, loop), schedule,
                                         out/f'probe_t{int(t):04d}_{d:+d}_{loop}')
                                     for t in probe_times for d in (-1, 1) for loop in ('closed', 'open')],
                              chunksize=1)
        else:
            # No full nominal schedule exists; do not run or invent a perturbed continuation.
            probes = [dict(kind='probe', t=t, delta=d, loop=loop, skipped=True, completed=False,
                           skip_reason='NOMINAL_MISSION_NOT_COMPLETED')
                      for t in probe_times for d in (-1, 1) for loop in ('closed', 'open')]
    ref = {(r['dh'], r['dV']): r for r in refs if not r.get('error')}
    rows = []
    initial_missing = []
    for offset, r, reference in zip(offsets, inits, refs):
        if r.get('error') or reference.get('error'):
            initial_missing.append(dict(dh=offset[0], dV=offset[1],
                                        reason='INITIAL_OR_REFERENCE_JOB_ERROR'))
            continue
        rows.append(dict(dh=r['dh'], dV=r['dV'], range_km=r['range_km'],
                         gain_km=r['range_km']-reference['range_km'], switches=r['switches'],
                         cl_action_changes=r['cl_action_changes'], stop=r['stop'],
                         completed=r['completed'], outcome=r['outcome'], reference_stop=reference['stop'],
                         reference_completed=reference['completed'], reference_outcome=reference['outcome']))
    perturbed = [x for x in rows if (x['dh'], x['dV']) != (0., 0.)]
    gains = [x['gain_km'] for x in perturbed]
    gains_ref_done = [x['gain_km'] for x in perturbed if x['reference_completed']]
    by = {}
    for requested, r in zip([(t, d, loop) for t in probe_times for d in (-1, 1) for loop in ('closed', 'open')], probes):
        by[requested] = r
    pairs, missing_pairs = [], []
    for t in probe_times:
        for d in (-1, 1):
            c, o = by[(t, d, 'closed')], by[(t, d, 'open')]
            if nominal_ok and c.get('valid') and o.get('valid'):
                pairs.append(dict(t=t, delta=d, A_closed=LABEL_SCALE*(c['G']-nominal['G']),
                                  A_open=LABEL_SCALE*(o['G']-nominal['G']),
                                  redecisions_changed=c['redecisions_changed'], stops=[c['stop'], o['stop']]))
            else:
                missing_pairs.append(dict(t=t, delta=d, reasons=[
                    r.get('failure_reason') or r.get('skip_reason') or
                    ('JOB_ERROR' if r.get('error') else r.get('stop') or 'INVALID_RESULT')
                    for r in (c, o)]))
    nominal_ref = ref.get((0., 0.))
    subset = dict(perturbed_mean_gain_km=_mean(gains), perturbed_gain_ci95_km=paired_bootstrap_ci(gains),
                  perturbed_worst_gain_km=float(min(gains)) if gains else None,
                  reference_completed_mean_gain_km=_mean(gains_ref_done),
                  reference_completed_gain_ci95_km=paired_bootstrap_ci(gains_ref_done),
                  perturbed_mean_switches=_mean([x['switches'] for x in perturbed]),
                  probe_mean_A_closed=_mean([p['A_closed'] for p in pairs]),
                  probe_mean_A_open=_mean([p['A_open'] for p in pairs]))
    summary = dict(schema=EVALUATION_SCHEMA, identity=summary_identity, comparison_identity=comparison_identity,
                   test_set=test_set, reference_cl=reference_cl,
                   complete=not initial_missing and not missing_pairs,
                   incomplete_reasons=([f'{len(initial_missing)} initial/reference pairs unavailable'] if initial_missing else [])+
                                      ([f'{len(missing_pairs)} probe pairs unavailable'] if missing_pairs else []),
                   coverage=dict(initial=_counts(inits, len(offsets)), reference=_counts(refs, len(offsets)),
                                 probes=_counts(probes, len(probe_times)*4)),
                   initial_missing=initial_missing, probe_missing_pairs=missing_pairs,
                   nominal_range_km=nominal['range_km'] if nominal else None,
                   nominal_G=nominal['G'] if nominal else None,
                   nominal_gain_km=nominal['range_km']-nominal_ref['range_km'] if nominal and nominal_ref else None,
                   nominal_switches=nominal['switches'] if nominal else None,
                   nominal_cl_action_changes=nominal['cl_action_changes'] if nominal else None,
                   nominal_completed=bool(nominal_ok), perturbed_states=len(gains),
                   perturbed_requested_states=sum(d != (0., 0.) for d in offsets),
                   perturbed_wins=int(sum(g > 0 for g in gains)),
                   perturbed_completed=int(sum(x['completed'] for x in perturbed)),
                   perturbed_reference_completed=int(sum(x['reference_completed'] for x in perturbed)),
                   probe_requested_pairs=len(probe_times)*2, probe_valid_pairs=len(pairs),
                   probe_closed_not_worse=int(sum(p['A_closed'] >= p['A_open'] for p in pairs)),
                   subset_statistics=subset, initial_conditions=rows, probes=pairs,
                   job_errors=[dict(kind=r['kind'], p=r.get('p'), error=r['error']) for r in res+probes if r.get('error')])
    # Partial subsets remain inspectable, but cannot masquerade as the requested test-set means.
    for key, value in subset.items():
        summary[key] = value if (not missing_pairs if key.startswith('probe_') else not initial_missing) else None
    write_json(out/'summary.json', summary)
    log(f"evaluation {'complete' if summary['complete'] else 'INCOMPLETE'}: "
        f"initial pairs {len(rows)}/{len(offsets)}, probe pairs {len(pairs)}/{len(probe_times)*2}")
    if nominal:
        log(f"nominal {nominal['range_km']:.2f} km; completed={nominal['completed']}; "
            f"geometry switches={nominal['switches']}, CL action changes={nominal['cl_action_changes']}")
    for reason in summary['incomplete_reasons']:
        log('  '+reason+'; aggregate withheld, inspect coverage, job_errors and subset_statistics')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description='Generalisation tests of a trained policy (state.json of a run).')
    p.add_argument('state', help='state.json written by training')
    p.add_argument('--out', required=True)
    p.add_argument('--reference-cl', type=float, default=.025, help='constant-CL reference policy')
    p.add_argument('--workers', type=int, default=min(14, os.cpu_count() or 1))
    p.add_argument('--test-set', choices=TEST_SETS, default=TEST_SETS[0],
                   help='held-out-3 (default): test set of the proposed method; held-out-1, held-out-2: earlier '
                        'held-out sets; development: tests used during development')
    p.add_argument('--reference-out', help='folder caching the reference flights (shared between policies)')
    p.add_argument('--discard-evidence', action='store_true',
                   help='explicitly delete each new job flight evidence after its result is saved')
    a = p.parse_args(argv)
    chain = json.loads(Path(a.state).read_text(encoding='utf-8-sig'))['chain']
    summary = evaluate(chain, a.out, a.reference_cl, a.workers, log=lambda s: print(s, flush=True), test_set=a.test_set,
             reference_out=a.reference_out, discard_evidence=a.discard_evidence)
    return 0 if summary['complete'] else 2
