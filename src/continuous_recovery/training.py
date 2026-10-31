"""Independent policy improvement on the continuously audited recovery controller.

Only an initial policy chain may cross from a previous controller. All returns,
labels, recovery state, certification and stopping evidence are generated anew.
Network fitting and policy-improvement helpers remain the original algorithm.
"""
from copy import deepcopy
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np

from certified_flight import CL_ACTIONS, integrity
from certified_flight.build import ROOT
from continuous_surrogate.model import FEATURE_SCHEMA, OUTPUT_SCHEMA
from continuous_surrogate.teacher import PROTOCOL as TEACHER_PROTOCOL
from rollout_improvement.ensemble import (HIDDEN, LABEL_DECAY, LABEL_KEEP, LABEL_SCALE,
    fit_ensemble, parse_hidden, retained_labels)
from rollout_improvement.features import STATE_INPUTS, input_names, parse_inputs
from rollout_improvement.identity import digest_json, experiment_identity
from rollout_improvement.policy import DECISION_S, constant_stage
from rollout_improvement.training import (save_json, load_json, pair_signature,
    previous_prediction_checks, prepare_iteration_directory, state_times, state_cost,
    margin_grid, positive_overrides, override_sets, improvable_fraction, stop_status, prune)
from .learned_policy import _validate_chain
from .local_model import MODEL_SCHEMA
from .preparation import DEFAULT_CONFIG, CONTROLLER_PROTOCOL

TRAINING_PROTOCOL = 'recovery-policy-improvement/1'


def read_initial_policy(path, initial_cl):
    """Import only a validated chain; never migrate legacy labels or return caches."""
    if path is None:
        return ([constant_stage(initial_cl)] if initial_cl != .04 else []), None
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    saved = json.loads(content.decode('utf-8-sig'))
    chain = saved.get('chain') if isinstance(saved, dict) else saved
    _validate_chain(chain)
    return deepcopy(chain), dict(path=str(path), sha256=integrity.sha256(path),
        chain_sha256=digest_json(chain), imported_fields=['chain'],
        migrated_labels=0, migrated_returns=0, migrated_certificates=0,
        migrated_stopping_records=0)


def inspect_inputs(config, model):
    """Validate the explicit immutable input pair before creating an output folder."""
    config, model = Path(config).resolve(strict=True), Path(model).resolve(strict=True)
    metadata = model.with_suffix('.json').resolve(strict=True)
    cfg, meta = load_json(config), load_json(metadata)
    if cfg.get('controller_protocol') != CONTROLLER_PROTOCOL:
        raise ValueError('Incompatible continuous-recovery controller config')
    for key, expected in (('protocol', TEACHER_PROTOCOL), ('model_schema', MODEL_SCHEMA),
                          ('feature_schema', FEATURE_SCHEMA), ('output_schema', OUTPUT_SCHEMA)):
        if meta.get(key) != expected:
            raise ValueError('Old or incompatible surrogate '+key)
    if meta.get('model_sha256') != integrity.sha256(model):
        raise ValueError('Local model hash mismatch')
    return dict(config=dict(path=str(config), sha256=integrity.sha256(config)),
                model=dict(path=str(model), sha256=integrity.sha256(model)),
                model_metadata=dict(path=str(metadata), sha256=integrity.sha256(metadata)))


def runtime_binding(inputs, initial_chain, settings, runtime=None):
    runtime = experiment_identity() if runtime is None else runtime
    payload = dict(protocol=TRAINING_PROTOCOL, runtime=runtime,
                   inputs={k: v['sha256'] for k, v in inputs.items()},
                   initial_chain_sha256=digest_json(initial_chain), settings=settings,
                   task=dict(mission_s=1200., execution_s=3., policy_decision_s=DECISION_S,
                             actions=CL_ACTIONS, fixed_config_flight_CL_used=False))
    return dict(payload=payload, sha256=digest_json(payload))


def validate_resume(st, binding):
    previous = st.get('experiment_binding')
    if st.get('training_protocol') != TRAINING_PROTOCOL or not isinstance(previous, dict):
        raise ValueError('checkpoint has no compatible recovery experiment fingerprint; '
                         'legacy returns, labels and stopping records cannot be resumed')
    if not isinstance(previous.get('payload'), dict) or previous.get('sha256') != digest_json(previous['payload']):
        raise ValueError('checkpoint recovery experiment fingerprint is invalid')
    if previous != binding or st.get('seed') != binding['payload']['settings']['seed']:
        raise ValueError('resume experiment mismatch: code, software, physics inputs, config, '
                         'model, metadata, initial policy and algorithm settings must match')
    spec = st.get('runtime_spec', {})
    hashes = binding['payload']['inputs']
    if (spec.get('config_sha256') != hashes['config']
            or spec.get('model_sha256') != hashes['model']
            or spec.get('model_metadata_sha256') != hashes['model_metadata']
            or spec.get('source_sha256') != binding['payload']['runtime']['sha256']
            or spec.get('runtime_identity') != binding['payload']['runtime']
            or spec.get('training_protocol') != TRAINING_PROTOCOL):
        raise ValueError('checkpoint runtime specification disagrees with its bound experiment')
    if st.get('G') is not None and not complete_candidate(st.get('nominal') or {}):
        raise ValueError('checkpoint lacks a successful full-horizon nominal policy result')
    nominal = st.get('nominal')
    if nominal is not None and (st.get('G') != nominal.get('G')
            or st.get('schedule') != nominal.get('schedule')
            or nominal.get('chain_sha256') != digest_json(st.get('chain'))
            or nominal.get('runtime_spec_sha256') != digest_json(spec)):
        raise ValueError('checkpoint cached return or nominal prefix belongs to another policy/runtime')
    _validate_chain(st.get('chain'))


def complete_candidate(result):
    """A short flight or UNKNOWN outcome never qualifies as a training score."""
    return (not result.get('error') and result.get('success') is True
            and result.get('completed_1200s') is True
            and float(result.get('time_s', -1.)) >= 1200.-1e-9
            and np.isfinite(result.get('G', np.nan)))


def _copy_checked(source, destination, expected):
    if integrity.sha256(source) != expected:
        raise ValueError('Input changed before snapshot: '+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if integrity.sha256(destination) != expected:
        raise ValueError('Input snapshot hash mismatch: '+str(destination))


def snapshot_inputs(out, inputs, chain, initial_reference, binding):
    folder = out/'inputs'
    folder.mkdir(parents=True, exist_ok=False)
    destinations = dict(config=folder/'controller.json', model=folder/'model.npz',
                        model_metadata=folder/'model.json')
    for name, record in inputs.items():
        _copy_checked(record['path'], destinations[name], record['sha256'])
    policy_path = folder/'initial_policy.json'
    save_json(policy_path, dict(chain=chain))
    spec = dict(config=str(destinations['config']), model=str(destinations['model']),
        config_sha256=inputs['config']['sha256'], model_sha256=inputs['model']['sha256'],
        model_metadata_sha256=inputs['model_metadata']['sha256'],
        source_sha256=binding['payload']['runtime']['sha256'],
        runtime_identity=binding['payload']['runtime'], training_protocol=TRAINING_PROTOCOL,
        initial_policy=str(policy_path), initial_policy_sha256=integrity.sha256(policy_path))
    save_json(out/'provenance.json', dict(protocol=TRAINING_PROTOCOL, inputs=inputs,
        initial_policy=initial_reference, initial_chain_sha256=digest_json(chain), runtime_spec=spec,
        binding=binding, labels_and_returns_recomputed=True, policy_driven=True,
        fixed_config_flight_CL_used=False, legacy_checkpoint_resume=False))
    manifest = {}
    runtime = binding['payload']['runtime']
    for rel, expected in runtime.get('sources', {}).items():
        _copy_checked(ROOT/rel, out/'source_snapshot'/rel, expected)
        manifest[rel] = expected
    save_json(out/'source_snapshot'/'manifest.json', manifest)
    return spec


def verify_snapshot(spec):
    for field, expected in (('config', 'config_sha256'), ('model', 'model_sha256'),
                            ('initial_policy', 'initial_policy_sha256')):
        if integrity.sha256(spec[field]) != spec[expected]:
            raise ValueError('Run input snapshot changed: '+field)
    if integrity.sha256(Path(spec['model']).with_suffix('.json')) != spec['model_metadata_sha256']:
        raise ValueError('Run input snapshot changed: model metadata')


def prepare_runtime(out, model, config=DEFAULT_CONFIG, initial_policy=None, initial_cl=.025, settings=None):
    """Create only frozen input/source evidence, without a flight or learner.

    This also serves bounded integration checks. Such a directory has no training
    state and cannot later masquerade as a completed learning iteration.
    """
    out = Path(out).resolve()
    if out == ROOT or not out.is_relative_to(ROOT):
        raise ValueError('Recovery output must be a descendant of the project root')
    if out.exists():
        raise ValueError('Runtime output already exists; preserve it and use a new directory')
    if initial_cl not in CL_ACTIONS:
        raise ValueError('initial_cl must be one of the 19 actions')
    refs = inspect_inputs(config, model)
    chain, reference = read_initial_policy(initial_policy, initial_cl)
    binding = runtime_binding(refs, chain, dict(settings or {}))
    out.mkdir(parents=True, exist_ok=False)
    spec = snapshot_inputs(out, refs, chain, reference, binding)
    return spec, chain, binding


def run_jobs(jobs, workers, costs=None):
    """One worker is genuinely serial; parallel use requires an explicit worker count."""
    from .training_rollouts import dispatch
    order = sorted(range(len(jobs)), key=lambda k: -costs[k]) if costs is not None else list(range(len(jobs)))
    out = [None]*len(jobs)
    if workers == 1:
        for k in order:
            out[k] = dispatch(jobs[k])
    elif jobs:
        with mp.get_context('spawn').Pool(min(workers, len(jobs))) as pool:
            for k, result in zip(order, pool.imap(dispatch, [jobs[k] for k in order], chunksize=1)):
                out[k] = result
    return out


def train(out, model, *, iterations, config=DEFAULT_CONFIG, initial_policy=None, seed=17, workers=1, states=25, stride=48., window=48., candidates=14, patience=None,
          min_advantage=1e-5, continuation='open-loop', inputs=STATE_INPUTS, overrides=False, initial_cl=.025,
          label_decay=LABEL_DECAY, label_keep=LABEL_KEEP, use_stop_rule=True, keep_evidence=True, hidden=HIDDEN,
          log=print):
    out = Path(out).resolve()
    if out == ROOT or not out.is_relative_to(ROOT):
        raise ValueError('New recovery training output must be a descendant of the project root')
    if iterations < 0 or workers < 1:
        raise ValueError('iterations must be nonnegative and workers must be positive')
    if stride % DECISION_S or stride <= 0 or states < 1:
        raise ValueError('stride must be a positive multiple of 12 s')
    if continuation not in ('open-loop', 'closed-loop'):
        raise ValueError(continuation)
    if initial_cl not in CL_ACTIONS:
        raise ValueError('initial_cl must be one of the 19 actions')
    hidden = parse_hidden(hidden)
    patience = patience or int(stride//DECISION_S)
    grid = margin_grid(candidates)
    inputs = parse_inputs(inputs)
    bound_settings = dict(seed=seed, states=states, stride=stride, window=window, candidates=candidates,
                          margins=grid, patience=patience, min_advantage=min_advantage, continuation=continuation,
                          inputs=input_names(inputs), overrides=overrides, initial_cl=initial_cl,
                          label_decay=label_decay, label_keep=label_keep, hidden=list(hidden))
    path = out/'state.json'
    st = load_json(path) if path.exists() else None
    if st is None and out.exists() and any(out.iterdir()):
        raise ValueError('output directory is nonempty without state.json; preserve it and use a new directory')
    inputs_ref = inspect_inputs(config, model)
    if st is not None and initial_policy is None:
        if st.get('training_protocol') != TRAINING_PROTOCOL or 'runtime_spec' not in st:
            raise ValueError('checkpoint has no compatible recovery experiment fingerprint')
        verify_snapshot(st['runtime_spec'])
        initial_chain, initial_ref = read_initial_policy(st['runtime_spec']['initial_policy'], initial_cl)
    else:
        initial_chain, initial_ref = read_initial_policy(initial_policy, initial_cl)
    binding = runtime_binding(inputs_ref, initial_chain, bound_settings)
    if st is not None:
        validate_resume(st, binding)
        verify_snapshot(st['runtime_spec'])
        runtime_spec = st['runtime_spec']
    else:
        out.mkdir(parents=True, exist_ok=True)
        runtime_spec = snapshot_inputs(out, inputs_ref, initial_chain, initial_ref, binding)
        st = dict(seed=seed, iteration=0, G=None, schedule=None, nominal=None,
                  policy_version=0, labels_current=[], label_archive=[], chain=initial_chain,
                  history=[], settings=[], training_protocol=TRAINING_PROTOCOL,
                  experiment_binding=binding, runtime_spec=runtime_spec)
    def job(kind, folder, chain, **extra):
        return kind, dict(folder=str(folder), chain=chain, runtime_spec=runtime_spec, **extra)
    stopped = st.get('stopped')
    if stopped and stopped['category'] != 'budget_limit' and use_stop_rule:
        log(f"already stopped ({stopped['category']}) at iteration {stopped['iteration']}: {stopped['reason']}; "
            'use --ignore-stop-rule to continue with an explicit iteration budget')
        return st
    if stopped:
        st.setdefault('stop_history', []).append(dict(st.pop('stopped'), ignored_rule=not use_stop_rule))
    out.mkdir(parents=True, exist_ok=True)
    st['settings'].append(dict(at_iteration=st['iteration'], states=states, stride=stride, window=window,
                               candidates=candidates, margins=grid, patience=patience, min_advantage=min_advantage,
                               continuation=continuation, inputs=input_names(inputs), overrides=overrides,
                               initial_cl=initial_cl, label_decay=label_decay, label_keep=label_keep,
                               stop_rule=use_stop_rule, workers=workers, hidden=list(hidden),
                               keep_evidence=keep_evidence, invocation_iterations=iterations))
    save_json(path, st)
    log(f"start: iteration {st['iteration']}, seed {st['seed']}, {states} states every {stride:g} s, "
        f"continuation={continuation}, inputs={','.join(input_names(inputs))}, overrides={overrides}, "
        f"hidden={','.join(map(str, hidden))}, workers={workers}")
    target = st['iteration']+iterations
    while st['iteration'] < target:
        i = st['iteration']
        idir = out/f'iter_{i:03d}'
        t0 = time.perf_counter()
        rec = dict(iteration=i)
        prepare_iteration_directory(idir, keep_evidence, log)
        times = state_times(i, states, stride)
        sweep = i//patience               # the neighbour radius (1 or 2) alternates per sweep

        # The nominal result is new-controller evidence, never imported from a checkpoint.
        if st['G'] is None or not st.get('nominal'):
            init = run_jobs([job('candidate', idir/'initial', st['chain'])], 1)[0]
            save_json(idir/'initial_result.json', init)
            if not complete_candidate(init):
                raise RuntimeError('initial policy did not complete 1200 s; evidence: '+str(idir/'initial_result.json'))
            st['G'], st['schedule'], st['nominal'] = init['G'], init['schedule'], init
            rec['initial'] = {k: v for k, v in init.items() if k not in ('schedule', 'decision_prefix')}
            # Save a completed initial flight before expensive branches. An interrupted
            # iteration still has zero completed learning iterations and empty labels.
            save_json(path, st)
            log(f"iter {i}: new-controller nominal policy G={init['G']:.6f} ({init['range_km']:.2f} km)")
        sched = st['schedule'] if continuation == 'open-loop' else None
        jobs = [job('branch', idir/f'branch_t{int(t):04d}', st['chain'], times=[t],
                    sweep=sweep, schedule=sched, nominal=st['nominal']) for t in times]
        costs = [state_cost(t) for t in times]
        res = run_jobs(jobs, workers, costs)
        save_json(idir/'branch_results.json', res)
        errors = [r['error'] for r in res if 'error' in r]
        if errors:
            raise RuntimeError('branch job failed:\n'+errors[0])
        rows = [r for w in res for r in w['rows']]
        if any(not np.isfinite(r.get('advantage', np.nan)) for r in rows if r['valid']):
            raise ValueError('A valid branch lacks a finite, verified same-prefix advantage')
        t_branch = time.perf_counter()-t0
        prefix = {float(t): x for w in res for t, x in w['prefix'].items()}
        new = [dict(features=f, action=a0, base_action=a0, t=t, advantage=0.) for t, (g, a0, f) in prefix.items()]
        new += [dict(features=r['features'], action=r['action'], base_action=r['base_action'], t=r['t'],
                     advantage=r['advantage']) for r in rows if r['valid']]
        save_json(idir/'labels.json', dict(rows=rows, labels=new))
        version = st['policy_version']
        for x in new:
            x['v'] = version
        st['labels_current'] += new
        st['label_archive'] += new
        if label_decay > 0:
            fit_labels, fit_weights = retained_labels(st['label_archive'], version, label_keep, label_decay)
            st['label_archive'] = fit_labels
        else:
            fit_labels, fit_weights = st['labels_current'], None
            st['label_archive'] = list(st['labels_current'])
        frac = improvable_fraction(new, min_advantage)
        n_pos = sum(1 for x in new if x['action'] != x['base_action'] and x['advantage'] > min_advantage)
        n_invalid = sum(not r['valid'] for r in rows)
        log(f"iter {i}: {len(new)} labels ({len(rows)} branches, {n_invalid} invalid) in {t_branch:.0f} s; "
            f"positive pairs {n_pos}, improvable states {frac:.0%}")

        # 2. out-of-sample check of the previous ensemble, then fit
        check, check_split = previous_prediction_checks(st, new, min_advantage)
        if check:
            log(f"iter {i}: previous ensemble on {check['pairs']} unseen pairs: Spearman "
                + (f"{check['spearman']:.2f}" if check['spearman'] is not None else '-')
                + (f", R2 {check['r2_clipped']:.2f}" if check['r2_clipped'] is not None else ''))
        if check_split['seen_pairs']:
            log(f"iter {i}: excluded {check_split['seen_pairs']} previously fitted pairs from unseen diagnostics")
        tick = time.perf_counter()
        members, fit = fit_ensemble(fit_labels, st['seed']*1000+i, fit_weights, inputs, hidden)
        fit['versions'] = sorted({x['v'] for x in fit_labels})
        t_fit = time.perf_counter()-tick
        st['check_members'], st['check_chain_len'], st['check_inputs'] = members, len(st['chain']), inputs
        st['check_pair_signatures'] = sorted({pair_signature(x, inputs) for x in fit_labels})
        label_times = sorted({float(x['t']) for x in fit_labels})

        # 3. candidates
        ov = override_sets(positive_overrides(st['labels_current'], min_advantage)) if overrides else []
        n_model = max(1, candidates-len(ov)) if ov else len(grid)
        mgrid = margin_grid(n_model) if n_model < len(grid) else grid
        specs = [dict(kind='model', name=f'candidate_m{m:g}', margin=m,
                      chain=st['chain']+[dict(kind='model', members=members, margin=m, iteration=i, times=label_times,
                                              window=window, **({} if inputs is None else dict(inputs=inputs)))])
                 for m in mgrid]
        for es in ov:
            entries = [dict(t=float(x['t']), features=[float(z) for z in x['features']], action=int(x['action']),
                            base_action=int(x['base_action']), advantage=float(x['advantage'])) for x in es]
            specs.append(dict(kind='override', name=f'candidate_override{len(es)}', n=len(es),
                              predicted_gain=sum(x['advantage'] for x in es),
                              chain=st['chain']+[dict(kind='override', entries=entries, match='t', iteration=i)]))
        tick = time.perf_counter()
        cres = run_jobs([job('candidate', idir/c['name'], c['chain']) for c in specs], min(len(specs), workers))
        t_cand = time.perf_counter()-tick
        candidate_errors = [r['error'] for r in cres if 'error' in r]
        candidate_evidence = idir/'candidate_results.json'
        save_json(candidate_evidence, dict(iteration=i, complete=not candidate_errors, wall_s=t_cand,
                                           specs=specs, results=cres))
        for c, r in zip(specs, cres):
            what = f"margin {c['margin']:g}" if c['kind'] == 'model' else f"override x{c['n']}"
            if 'error' in r:
                log(f"  candidate {what}: ERROR\n{r['error']}")
            else:
                log(f"  candidate {what}: G={r['G']:.6f} ({(r['G']-st['G'])*LABEL_SCALE:+.3f}e-4) {r['range_km']:.2f} km "
                    f"stop={r['stop']} switches={r['switches']}")
        if candidate_errors:
            raise RuntimeError(f'{len(candidate_errors)} candidate jobs failed; iteration {i} is incomplete and '
                               f'was not counted toward acceptance or stopping. Evidence: {candidate_evidence}\n'
                               + str(candidate_errors[0]))

        # 4. acceptance
        ok = [(r['G'], c, r) for c, r in zip(specs, cres) if complete_candidate(r)]
        best = max(ok, key=lambda x: x[0]) if ok else None
        accepted = bool(best and best[0] > st['G']+1e-9)
        invalid_stops = {}
        for row in rows:
            if not row['valid']:
                reason = str(row.get('stop') or row.get('outcome') or 'UNSPECIFIED')
                invalid_stops[reason] = invalid_stops.get(reason, 0)+1
        failed_candidates = sum(not complete_candidate(r) for r in cres)
        rec.update(labels=len(new), invalid=n_invalid+failed_candidates, branch_invalid=n_invalid,
                   candidate_invalid=failed_candidates, invalid_stops=invalid_stops,
                   improvable_fraction=frac, positive_pairs=n_pos, sweep=sweep,
                   prediction_check=check, prediction_diagnostics=check_split, fit=fit, G_before=st['G'], accepted=accepted,
                   candidates=[dict(kind=c['kind'], margin=c.get('margin'), overrides=c.get('n'),
                                    **{k: v for k, v in r.items() if k not in ('schedule', 'decision_prefix')}) for c, r in zip(specs, cres)],
                   wall_s=dict(branch=t_branch, fit=t_fit, candidates=t_cand, total=time.perf_counter()-t0))
        if accepted:
            c = best[1]
            st['chain'], st['G'], st['schedule'] = c['chain'], best[0], best[2]['schedule']
            st['nominal'] = best[2]
            st['labels_current'] = []
            st['policy_version'] = version+1
            rec.update(accepted_kind=c['kind'], accepted_margin=c.get('margin'), accepted_overrides=c.get('n'),
                       range_km=best[2]['range_km'])
        if not keep_evidence:
            prune(idir, best[1]['name'] if accepted else None)
        st['history'].append(rec)
        st['iteration'] = i+1
        status = stop_status(st['history'], patience) if use_stop_rule else None
        if status:
            st['stopped'] = dict(status, iteration=i, G=st['G'])
        save_json(path, st)
        log(f"iter {i} done: " + (f"accepted {best[1]['name'].removeprefix('candidate_')}" if accepted else 'kept policy')
            + f"; G={st['G']:.6f} (wall {rec['wall_s']['total']:.0f} s)")
        if status:
            log(f"stopped at iteration {i} ({status['category']}): {status['reason']}")
            break
    if not st.get('stopped'):
        st['stopped'] = dict(category='budget_limit', reason=f'invocation budget of {iterations} iterations reached',
                             convergence_proven=False, iteration=st['iteration']-1, G=st['G'])
        save_json(path, st)
    return st


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='Independent run inside the repository; exact resume only')
    parser.add_argument('--model', required=True, type=Path, help='Frozen local-model NPZ with JSON metadata')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--initial-policy', type=Path, help='Import chain only; no old labels, returns or stop state')
    parser.add_argument('--initial-cl', type=float, default=.025, help='Constant policy if no initial chain is given')
    parser.add_argument('--iterations', required=True, type=int,
                        help='Explicit maximum learning iterations for this invocation; 0 initializes without flight')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--workers', type=int, default=1, help='Certified rollout workers; serial by default')
    parser.add_argument('--states', type=int, default=25)
    parser.add_argument('--stride', type=float, default=48.)
    parser.add_argument('--window', type=float, default=48.)
    parser.add_argument('--candidates', type=int, default=14)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--min-advantage', type=float, default=1e-5)
    parser.add_argument('--continuation', choices=['open-loop', 'closed-loop'], default='open-loop')
    parser.add_argument('--inputs', default=STATE_INPUTS)
    parser.add_argument('--overrides', action='store_true')
    parser.add_argument('--label-decay', type=float, default=LABEL_DECAY)
    parser.add_argument('--label-keep', type=int, default=LABEL_KEEP)
    parser.add_argument('--ignore-stop-rule', action='store_true')
    parser.add_argument('--hidden', default=','.join(map(str, HIDDEN)))
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument('--keep-evidence', dest='keep_evidence', action='store_true', default=True)
    evidence.add_argument('--discard-rejected-evidence', dest='keep_evidence', action='store_false',
                          help='Explicitly discard rejected branches/candidates; default keeps all evidence')
    args = parser.parse_args(argv)
    result = train(args.out, args.model, iterations=args.iterations, config=args.config,
        initial_policy=args.initial_policy, initial_cl=args.initial_cl, seed=args.seed,
        workers=args.workers, states=args.states, stride=args.stride, window=args.window,
        candidates=args.candidates, patience=args.patience, min_advantage=args.min_advantage,
        continuation=args.continuation, inputs=args.inputs, overrides=args.overrides,
        label_decay=args.label_decay, label_keep=args.label_keep, hidden=args.hidden,
        use_stop_rule=not args.ignore_stop_rule, keep_evidence=args.keep_evidence,
        log=lambda message: print(message, flush=True))
    print(json.dumps(dict(protocol=TRAINING_PROTOCOL, iteration=result['iteration'], G=result['G'],
                          stopped=result.get('stopped'), state=str(Path(args.out).resolve()/'state.json'))),
          flush=True)
    return 0

