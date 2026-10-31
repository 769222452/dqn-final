"""One explicitly budgeted iteration with exact action-pair support.

The imported chain is a frozen reference, not a claim that its historical
stages have state support. Only newly appended stages use this protocol.
No cross-controller or cross-reference advantage labels are imported.
"""
from copy import deepcopy
import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np

from certified_flight import CL_ACTIONS
from continuous_recovery.preparation import DEFAULT_CONFIG
from continuous_recovery.training import prepare_runtime, complete_candidate
from rollout_improvement.ensemble import HIDDEN, fit_ensemble
from rollout_improvement.features import STATE_INPUTS, parse_inputs
from rollout_improvement.rollouts import alternatives
from rollout_improvement.training import margin_grid, state_times
from continuous_tracking.experiment import clean
from .support import (build_support_label, canonical_hash, continuation_identity,
                      context_hash, prefix_chain_hash)

PROTOCOL = 'supported-recovery-policy-improvement/1'
ACTION_SEMANTICS = 'certified_policy_request'


def save(path, value):
    """Atomic finite JSON; a failed write never replaces a completed checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(data, encoding='utf-8', newline='\n')
    temporary.replace(path)


def _dispatch(job):
    from .rollouts import paired_job, candidate_job
    kind, args = job
    return (paired_job if kind == 'pair' else candidate_job)(args)


def jobs(batch, workers):
    if workers == 1:
        return [_dispatch(job) for job in batch]
    with mp.get_context('spawn').Pool(min(workers, len(batch))) as pool:
        return list(pool.imap(_dispatch, batch, chunksize=1))


class Budget:
    """Charge launches, including failed attempts, before any flight starts."""
    def __init__(self, pairs, candidates):
        self.limits = dict(pair=pairs, candidate=candidates, baseline=1)
        self.used = dict.fromkeys(self.limits, 0)

    def charge(self, kind, count=1):
        if count < 0 or self.used[kind]+count > self.limits[kind]:
            raise ValueError('Launch would exceed '+kind+' budget')
        self.used[kind] += count

    def state(self):
        return dict(limits=dict(self.limits), launched=dict(self.used),
                    paired_continuations_launched=2*self.used['pair'],
                    baseline_and_candidates_separate=True)


def nominal_queries(nominal, count):
    """Spread the bounded sample across the original 48 s sweep, not a prefix."""
    times = state_times(0, 25, 48.)
    if not 1 <= count <= len(times):
        raise ValueError('Nominal pair count must be from 1 to 25')
    selected = np.linspace(0, len(times)-1, count, dtype=int)
    result = []
    for j, index in enumerate(selected):
        tick = times[int(index)]
        binding = nominal['decision_prefix'][str(int(tick))]
        base = int(binding['action'])
        options = alternatives(base, tick, 0)
        result.append(dict(binding=deepcopy(binding), base_action=base,
                           action=int(options[j % len(options)]), origin='nominal'))
    return result


def labels_from_pairs(results, reference_chain, continuation):
    """Only complete, same-origin paired returns produce a supervised target."""
    labels = []
    prefix = prefix_chain_hash(reference_chain)
    for result in results:
        if result.get('error'):
            raise RuntimeError('Paired rollout infrastructure error: '+result['error'])
        if result.get('valid') is not True:
            continue
        if (result.get('reference_prefix_sha256') != prefix
                or result.get('continuation_sha256') != continuation['sha256']
                or result.get('action_semantics') != ACTION_SEMANTICS):
            raise ValueError('Paired label belongs to another reference/continuation/semantics')
        binding = result['binding']
        base, action = int(result['base_action']), int(result['action'])
        advantage = float(result['advantage'])
        if not np.isfinite(advantage):
            raise ValueError('Valid paired target must be finite')
        for candidate, target, kind in ((base, 0., 'baseline_zero'), (action, advantage, 'branch')):
            evidence = build_support_label(binding['state'],
                reference_prefix_sha256=prefix, continuation_sha256=continuation['sha256'],
                base_action=base, candidate_action=candidate, label_kind=kind, advantage=target)
            labels.append(dict(features=deepcopy(binding['features']), action=candidate,
                base_action=base, t=float(binding['t']), advantage=target,
                support=evidence, reference_version=0,
                evidence_path=result.get('evidence_path'), origin=result.get('query_origin')))
    return labels


def fitting_rows(labels, inputs):
    """Do not silently merge incompatible hidden states into one regression cell."""
    unique, observed = {}, {}
    for row in labels:
        key = (row['support']['context_sha256'], row['base_action'], row['action'])
        if key in unique and unique[key]['advantage'] != row['advantage']:
            raise ValueError('Repeated exact action pair has conflicting measured returns')
        unique[key] = row
        ix = range(len(row['features'])) if inputs is None else inputs
        feature_key = (tuple(row['features'][i] for i in ix), row['action'])
        previous = observed.get(feature_key)
        if previous is not None and (previous['base_action'] != row['base_action']
                                      or previous['advantage'] != row['advantage']):
            raise ValueError('Network-input alias: incompatible action targets; no silent averaging')
        observed[feature_key] = row
    return list(unique.values())


def make_stages(reference_chain, labels, continuation, seed, inputs, margins):
    rows = fitting_rows(labels, inputs)
    if not rows:
        return [], dict(reason='NO_COMPLETE_PAIRED_LABELS', fitted=False)
    identities = {r['support']['runtime_identity'] for r in rows}
    if len(identities) != 1:
        raise ValueError('Mixed-controller labels cannot fit a supported stage')
    for row in rows:
        support = row['support']
        if (support.get('reference_prefix_sha256') != prefix_chain_hash(reference_chain)
                or support.get('continuation_sha256') != continuation['sha256']
                or support.get('action_semantics') != ACTION_SEMANTICS
                or support.get('valid') is not True
                or support.get('candidate_action') != row['action']
                or support.get('base_action') != row['base_action']
                or support.get('advantage') != row['advantage']
                or support.get('context_sha256') != canonical_hash(support.get('context'))):
            raise ValueError('Incompatible reference/continuation/action support in fit labels')
    started = time.perf_counter()
    members, diagnostics = fit_ensemble(rows, seed, None, inputs, HIDDEN)
    prototype = dict(kind='supported_model', members=members, inputs=inputs,
        support_mode='exact', action_semantics=ACTION_SEMANTICS,
        reference_prefix_sha256=prefix_chain_hash(reference_chain),
        runtime_identity=next(iter(identities)), continuation=deepcopy(continuation),
        continuation_sha256=continuation['sha256'],
        support_records=[deepcopy(r['support']) for r in rows])
    chains = [deepcopy(reference_chain)+[dict(deepcopy(prototype), margin=float(m))] for m in margins]
    return chains, dict(diagnostics, fitted=True, labels=len(rows), input_indices=inputs,
                        wall_s=time.perf_counter()-started,
                        label_policy='CURRENT_REFERENCE_ONLY', network_seed=seed)


def active_queries(candidates, reference_chain, measured, limit):
    """Query unsupported requests on actual visited states; no hypothetical rollout."""
    used = {(context_hash(r['binding']['state']), int(r['action'])) for r in measured}
    prefix = prefix_chain_hash(reference_chain)
    proposals = {}
    for candidate in candidates:
        for entry in candidate.get('support_logs', []):
            key = str(int(round(entry['t'])))
            binding = candidate['decision_prefix'].get(key)
            if binding is None:
                raise ValueError('Active query lacks a real visited-state binding')
            for stage in entry.get('stages', []):
                if stage['reference_prefix_sha256'] != prefix:
                    continue
                base = int(stage['base_action'])
                for item in stage['candidates']:
                    if item['supported'] or item['action'] == base:
                        continue
                    pair = (context_hash(binding['state']), int(item['action']))
                    score = float(item['score'])
                    if pair in used or not np.isfinite(score):
                        continue
                    proposals.setdefault(pair, dict(binding=deepcopy(binding), base_action=base,
                        action=int(item['action']), origin='active_visited_state',
                        proposal_score=score, prediction_only=True))
    ordered = sorted(proposals.values(), key=lambda x: (-x['proposal_score'], x['binding']['t'], x['action']))
    # Spread queries across physical states rather than spending the batch on
    # several alternatives at the same exact state. Every request remains ±2.
    result, states = [], set()
    for query in ordered:
        state = context_hash(query['binding']['state'])
        if state not in states:
            result.append(query)
            states.add(state)
        if len(result) == limit:
            break
    return result if limit else []


def support_statistics(result):
    stages = [s for x in result.get('support_logs', []) for s in x.get('stages', [])]
    attempts = [r for s in stages for r in s['candidates'] if r['action'] != s['base_action']]
    changed = [s for s in stages if s['selected_action'] != s['base_action']]
    matched = set()
    for s in stages:
        for row in s['candidates']:
            for index in row.get('matched_label_indices', []):
                matched.add((s['stage_index'], index))
    negative = []
    for s in changed:
        row = next(r for r in s['candidates'] if r['action'] == s['selected_action'])
        if row.get('measured_advantage', 0.) < 0:
            negative.append(dict(t=next(x['t'] for x in result['support_logs'] if s in x['stages']),
                                 action=s['selected_action'], advantage=row['measured_advantage']))
    times = [entry['t'] for entry in result.get('support_logs', [])
             if any(s['selected_action'] != s['base_action'] for s in entry.get('stages', []))]
    return dict(candidate_action_pairs=len(attempts), supported_action_pairs=sum(x['supported'] for x in attempts),
        proposed_changes_blocked=sum(s['proposed_action'] != s['base_action'] and
                                    not next(r for r in s['candidates'] if r['action'] == s['proposed_action'])['supported']
                                    for s in stages), actual_new_stage_changes=len(changed),
        matched_support_records=len(matched), first_new_stage_change_s=min(times) if times else None,
        measured_negative_requests=negative,
        refusal_reasons=dict(Counter(r['reason'] for r in attempts if not r['supported'])))


def train(out, model, *, config=DEFAULT_CONFIG, initial_policy=None, initial_cl=.025,
          seed=17, pairs=25, active_pairs=5, candidates=14, workers=1,
          continuation_mode='open-loop', inputs=STATE_INPUTS, log=print):
    """One learning iteration, no implicit additional rounds or old-label resume."""
    if (type(pairs) is not int or type(active_pairs) is not int or not 0 <= active_pairs < pairs
            or pairs-active_pairs > 25 or pairs < 1 or type(candidates) is not int or candidates < 2
            or type(workers) is not int or workers < 1):
        raise ValueError('Require positive budget, 1..25 nominal pairs and at least two candidate slots')
    if continuation_mode not in ('open-loop', 'closed-loop'):
        raise ValueError('Invalid continuation mode')
    settings = dict(protocol=PROTOCOL, seed=seed, pairs=pairs, active_pairs=active_pairs,
        candidates=candidates, workers=workers, continuation=continuation_mode,
        inputs=inputs, iterations=1, support_mode='exact', label_policy='CURRENT_REFERENCE_ONLY',
        action_semantics=ACTION_SEMANTICS, hidden=list(HIDDEN), legacy_stage_behavior='FROZEN_REFERENCE')
    out = Path(out).resolve()
    # Conservative reserve checked again before each launched batch. No files
    # are deleted/archived to make a run fit its disk budget.
    reserve = 1_500_000_000
    spec, reference, binding = prepare_runtime(out, model, config, initial_policy, initial_cl, settings)
    budget = Budget(pairs, candidates)
    started = time.perf_counter()
    state = dict(protocol=PROTOCOL, settings=settings, runtime_spec=spec,
        experiment_binding=binding, reference_chain=reference, chain=deepcopy(reference),
        iteration=0, phase='INITIALIZED', labels=[], pairs=[], candidates=[], fits=[],
        seed=seed, policy_version=0, rng_state=np.random.default_rng(seed).bit_generator.state,
        convergence_proven=False, budget=budget.state())

    def checkpoint(phase):
        state.update(phase=phase, budget=budget.state(), wall_s=time.perf_counter()-started)
        save(out/'state.json', state)
        log(f'{phase}: pairs {budget.used["pair"]}/{pairs}, candidate flights {budget.used["candidate"]}/{candidates}',)

    def launch(kind, batch):
        # Existing complete audited flights occupy about 0.61 GB here. Reserve
        # 1 GB per full-horizon equivalent before a batch, including both paired
        # arms. This is a disk planning allowance, not a solver/physics budget.
        equivalents = sum(2*(1200.-float(args['binding']['t']))/1200.
                          if job_kind == 'pair' else 1. for job_kind, args in batch)
        if shutil.disk_usage(out).free < reserve+equivalents*1_000_000_000:
            checkpoint('PAUSED_DISK_SPACE')
            raise RuntimeError('Insufficient free disk for next batch plus 1.5 GB reserve; evidence retained')
        budget.charge(kind, len(batch))
        checkpoint('RUNNING_'+kind.upper())
        result = jobs(batch, workers)
        for (_, args), value in zip(batch, result):
            if value.get('error'):
                save(out/'infrastructure_failure.json', dict(folder=args['folder'], result=value))
                checkpoint('FAILED_INFRASTRUCTURE')
                raise RuntimeError(value['error'])
        return result

    def pair_batch(queries, name):
        batch = [('pair', dict(folder=str(out/name/f'pair_{i:03d}'), runtime_spec=spec,
            reference_chain=reference, binding=q['binding'], base_action=q['base_action'], action=q['action'],
            continuation=continuation_mode, schedule=schedule, query_origin=q['origin']))
            for i, q in enumerate(queries)]
        save(out/name/'queries.json', queries)
        results = launch('pair', batch) if batch else []
        state['pairs'].extend(results)
        new = labels_from_pairs(results, reference, continuation)
        state['labels'].extend(new)
        save(out/name/'results.json', results)
        checkpoint('PAIRS_RECORDED')
        return results

    def evaluate(chains, name):
        batch = [('candidate', dict(folder=str(out/name/f'candidate_{i:02d}'), chain=chain,
                                     runtime_spec=spec, max_decisions=400)) for i, chain in enumerate(chains)]
        # Store actual weights and frozen support tables before the first flight,
        # not just a hash that would be unrecoverable after a process interruption.
        save(out/name/'candidate_requests.json', [args for _, args in batch])
        state['pending_candidates'] = [dict(chain_sha256=canonical_hash(chain),
            request_file=str(out/name/'candidate_requests.json'), index=i)
            for i, chain in enumerate(chains)]
        results = launch('candidate', batch) if batch else []
        for chain, result in zip(chains, results):
            result['support_statistics'] = support_statistics(result)
            state['candidates'].append(dict(chain=chain, result=result))
        save(out/name/'results.json', results)
        state['pending_candidates'] = []
        checkpoint('CANDIDATES_RECORDED')
        return results

    checkpoint('INITIALIZED')
    nominal = launch('baseline', [('candidate', dict(folder=str(out/'baseline'), chain=reference,
                                                    runtime_spec=spec, max_decisions=400))])[0]
    state['baseline'] = nominal
    checkpoint('BASELINE_RECORDED')
    if not complete_candidate(nominal):
        checkpoint('BASELINE_NOT_COMPLETE')
        return state
    schedule = nominal['schedule'] if continuation_mode == 'open-loop' else None
    continuation = continuation_identity(reference, continuation_mode, schedule)
    state['continuation'] = continuation
    queries = nominal_queries(nominal, pairs-active_pairs)
    pair_batch(queries, 'nominal_pairs')
    if not state['labels']:
        state.update(accepted=False, final_result=nominal, label_valid_pairs=0,
                     support_limited=True, stopping_reason='NO_COMPLETE_PAIRED_LABELS')
        checkpoint('NO_COMPLETE_PAIRED_LABELS')
        save(out/'summary.json', dict(protocol=PROTOCOL, phase=state['phase'], iterations=0,
            accepted=False, baseline_range_km=nominal['range_km'], labels=0,
            budget=budget.state(), wall_s=state['wall_s'], convergence_proven=False))
        return state
    ix = parse_inputs(inputs)
    first_count = max(1, candidates//2) if active_pairs else candidates
    first_margins = margin_grid(first_count)[:first_count]
    chains, fit = make_stages(reference, state['labels'], continuation, seed*1000, ix, first_margins)
    state['fits'].append(fit)
    initial_results = evaluate(chains, 'initial_candidates')
    active = active_queries(initial_results, reference, state['pairs'], active_pairs)
    pair_batch(active, 'active_pairs')
    remaining = candidates-budget.used['candidate']
    if active and remaining:
        margins = margin_grid(remaining)[:remaining]
        chains, fit = make_stages(reference, state['labels'], continuation, seed*1000+1, ix, margins)
        state['fits'].append(fit)
        evaluate(chains, 'final_candidates')
    elif remaining and state['candidates']:
        # Reserve an independently re-flown frozen candidate even when there
        # were no new queries. Its results count within the candidate budget.
        best_so_far = max(state['candidates'], key=lambda c: c['result'].get('G', -np.inf)
                          if complete_candidate(c['result']) else -np.inf)
        evaluate([best_so_far['chain']], 'final_verification')
    latest = {canonical_hash(c['chain']): c for c in state['candidates']}
    eligible = [c for c in latest.values() if complete_candidate(c['result'])
                and c['result']['G'] > nominal['G']+1e-9]
    best = max(eligible, key=lambda c: c['result']['G']) if eligible else None
    state.update(iteration=1, accepted=best is not None, policy_version=int(best is not None),
        chain=deepcopy(best['chain']) if best else reference,
        final_result=best['result'] if best else nominal,
        stopping_reason=('ONE_ITERATION_BUDGET_REACHED' if state['labels'] else 'NO_COMPLETE_PAIRED_LABELS'),
        convergence_proven=False,
        support_limited=not state['labels'] or any(c['result']['support_statistics']['proposed_changes_blocked']
                            for c in state['candidates']),
        label_valid_pairs=sum(p.get('valid') is True for p in state['pairs']))
    checkpoint('COMPLETE')
    final = state['final_result']
    summary = dict(protocol=PROTOCOL, iterations=1, accepted=state['accepted'],
        baseline={k:v for k,v in nominal.items() if k not in ('decision_prefix', 'support_logs', 'schedule', 'final_policy_state')},
        final={k:v for k,v in final.items() if k not in ('decision_prefix', 'support_logs', 'schedule', 'final_policy_state')},
        range_delta_km=final['range_km']-nominal['range_km'], reward_delta=final['G']-nominal['G'],
        labels=len(state['labels']), valid_pairs=state['label_valid_pairs'], pair_count=len(state['pairs']),
        budget=budget.state(), wall_s=state['wall_s'], support_limited=state['support_limited'],
        convergence_proven=False, local_generalization_validated=False,
        candidates=[dict(range_km=c['result'].get('range_km'), G=c['result'].get('G'),
                         completed=c['result'].get('completed_1200s'), wall_s=c['result'].get('wall_s'),
                         support_statistics=c['result']['support_statistics']) for c in state['candidates']])
    save(out/'summary.json', summary)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--initial-policy', type=Path)
    parser.add_argument('--initial-cl', type=float, default=.025)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--pairs', type=int, default=25, help='Total paired queries; each launches two continuations')
    parser.add_argument('--active-pairs', type=int, default=5, help='Included in --pairs, not additional')
    parser.add_argument('--candidates', type=int, default=14, help='All acquisition and final full-flight candidates')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--continuation', choices=['open-loop', 'closed-loop'], default='open-loop')
    parser.add_argument('--inputs', default=STATE_INPUTS)
    args = parser.parse_args(argv)
    result = train(args.out, args.model, config=args.config, initial_policy=args.initial_policy,
        initial_cl=args.initial_cl, seed=args.seed, pairs=args.pairs, active_pairs=args.active_pairs,
        candidates=args.candidates, workers=args.workers, continuation_mode=args.continuation,
        inputs=args.inputs, log=lambda message: print(message, flush=True))
    print(json.dumps(dict(phase=result['phase'], iteration=result['iteration'],
                         state=str(Path(args.out).resolve()/'state.json'))), flush=True)
    return 0 if result['phase'] == 'COMPLETE' else 1


if __name__ == '__main__':
    main()
