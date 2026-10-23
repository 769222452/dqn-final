"""Rollout-based approximate policy iteration with certified flights.

One iteration with the current policy pi (exact return G_pi):

1. Labels: branch at `states` decision times (spacing `stride`, offset rotating every iteration) and measure the
   exact advantages of two alternative actions per state (rollouts.branch_job).
2. Fit: bootstrap ensemble of advantage networks on the labels of the last LABEL_KEEP policy versions.
3. Candidates: pi extended by one model stage for several margins (and, optionally, exact override stages built
   from the positive labels of the current policy). Every candidate is flown with full certification.
4. Acceptance: the best candidate replaces pi only if its exact return is higher (monotone improvement);
   otherwise pi is kept and labels keep accumulating.

Stop rule: two consecutive sweeps over all state offsets without any positive pair, invalid branch or
acceptance, or three sweeps without acceptance.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from certified_flight import CL_ACTIONS

from .ensemble import (HIDDEN, LABEL_DECAY, LABEL_KEEP, LABEL_SCALE, fit_ensemble, parse_hidden, prediction_check,
                       retained_labels)
from .features import STATE_INPUTS, input_names, parse_inputs
from .policy import DECISION_S, constant_stage
from .rollouts import MISSION_S, dispatch

MARGINS = (1., .5, .25, .75, 1.25, 1.5)       # LCB thresholds of model candidates, units of 1e-4 return
OVERRIDE_SETS = (1, 2, 3, 4, 5, 6, 8, 10, 12)  # sizes of the exact override candidates (plus all)


# ----------------------------------------------------------------------------------------------- utilities
def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def save_json(path, value):
    """Atomic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name('.' + path.name + '.tmp')
    with pending.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(_jsonable(value), stream, indent=1, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(40):          # tolerate transient sharing locks on Windows
        try:
            os.replace(pending, path)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 39:
                raise
            time.sleep(.25)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_jobs(jobs, workers, costs=None):
    """Run jobs in spawned worker processes (longest first when costs are given); results in job order."""
    order = sorted(range(len(jobs)), key=lambda k: -costs[k]) if costs is not None else list(range(len(jobs)))
    out = [None]*len(jobs)
    with mp.get_context('spawn').Pool(min(workers, len(jobs))) as pool:
        for k, r in zip(order, pool.imap(dispatch, [jobs[k] for k in order], chunksize=1)):
            out[k] = r
    return out


# ----------------------------------------------------------------------------------------------- schedule
def state_times(iteration, states, stride):
    """Branched decision times; the offset rotates by one decision per iteration."""
    offset = DECISION_S*(iteration % int(stride/DECISION_S))
    return [offset+stride*j for j in range(states) if offset+stride*j <= MISSION_S-DECISION_S+1e-9]


def state_cost(t):
    """Certified steps of one branch job: prefix t/3 plus two continuations."""
    return t/3 + 2*(MISSION_S-t)/3


def margin_grid(n):
    """n distinct margins (at least 3); the first ones bracket margin 1."""
    return sorted(MARGINS[:max(3, min(n, len(MARGINS)))])


def positive_overrides(labels, min_advantage):
    """Best positive alternative per labelled state, largest advantage first."""
    best = {}
    for x in labels:
        if x['action'] != x['base_action'] and x['advantage'] > min_advantage and \
                x['advantage'] > best.get(x['t'], {'advantage': -1})['advantage']:
            best[x['t']] = x
    return sorted(best.values(), key=lambda x: -x['advantage'])


def override_sets(entries):
    n = len(entries)
    return [entries[:k] for k in sorted({min(k, n) for k in OVERRIDE_SETS+(n,)}) if k > 0] if n else []


def improvable_fraction(new, min_advantage):
    states = sorted({x['t'] for x in new})
    if not states:
        return 1.
    return sum(1 for t in states if max(x['advantage'] for x in new if x['t'] == t) > min_advantage)/len(states)


def stop_reason(history, patience, sweeps=2, stall=3):
    def clean(r):
        return not r['accepted'] and r['positive_pairs'] == 0 and r['invalid'] == 0
    n = 0
    for r in reversed(history):
        if not clean(r):
            break
        n += 1
    if n >= patience*sweeps:
        return f'{sweeps} clean sweeps ({n} iterations): no positive pair, no invalid branch, no acceptance'
    if len(history) >= patience*stall and not any(r['accepted'] for r in history[-patience*stall:]):
        return f'no candidate accepted in {patience*stall} iterations'
    return None


def prune(idir, keep):
    """Remove re-flyable proof records of a finished iteration (branches and rejected candidates)."""
    for d in list(idir.glob('branch_t*'))+[d for d in idir.glob('candidate_*') if d.name != keep]:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------------------------- main loop
def train(out, seed=17, iterations=40, workers=14, states=25, stride=48., window=48., candidates=14, patience=None,
          min_advantage=1e-5, continuation='open-loop', inputs=STATE_INPUTS, overrides=False, initial_cl=.04,
          label_decay=LABEL_DECAY, label_keep=LABEL_KEEP, use_stop_rule=True, keep_evidence=False, hidden=HIDDEN,
          log=print):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
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
    path = out/'state.json'
    st = load_json(path) if path.exists() else dict(
        seed=seed, iteration=0, G=None, schedule=None, policy_version=0, labels_current=[], label_archive=[],
        chain=[constant_stage(initial_cl)] if initial_cl != .04 else [], history=[], settings=[])
    if st.get('converged'):
        log(f"already converged at iteration {st['converged']['iteration']}: {st['converged']['reason']}")
        return st
    st['settings'].append(dict(at_iteration=st['iteration'], states=states, stride=stride, window=window,
                               candidates=candidates, margins=grid, patience=patience, min_advantage=min_advantage,
                               continuation=continuation, inputs=input_names(inputs), overrides=overrides,
                               initial_cl=initial_cl, label_decay=label_decay, label_keep=label_keep,
                               stop_rule=use_stop_rule, workers=workers, hidden=list(hidden)))
    log(f"start: iteration {st['iteration']}, seed {st['seed']}, {states} states every {stride:g} s, "
        f"continuation={continuation}, inputs={','.join(input_names(inputs))}, overrides={overrides}, "
        f"hidden={','.join(map(str, hidden))}, workers={workers}")
    target = st['iteration']+iterations
    while st['iteration'] < target:
        i = st['iteration']
        idir = out/f'iter_{i:03d}'
        t0 = time.perf_counter()
        rec = dict(iteration=i)
        if idir.exists():                 # leftovers of an interrupted attempt are never mixed in
            shutil.rmtree(idir, ignore_errors=True)
        times = state_times(i, states, stride)
        sweep = i//patience               # the neighbour radius (1 or 2) alternates per sweep

        # 1. labels
        if continuation == 'open-loop' and (st['G'] is None or not st['schedule']):
            init = run_jobs([('candidate', (str(idir/'initial'), st['chain']))], 1)[0]
            if 'error' in init or not init['success']:
                raise RuntimeError(f'initial policy failed: {init}')
            if st['G'] is not None and abs(init['G']-st['G']) > 1e-12:
                raise RuntimeError('return of the current policy not reproduced')
            st['G'], st['schedule'] = init['G'], init['schedule']
            rec['initial'] = {k: v for k, v in init.items() if k != 'schedule'}
            log(f"iter {i}: nominal decisions of the current policy, G={init['G']:.6f} ({init['range_km']:.2f} km)")
        sched = st['schedule'] if continuation == 'open-loop' else None
        jobs = [('branch', (str(idir/f'branch_t{int(t):04d}'), st['chain'], [t], sweep, sched)) for t in times]
        costs = [state_cost(t) for t in times]
        if st['G'] is None:               # closed loop: the initial return is flown alongside the branches
            res = run_jobs(jobs+[('candidate', (str(idir/'initial'), st['chain']))], workers, costs+[400.])
            init = res.pop()
            if 'error' in init or not init['success']:
                raise RuntimeError(f'initial policy failed: {init}')
            st['G'], st['schedule'] = init['G'], init['schedule']
            rec['initial'] = {k: v for k, v in init.items() if k != 'schedule'}
            log(f"iter {i}: initial policy G={init['G']:.6f} ({init['range_km']:.2f} km)")
        else:
            res = run_jobs(jobs, workers, costs)
        errors = [r['error'] for r in res if 'error' in r]
        if errors:
            raise RuntimeError('branch job failed:\n'+errors[0])
        rows = [r for w in res for r in w['rows']]
        t_branch = time.perf_counter()-t0
        prefix = {float(t): x for w in res for t, x in w['prefix'].items()}
        base = {t: st['G']-g for t, (g, a0, f) in prefix.items()}
        new = [dict(features=f, action=a0, base_action=a0, t=t, advantage=0.) for t, (g, a0, f) in prefix.items()]
        new += [dict(features=r['features'], action=r['action'], base_action=r['base_action'], t=r['t'],
                     advantage=r['G_rest']-base[r['t']]) for r in rows if r['valid']]
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
        check = prediction_check(st.get('check_members'), new, min_advantage, st.get('check_inputs'))
        if check:
            check['policy_changed_since_fit'] = bool(st.get('check_chain_len') != len(st['chain']))
            log(f"iter {i}: previous ensemble on {check['pairs']} unseen pairs: Spearman "
                + (f"{check['spearman']:.2f}" if check['spearman'] is not None else '-')
                + (f", R2 {check['r2_clipped']:.2f}" if check['r2_clipped'] is not None else ''))
        tick = time.perf_counter()
        members, fit = fit_ensemble(fit_labels, st['seed']*1000+i, fit_weights, inputs, hidden)
        fit['versions'] = sorted({x['v'] for x in fit_labels})
        t_fit = time.perf_counter()-tick
        st['check_members'], st['check_chain_len'], st['check_inputs'] = members, len(st['chain']), inputs
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
        cres = run_jobs([('candidate', (str(idir/c['name']), c['chain'])) for c in specs], min(len(specs), workers))
        t_cand = time.perf_counter()-tick
        for c, r in zip(specs, cres):
            what = f"margin {c['margin']:g}" if c['kind'] == 'model' else f"override x{c['n']}"
            if 'error' in r:
                log(f"  candidate {what}: ERROR\n{r['error']}")
            else:
                log(f"  candidate {what}: G={r['G']:.6f} ({(r['G']-st['G'])*LABEL_SCALE:+.3f}e-4) {r['range_km']:.2f} km "
                    f"stop={r['stop']} switches={r['switches']}")

        # 4. acceptance
        ok = [(r['G'], c, r) for c, r in zip(specs, cres) if 'error' not in r and r['success']]
        best = max(ok, key=lambda x: x[0]) if ok else None
        accepted = bool(best and best[0] > st['G']+1e-9)
        rec.update(labels=len(new), invalid=n_invalid, improvable_fraction=frac, positive_pairs=n_pos, sweep=sweep,
                   prediction_check=check, fit=fit, G_before=st['G'], accepted=accepted,
                   candidates=[dict(kind=c['kind'], margin=c.get('margin'), overrides=c.get('n'),
                                    **({k: r[k] for k in ('G', 'range_km', 'stop', 'switches')} if 'error' not in r
                                       else dict(error=True))) for c, r in zip(specs, cres)],
                   wall_s=dict(branch=t_branch, fit=t_fit, candidates=t_cand, total=time.perf_counter()-t0))
        if accepted:
            c = best[1]
            st['chain'], st['G'], st['schedule'] = c['chain'], best[0], best[2]['schedule']
            st['labels_current'] = []
            st['policy_version'] = version+1
            rec.update(accepted_kind=c['kind'], accepted_margin=c.get('margin'), accepted_overrides=c.get('n'),
                       range_km=best[2]['range_km'])
        if not keep_evidence:
            prune(idir, best[1]['name'] if accepted else None)
        st['history'].append(rec)
        st['iteration'] = i+1
        reason = stop_reason(st['history'], patience) if use_stop_rule else None
        if reason:
            st['converged'] = dict(iteration=i, reason=reason, G=st['G'])
        save_json(path, st)
        log(f"iter {i} done: " + (f"accepted {best[1]['name'].removeprefix('candidate_')}" if accepted else 'kept policy')
            + f"; G={st['G']:.6f} (wall {rec['wall_s']['total']:.0f} s)")
        if reason:
            log(f'stopped at iteration {i}: {reason}')
            break
    return st


def main(argv=None):
    p = argparse.ArgumentParser(description='Rollout-based policy iteration with certified flights. Defaults are '
                                            'the proposed method; see README for the baseline settings.')
    p.add_argument('--out', required=True, help='run directory (resumed if it contains state.json)')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--iterations', type=int, default=40, help='maximum iterations of this invocation')
    p.add_argument('--workers', type=int, default=min(14, os.cpu_count() or 1))
    p.add_argument('--states', type=int, default=25, help='decision states branched per iteration')
    p.add_argument('--stride', type=float, default=48., help='seconds between branched states (multiple of 12)')
    p.add_argument('--window', type=float, default=48., help='model stages act within +-window s of labelled states')
    p.add_argument('--candidates', type=int, default=14, help='candidate budget per iteration (shared with overrides)')
    p.add_argument('--patience', type=int, default=None, help='iterations per sweep of all offsets (default stride/12)')
    p.add_argument('--min-advantage', type=float, default=1e-5, help='advantage counted as improvement')
    p.add_argument('--continuation', choices=['open-loop', 'closed-loop'], default='open-loop',
                   help="branch continuation: replay pi's nominal decisions (open-loop) or let pi re-decide (closed-loop)")
    p.add_argument('--inputs', default=STATE_INPUTS, help="network inputs: subset of h,V,gamma,alpha,t,prev or 'all'")
    p.add_argument('--overrides', action='store_true', help='add exact override candidates from positive labels')
    p.add_argument('--initial-cl', type=float, default=.04, help='constant initial policy')
    p.add_argument('--label-decay', type=float, default=LABEL_DECAY)
    p.add_argument('--label-keep', type=int, default=LABEL_KEEP)
    p.add_argument('--ignore-stop-rule', action='store_true', help='run all iterations')
    p.add_argument('--keep-evidence', action='store_true', help='keep proof records of branches and rejected candidates')
    p.add_argument('--hidden', default=','.join(map(str, HIDDEN)),
                   help='hidden layer widths of the advantage networks, e.g. 64,64,64,64')
    a = p.parse_args(argv)
    train(a.out, a.seed, a.iterations, a.workers, a.states, a.stride, a.window, a.candidates, a.patience,
          a.min_advantage, a.continuation, a.inputs, a.overrides, a.initial_cl, a.label_decay, a.label_keep,
          not a.ignore_stop_rule, a.keep_evidence, a.hidden, log=lambda s: print(s, flush=True))
    return 0
