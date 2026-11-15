"""Rollout-based policy iteration over a set of disturbed scenarios (fast-model stage of experiments/perturb).

Same method as `training.py` (exact counterfactual branch labels, open-loop continuation, bootstrap advantage
ensemble, conservative model stages, monotone acceptance), with three differences fixed in
experiments/perturb/PREREGISTRATION.md:

* the environment is `certified_flight.fastflight` (disturbed truth, 0.5 s inner loop, dynamic-pressure set
  point actions R_LEVELS);
* labels come from a rotating pair of training scenarios per iteration (a scenario is deterministic, so the
  counterfactuals stay exact);
* a candidate is accepted only if its MEAN return over all training scenarios is higher and every one of them
  is flown without a limit violation.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

from certified_flight import fastflight as F

from .ensemble import HIDDEN, LABEL_DECAY, LABEL_KEEP, LABEL_SCALE, fit_ensemble, parse_hidden, retained_labels
from .evaluation import paired_bootstrap_ci
from .features import features, input_names, parse_inputs
from .policy import chain_action
from .rollouts import alternatives
from .training import MARGINS, improvable_fraction, load_json, save_json, state_times, stop_reason

FULL_INPUTS = 'h,V,gamma,t,rho,q,kL,kD,rho_prev'


def constant(action):
    return dict(kind='constant', action=int(action), r=F.R_LEVELS[int(action)])


# ----------------------------------------------------------------------------------------------- flights
def fly(env, chain, override=None, schedule=None, until=None, log=None):
    """Fly `env` until `until` (s) or the end. override=(t, a); schedule={t: a} (open-loop continuation)."""
    while not env.done and (until is None or env.t < until-1e-9):
        t = int(round(env.t))
        if override is not None and abs(env.t-override[0]) < 1e-6:
            a = int(override[1])
        elif schedule is not None:
            a = int(schedule[t])
        else:
            a = int(chain_action(chain, features(env)))
        if log is not None:
            log.append((float(env.t), a))
        env.step(a)
    return env


def make_env(level, seed, inner):
    return F.FastFlight(F.scenario(level, seed), **inner)


def candidate_job(args):
    level, seed, inner, chain = args
    try:
        dec = []
        env = fly(make_env(level, seed, inner), chain, log=dec)
        return dict(env.summary(), schedule=dec)
    except BaseException:
        return dict(error=traceback.format_exc())


def branch_job(args):
    """Advantages of the alternatives at decision times `times` of one scenario (open-loop continuation)."""
    level, seed, inner, chain, times, sweep, schedule = args
    try:
        sched = {int(round(t)): int(a) for t, a in schedule}
        env = make_env(level, seed, inner)
        rows, prefix = [], {}
        for t_b in times:
            fly(env, chain, schedule=sched, until=t_b)
            if env.done:
                return dict(error=f'nominal flight ended before t={t_b}', rows=rows)
            f = features(env).tolist()
            a0 = int(sched[int(round(t_b))])
            prefix[t_b] = (env.G, a0, f)
            for a in alternatives(a0, t_b, sweep):
                b = fly(env.fork(), chain, override=(t_b, a), schedule=sched)
                # A limit violation is a real outcome with a known return (truncated range + failure penalty), so the
                # branch is a valid label; `failed` only counts it.
                rows.append(dict(scenario=seed, t=t_b, action=int(a), base_action=a0, features=f, G_end=b.G,
                                 valid=True, failed=bool(b.failed), stop=b.reason))
        return dict(rows=rows, prefix={str(t): list(v) for t, v in prefix.items()})
    except BaseException:
        return dict(error=traceback.format_exc(), rows=[])


def _dispatch(job):
    kind, args = job
    return branch_job(args) if kind == 'branch' else candidate_job(args)


def _init_worker():
    for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[k] = '1'
    F.world()


class Runner:
    """Persistent spawned worker pool (the aerodynamic world is loaded once per worker)."""

    def __init__(self, workers):
        self.pool = mp.get_context('spawn').Pool(workers, initializer=_init_worker)

    def map(self, jobs):
        return self.pool.map(_dispatch, jobs, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


def fly_set(runner, chain, level, seeds, inner):
    res = runner.map([('candidate', (level, s, inner, chain)) for s in seeds])
    errs = [r['error'] for r in res if 'error' in r]
    if errs:
        raise RuntimeError(errs[0])
    return res


def set_return(res):
    return float(np.mean([r['G'] for r in res])), all(r['success'] for r in res)


# ----------------------------------------------------------------------------------------------- training
def train(out, level, seed, inner, initial_action, inputs=FULL_INPUTS, workers=4, iterations=30, states=25,
          stride=48., window=48., scenarios_per_iteration=2, score='lcb', hidden=HIDDEN, min_advantage=1e-5,
          label_decay=LABEL_DECAY, label_keep=LABEL_KEEP, train_seeds=F.TRAIN_SEEDS, runner=None, log=print):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hidden = parse_hidden(hidden)
    inputs_idx = parse_inputs(inputs)
    patience = int(stride//F.DECISION_S)
    path = out/'state.json'
    own = runner is None
    runner = runner or Runner(workers)
    try:
        st = load_json(path) if path.exists() else dict(
            level=level, seed=seed, inner=inner, iteration=0, chain=[constant(initial_action)], policy_version=0,
            label_archive=[], labels_current=[], history=[], settings=dict(
                inputs=input_names(inputs_idx), states=states, stride=stride, window=window, score=score,
                scenarios_per_iteration=scenarios_per_iteration, hidden=list(hidden), margins=list(MARGINS),
                train_seeds=list(train_seeds), initial_action=int(initial_action)))
        if st.get('converged'):
            log(f"already converged: {st['converged']['reason']}")
            return st
        if 'G' not in st:
            res = fly_set(runner, st['chain'], level, train_seeds, inner)
            G, ok = set_return(res)
            if not ok:
                raise RuntimeError('initial policy violates a limit on a training scenario')
            st['G'], st['schedules'] = G, {str(r['scenario']['seed']): r['schedule'] for r in res}
            st['scenario_G'] = {str(r['scenario']['seed']): r['G'] for r in res}
            st['train_range_km'] = float(np.mean([r['range_km'] for r in res]))
            log(f"[{level} seed {seed}] initial policy: mean G {G:.6f}, mean range {st['train_range_km']:.2f} km")
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(train_seeds)))
        target = st['iteration']+iterations
        while st['iteration'] < target:
            i = st['iteration']
            t0 = time.perf_counter()
            k = scenarios_per_iteration
            pick = [train_seeds[order[(i*k+j) % len(order)]] for j in range(k)]
            times = state_times(i, states, stride)
            sweep = i//patience
            jobs = [('branch', (level, s, inner, st['chain'], [t], sweep, st['schedules'][str(s)]))
                    for s in pick for t in times]
            res = runner.map(jobs)
            errs = [r['error'] for r in res if 'error' in r]
            if errs:
                raise RuntimeError(errs[0])
            rows = [r for w in res for r in w['rows']]
            # G_rest(a) - G_rest(pi) = G_end(a) - G_end(pi); the open-loop continuation replays pi exactly
            G_nom = {int(x): float(st['scenario_G'][str(x)]) for x in pick}
            new = []
            for w in res:
                for t, (g, a0, f) in w['prefix'].items():
                    new.append(dict(features=f, action=a0, base_action=a0, t=float(t), advantage=0.))
            new += [dict(features=r['features'], action=r['action'], base_action=r['base_action'], t=r['t'],
                         advantage=r['G_end']-G_nom[r['scenario']]) for r in rows if r['valid']]
            n_invalid = sum(r['failed'] for r in rows)        # branches ending in a violation (kept as labels)
            n_pos = sum(1 for x in new if x['action'] != x['base_action'] and x['advantage'] > min_advantage)
            version = st['policy_version']
            for x in new:
                x['v'] = version
            st['labels_current'] += new
            st['label_archive'] += new
            fit_labels, fit_w = retained_labels(st['label_archive'], version, label_keep, label_decay)
            st['label_archive'] = fit_labels
            members, fit = fit_ensemble(fit_labels, seed*1000+i, fit_w, inputs_idx, hidden)
            label_times = sorted({float(x['t']) for x in fit_labels})
            specs = [dict(margin=m, chain=st['chain']+[dict(kind='model', members=members, margin=m, iteration=i,
                                                            times=label_times, window=window, score=score,
                                                            **({} if inputs_idx is None else dict(inputs=inputs_idx)))])
                     for m in sorted(MARGINS)]
            cres = runner.map([('candidate', (level, s, inner, c['chain'])) for c in specs for s in train_seeds])
            errs = [r['error'] for r in cres if 'error' in r]
            if errs:
                raise RuntimeError(errs[0])
            n = len(train_seeds)
            results = []
            for j, c in enumerate(specs):
                part = cres[j*n:(j+1)*n]
                G, ok = set_return(part)
                results.append((G, ok, c, part))
            ok_res = [x for x in results if x[1]]
            best = max(ok_res, key=lambda x: x[0]) if ok_res else None
            accepted = bool(best and best[0] > st['G']+1e-9)
            rec = dict(iteration=i, scenarios=[int(s) for s in pick], labels=len(new), invalid=n_invalid,
                       positive_pairs=n_pos, improvable_fraction=improvable_fraction(new, min_advantage),
                       G_before=st['G'], accepted=accepted, fit=fit,
                       candidates=[dict(margin=c['margin'], G=G, all_success=ok,
                                        failures=int(sum(not r['success'] for r in part)),
                                        mean_range_km=float(np.mean([r['range_km'] for r in part])))
                                   for G, ok, c, part in results],
                       wall_s=time.perf_counter()-t0)
            if accepted:
                G, _, c, part = best
                st['chain'], st['G'] = c['chain'], G
                st['schedules'] = {str(r['scenario']['seed']): r['schedule'] for r in part}
                st['scenario_G'] = {str(r['scenario']['seed']): r['G'] for r in part}
                st['train_range_km'] = float(np.mean([r['range_km'] for r in part]))
                st['labels_current'] = []
                st['policy_version'] = version+1
                rec.update(accepted_margin=c['margin'], train_range_km=st['train_range_km'])
            st['history'].append(rec)
            st['iteration'] = i+1
            reason = stop_reason(st['history'], patience)
            if reason:
                st['converged'] = dict(iteration=i, reason=reason, G=st['G'])
            save_json(path, st)
            log(f"[{level} seed {seed}] iter {i}: {len(new)} labels, {n_pos} positive, {n_invalid} invalid; "
                + ('accepted margin %g' % best[2]['margin'] if accepted else 'kept')
                + f"; mean train range {st['train_range_km']:.2f} km ({rec['wall_s']:.0f} s)")
            if reason:
                log(f"[{level} seed {seed}] stopped: {reason}")
                break
        return st
    finally:
        if own:
            runner.close()


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(runner, chains, level, inner_by_name, seeds=F.TEST_SEEDS):
    """Fly every named chain on the scenarios; returns {name: [summary per seed]}."""
    out = {}
    for name, chain in chains.items():
        out[name] = fly_set(runner, chain, level, seeds, inner_by_name[name])
    return out


def paired(results, name, ref, scale=1.):
    a = np.array([r['range_km'] for r in results[name]])
    b = np.array([r['range_km'] for r in results[ref]])
    d = a-b
    lo, hi = paired_bootstrap_ci(d)
    return dict(mean=float(d.mean()), ci95=[lo, hi], median=float(np.median(d)), wins=int((d > 0).sum()),
                worst=float(d.min()), n=len(d),
                failures=int(sum(not r['success'] for r in results[name])),
                ref_failures=int(sum(not r['success'] for r in results[ref])),
                worst10_mean_km=float(np.mean(np.sort(a)[:max(1, len(a)//10)])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--level', required=True, choices=sorted(F.LEVELS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--baselines', required=True, help='baselines.json written by scripts/perturb_study.py tune')
    p.add_argument('--inputs', default=FULL_INPUTS)
    p.add_argument('--score', choices=['lcb', 'mean'], default='lcb')
    p.add_argument('--initial', choices=['B2', 'untuned'], default='B2')
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    a = p.parse_args(argv)
    base = json.loads(Path(a.baselines).read_text())[a.level]
    inner = dict(kp=base['B2']['kp'], kg=base['B2']['kg'])
    init = base['B2']['action'] if a.initial == 'B2' else F.R_LEVELS.index(.80)
    train(a.out, a.level, a.seed, inner, init, inputs=a.inputs, workers=a.workers, iterations=a.iterations,
          score=a.score)


if __name__ == '__main__':
    main()
