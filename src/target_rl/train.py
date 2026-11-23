"""Counterfactual-advantage policy iteration on the training cases (PREREGISTRATION.md section 5).

Same method as the original project (rollout_improvement.scenario_training): exact counterfactual branch labels
with open-loop continuation, bootstrap advantage ensemble, conservative model stages, monotone acceptance.
A case = disturbance scenario + target; a case is deterministic, so a branch label (change one decision, replay
the recorded actions to the end, compare the returns) is exact. Differences to the previous stage: two-dimensional
actions; a candidate is accepted if its mean return over all training cases is higher AND it does not have more
cases ending in a limit violation than the current policy.

Variants (round 2, PREREGISTRATION.md amendment 5): every variant flies with the execution feedback
CL target - kg gamma (kg from configs/task.yaml execution) and starts from the target lookup table, except:
  full     inputs h,V,gamma,t,range_to_go,speed_error; morphing; LCB          (the method)
  nodamp   kg = 0 (round-1 execution)
  single   initial policy = one constant action for every target (round-1 start)
  fixed    no morphing
  dist     + measured density ratio, dynamic pressure, online bias estimates, previous density ratio (not run)
  mean     ensemble mean instead of the LCB (not run)
Study A (PREREGISTRATION.md amendment 6, configs/study_a.yaml):
  pcguided the cheap strong predictor-corrector flies (guided.py); the policy chooses an offset to its planned
           morph level; inputs BASE + the corrector's predicted errors and plan; initial policy = offset 0
Study R (PREREGISTRATION.md amendment 7, configs/study_r.yaml):
  rel      the policy flies (CL target increments + w), one coarse prediction as input (guided.RelPredFlight);
           initial policy = the feedback rule of configs/study_r.yaml start_rule
  relfixed the same without morphing

    python -m target_rl.train --variant full --seed 17 --workers 14
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from . import config, flight as F
from .ensemble import LABEL_DECAY, LABEL_KEEP, fit_ensemble, retained_labels
from .parallel import pool_map
from .policy import BASE, EXTRA, GUIDE, alternatives, chain_action, input_index
from .scenario import Scenario, Target
from .util import load_json, save_json

VARIANTS = dict(full=dict(inputs=BASE, morph=True, score='lcb', kg=None, start='lookup'),
                nodamp=dict(inputs=BASE, morph=True, score='lcb', kg=0., start='lookup'),
                single=dict(inputs=BASE, morph=True, score='lcb', kg=None, start='single'),
                fixed=dict(inputs=BASE, morph=False, score='lcb', kg=None, start='lookup'),
                dist=dict(inputs=BASE+EXTRA, morph=True, score='lcb', kg=None, start='lookup'),
                mean=dict(inputs=BASE, morph=True, score='mean', kg=None, start='lookup'),
                pcguided=dict(inputs=BASE+GUIDE, morph=True, score='lcb', kg=None, start='plan'),
                rel=dict(inputs=BASE+GUIDE, morph=True, score='lcb', kg=None, start='rule'),
                relfixed=dict(inputs=BASE+GUIDE, morph=False, score='lcb', kg=None, start='rule'))


def variant_kg(v):
    """Execution feedback gain of a variant (None = the configured gain)."""
    return float(config.load()['execution']['kg'] if v['kg'] is None else v['kg'])


# ------------------------------------------------------------------------------------------- cases
def case_to_json(case):
    sc, tg = case
    return dict(scenario=sc.__dict__, target=tg.__dict__)


def case_from_json(d):
    return Scenario(**d['scenario']), Target(**d['target'])


# ------------------------------------------------------------------------------------------- flights
def fly(env, chain, override=None, schedule=None, until=None, log=None):
    while not env.done and (until is None or env.t < until-1e-9):
        t = int(round(env.t))
        if override is not None and abs(env.t-override[0]) < 1e-6:
            a = int(override[1])
        elif schedule is not None and t in schedule:
            a = int(schedule[t])                     # open-loop continuation: replay the recorded actions
        else:            # closed loop; with a schedule only past the end of a recorded flight that ended in a violation
            a = chain_action(chain, env.features(extra=True), env.morph)
        if log is not None:
            log.append((float(env.t), int(a)))
        env.step(a)
    return env


def make_env(sc, tg, morph=True, kg=None):
    from .guided import make
    return make(sc, tg, morph, kg=kg)


def candidate_job(kw):
    sc, tg = kw['case']
    dec = []
    env = fly(make_env(sc, tg, kw['morph'], kg=kw.get('kg')), kw['chain'], log=dec)
    return dict(env.summary(), schedule=dec)


def branch_job(kw):
    sc, tg = kw['case']
    morph = kw['morph']
    sched = {int(round(t)): int(a) for t, a in kw['schedule']}
    env = make_env(sc, tg, morph, kg=kw.get('kg'))
    rows, prefix = [], []
    for t_b in kw['times']:
        fly(env, kw['chain'], schedule=sched, until=t_b)
        if env.done:
            break
        f = env.features(extra=True)
        a0 = sched[int(round(t_b))]
        prefix.append((t_b, a0, f))
        for a in alternatives(a0, t_b, kw['sweep'], morph):
            b = fly(env.fork(), kw['chain'], override=(t_b, a), schedule=sched)
            rows.append(dict(case=kw['case_id'], t=t_b, action=int(a), base_action=a0, features=f, G_end=b.G,
                             failed=bool(b.failed)))
    return dict(rows=rows, prefix=prefix)


def fly_set(chain, cases, morph, workers, kg=None):
    return pool_map([('candidate', dict(case=c, chain=chain, morph=morph, kg=kg)) for c in cases], workers)


def set_stats(res):
    return float(np.mean([r['G'] for r in res])), int(sum(not r['feasible'] for r in res)), \
        int(sum(r['hit'] for r in res))


# ------------------------------------------------------------------------------------------- schedule
def state_times(iteration, states, stride, decision=12., horizon=1200.):
    offset = decision*(iteration % int(stride/decision))
    return [offset+stride*j for j in range(states) if offset+stride*j <= horizon-decision+1e-9]


def improvable_fraction(new, min_advantage):
    states = sorted({(x['case'], x['t']) for x in new})
    if not states:
        return 1.
    return sum(1 for s in states if max(x['advantage'] for x in new if (x['case'], x['t']) == s) > min_advantage)/len(states)


def stop_reason(history, patience, sweeps=2, stall=3, no_acceptance_iterations=None):
    n = 0
    for r in reversed(history):
        if r['accepted'] or r['positive_pairs'] or r['invalid']:
            break
        n += 1
    if n >= patience*sweeps:
        return f'{sweeps} clean sweeps ({n} iterations): no positive pair, no violating branch, no acceptance'
    limit = patience*stall if no_acceptance_iterations is None else int(no_acceptance_iterations)
    if len(history) >= limit and not any(r['accepted'] for r in history[-limit:]):
        return f'no candidate accepted in {limit} iterations'
    return None


# ------------------------------------------------------------------------------------------- pi0
def initial_stage(morph, kg, start, workers, cases):
    """First policy stage, cached in runs/<tag>/<library>/start_<morph|fixed>_kg<kg>.json.

    Every constant action is flown on every training case (with the variant's execution gain):
      single  the action with the highest mean return             -> dict(kind='constant', action)
      lookup  for every training case the action with the highest return; a flight uses the entry whose target
              is nearest to its own target                        -> dict(kind='lookup', targets, actions)"""
    from .baselines import constant_table
    path = config.runs_dir()/f"start_{'morph' if morph else 'fixed'}_kg{kg:g}.json"
    if path.exists():
        tab = load_json(path)
    else:
        G, viol = constant_table(cases, workers, morph, kg)
        tab = dict(G=G.tolist(), violations=viol.tolist(), targets=[[c[1].range_km, c[1].speed_mps] for c in cases],
                   kg=kg, morph=morph)
        save_json(path, tab)
    G = np.asarray(tab['G'])                                   # [action, case]
    if start == 'single':
        a = int(np.argmax(G.mean(1)))
        return dict(kind='constant', action=a, decoded=F.decode(a, morph))
    return dict(kind='lookup', targets=tab['targets'], actions=[int(a) for a in np.argmax(G, 0)])


# ------------------------------------------------------------------------------------------- training
def _log(msg):
    print(msg, flush=True)


def train(variant, seed, workers=4, out=None, iterations=None, cases=None, pi0=None, overrides=None, log=_log):
    """overrides / cases / pi0 (an action index or a stage dict): only for the tests; runs use the configuration."""
    from .region import training_cases
    cfg = config.load()
    tr = dict(cfg['training'], **(overrides or {}))
    v = VARIANTS[variant]
    morph, inputs = v['morph'], input_index(v['inputs'])
    kg = variant_kg(v)
    out = Path(out or config.runs_dir()/'train'/f'{variant}_s{seed}')
    out.mkdir(parents=True, exist_ok=True)
    path = out/'state.json'
    iterations = tr['iterations'] if iterations is None else iterations
    patience = int(tr['stride_s']//cfg['mission']['decision_s'])
    n_actions = F.n_actions(morph)
    if path.exists():
        st = load_json(path)
        cases = [case_from_json(c) for c in st['cases']]
    else:
        cases = cases or training_cases()
        if v['start'] == 'rule':
            cfg_ = config.load()
            if cfg_.get('guidance', {}).get('kind') != 'rel_pred':
                raise ValueError('variants rel/relfixed need the study R configuration (TARGET_RL_CONFIG=configs/study_r.yaml)')
            first = dict(kind='pfb', morph=morph, **cfg_['start_rule'])
        elif v['start'] == 'plan':
            from .guided import enabled, plan_action
            if not enabled():
                raise ValueError('variant pcguided needs the study A configuration (TARGET_RL_CONFIG=configs/study_a.yaml)')
            first = dict(kind='plan', action=plan_action(morph))
        elif pi0 is None:
            first = initial_stage(morph, kg, v['start'], workers, cases)
        else:
            first = pi0 if isinstance(pi0, dict) else dict(kind='constant', action=int(pi0))
        st = dict(variant=variant, seed=seed, iteration=0, policy_version=0, chain=[first],
                  cases=[case_to_json(c) for c in cases], label_archive=[], history=[],
                  settings=dict(inputs=list(v['inputs']), morph=morph, score=v['score'], kg=kg, start=v['start'],
                                training=tr))
    if st.get('converged'):
        log(f"[{variant} s{seed}] already finished: {st['converged']['reason']}")
        return st
    held = bool(tr.get('held_out_acceptance'))      # v3b: a candidate must also improve on held-out acceptance cases
    if held and 'acc_cases' not in st:
        from .region import v2_cases
        st['acc_cases'] = [case_to_json(c) for c in v2_cases('accept')]
    acc_cases = [case_from_json(c) for c in st['acc_cases']] if held else []
    if held and 'acc_G' not in st:
        st['acc_G'], st['acc_violations'], st['acc_hits'] = set_stats(fly_set(st['chain'], acc_cases, morph, workers, kg))
        save_json(path, st)
        log(f"[{variant} s{seed}] pi0 on {len(acc_cases)} acceptance cases: mean G {st['acc_G']:.4f}, hits {st['acc_hits']}, "
            f"violations {st['acc_violations']}")
    if 'G' not in st:
        res = fly_set(st['chain'], cases, morph, workers, kg)
        st['G'], st['violations'], st['hits'] = set_stats(res)
        st['schedules'] = [r['schedule'] for r in res]
        st['case_G'] = [r['G'] for r in res]
        save_json(path, st)
        log(f"[{variant} s{seed}] pi0: mean G {st['G']:.4f}, hits {st['hits']}/{len(cases)}, "
            f"violations {st['violations']}")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(cases)))
    stop_at = min(st['iteration']+iterations, tr['iterations'])
    while st['iteration'] < stop_at:
        i = st['iteration']
        t0 = time.perf_counter()
        k = tr['scenarios_per_iteration']
        pick = [int(order[(i*k+j) % len(order)]) for j in range(k)]
        times = state_times(i, tr['states'], tr['stride_s'], cfg['mission']['decision_s'], cfg['mission']['duration_s'])
        sweep = i//patience
        jobs = [('branch', dict(case=cases[c], case_id=c, chain=st['chain'], times=[t], sweep=sweep,
                                schedule=st['schedules'][c], morph=morph, kg=kg)) for c in pick for t in times]
        res = pool_map(jobs, workers)
        new = []
        for w in res:
            for t, a0, f in w['prefix']:
                new.append(dict(case=None, t=float(t), features=f, action=a0, base_action=a0, advantage=0.))
        rows = [r for w in res for r in w['rows']]
        for r in rows:
            new.append(dict(case=r['case'], t=r['t'], features=r['features'], action=r['action'],
                            base_action=r['base_action'], advantage=r['G_end']-st['case_G'][r['case']]))
        n_fail = sum(r['failed'] for r in rows)
        n_pos = sum(1 for x in new if x['action'] != x['base_action'] and x['advantage'] > tr['min_advantage'])
        version = st['policy_version']
        for x in new:
            x['v'] = version
        st['label_archive'] += new
        fit_labels, fit_w = retained_labels(st['label_archive'], version, LABEL_KEEP, LABEL_DECAY)
        st['label_archive'] = fit_labels
        members, fit = fit_ensemble(fit_labels, seed*1000+i, n_actions, inputs, tr['hidden'], fit_w, workers)
        label_times = sorted({float(x['t']) for x in fit_labels})
        specs = [st['chain']+[dict(kind='model', run=f'{variant}_s{seed}', members=members, margin=m, iteration=i,
                                   times=label_times, window=tr['window_s'], score=v['score'], inputs=inputs)]
                 for m in sorted(tr['margins'])]
        allc = cases+acc_cases
        cres = pool_map([('candidate', dict(case=c, chain=ch, morph=morph, kg=kg)) for ch in specs for c in allc],
                        workers)
        n, na = len(cases), len(allc)
        cands = []
        for j, ch in enumerate(specs):
            part = cres[j*na:j*na+n]
            G, viol, hits = set_stats(part)
            c = dict(chain=ch, G=G, violations=viol, hits=hits, part=part, margin=ch[-1]['margin'])
            if held:
                c['acc_G'], c['acc_violations'], c['acc_hits'] = set_stats(cres[j*na+n:(j+1)*na])
            cands.append(c)
        ok = [c for c in cands if c['violations'] <= st['violations']]
        if held:     # better on the training cases AND on the held-out acceptance cases, no more violations on either; best on the held-out
            ok = [c for c in ok if c['G'] > st['G']+1e-9 and c['acc_G'] > st['acc_G']+1e-9
                  and c['acc_violations'] <= st['acc_violations']]
            best = max(ok, key=lambda c: c['acc_G']) if ok else None
        else:
            best = max(ok, key=lambda c: c['G']) if ok else None
        accepted = bool(best and best['G'] > st['G']+1e-9)
        rec = dict(iteration=i, cases=pick, labels=len(new), invalid=n_fail, positive_pairs=n_pos,
                   improvable_fraction=improvable_fraction([x for x in new if x['case'] is not None],
                                                           tr['min_advantage']),
                   G_before=st['G'], accepted=accepted, fit=fit, wall_s=None,
                   candidates=[dict(margin=c['margin'], G=c['G'], violations=c['violations'], hits=c['hits'],
                                    **({k: c[k] for k in ('acc_G', 'acc_violations', 'acc_hits')} if held else {}))
                               for c in cands])
        if accepted:
            st['chain'], st['G'], st['violations'], st['hits'] = best['chain'], best['G'], best['violations'], best['hits']
            st['schedules'] = [r['schedule'] for r in best['part']]
            st['case_G'] = [r['G'] for r in best['part']]
            st['policy_version'] = version+1
            rec['accepted_margin'] = best['margin']
            if held:
                st['acc_G'], st['acc_violations'], st['acc_hits'] = best['acc_G'], best['acc_violations'], best['acc_hits']
        rec['wall_s'] = time.perf_counter()-t0
        st['history'].append(rec)
        st['iteration'] = i+1
        reason = stop_reason(st['history'], patience, no_acceptance_iterations=tr.get('no_acceptance_iterations'))
        if reason or st['iteration'] >= tr['iterations']:
            st['converged'] = dict(iteration=i, reason=reason or 'iteration limit', G=st['G'])
        save_json(path, st)
        log(f"[{variant} s{seed}] iter {i}: {len(new)} labels, {n_pos} positive, {n_fail} violating branches; "
            + (f"accepted margin {best['margin']:g}" if accepted else 'kept')
            + f"; mean G {st['G']:.4f}, hits {st['hits']}/{n}, violations {st['violations']}"
            + (f"; acceptance set G {st['acc_G']:.4f}, hits {st['acc_hits']}/{len(acc_cases)}, violations {st['acc_violations']}"
               if held else '') + f" ({rec['wall_s']:.0f} s)")
        if st.get('converged'):
            break
    return st


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=sorted(VARIANTS))
    p.add_argument('--seed', type=int)
    p.add_argument('--pi0', action='store_true',
                   help='only build the initial-policy tables of the configured variants (constant actions on all cases)')
    p.add_argument('--variants', default=None, help='with --pi0: variants to build tables for (default: configured)')
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--iterations', type=int, default=None, help='iterations in this call (default: up to the limit)')
    a = p.parse_args(argv)
    if not a.pi0 and (a.variant is None or a.seed is None):
        p.error('--variant and --seed are required')
    try:
        if a.pi0:
            from .region import training_cases
            cases = training_cases()
            done = set()
            names = (a.variants.replace(',', ' ').split() if a.variants else config.load()['experiment']['variants'])
            for name in names:
                v = VARIANTS[name]
                key = (v['morph'], variant_kg(v))
                if key in done:
                    continue
                done.add(key)
                for start in ('single', 'lookup'):
                    st = initial_stage(v['morph'], variant_kg(v), start, a.workers, cases)
                    desc = (f"action {st['action']} = (CL level, w level) {F.decode(st['action'], v['morph'])}"
                            if st['kind'] == 'constant' else f"lookup table of {len(st['actions'])} targets, "
                            f"{len(set(st['actions']))} distinct actions")
                    print(f"pi0 ({'morphing' if v['morph'] else 'fixed'}, kg {variant_kg(v):g}, {start}): {desc}", flush=True)
        else:
            train(a.variant, a.seed, a.workers, iterations=a.iterations)
    finally:
        from .parallel import close
        close()


if __name__ == '__main__':
    main()
