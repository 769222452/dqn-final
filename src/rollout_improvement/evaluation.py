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

from .ensemble import LABEL_SCALE
from .features import features
from .policy import MAX_ACTION, ChainPolicy, chain_action, constant_stage
from .rollouts import certified_fly

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
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    return [float(np.quantile(means, (1-level)/2)), float(np.quantile(means, (1+level)/2))]


def _name(d):
    return f'dh{d[0]:g}_dv{d[1]:g}'


def _job(args):
    kind, chain, p, schedule, out = args
    out = Path(out)
    done = out/'result.json'
    if done.exists():
        return json.loads(done.read_text(encoding='utf-8'))
    try:
        tick = time.perf_counter()
        if kind == 'initial':
            env, certifier = build_flight(out/'run', 'evaluation', initial_offset=p)
            pi = ChainPolicy(chain, env)
            g, stop, acts = certified_fly(certifier, pi)
            res = dict(kind=kind, dh=p[0], dV=p[1], G=g, stop=stop, range_km=float(env.x[3]/1000),
                       switches=int(sum(a != b for a, b in zip(acts, acts[1:]))),
                       schedule=[[float(t), int(a)] for t, a in pi.decisions])
        else:
            t_b, delta, loop = p
            env, certifier = build_flight(out/'run', 'evaluation')
            g0, stop, _ = certified_fly(certifier, ChainPolicy(chain, env), until=t_b)
            a = chain_action(chain, features(env))+delta
            if not 0 <= a <= MAX_ACTION:
                res = dict(kind=kind, t=t_b, delta=delta, loop=loop, skipped=True)
            else:
                pol = ChainPolicy(chain, env, override=(t_b, a), schedule=schedule if loop == 'open' else None)
                g1, stop, _ = certified_fly(certifier, pol)
                changed = sum(1 for t, x in pol.decisions if t > t_b+1e-9 and x != schedule[int(round(t))]) \
                    if loop == 'closed' else 0
                res = dict(kind=kind, t=t_b, delta=delta, loop=loop, G=g0+g1, stop=stop, redecisions_changed=changed)
        res['wall_s'] = time.perf_counter()-tick
    except BaseException:
        res = dict(kind=kind, p=list(p), error=traceback.format_exc())
    out.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(res, indent=1), encoding='utf-8')
    shutil.rmtree(out/'run', ignore_errors=True)
    return res


def evaluate(chain, out, reference_cl=.025, workers=8, log=print, test_set='held-out-2', reference_out=None):
    """Run the tests of `test_set` (see TEST_SETS); reference flights are cached in reference_out (default: out)
    so that several policies can share them."""
    offsets, probe_times = test_set_points(test_set)
    out = Path(out)
    ref_dir = Path(reference_out) if reference_out else out
    ref_chain = [constant_stage(reference_cl)]
    with mp.get_context('spawn').Pool(workers) as pool:
        jobs = [('initial', chain, d, None, str(out/f'initial_{_name(d)}')) for d in offsets]
        jobs += [('initial', ref_chain, d, None, str(ref_dir/f'reference_cl{reference_cl:g}_{_name(d)}')) for d in offsets]
        res = pool.map(_job, jobs, chunksize=1)
        inits, refs = res[:len(offsets)], res[len(offsets):]
        for r in res:
            if 'error' in r:
                raise RuntimeError(r['error'])
        nominal = next(r for r in inits if r['dh'] == 0 and r['dV'] == 0)
        schedule = {int(round(t)): a for t, a in nominal['schedule']}
        probes = pool.map(_job, [('probe', chain, (t, d, loop), schedule, str(out/f'probe_t{int(t):04d}_{d:+d}_{loop}'))
                                 for t in probe_times for d in (-1, 1) for loop in ('closed', 'open')], chunksize=1)
    ref = {(r['dh'], r['dV']): r for r in refs}
    rows = [dict(dh=r['dh'], dV=r['dV'], range_km=r['range_km'],
                 gain_km=r['range_km']-ref[(r['dh'], r['dV'])]['range_km'], switches=r['switches'], stop=r['stop'],
                 reference_stop=ref[(r['dh'], r['dV'])]['stop']) for r in inits]
    perturbed = [x for x in rows if (x['dh'], x['dV']) != (0., 0.)]
    gains = [x['gain_km'] for x in perturbed]
    success = {k: r['stop'] == 'DONE' for k, r in ref.items()}
    gains_ref_done = [x['gain_km'] for x in perturbed if success[(x['dh'], x['dV'])]]
    by = {(r['t'], r['delta'], r['loop']): r for r in probes if 'error' not in r and not r.get('skipped')}
    pairs = []
    for t in probe_times:
        for d in (-1, 1):
            c, o = by.get((t, d, 'closed')), by.get((t, d, 'open'))
            if c and o:
                pairs.append(dict(t=t, delta=d, A_closed=LABEL_SCALE*(c['G']-nominal['G']),
                                  A_open=LABEL_SCALE*(o['G']-nominal['G']), redecisions_changed=c['redecisions_changed'],
                                  stops=[c['stop'], o['stop']]))
    summary = dict(test_set=test_set, reference_cl=reference_cl, nominal_range_km=nominal['range_km'],
                   nominal_G=nominal['G'], nominal_gain_km=nominal['range_km']-ref[(0., 0.)]['range_km'],
                   nominal_switches=nominal['switches'], perturbed_states=len(gains),
                   perturbed_mean_gain_km=float(np.mean(gains)), perturbed_gain_ci95_km=paired_bootstrap_ci(gains),
                   perturbed_wins=int(sum(g > 0 for g in gains)), perturbed_worst_gain_km=float(min(gains)),
                   perturbed_completed=int(sum(x['stop'] == 'DONE' for x in perturbed)),
                   perturbed_reference_completed=int(sum(success[(x['dh'], x['dV'])] for x in perturbed)),
                   reference_completed_mean_gain_km=float(np.mean(gains_ref_done)) if gains_ref_done else None,
                   reference_completed_gain_ci95_km=paired_bootstrap_ci(gains_ref_done) if gains_ref_done else None,
                   perturbed_mean_switches=float(np.mean([x['switches'] for x in perturbed])),
                   probe_mean_A_closed=float(np.mean([p['A_closed'] for p in pairs])),
                   probe_mean_A_open=float(np.mean([p['A_open'] for p in pairs])),
                   probe_closed_not_worse=int(sum(p['A_closed'] >= p['A_open'] for p in pairs)),
                   initial_conditions=rows, probes=pairs)
    (out/'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    log(f"nominal {nominal['range_km']:.2f} km ({summary['nominal_gain_km']:+.2f} km vs constant CL {reference_cl:g})")
    ci = summary['perturbed_gain_ci95_km']
    log(f"perturbed initial states ({test_set}): mean gain {summary['perturbed_mean_gain_km']:+.2f} km "
        f"(95% CI {ci[0]:+.2f} to {ci[1]:+.2f}), wins {summary['perturbed_wins']}/{len(gains)}, worst "
        f"{summary['perturbed_worst_gain_km']:+.2f} km, completed {summary['perturbed_completed']}/{len(gains)} "
        f"(reference {summary['perturbed_reference_completed']}/{len(gains)})")
    log(f"one-decision probes (1e-4 return): mean A_closed {summary['probe_mean_A_closed']:+.2f}, mean A_open "
        f"{summary['probe_mean_A_open']:+.2f}, closed >= open in {summary['probe_closed_not_worse']}/{len(pairs)}")
    for p in pairs:
        log(f"  t={p['t']:5.0f} {p['delta']:+d}: closed {p['A_closed']:+8.3f}  open {p['A_open']:+8.3f}  "
            f"re-decisions changed {p['redecisions_changed']}")
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
    a = p.parse_args(argv)
    chain = json.loads(Path(a.state).read_text(encoding='utf-8-sig'))['chain']
    evaluate(chain, a.out, a.reference_cl, a.workers, log=lambda s: print(s, flush=True), test_set=a.test_set,
             reference_out=a.reference_out)
    return 0
