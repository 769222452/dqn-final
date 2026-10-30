"""Paired comparison of evaluated policies on the same initial states.

    python scripts/compare.py --out comparison.json NAME=EVALUATION_DIR [NAME=EVALUATION_DIR ...]

Each EVALUATION_DIR holds the summary.json written by scripts/evaluate.py for the same test set and reference
policy. For every policy the gains over the reference are summarised; for every pair of policies (the first
named policy against each other one) the paired range differences are summarised.
"""
import os
import sys

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from certified_flight.serialization import write_json  # noqa: E402
from rollout_improvement.evaluation import paired_bootstrap_ci  # noqa: E402
from rollout_improvement.identity import digest_json  # noqa: E402


def describe(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return dict(n=0, mean=None, ci95=None, wins=0, worst=None, best=None, median=None)
    return dict(n=len(x), mean=float(x.mean()), ci95=paired_bootstrap_ci(x), wins=int((x > 0).sum()),
                worst=float(x.min()), best=float(x.max()), median=float(np.median(x)))


def compare(summaries):
    names = list(summaries)
    if not names:
        raise ValueError('at least one evaluated policy is required')
    if any(s.get('complete') is False for s in summaries.values()):
        raise ValueError('incomplete evaluation: inspect coverage and missing pairs before comparing policies')
    identities = [summaries[n].get('comparison_identity') for n in names]
    if any(identity is not None for identity in identities):
        if any(identity is None for identity in identities) or len({digest_json(i) for i in identities}) != 1:
            raise ValueError('evaluation runtime/protocol identities differ or are missing')
    sets = [summaries[n]['test_set'] for n in names]
    refs = [summaries[n]['reference_cl'] for n in names]
    if len(set(sets)) != 1 or len(set(refs)) != 1:
        raise ValueError('all policies must be evaluated on the same test set and reference')
    states = {n: {(r['dh'], r['dV']): r for r in summaries[n]['initial_conditions'] if (r['dh'], r['dV']) != (0., 0.)}
              for n in names}
    keys = sorted(states[names[0]])
    if any(sorted(states[n]) != keys for n in names):
        raise ValueError('policies were evaluated on different initial states')
    result = dict(test_set=sets[0], reference_cl=refs[0], states=len(keys), policies={}, pairs={},
                  identity_verified=all(i is not None for i in identities), comparison_identity=identities[0])
    for n in names:
        s = summaries[n]
        done = [k for k in keys if states[n][k].get('reference_completed',
                                                  states[n][k].get('reference_stop', 'DONE') == 'DONE')]
        result['policies'][n] = dict(
            gain_over_reference_km=describe([states[n][k]['gain_km'] for k in keys]),
            gain_where_reference_completed_km=describe([states[n][k]['gain_km'] for k in done]),
            reference_completed=len(done),
            completed=int(sum(states[n][k].get('completed', states[n][k]['stop'] == 'DONE') for k in keys)),
            nominal_range_km=s['nominal_range_km'], nominal_gain_km=s['nominal_gain_km'],
            probe_mean_A_closed=s['probe_mean_A_closed'], probe_mean_A_open=s['probe_mean_A_open'],
            probe_closed_not_worse=s['probe_closed_not_worse'], probes=len(s['probes']))
    for n in names[1:]:
        diff = [states[names[0]][k]['range_km']-states[n][k]['range_km'] for k in keys]
        result['pairs'][f'{names[0]} - {n}'] = describe(diff)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('policies', nargs='+', help='NAME=EVALUATION_DIR')
    p.add_argument('--out', required=True, help='output JSON file')
    a = p.parse_args(argv)
    summaries = {}
    for item in a.policies:
        name, folder = item.split('=', 1)
        summaries[name] = json.loads((Path(folder)/'summary.json').read_text(encoding='utf-8'))
    result = compare(summaries)
    write_json(a.out, result)
    for n, r in result['policies'].items():
        g = r['gain_over_reference_km']
        if not g['n']:
            print(f'{n:12s} no perturbed initial-state pairs; statistics unavailable')
            continue
        print(f"{n:12s} gain over CL {result['reference_cl']:g}: mean {g['mean']:+.2f} km (95% CI {g['ci95'][0]:+.2f} to "
              f"{g['ci95'][1]:+.2f}), wins {g['wins']}/{g['n']}, worst {g['worst']:+.2f} km, completed "
              f"{r['completed']}/{g['n']}; probes A_closed {_number(r['probe_mean_A_closed'])}, "
              f"A_open {_number(r['probe_mean_A_open'])}")
        h = r['gain_where_reference_completed_km']
        if not h['n']:
            print(f"{'':12s} reference completed 0 states; conditional gain statistics unavailable")
            continue
        print(f"{'':12s} where the reference completed ({h['n']} states): mean {h['mean']:+.2f} km (95% CI "
              f"{h['ci95'][0]:+.2f} to {h['ci95'][1]:+.2f}), wins {h['wins']}/{h['n']}, worst {h['worst']:+.2f} km")
    for pair, d in result['pairs'].items():
        if not d['n']:
            print(f'{pair}: no paired states; statistics unavailable')
            continue
        print(f"{pair}: mean {d['mean']:+.2f} km (95% CI {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f}), "
              f"first better in {d['wins']}/{d['n']}, worst {d['worst']:+.2f} km")
    return 0


def _number(value):
    return 'unavailable' if value is None else f'{value:+.2f}'


if __name__ == '__main__':
    sys.exit(main())
