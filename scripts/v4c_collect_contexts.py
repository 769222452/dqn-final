"""v4c surrogate refit, step 1: record decision contexts on the certified chain with the F8D2000B library.

    TARGET_RL_CONFIG=configs/v4c.yaml PYTHONPATH=src python scripts/v4c_collect_contexts.py --flights 64 --jobs 4 \
        --out runs/v4c/F8D2000B/surrogate/contexts.jsonl

The exact-physics baseline flies (no surrogate exists yet for this snapshot), with the post corrector and the audits; the
look-ahead guard is off. Disturbed scenarios (level L2, seeds 5000+, used nowhere else) and the plant mismatch of the
configuration. Policy: the start rule of the configuration with exploration (each 12 s decision, with probability 0.4,
a random CL increment and a random morph level), so the contexts cover the brake shapes. Targets are uniform in
D 1700-2600 km, V 1100-1800 m/s (only the policy's inputs depend on them). From all committed decisions, --samples
contexts are drawn; splits are by flight (75 % train, 6 % validation, 19 % test), as in experiments/aoa_fix.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def fly(i):
    for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[k] = '1'
    from target_rl import config, flight as F
    from target_rl.chain import ChainFlight
    from target_rl.policy import chain_action
    from target_rl.scenario import Target, scenario
    rng = np.random.default_rng([20261123, i])
    tg = Target(f'ctx{i}', float(rng.uniform(1700, 2600)), float(rng.uniform(1100, 1800)))
    rule = [dict(kind='pfb', morph=True, **config.load()['start_rule'])]
    rec = []
    env = ChainFlight(scenario('L2', 5000+i), tg, True, physics_controller=True, record=rec)
    n_cl, n_w = len(env.w['cl_levels']), len(env.w['w_levels'])
    while not env.done:
        a = chain_action(rule, env.features(extra=True), True)
        if rng.random() < .4:
            a = F.encode(int(rng.integers(n_cl)), int(rng.integers(n_w)), True)
        env.step(a)
    return dict(flight=i, summary=env.summary(), contexts=rec)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--flights', type=int, default=64)
    p.add_argument('--samples', type=int, default=1500)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    with mp.get_context('spawn').Pool(a.jobs) as pool:
        res = pool.map(fly, range(a.flights), chunksize=1)
    rng = np.random.default_rng(20261124)
    order = rng.permutation(a.flights)
    n_tr, n_va = int(round(.75*a.flights)), max(2, int(round(.06*a.flights)))
    split = {int(f): ('train' if k < n_tr else 'validation' if k < n_tr+n_va else 'test') for k, f in enumerate(order)}
    rows = [dict(flight=f"v4c/{r['flight']}", step=j, split=split[r['flight']], t=c['t'], CL_raw=c['CL_raw'],
                 channel=c['channel'], pre_context={k: v for k, v in c['pre_context'].items() if k != 'control_contract'},
                 id=f"v4c/{r['flight']}#{j:03d}")
            for r in res for j, c in enumerate(r['contexts'])]
    pick = sorted(rng.choice(len(rows), size=min(a.samples, len(rows)), replace=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='\n') as f:
        for k in pick:
            f.write(json.dumps(rows[k])+'\n')
    summ = dict(flights=a.flights, decisions=len(rows), sampled=len(pick),
                splits={s: sum(rows[k]['split'] == s for k in pick) for s in ('train', 'validation', 'test')},
                flight_outcomes={r['flight']: dict(reason=r['summary']['reason'], dD_km=r['summary']['dD_km'],
                                                   dV_mps=r['summary']['dV_mps'], switches=r['summary']['n_switches'])
                                 for r in res},
                brake_shape_share=float(np.mean([not str(r['contexts'][0]['pre_context'].get('target', '')).startswith('x')
                                                 for r in res])) if res else None)
    out.with_suffix('.summary.json').write_text(json.dumps(summ, indent=1)+'\n', encoding='utf-8')
    print(json.dumps({k: summ[k] for k in ('flights', 'decisions', 'sampled', 'splits')}))
    print('reasons', {k: sum(v['reason'] == k for v in summ['flight_outcomes'].values())
                      for k in {v['reason'] for v in summ['flight_outcomes'].values()}})


if __name__ == '__main__':
    main()
