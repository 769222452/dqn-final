"""Disturbance study, fast-model stage (experiments/perturb/PREREGISTRATION.md).

    python scripts/perturb_study.py tune --out experiments/perturb/baselines.json --levels L1 L2 L3 L3b
    python scripts/perturb_study.py eval --baselines experiments/perturb/baselines.json --level L2 \
        --runs runs/perturb/L2_full_s17 runs/perturb/L2_full_s23 ... --out experiments/perturb/eval_L2.json

tune: baselines on the 16 training scenarios only (B2 dynamic-pressure tracking law, B1 constant CL with the same
      protection loop); the selected B2 inner-loop gains are shared by every policy.
eval: the trained policies and the baselines on the 40 test scenarios; paired differences against B2 and B1.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight import fastflight as F  # noqa: E402
from rollout_improvement.scenario_training import Runner, constant, fly_set, paired  # noqa: E402
from rollout_improvement.training import load_json, save_json  # noqa: E402

R_TUNE = [k for k, r in enumerate(F.R_LEVELS) if r >= .75]
KP, KG = (1., 2., 4.), (.3, 1., 3.)
CL_CONST = (.020, .0225, .025, .0275, .030, .035)


def _select(rows):
    ok = [r for r in rows if r['failures'] == 0]
    return max(ok, key=lambda r: r['mean_range_km']) if ok else None


def tune(runner, level, log=print):
    seeds = F.TRAIN_SEEDS
    rows = []
    for kp in KP:
        for kg in KG:
            for a in R_TUNE:
                res = fly_set(runner, [constant(a)], level, seeds, dict(kp=kp, kg=kg))
                rows.append(dict(action=a, r=F.R_LEVELS[a], kp=kp, kg=kg, failures=int(sum(not x['success'] for x in res)),
                                 mean_range_km=float(np.mean([x['range_km'] for x in res]))))
    b2 = _select(rows)
    if b2 is None:
        raise RuntimeError(f'{level}: no B2 setting without violations on the training scenarios')
    log(f"{level} B2: r={b2['r']:.2f} kp={b2['kp']:g} kg={b2['kg']:g} -> {b2['mean_range_km']:.2f} km "
        f"({sum(r['failures'] == 0 for r in rows)}/{len(rows)} settings feasible)")
    rows1 = []
    for cl in CL_CONST:
        for a in R_TUNE:
            inner = dict(kp=b2['kp'], kg=b2['kg'], mode='const', cl_const=cl)
            res = fly_set(runner, [constant(a)], level, seeds, inner)
            rows1.append(dict(action=a, r=F.R_LEVELS[a], cl_const=cl, failures=int(sum(not x['success'] for x in res)),
                              mean_range_km=float(np.mean([x['range_km'] for x in res]))))
    b1 = _select(rows1)
    if b1 is not None:
        b1.update(kp=b2['kp'], kg=b2['kg'])
        log(f"{level} B1: CL={b1['cl_const']:g} protection r={b1['r']:.2f} -> {b1['mean_range_km']:.2f} km")
    return dict(B2=b2, B1=b1, grid_B2=rows, grid_B1=rows1)


def baseline_entries(base):
    chains = {'B2': [constant(base['B2']['action'])]}
    inners = {'B2': dict(kp=base['B2']['kp'], kg=base['B2']['kg'])}
    if base.get('B1'):
        b1 = base['B1']
        chains['B1'] = [constant(b1['action'])]
        inners['B1'] = dict(kp=b1['kp'], kg=b1['kg'], mode='const', cl_const=b1['cl_const'])
    return chains, inners


def evaluate(runner, base, level, runs, seeds=F.TEST_SEEDS):
    chains, inners = baseline_entries(base)
    names = []
    for run in runs:
        st = load_json(Path(run)/'state.json')
        if st['level'] != level:
            raise ValueError(f'{run}: level {st["level"]} != {level}')
        name = Path(run).name
        chains[name], inners[name] = st['chain'], st['inner']
        names.append(name)
    results = {n: fly_set(runner, c, level, seeds, inners[n]) for n, c in chains.items()}
    summary = {}
    for n in names:
        summary[n] = dict(vs_B2=paired(results, n, 'B2'), **({'vs_B1': paired(results, n, 'B1')} if 'B1' in results else {}))
    if 'B1' in results:
        summary['B1'] = dict(vs_B2=paired(results, 'B1', 'B2'))
    summary['B2'] = dict(mean_range_km=float(np.mean([r['range_km'] for r in results['B2']])),
                         failures=int(sum(not r['success'] for r in results['B2'])))
    return dict(level=level, seeds=list(seeds), summary=summary,
                flights={n: [{k: r[k] for k in ('scenario', 'range_km', 'success', 'reason', 'min_margin')} for r in rs]
                         for n, rs in results.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tune')
    t.add_argument('--out', required=True)
    t.add_argument('--levels', nargs='+', default=['L1', 'L2', 'L3', 'L3b'])
    t.add_argument('--workers', type=int, default=4)
    e = sub.add_parser('eval')
    e.add_argument('--baselines', required=True)
    e.add_argument('--level', required=True)
    e.add_argument('--runs', nargs='*', default=[])
    e.add_argument('--out', required=True)
    e.add_argument('--workers', type=int, default=4)
    a = p.parse_args(argv)
    runner = Runner(a.workers)
    try:
        if a.cmd == 'tune':
            path = Path(a.out)
            out = load_json(path) if path.exists() else {}
            for lvl in a.levels:
                if lvl not in out:
                    out[lvl] = tune(runner, lvl)
                    save_json(path, out)
        else:
            base = json.loads(Path(a.baselines).read_text())[a.level]
            res = evaluate(runner, base, a.level, a.runs)
            save_json(Path(a.out), res)
            for n, s in res['summary'].items():
                print(n, json.dumps(s))
    finally:
        runner.close()


if __name__ == '__main__':
    main()
