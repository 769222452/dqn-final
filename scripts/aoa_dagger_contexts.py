"""DAgger contexts: pre contexts visited by flights of the round-0 surrogate (development states only).

    python scripts/aoa_dagger_contexts.py runs/aoa/dagger --out runs/aoa/surrogate/dagger_contexts.jsonl

Reads every <dir>/**/steps.jsonl written by aoa_gate_experiment.py --inner surrogate. Keeps all decisions
that left the surrogate channel (physics baseline or true OCP after an OOD refusal) and a uniform random
sample of the rest, up to --count. Splits are assigned by flight with the hash rule of
aoa_label_contexts.py (70/15/15), so the length-scale selection and the qualification also see
on-policy validation/test labels from flights never used for fitting.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aoa_label_contexts import split_of


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('flights', help='directory holding the round-0 flights')
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=400)
    p.add_argument('--seed', type=int, default=20261101)
    a = p.parse_args(argv)
    root = Path(a.flights)
    rows = []
    for steps in sorted(root.glob('**/steps.jsonl')):
        flight = steps.parent.relative_to(root).as_posix()
        for line in steps.read_text(encoding='utf-8').splitlines():
            s = json.loads(line)
            rows.append(dict(id=f"dagger/{flight}#{s['step']:03d}", flight='dagger/'+flight, step=s['step'],
                             split=split_of('dagger/'+flight), source='dagger', t=s['t_start_s'], channel=s.get('channel'),
                             pre_context=s['pre_context']))
    off = [i for i, r in enumerate(rows) if r['channel'] != 'FAST_APPROXIMATE']
    rest = [i for i, r in enumerate(rows) if r['channel'] == 'FAST_APPROXIMATE']
    rng = np.random.default_rng(a.seed)
    take = max(0, a.count-len(off))
    chosen = sorted(off[:a.count]+rng.choice(rest, min(take, len(rest)), replace=False).tolist())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='\n') as stream:
        for i in chosen:
            stream.write(json.dumps(rows[i])+'\n')
    print(json.dumps(dict(flights=len({r['flight'] for r in rows}), decisions=len(rows), off_surrogate=len(off),
                          sampled=len(chosen), splits={s: sum(rows[i]['split'] == s for i in chosen)
                                                       for s in ('train', 'validation', 'test')})))


if __name__ == '__main__':
    main()
