"""Sample teacher-label contexts from the archived V16 flights (run once; output is committed).

Each archived accepted-candidate/initial flight in v14_final/v16_b16/runs/seed*/iter_*/*/run holds the
proof of every executed 3 s decision, including the controller's pre context (state, alpha, geometry
transition and CL command before the generator). These states come from the V16 closed loop; the
DAgger stage adds states visited by the new inner loop.

Split by flight (never by decision), so validation/test contexts come from flights not used for fitting.

    python scripts/aoa_label_contexts.py --archive <extracted>/v14_final --out experiments/aoa_fix/surrogate/contexts.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def flights(archive):
    for run in sorted(Path(archive).glob('v16_b16/runs/seed*/iter_*/*/run')):
        if (run/'pools.jsonl').is_file() and (run/'proofs').is_dir():
            yield run


def executed(run):
    for line in (run/'pools.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        passed = [i for i, ok in enumerate(row['mask']) if ok]
        if not passed:
            return
        order = np.argsort(-np.asarray(row['preference']))
        action = next(int(a) for a in order if row['mask'][a])
        info = json.loads((run/'proofs'/(row['proof_ids'][action]+'.json')).read_text(encoding='utf-8'))['info']
        yield info


def split_of(flight_id, fractions=(.7, .15)):
    u = int(hashlib.sha256(flight_id.encode()).hexdigest()[:8], 16)/2**32
    return 'train' if u < fractions[0] else 'validation' if u < sum(fractions) else 'test'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--archive', required=True, help='extracted v14_final directory of the result archive')
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=1500)
    p.add_argument('--seed', type=int, default=20261031)
    a = p.parse_args(argv)
    archive = Path(a.archive)
    rows = []
    for run in flights(archive):
        fid = run.relative_to(archive/'v16_b16'/'runs').parent.as_posix()
        for k, info in enumerate(executed(run)):
            c = dict(info['pre_context'])
            c.pop('control_contract', None)
            rows.append(dict(flight=fid, step=k, split=split_of(fid), t=float(c['t']),
                             CL_raw=float(info['CL_raw']), channel=info['channel'], pre_context=c))
    rng = np.random.default_rng(a.seed)
    # Stratify by mission time (20 bins of 60 s) so early descent and late flight are both covered.
    bins = np.minimum((np.array([r['t'] for r in rows])//60).astype(int), 19)
    chosen = []
    for b in range(20):
        idx = np.flatnonzero(bins == b)
        take = min(len(idx), int(round(a.count*len(idx)/len(rows))))
        chosen.extend(rng.choice(idx, take, replace=False).tolist())
    chosen = sorted(chosen)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='\n') as stream:
        for i in chosen:
            stream.write(json.dumps(dict(rows[i], id=f"{rows[i]['flight']}#{rows[i]['step']:03d}"))+'\n')
    summary = dict(source_decisions=len(rows), flights=len({r['flight'] for r in rows}), sampled=len(chosen),
                   splits={s: sum(rows[i]['split'] == s for i in chosen) for s in ('train', 'validation', 'test')},
                   flights_per_split={s: len({r['flight'] for r in rows if r['split'] == s})
                                      for s in ('train', 'validation', 'test')},
                   archive_sha256='1a8d1a38e0153a9349cf77ce97248a6f828c9c176f18c1ffdb2fb50380b99130',
                   sampling_seed=a.seed, contexts_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
