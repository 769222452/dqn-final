"""Teacher labels (true pre + post OCP of an objective) for recorded pre contexts. Parallel and resumable.

    python scripts/aoa_generate_labels.py --objective consistent \
        --contexts experiments/aoa_fix/surrogate/contexts.jsonl --out runs/aoa/surrogate/labels.jsonl --jobs 47

Each output line: id, flight, split, pre_context, pre_mach, post_context, post_alpha, ok/reason, solver
statuses, wall_s. Ids already in --out are skipped, so an interrupted run resumes. See
certified_flight/distill.py for the teacher definition.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight.distill import Teacher, read_jsonl  # noqa: E402
from certified_flight.objectives import OBJECTIVES  # noqa: E402
from certified_flight.serialization import clean  # noqa: E402

_teacher = None


def _init(objective):
    global _teacher
    _teacher = Teacher(objective)


def _label(row):
    out = _teacher.label(row['pre_context'])
    return dict({k: row[k] for k in ('id', 'flight', 'split', 'source') if k in row}, **out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--objective', choices=sorted(OBJECTIVES), required=True)
    p.add_argument('--contexts', required=True, nargs='+', help='jsonl files with id, split, pre_context')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=max(1, os.cpu_count()-1))
    p.add_argument('--limit', type=int, help='label only the first N pending contexts (smoke test)')
    a = p.parse_args(argv)
    rows = [r for f in a.contexts for r in read_jsonl(f)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r['id'] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r['id'] not in done][:a.limit]
    print(f'{len(rows)} contexts, {len(done)} labelled, {len(todo)} to label with {a.jobs} workers', flush=True)
    started, failed = time.perf_counter(), 0
    with out.open('a', encoding='utf-8', newline='\n') as stream, \
            mp.get_context('spawn').Pool(a.jobs, _init, (a.objective,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_label, todo), 1):
            stream.write(json.dumps(clean(r))+'\n')
            stream.flush()
            failed += not r['ok']
            if k % 25 == 0 or k == len(todo):
                el = time.perf_counter()-started
                print(f'{k}/{len(todo)} labelled, {failed} failed, {el/60:.1f} min, '
                      f'eta {el/k*(len(todo)-k)/60:.1f} min', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
