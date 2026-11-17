"""Evaluation on the test cases (20 test scenarios x 16 targets: T1-T6 and R1-R10).

Methods: constant (pi0), pc_fair, pc_strong, pc_fixed, reference, and every finished training run
(rl:<variant>:<seed>, read from runs/<library>/train/<variant>_s<seed>/state.json). One file per method:
runs/<library>/eval/<level>/<method>.json, resumable (finished cases are kept).

    python -m target_rl.evaluate --methods all --workers 14
    python -m target_rl.evaluate --methods pc_fair,rl:full:17 --level L2
"""
import argparse
import os

from . import config
from .parallel import pool_imap
from .util import load_json, save_json

BASELINES = ('constant', 'pc_fair', 'pc_strong', 'pc_fixed', 'reference')


def fly_method(method, case, spec):
    from .baselines import PredictorCorrector, fly_constant, reference
    sc, tg = case
    if method == 'constant':
        return fly_constant(sc, tg, spec['action'], True).summary()
    if method in ('pc_fair', 'pc_strong', 'pc_fixed'):
        return PredictorCorrector(method).fly(sc, tg).summary()
    if method == 'reference':
        return reference(sc, tg)
    if method.startswith('rl:'):
        from . import flight as F
        from .train import fly
        return fly(F.TargetFlight(sc, tg, spec['morph']), spec['chain']).summary()
    raise ValueError(method)


def trained_runs():
    out = []
    for p in sorted((config.runs_dir()/'train').glob('*_s*/state.json')):
        st = load_json(p)
        if st.get('converged'):
            out.append(f"rl:{st['variant']}:{st['seed']}")
    return out


def spec_for(method):
    if method == 'constant':
        return dict(action=load_json(config.runs_dir()/'pi0_morph.json')['action'])
    if method.startswith('rl:'):
        _, variant, seed = method.split(':')
        st = load_json(config.runs_dir()/'train'/f'{variant}_s{seed}'/'state.json')
        return dict(chain=st['chain'], morph=st['settings']['morph'])
    return None


def evaluate(method, level, workers, cases=None, log=print):
    from .region import test_cases
    cases = cases or test_cases(level=level)
    path = config.runs_dir()/'eval'/level/f"{method.replace(':', '_')}.json"
    done = load_json(path)['rows'] if path.exists() else []
    have = {(r['seed'], r['target']) for r in done}
    todo = [c for c in cases if (c[0].seed, c[1].name) not in have]
    if not todo:
        log(f'{method}: {len(done)} cases already evaluated')
        return done
    spec = spec_for(method)
    rows = list(done)
    for k, r in enumerate(pool_imap([('method', dict(method=method, case=c, spec=spec)) for c in todo], workers)):
        if 'error' in r:
            raise RuntimeError(r['error'])
        rows.append(r)
        if (k+1) % 20 == 0 or k+1 == len(todo):
            save_json(path, dict(method=method, level=level, rows=rows))
            log(f'{method}: {len(rows)}/{len(cases)}')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', default='all', help="comma list, 'baselines', 'rl' or 'all'")
    p.add_argument('--level', default=None)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)
    level = a.level or config.load()['disturbance']['main_level']
    if a.methods in ('all', 'baselines', 'rl'):
        methods = (list(BASELINES) if a.methods != 'rl' else [])+(trained_runs() if a.methods != 'baselines' else [])
    else:
        methods = a.methods.split(',')
    try:
        for m in methods:
            evaluate(m, level, a.workers, log=lambda s: print(s, flush=True))
    finally:
        from .parallel import close
        close()


if __name__ == '__main__':
    main()
