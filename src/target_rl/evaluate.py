"""Evaluation on the test cases (20 test scenarios x 16 targets: T1-T6 and R1-R10).

Methods: constant (pi0), pc_fair, pc_strong, pc_fixed, pc_oracle (full-information predictor-corrector),
reference (open-loop clairvoyant search; not in the default sets), and every finished training run
(rl:<variant>:<seed>, read from runs/<library>/train/<variant>_s<seed>/state.json).

Method sets: all = constant, pc_fair, pc_strong, pc_fixed, pc_oracle + every run; core = constant, pc_fair,
pc_strong, pc_oracle + the full-method runs; baselines; rl.

Every row records the wall time of the flight (one process, one BLAS thread) and, for predictor-correctors, the
number of trajectory predictions. The cases of experiment.trace_cases (configs/task.yaml) are also flown with a
per-step trajectory log, stored in runs/<library>/traces/<level>/<method>.json.

One file per method: runs/<library>/eval/<level>/<method>.json, resumable (finished cases are kept).

    python -m target_rl.evaluate --methods all --workers 14
    python -m target_rl.evaluate --methods core --level L3
"""
import argparse
import os
import time

from . import config
from .parallel import pool_imap
from .util import load_json, save_json

BASELINES = ('constant', 'pc_fair', 'pc_strong', 'pc_fixed', 'pc_oracle')
CORE_BASELINES = ('constant', 'pc_fair', 'pc_strong', 'pc_oracle')
PCS = ('pc_fair', 'pc_strong', 'pc_fixed', 'pc_oracle')


def fly_method(method, case, spec, trace=False):
    from .baselines import PredictorCorrector, fly_constant, reference
    sc, tg = case
    t0 = time.perf_counter()
    if method == 'constant':
        env = fly_constant(sc, tg, spec['action'], True, trace=trace)
    elif method in PCS:
        env = PredictorCorrector(method).fly(sc, tg, trace=trace)
    elif method == 'reference':
        r = reference(sc, tg)
        r['wall_s'] = time.perf_counter()-t0
        return r
    elif method.startswith('rl:'):
        from . import flight as F
        from .train import fly
        env = fly(F.TargetFlight(sc, tg, spec['morph'], trace=trace), spec['chain'])
    else:
        raise ValueError(method)
    r = env.summary()
    r['wall_s'] = time.perf_counter()-t0
    r['decisions'] = int(round(env.t/env.w['decision']))
    if trace:
        r['trace'] = env.trace
    return r


def trained_runs(variants=None):
    out = []
    for p in sorted((config.runs_dir()/'train').glob('*_s*/state.json')):
        st = load_json(p)
        if st.get('converged') and (variants is None or st['variant'] in variants):
            out.append(f"rl:{st['variant']}:{st['seed']}")
    return out


def method_set(name):
    if name == 'all':
        return list(BASELINES)+trained_runs()
    if name == 'core':
        return list(CORE_BASELINES)+trained_runs(('full',))
    if name == 'baselines':
        return list(BASELINES)
    if name == 'rl':
        return trained_runs()
    return name.split(',')


def spec_for(method):
    if method == 'constant':
        return dict(action=load_json(config.runs_dir()/'pi0_morph.json')['action'])
    if method.startswith('rl:'):
        _, variant, seed = method.split(':')
        st = load_json(config.runs_dir()/'train'/f'{variant}_s{seed}'/'state.json')
        return dict(chain=st['chain'], morph=st['settings']['morph'])
    return None


def trace_keys(cfg=None):
    cfg = cfg or config.load()
    return {(int(s), str(t)) for s, t in cfg['experiment']['trace_cases']}


def evaluate(method, level, workers, cases=None, log=print):
    from .region import test_cases
    cases = cases or test_cases(level=level)
    name = method.replace(':', '_')
    path = config.runs_dir()/'eval'/level/f'{name}.json'
    tpath = config.runs_dir()/'traces'/level/f'{name}.json'
    done = load_json(path)['rows'] if path.exists() else []
    traces = load_json(tpath)['traces'] if tpath.exists() else []
    have = {(r['seed'], r['target']) for r in done}
    tkeys = trace_keys()
    todo = [c for c in cases if (c[0].seed, c[1].name) not in have]
    if not todo:
        log(f'{method} ({level}): {len(done)} cases already evaluated')
        return done
    spec = spec_for(method)
    rows = list(done)
    jobs = [('method', dict(method=method, case=c, spec=spec, trace=(c[0].seed, c[1].name) in tkeys)) for c in todo]
    for k, r in enumerate(pool_imap(jobs, workers)):
        if 'error' in r:
            raise RuntimeError(r['error'])
        if 'trace' in r:
            traces.append(dict(seed=r['seed'], target=r['target'], columns=[
                't', 'h', 'V', 'gamma', 'range_km', 'CL', 'w', 'q_ratio', 'protected'], trace=r.pop('trace')))
        rows.append(r)
        if (k+1) % 20 == 0 or k+1 == len(todo):
            save_json(path, dict(method=method, level=level, rows=rows))
            if traces:
                save_json(tpath, dict(method=method, level=level, traces=traces))
            log(f'{method} ({level}): {len(rows)}/{len(cases)}')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', default='all', help="all, core, baselines, rl, or a comma list")
    p.add_argument('--level', default=None)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)
    level = a.level or config.load()['disturbance']['main_level']
    try:
        for m in method_set(a.methods):
            evaluate(m, level, a.workers, log=lambda s: print(s, flush=True))
    finally:
        from .parallel import close
        close()


if __name__ == '__main__':
    main()
