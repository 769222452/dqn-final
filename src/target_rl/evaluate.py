"""Evaluation on the test cases (round 2: test scenarios 2300-2319 x 16 targets: T1-T6 and R1-R10).

Methods:
  constant   one constant action for every target (best mean return on the training cases), round-2 execution
  lookup     the target lookup table alone (the initial policy of the method, "learning removed")
  pc_fair, pc_strong, pc_fixed, pc_oracle   predictor-correctors (baselines.py)
  reference  open-loop clairvoyant search (not in the default sets)
  rl:<variant>:<seed>   round-2 training runs (runs/r2/<library>/train/<variant>_s<seed>/state.json)
  r1:<variant>:<seed>   round-1 policies (runs/<library>/train/...) flown as trained in round 1 (kg = 0)

Method sets: all = constant, lookup, pc_fair, pc_strong, pc_fixed, pc_oracle + every round-2 run + the round-1
full runs; core = constant, lookup, pc_fair, pc_strong, pc_oracle + the round-2 full runs; baselines; rl.

Case sets: test (default) or dev (pilot scenarios x T1-T6; only for scripts/pilot_r2.sh, never reported as test).

Every row records the wall time of the flight (one process, one BLAS thread) and, for predictor-correctors, the
number of trajectory predictions. The cases of experiment.trace_cases are also flown with a per-step trajectory log
(runs/r2/<library>/traces/<level>/<method>.json).

One file per method: runs/r2/<library>/eval/<level>/<method>.json (dev: eval_dev/), resumable.

    python -m target_rl.evaluate --methods all --workers 14
    python -m target_rl.evaluate --methods core --level L3
"""
import argparse
import os
import time

from . import config
from .parallel import pool_imap
from .util import load_json, save_json

BASELINES = ('constant', 'lookup', 'pc_fair', 'pc_strong', 'pc_fixed', 'pc_oracle')
CORE_BASELINES = ('constant', 'lookup', 'pc_fair', 'pc_strong', 'pc_oracle')
PCS = ('pc_fair', 'pc_strong', 'pc_fixed', 'pc_oracle')


def fly_method(method, case, spec, trace=False):
    from .baselines import PredictorCorrector, reference
    sc, tg = case
    t0 = time.perf_counter()
    if method in PCS:
        env = PredictorCorrector(method).fly(sc, tg, trace=trace)
    elif method == 'reference':
        r = reference(sc, tg)
        r['wall_s'] = time.perf_counter()-t0
        return r
    elif method in ('constant', 'lookup') or method.startswith(('rl:', 'r1:')):
        from . import flight as F
        from .train import fly
        env = fly(F.TargetFlight(sc, tg, spec['morph'], trace=trace, kg=spec['kg']), spec['chain'])
    else:
        raise ValueError(method)
    r = env.summary()
    r['wall_s'] = time.perf_counter()-t0
    r['decisions'] = int(round(env.t/env.w['decision']))
    if trace:
        r['trace'] = env.trace
    return r


def _runs(base, prefix, variants=None):
    out = []
    for p in sorted((base/'train').glob('*_s*/state.json')):
        st = load_json(p)
        if st.get('converged') and (variants is None or st['variant'] in variants):
            out.append(f"{prefix}:{st['variant']}:{st['seed']}")
    return out


def trained_runs(variants=None):
    return _runs(config.runs_dir(), 'rl', variants)


def round1_runs(variants=('full',)):
    return _runs(config.round1_dir(), 'r1', variants)


def method_set(name):
    if name == 'all':
        return list(BASELINES)+trained_runs()+round1_runs()
    if name == 'core':
        return list(CORE_BASELINES)+trained_runs(('full',))
    if name == 'baselines':
        return list(BASELINES)
    if name == 'rl':
        return trained_runs()
    return name.split(',')


def spec_for(method):
    from .train import VARIANTS, initial_stage, variant_kg
    if method in ('constant', 'lookup'):
        from .region import training_cases
        v = VARIANTS['full']
        kg = variant_kg(v)
        stage = initial_stage(True, kg, 'single' if method == 'constant' else 'lookup', 1, training_cases())
        return dict(chain=[stage], morph=True, kg=kg)
    if method.startswith(('rl:', 'r1:')):
        tag, variant, seed = method.split(':')
        base = config.runs_dir() if tag == 'rl' else config.round1_dir()
        st = load_json(base/'train'/f'{variant}_s{seed}'/'state.json')
        kg = st['settings'].get('kg', 0.)          # round-1 states have no kg: they were trained with kg = 0
        return dict(chain=st['chain'], morph=st['settings']['morph'], kg=kg)
    return None


def trace_keys(cfg=None):
    cfg = cfg or config.load()
    return {(int(s), str(t)) for s, t in cfg['experiment']['trace_cases']}


def evaluate(method, level, workers, cases=None, which='test', log=print):
    from .region import test_cases
    cases = cases or test_cases(level=level, which=which)
    name = method.replace(':', '_')
    sub = 'eval' if which == 'test' else f'eval_{which}'
    path = config.runs_dir()/sub/level/f'{name}.json'
    tpath = config.runs_dir()/'traces'/level/f'{name}.json'
    done = load_json(path)['rows'] if path.exists() else []
    traces = load_json(tpath)['traces'] if tpath.exists() and which == 'test' else []
    have = {(r['seed'], r['target']) for r in done}
    tkeys = trace_keys() if which == 'test' else set()
    todo = [c for c in cases if (c[0].seed, c[1].name) not in have]
    if not todo:
        log(f'{method} ({level}, {which}): {len(done)} cases already evaluated')
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
            save_json(path, dict(method=method, level=level, cases=which, rows=rows))
            if traces:
                save_json(tpath, dict(method=method, level=level, traces=traces))
            log(f'{method} ({level}, {which}): {len(rows)}/{len(cases)}')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', default='all', help="all, core, baselines, rl, or a comma list")
    p.add_argument('--level', default=None)
    p.add_argument('--cases', choices=['test', 'dev'], default='test')
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)
    level = a.level or config.load()['disturbance']['main_level']
    try:
        for m in method_set(a.methods):
            evaluate(m, level, a.workers, which=a.cases, log=lambda s: print(s, flush=True))
    finally:
        from .parallel import close
        close()


if __name__ == '__main__':
    main()
