"""Process pool for flights and network fits (spawned workers, one BLAS thread each)."""
import multiprocessing as mp
import os
import traceback

_POOL = {}


def _init():
    for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[k] = '1'
    from . import flight
    flight.world(True)
    flight.world(False)


def _run(job):
    kind, kw = job
    try:
        if kind == 'constant':
            from .baselines import fly_constant
            sc, tg = kw['case']
            return fly_constant(sc, tg, kw['action'], kw.get('morph', True), kg=kw.get('kg')).summary()
        if kind == 'method':
            from .evaluate import fly_method
            return fly_method(kw['method'], kw['case'], kw.get('spec'), kw.get('trace', False))
        if kind == 'branch':
            from .train import branch_job
            return branch_job(kw)
        if kind == 'candidate':
            from .train import candidate_job
            return candidate_job(kw)
        if kind == 'fit_member':
            from .ensemble import fit_member_job
            return fit_member_job(kw)
        raise ValueError(kind)
    except BaseException:
        return dict(error=traceback.format_exc())


def get_pool(workers):
    if workers not in _POOL:
        _POOL[workers] = mp.get_context('spawn').Pool(workers, initializer=_init)
    return _POOL[workers]


def pool_map(jobs, workers, chunksize=1):
    if workers <= 1:
        _init()
        res = [_run(j) for j in jobs]
    else:
        res = get_pool(workers).map(_run, jobs, chunksize=chunksize)
    errs = [r['error'] for r in res if isinstance(r, dict) and 'error' in r]
    if errs:
        raise RuntimeError(errs[0])
    return res


def pool_imap(jobs, workers):
    if workers <= 1:
        _init()
        for j in jobs:
            yield _run(j)
        return
    yield from get_pool(workers).imap(_run, jobs, chunksize=1)


def close():
    for p in _POOL.values():
        p.close()
        p.join()
    _POOL.clear()
