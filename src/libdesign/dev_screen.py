"""Development screen on TRAINING scenarios only: do the baselines leave room on the morphing-only targets?

For a library (TARGET_RL_LIBRARY): pc_fair, pc_strong and the clairvoyant reference on T4-T6 for training
scenarios 1100-1105 (L2). Test scenarios are not touched, so the preregistered evaluation stays clean.

    TARGET_RL_LIBRARY=F8D2000B PYTHONPATH=src python -m libdesign.dev_screen --workers 4
"""
import argparse
import multiprocessing as mp
import os

from target_rl import baselines as B, config, region as RG
from target_rl.scenario import scenario

from .paths import OUT, load, save

SEEDS = range(1100, 1106)


def job(a):
    method, seed, tname = a
    T = {t.name: t for t in RG.named_targets()}
    sc = scenario('L2', seed)
    if method == 'reference':
        r = B.reference(sc, T[tname])
    else:
        r = B.PredictorCorrector(method).fly(sc, T[tname]).summary()
    return dict(method=method, seed=seed, target=tname, hit=bool(r['hit']), feasible=bool(r['feasible']),
                dD_km=float(r['dD_km']), dV_mps=float(r['dV_mps']))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    jobs = [(m, s, t) for m in ('pc_fair', 'pc_strong', 'reference') for s in SEEDS for t in ('T4', 'T5', 'T6')]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    lib = config.library_name()
    path = OUT/'dev_screen.json'
    out = load(path) if path.exists() else {}
    out[lib] = rows
    save(path, out)
    for m in ('pc_fair', 'pc_strong', 'reference'):
        rr = [r for r in rows if r['method'] == m]
        print(lib, m, 'hits %d/%d' % (sum(r['hit'] for r in rr), len(rr)),
              'violations %d' % sum(not r['feasible'] for r in rr))


if __name__ == '__main__':
    main()
