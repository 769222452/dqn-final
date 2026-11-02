"""Headroom under uncertainty (src/certified_flight/headroom.py). One result file per scenario and strategy.

    python scripts/headroom_study.py --out experiments/headroom/runs --sigma 0.06 --corr-km 200 \
        --seeds 0 1 2 3 --backs 0 0.01 0.02 0.04 0.06 --jobs 4

Strategies per scenario (density field seed, constant aero biases k_L, k_D):
    clair         clairvoyant full-horizon plan that knows the whole field and the aero biases
    nominal_ol    the nominal plan (k = 1) flown without feedback
    mpc_b<b>_l<lam>  receding-horizon re-planning every --period s, back-off b on the q, n and heat limits, density
                     forecast persistence (lam = 0) or mean reversion with length 1/lam (lam given in 1/km)
Resumable: existing result files are kept.
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

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight import headroom as H  # noqa: E402
from certified_flight.aero import AeroDatabase  # noqa: E402
from certified_flight.build import runtime_config  # noqa: E402
from certified_flight.mission import Mission  # noqa: E402
from certified_flight.serialization import clean  # noqa: E402

POLAR = ROOT/'experiments/headroom/polar_table.npz'


def scenario(sigma, corr_km, seed, aero_sigma):
    rng = np.random.default_rng(10_000+seed)
    k_l, k_d = (float(np.clip(1+aero_sigma*rng.standard_normal(), 1-2*aero_sigma, 1+2*aero_sigma)) for _ in range(2))
    field = H.density_field(sigma, corr_km*1e3, seed)
    return field, k_l, k_d


def world():
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    z = np.load(POLAR)
    return mission, H.PolarSurface(z['knots'], z['cd']), (float(db.M[0]), float(db.M[-1]))


def run(job):
    out, sigma, corr_km, seed, aero_sigma, strategy, period = job
    path = Path(out)/f'sig{sigma:g}_L{corr_km:g}_a{aero_sigma:g}_seed{seed}_{strategy}.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.perf_counter()
    mission, surface, support = world()
    x0 = np.array(mission.data['initial_state']['x_h_V_gamma_s'], float)
    field, k_l, k_d = scenario(sigma, corr_km, seed, aero_sigma)
    truth = H.Truth(mission, surface, support, field, k_l, k_d)
    base = H.Planner.BASE_BACKOFF
    res = dict(sigma=sigma, corr_km=corr_km, seed=seed, aero_sigma=aero_sigma, k_lift=k_l, k_drag=k_d, strategy=strategy,
               period_s=period)
    if strategy == 'clair':
        # 5 s uniform grid (the field varies on ~80 s); the back-off is tightened until the flown plan is feasible
        planner = H.Planner(mission, surface, support, kind='profile', profile=field, split=((240, 1.0),),
                            ipopt=dict(max_iter=3000))
        back, tries = base, []
        for _ in range(4):
            plan = planner.plan(x0, 1200., back=back, k_lift=k_l, k_drag=k_d)
            flown = H.fly(truth, x0, H.open_loop_controller(plan), period=period)
            tries.append(dict(back=back, plan_status=plan['status'], plan_range_km=plan['range_km'],
                              flown_km=flown['range_km'], min_margin=flown['min_margin']))
            if flown['feasible'] and plan['ok']:
                break
            back *= 1+1.1*min(flown['min_margin'], 0.)
        res.update(plan_status=plan['status'], plan_range_km=plan['range_km'], tries=tries, final_back=back)
    elif strategy == 'nominal_ol':
        planner = H.Planner(mission, surface, support)
        plan = planner.plan(x0, 1200., back=base)
        res.update(plan_status=plan['status'], plan_range_km=plan['range_km'])
        flown = H.fly(truth, x0, H.open_loop_controller(plan), period=period)
    else:
        _, b, lam = strategy.split('_')
        b, lam = float(b[1:]), float(lam[1:])
        planner = H.Planner(mission, surface, support)
        controller = H.mpc_controller(planner, back=base*(1-b), lam=lam*1e-3)
        flown = H.fly(truth, x0, controller, period=period)
        res.update(plans=getattr(controller, 'plans', 0), plan_failures=getattr(controller, 'failures', 0))
    log = flown.pop('log')
    res.update(flown, wall_s=time.perf_counter()-started, min_q_ratio_note='margins are normalised: >= 0 feasible',
               log_t_h_V_s_c=log[::4].tolist())
    path.write_text(json.dumps(clean(res))+'\n')
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--sigma', type=float, nargs='+', default=[.06])
    p.add_argument('--corr-km', type=float, nargs='+', default=[200.])
    p.add_argument('--aero-sigma', type=float, default=.05)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--backs', type=float, nargs='+', default=[0., .01, .02, .04, .06])
    p.add_argument('--lams', type=float, nargs='+', default=[0.], help='mean-reversion rate, 1/km x 1e-3 (0 = persistence)')
    p.add_argument('--period', type=float, default=15.)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--skip', nargs='*', default=[], help="strategies to skip, e.g. nominal_ol")
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    todo = []
    for sg in a.sigma:
        for lc in a.corr_km:
            for seed in a.seeds:
                strategies = ['clair', 'nominal_ol']+[f'mpc_b{b:g}_l{lam:g}' for b in a.backs for lam in a.lams]
                for s in strategies:
                    if s not in a.skip:
                        todo.append((a.out, sg, lc, seed, a.aero_sigma, s, a.period))
    # long jobs first
    todo.sort(key=lambda j: j[5].startswith('mpc'), reverse=True)
    with mp.get_context('spawn').Pool(a.jobs) as pool:
        for r in pool.imap_unordered(run, todo):
            print(json.dumps(dict(sigma=r['sigma'], L=r['corr_km'], seed=r['seed'], strategy=r['strategy'],
                                  range_km=round(r['range_km'], 2), feasible=r['feasible'],
                                  min_margin=round(r['min_margin'], 5), wall_s=round(r['wall_s']))), flush=True)


if __name__ == '__main__':
    main()
