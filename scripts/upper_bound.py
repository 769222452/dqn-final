"""Physical range bounds (src/certified_flight/bound.py) for initial-state offsets.

    python scripts/upper_bound.py --out experiments/improve/bounds --states nominal heldout5 development

Writes one JSON per state with the relaxed upper bound (and its 0.1 s re-integration check) and, with
--reference, the achievable fixed-geometry reference.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight.aero import AeroDatabase  # noqa: E402
from certified_flight.bound import FrictionTable, relaxed_bound  # noqa: E402
from certified_flight.build import runtime_config  # noqa: E402
from certified_flight.mission import Mission  # noqa: E402
from certified_flight.serialization import clean  # noqa: E402
from rollout_improvement.evaluation import INITIAL_OFFSETS, test_set_points  # noqa: E402

WARM = ROOT/'experiments/aoa_fix/surrogate_local/flights/final_consistent_c025/steps.jsonl.gz'


def warm_start():
    st = [json.loads(line) for line in gzip.open(WARM, 'rt')]
    t = [s['t_start_s'] for s in st]+[st[-1]['t_end_s']]
    x = [s['pre_context']['x'] for s in st]+[st[-1]['pre_context']['x']]
    return dict(t=np.asarray(t), x=np.asarray(x), t_ctrl=np.asarray(t[:-1]), CL=np.asarray([s['executed_CL_mean'] for s in st]))


def states(names):
    out = []
    held, _ = test_set_points('held-out-3')
    for n in names:
        if n == 'nominal':
            out.append((0., 0.))
        elif n == 'heldout5':
            out += held[1:6]
        elif n == 'heldout':
            out += held[1:]
        elif n == 'development':
            out += [p for p in INITIAL_OFFSETS if p != (0., 0.)]
    return list(dict.fromkeys(out))


def job(args):
    out, (dh, dv), geoms, friction, constant_cl = args
    tag = ('' if not geoms else '_'+'+'.join(geoms))+('_friction' if friction else '')+('_constcl' if constant_cl else '')
    path = Path(out)/f'dh{dh:g}_dv{dv:g}{tag}.json'
    if path.exists():
        return json.loads(path.read_text())
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    x0 = np.array(mission.data['initial_state']['x_h_V_gamma_s'], float)
    x0[0] += dh
    x0[1] += dv
    w = warm_start()
    w['x'] = w['x']+np.array([dh, dv, 0, 0])*np.linspace(1, 0, len(w['x']))[:, None]
    idx = None if not geoms else [db.index[g] for g in geoms]
    ft = FrictionTable(ROOT/friction, db) if friction else None
    res = dict(state=[dh, dv], x0=x0.tolist(), geometries=geoms or 'all', friction=friction, constant_cl=constant_cl,
               relaxed=relaxed_bound(mission, db, x0, warm=w, indices=idx, friction=ft, constant_cl=constant_cl))
    path.write_text(json.dumps(clean(res))+'\n')
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--states', nargs='+', default=['nominal'])
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--friction', help='friction table (scripts/build_friction_table.py), repository-relative')
    p.add_argument('--constant-cl', action='store_true', help='one CL for the whole flight')
    p.add_argument('--geometry-sets', nargs='*', default=['all'],
                   help="'all' and/or '+'-joined geometry ids, e.g. G42 S2_MD8_b38c94c34e")
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    sets = [None if g == 'all' else g.split('+') for g in a.geometry_sets]
    todo = [(a.out, s, g, a.friction, a.constant_cl) for g in sets for s in states(a.states)]
    with mp.get_context('spawn').Pool(a.jobs) as pool:
        for r in pool.imap_unordered(job, todo):
            rb = r['relaxed']
            print(json.dumps(dict(state=r['state'], geometries=r.get('geometries', 'all'), friction=r.get('friction'),
                                  constant_cl=r.get('constant_cl'), status=rb['status'], range_km=rb['range_km'],
                                  verify_km=rb['verification']['range_km'], verify_margin=rb['verification']['min_qhn_margin'],
                                  wall_s=round(rb['wall_s']))), flush=True)


if __name__ == '__main__':
    main()
