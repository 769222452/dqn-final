"""Evaluate libraries on the target-arrival task: reachable (range, final speed) region with and without morphing.

Registers the libraries through TARGET_RL_EXTRA_LIBRARIES and runs target_rl.region for each (nominal state,
three-phase plan grid). The fixed geometry of a design library is its best fixed shape on the 120 equal-lift
conditions (libdesign.metrics.static_gain), the analogue of F8D_01351 for F8D2000.

    python -m libdesign.task_eval --libraries f8d_x3,ladder --workers 4
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import yaml

from . import metrics as Mx
from .paths import OUT, ROOT, save


def register(names):
    lib = {}
    for n in names:
        p = OUT/'libraries'/f'{n}.npz'
        z = np.load(p, allow_pickle=False)
        cd = Mx.equal_lift(z['CL'], z['CD_pressure'])
        best = Mx.static_gain(cd)['best_fixed']
        lib[f'design_{n}'] = dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  fixed_geometry=str(z['geometry_ids'][best]))
    path = OUT/'extra_libraries.yaml'
    path.write_text(yaml.safe_dump(lib))
    return path, lib


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--libraries', required=True)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    names = a.libraries.split(',')
    path, lib = register(names)
    from .paths import load
    out = load(OUT/'task_eval.json') if (OUT/'task_eval.json').exists() else {}
    for name in lib:
        env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), TARGET_RL_EXTRA_LIBRARIES=str(path), TARGET_RL_LIBRARY=name)
        subprocess.run([sys.executable, '-m', 'target_rl.region', '--workers', str(a.workers)], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        r = json.loads((ROOT/'runs'/name/'region.json').read_text())
        out[name] = {k: r[k] for k in ('targets', 'notes', 'area_ratio', 'feasible', 'fixed_range')}
        out[name]['fixed_geometry'] = lib[name]['fixed_geometry']
        print(name, 'area ratio %.2f' % r['area_ratio'], [round(t.get('band_mps', 0)) for t in r['targets'][3:]], flush=True)
    save(OUT/'task_eval.json', out)


if __name__ == '__main__':
    main()
