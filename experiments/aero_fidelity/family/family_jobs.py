"""Job list of the controlled waverider family for the design-point decision experiment.

Only the design Mach and the (uniform, conical) shock angle change between family members; the leading-edge
and planform inputs (c1 = c2 = 1.0331, d1 = 0.116, design AOA = -4 deg, exactly as G42) are fixed. The shock
angle is mu(M_d) + delta with one common delta for all members. a_x (exit deflection coefficient) is scanned
to match the internal volume (selection rule in check_family.py, declared in ../README.md).

    python experiments/aero_fidelity/family/family_jobs.py --out runs/aero_fidelity/family --slices 4
"""
import argparse
import json
from pathlib import Path

import numpy as np

DESIGN_MACHS = (4., 6., 8.)
DELTAS_DEG = (-3., -2., -1., 0., 1., 2., 3., 4.)
AX = (0.97, 0.985, 1.0, 1.015, 1.03)
FIXED = dict(c1=1.0331, c2=1.0331, d1=0.11606537709080691, AOA=-4.)


def mach_angle_deg(M):
    return float(np.degrees(np.arcsin(1/M)))


def jobs():
    out = []
    for M in DESIGN_MACHS:
        for d in DELTAS_DEG:
            beta = mach_angle_deg(M)+d
            for ax in AX:
                jid = f'FAM_MD{M:g}_d{d:+g}_ax{ax:.3f}'
                out.append(dict(job_id=jid, Ma_design=M, delta_deg=d, shock_angle_deg=beta, a_x=ax,
                                inputs10=[M, beta, beta, beta, beta, FIXED['c1'], FIXED['c2'], FIXED['d1'],
                                          FIXED['AOA'], ax]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slices', type=int, default=4, help='parallel MATLAB processes')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = jobs()
    (out/'jobs.json').write_text(json.dumps(js, indent=1))
    for k in range(a.slices):
        (out/f'jobs_{k}.json').write_text(json.dumps(js[k::a.slices]))
    print(f'{len(js)} jobs in {a.slices} slices -> {out}')


if __name__ == '__main__':
    main()
