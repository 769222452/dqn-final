"""Stage A of the design-point library: the volume map (job list only; nothing is run here).

Recipe (README.md in this folder): one member per design Mach M_d = 4.0, 4.5, ..., 8.0 with the planform inputs of G42
(c1 = c2 = 1.0331, d1 = 0.11607, design AOA = -4 deg), a lower/upper shock-angle ratio r = 0.5 (the only ratio that gave
audit-VALID members, family versions 2 to 4), and the volume of G42 (+-3 %). Per M_d the free quantities are the shock
offset delta (mean shock angle = asin(1/M_d) + delta) and a_x. Stage A generates a coarse map of both, the audit and the
closed-surface volume of every member are computed with experiments/aero_fidelity/family/check_family.py, and stage B
(not yet written; it needs the map) interpolates delta and a_x per M_d to hit the volume and generates the final member.

    python experiments/aero_fidelity/library/library_jobs.py --out runs/aero_library/map
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'family'))

import family_jobs as fj  # noqa: E402

DESIGN_MACHS = tuple(4.+.5*i for i in range(9))
DELTAS = (1., 2., 3., 4., 5., 6.)
AX = (.95, .975, 1., 1.025, 1.05)
RATIO = .5


def jobs():
    out = []
    for M in DESIGN_MACHS:
        mu = fj.mach_angle_deg(M)
        for d in DELTAS:
            a = 2*(mu+d)/(1+RATIO)
            for ax in AX:
                out.append(fj._job(f'LIBA_MD{M:g}_pair_d{d:+g}_r{RATIO:g}_ax{ax:.3f}', M, d, (a, a, RATIO*a, RATIO*a), ax,
                                   design='pair', ratio=RATIO, design_key='pair_d%+g_r%g' % (d, RATIO),
                                   target_volume_m3=fj.TARGET_VOLUME_M3, selection='per_mach_closest_volume'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slices', type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = jobs()
    (out/'jobs.json').write_text(json.dumps(js, indent=1))
    for k in range(a.slices):
        (out/f'jobs_{k}.json').write_text(json.dumps(js[k::a.slices]))
    print(f'stage A: {len(js)} jobs in {a.slices} slices -> {out}')


if __name__ == '__main__':
    main()
