"""Job list of the controlled waverider family for the design-point decision experiment.

Version 1 (first run, result in ../results_family_v1/): a1 = a2 = b1 = b2 = asin(1/M_d) + delta. All 105
members that the generator produced fail the project's frozen native-mesh audit (upper and lower skins
intersect at the tail), so version 1 has no admissible member.

Version 2 (protocol amendment 1, declared before any CFD result): the shock is described by the mean angle
offset delta and the ratio r = b/a of the lower to the upper pair of shock angles,

    a1 = a2 = a, b1 = b2 = r a, with mean(a, r a) = asin(1/M_d) + delta,     r in {0.50, 0.65, 0.80},

plus the pattern of the project's validated reference geometry G42 (a1:a2:b1:b2 = 5.634:7.797:3.698:3.012)
scaled to the same mean angle. Everything else is as in version 1: leading-edge and planform inputs of G42
(c1 = c2 = 1.0331, d1 = 0.11607, design AOA = -4 deg), a_x in {0.97, 1.00, 1.03} to match the volume, design
Mach in {4, 6, 8}. Members are accepted only if the frozen audit classifies them VALID.

Version 3 (protocol amendment 2, declared before any CFD result): the volume knob is refined. In version 2 only
`pair_d+2_r0.5`, `pair_d+4_r0.5` and `pair_d+6_r0.5` had a VALID member at every design Mach, and none was
volume-matched because a_x was scanned in three steps (0.97, 1.00, 1.03) while the enclosed volume changes by
about 35 % over that range. Version 3 re-generates exactly those three shock parametrisations with
a_x = 0.90, 0.91, ..., 1.10 (21 values) and matches the volume to the enclosed volume of G42 (TARGET_VOLUME_M3).

    python experiments/aero_fidelity/family/family_jobs.py --out runs/aero_fidelity/family --version 3
"""
import argparse
import json
from pathlib import Path

import numpy as np

DESIGN_MACHS = (4., 6., 8.)
FIXED = dict(c1=1.0331, c2=1.0331, d1=0.11606537709080691, AOA=-4.)
G42_PATTERN = (5.633749605922896, 7.796675937828777, 3.698465133623518, 3.0116105313696075)

V1 = dict(deltas=(-3., -2., -1., 0., 1., 2., 3., 4.), ax=(0.97, 0.985, 1.0, 1.015, 1.03))
V2 = dict(deltas=(-2., 0., 2., 4., 6.), ratios=(0.5, 0.65, 0.8), ax=(0.97, 1.0, 1.03))
# Version 3: the design keys with a VALID member at all three design Machs in the version-2 audit
# (results_family_v2/family_audit.json; checked by tests/test_aero_fidelity.py), and the refined a_x scan.
V3 = dict(deltas=(2., 4., 6.), ratio=0.5, ax=tuple(round(0.90+0.01*i, 2) for i in range(21)))
# Enclosed volume of G42 (data/geometry_subset/geometries/G42, scripts/cfd/make_su2_cases.closed_surface): equal payload volume.
TARGET_VOLUME_M3 = 1.018813633812261


def mach_angle_deg(M):
    return float(np.degrees(np.arcsin(1/M)))


def _job(job_id, M, delta, beta4, ax, **extra):
    return dict(job_id=job_id, Ma_design=M, delta_deg=delta, shock_angles_deg=list(beta4), a_x=ax,
                inputs10=[M, *beta4, FIXED['c1'], FIXED['c2'], FIXED['d1'], FIXED['AOA'], ax], **extra)


def jobs(version=2):
    out = []
    for M in DESIGN_MACHS:
        mu = mach_angle_deg(M)
        if version == 3:
            r = V3['ratio']
            for d in V3['deltas']:
                a = 2*(mu+d)/(1+r)
                for ax in V3['ax']:
                    out.append(_job(f'FAM3_MD{M:g}_pair_d{d:+g}_r{r:g}_ax{ax:.2f}', M, d, (a, a, r*a, r*a), ax,
                                    design='pair', ratio=r, design_key='pair_d%+g_r%g' % (d, r), target_volume_m3=TARGET_VOLUME_M3))
            continue
        if version == 1:
            for d in V1['deltas']:
                b = mu+d
                for ax in V1['ax']:
                    out.append(_job(f'FAM_MD{M:g}_d{d:+g}_ax{ax:.3f}', M, d, (b, b, b, b), ax, design='conical', design_key='d%+g' % d))
            continue
        for d in V2['deltas']:
            mean = mu+d
            for r in V2['ratios']:
                a = 2*mean/(1+r)
                for ax in V2['ax']:
                    out.append(_job(f'FAM2_MD{M:g}_pair_d{d:+g}_r{r:g}_ax{ax:.2f}', M, d, (a, a, r*a, r*a), ax,
                                    design='pair', ratio=r, design_key='pair_d%+g_r%g' % (d, r)))
            s = mean/np.mean(G42_PATTERN)
            for ax in V2['ax']:
                out.append(_job(f'FAM2_MD{M:g}_g42_d{d:+g}_ax{ax:.2f}', M, d, tuple(s*p for p in G42_PATTERN), ax,
                                design='g42', design_key='g42_d%+g' % d))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slices', type=int, default=4, help='parallel MATLAB processes')
    ap.add_argument('--version', type=int, default=3, choices=(1, 2, 3))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(a.version)
    (out/'jobs.json').write_text(json.dumps(js, indent=1))
    for k in range(a.slices):
        (out/f'jobs_{k}.json').write_text(json.dumps(js[k::a.slices]))
    print(f'version {a.version}: {len(js)} jobs in {a.slices} slices -> {out}')


if __name__ == '__main__':
    main()
