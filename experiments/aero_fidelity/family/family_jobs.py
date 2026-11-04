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

Version 4 (protocol amendment 3, declared before any CFD result): version 3 produced 106 audit-VALID members but no
common shock offset delta whose volume ranges at the three design Machs intersect (the valid volume ranges at
delta = +2, +4, +6 miss each other by 28 %, 17 % and 10 % of G42's volume). The requirement that all three members
share one delta is dropped: each design Mach gets the weakest shock offset in {+2, +4, +6} (the audited set) whose
valid a_x window reaches G42's volume, which the version-3 audit gives as MD4: +4, MD6: +2, MD8: +2, with a_x scanned
on a fine grid around the volume match (29 generator calls).

    python experiments/aero_fidelity/family/family_jobs.py --out runs/aero_fidelity/family --version 4

Versions 5 and 6 (optimal-family test, README section 3h): version 5 is a map of design Mach x shock offset x shock-angle
ratio x d1 x a_x (1260 calls); version 6 takes the audited map (--map <family_audit.json of version 5>) and, for every shape
(design Mach, offset, ratio) whose valid a_x neighbours bracket G42's volume, generates three members at the interpolated a_x
and +-0.0025 around it. Both are audited without selection (check mode 'map'); the selection is the declared L/D rule of
experiments/aero_fidelity/optimum/evaluate_optimum.py.
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
# Version 4: shock offset per design Mach and fine a_x scan around the volume match (derived from volumes only).
V4 = {4.: dict(delta=4., ax=tuple(round(1.02+.005*i, 4) for i in range(9))),
      6.: dict(delta=2., ax=tuple(round(1.00+.005*i, 4) for i in range(9))),
      8.: dict(delta=2., ax=tuple(round(.975+.0025*i, 4) for i in range(11)))}
# Optimal-family map (version 5)
# d1 (the leading-edge curve control point, build_geometry_core.m line 71) is a design variable too: G42, whose value the
# earlier families inherited, is a Mach-15 design (geometry_inventory.csv: REG_G42_MD15), and the library's own d1 values are mostly 0.02-0.06.
OPT_A = dict(machs=(4., 5., 6., 7., 8., 9., 10.), deltas=(1., 2., 3., 4., 5., 6.), ratios=(.35, .5), d1s=(0.04, 0.08, FIXED['d1']),
             ax=(.95, .975, 1., 1.025, 1.05))
# Enclosed volume of G42 (data/geometry_subset/geometries/G42, scripts/cfd/make_su2_cases.closed_surface): equal payload volume.
TARGET_VOLUME_M3 = 1.018813633812261


def mach_angle_deg(M):
    return float(np.degrees(np.arcsin(1/M)))


def _job(job_id, M, delta, beta4, ax, **extra):
    return dict(job_id=job_id, Ma_design=M, delta_deg=delta, shock_angles_deg=list(beta4), a_x=ax,
                inputs10=[M, *beta4, FIXED['c1'], FIXED['c2'], FIXED['d1'], FIXED['AOA'], ax], **extra)


def _pair(job_id, M, d, r, ax, d1=None, **extra):
    d1 = FIXED['d1'] if d1 is None else d1
    a = 2*(mach_angle_deg(M)+d)/(1+r)
    job = _job(job_id, M, d, (a, a, r*a, r*a), ax, design='pair', ratio=r, design_key=f'MD{M:g}_d{d:+g}_r{r:g}_p{d1:.4f}',
               target_volume_m3=TARGET_VOLUME_M3, selection='map', d1=d1, **extra)
    job['inputs10'][7] = d1
    return job


def opt_stage_a():
    return [_pair(f'OPTA_MD{M:g}_d{d:+g}_r{r:g}_p{p:.4f}_ax{ax:.3f}', M, d, r, ax, d1=p)
            for M in OPT_A['machs'] for d in OPT_A['deltas'] for r in OPT_A['ratios'] for p in OPT_A['d1s'] for ax in OPT_A['ax']]


def opt_stage_b(map_rows, target=None, step=.0025):
    """Interpolated a_x per shape whose adjacent valid map members bracket the target volume (pure function)."""
    target = TARGET_VOLUME_M3 if target is None else target
    groups = {}
    for r in map_rows:
        groups.setdefault((r['Ma_design'], r['delta_deg'], r['ratio'], r['d1']), []).append(r)
    out = []
    for (M, d, ratio, d1), rows in sorted(groups.items()):
        by_ax = {round(r['a_x'], 6): r for r in rows}
        grid = sorted(by_ax)
        for a0, a1 in zip(grid[:-1], grid[1:]):
            r0, r1 = by_ax[a0], by_ax[a1]
            if not (r0['valid'] and r1['valid']):
                continue
            v0, v1 = r0['volume_m3'], r1['volume_m3']
            if (v0-target)*(v1-target) > 0 or v0 == v1:
                continue
            ax = a0+(target-v0)*(a1-a0)/(v1-v0)
            for k in (-1, 0, 1):
                x = round(ax+k*step, 5)
                out.append(_pair(f'OPTB_MD{M:g}_d{d:+g}_r{ratio:g}_p{d1:.4f}_ax{x:.5f}', M, d, ratio, x, d1=d1, bracket=[a0, a1]))
    return out


def jobs(version=2, map_rows=None):
    if version == 5:
        return opt_stage_a()
    if version == 6:
        if map_rows is None:
            raise ValueError('version 6 needs the audited version-5 map (--map)')
        return opt_stage_b(map_rows)
    out = []
    for M in DESIGN_MACHS:
        mu = mach_angle_deg(M)
        if version == 4:
            d, r = V4[M]['delta'], 0.5
            a = 2*(mu+d)/(1+r)
            for ax in V4[M]['ax']:
                out.append(_job(f'FAM4_MD{M:g}_pair_d{d:+g}_r{r:g}_ax{ax:.4f}', M, d, (a, a, r*a, r*a), ax,
                                design='pair', ratio=r, design_key='pair_d%+g_r%g' % (d, r), target_volume_m3=TARGET_VOLUME_M3,
                                selection='per_mach_closest_volume'))
            continue
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
    ap.add_argument('--version', type=int, default=4, choices=(1, 2, 3, 4, 5, 6))
    ap.add_argument('--map', help='version 6: family_audit.json of the audited version-5 map')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(a.version, json.loads(Path(a.map).read_text())['rows'] if a.map else None)
    (out/'jobs.json').write_text(json.dumps(js, indent=1))
    for k in range(a.slices):
        (out/f'jobs_{k}.json').write_text(json.dumps(js[k::a.slices]))
    print(f'version {a.version}: {len(js)} jobs in {a.slices} slices -> {out}')


if __name__ == '__main__':
    main()
