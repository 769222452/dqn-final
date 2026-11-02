"""Optimal-family test (experiments/aero_fidelity/README.md section 3h): which equal-volume shape is best at each flight Mach?

    python experiments/aero_fidelity/optimum/evaluate_optimum.py runs/aero_opt/b --audit runs/aero_opt/b_selected/family_audit.json \
        --out experiments/aero_fidelity/optimum/results

Candidates: audited members of the version-6 batch (family_jobs.py) that are VALID and within +-3 % of G42's volume; per shape
(design Mach, shock offset, ratio) the member closest to the volume. Aerodynamics: tangent-wedge pressure + friction
(src/certified_flight/friction.py; validated against Euler for this class in section 3g), flight Mach F in {4, 6, 8} at q = 50 kPa,
alpha 2..8 deg; quadratic CD(CL) through the alphas; L/D at CL 0.03, 0.04, 0.05.

Declared rule (fixed before any member of the batch exists):
  eligible   a candidate with all three CL targets inside its fitted range at all three flight Machs
  LD(m, F)   mean over the three CL targets
  w(F)       argmax_m LD(m, F); its design Mach is Md*(F)
  P1         Md*(4) < Md*(8)
  P2         per CL target c: s*_c = argmax_m mean_F LD(m, F, c); G_c = mean_F [max_m LD(m, F, c) / LD(s*_c, F, c)] - 1;
             P2 holds if G_c >= 0.03 for at least two of the three targets
  premise    holds in the optimal-family sense iff P1 and P2
  Euler set  {w(4), w(6), w(8), s*_0.04}: written to --out/confirm for an Euler check (make_su2_cases.py --base-marker, alpha
             2, 4, 6, 8); the wedge result counts as confirmed if Euler gives the same winner w(F) among this set at every F.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'/'cfd'))

from certified_flight import friction as F  # noqa: E402

FLIGHT_MACHS = (4., 6., 8.)
ALPHAS_DEG = (2., 3., 4., 5., 6., 7., 8.)
CL_TARGETS = (.03, .04, .05)
TOL = .03
GAIN = .03


def candidates(rows, target):
    ok = [r for r in rows if r['valid'] and abs(r['volume_m3']/target-1) <= TOL]
    best = {}
    for r in ok:
        k = (r['Ma_design'], r['delta_deg'], r['ratio'])
        if k not in best or abs(r['volume_m3']/target-1) < abs(best[k]['volume_m3']/target-1):
            best[k] = r
    return [best[k] for k in sorted(best)]


def polar(mat, mission):
    """{F: {CL target: L/D}} for one geometry file (targets outside the fitted CL range are left out)."""
    import make_su2_cases as cfd
    from collect_su2 import fit_equal_lift
    p = F.load_panels(mat)
    out = {}
    for M in FLIGHT_MACHS:
        fs = cfd.freestream(mission, M)
        cl, cd = [], []
        for a in ALPHAS_DEG:
            al = np.deg2rad(a)
            cw, dw = F.pressure_coefficients(p, M, al)
            cf, df = F.friction_coefficients(p, M, al, fs['T_K'], fs['rho_kgm3'])
            cl.append(cw+cf)
            cd.append(dw+df)
        out[M] = {t: t/v for t, v in fit_equal_lift(cl, cd, CL_TARGETS).items()}
    return out


def criteria(ld, design_mach):
    """ld[member][F][c] -> the declared rule (pure function). design_mach[member] -> Md."""
    eligible = [m for m in ld if all(len(ld[m].get(Fm, {})) == len(CL_TARGETS) for Fm in FLIGHT_MACHS)]
    if not eligible:
        return dict(status='NO_ELIGIBLE_CANDIDATE', premise_holds=False)
    mean = {m: {Fm: float(np.mean([ld[m][Fm][c] for c in CL_TARGETS])) for Fm in FLIGHT_MACHS} for m in eligible}
    winners = {Fm: max(eligible, key=lambda m: (mean[m][Fm], m)) for Fm in FLIGHT_MACHS}
    p1 = design_mach[winners[4.]] < design_mach[winners[8.]]
    gains, singles = {}, {}
    for c in CL_TARGETS:
        s = max(eligible, key=lambda m: (np.mean([ld[m][Fm][c] for Fm in FLIGHT_MACHS]), m))
        singles[c] = s
        gains[c] = float(np.mean([max(ld[m][Fm][c] for m in eligible)/ld[s][Fm][c] for Fm in FLIGHT_MACHS])-1)
    p2 = sum(g >= GAIN for g in gains.values()) >= 2
    curve = {}
    for Fm in FLIGHT_MACHS:
        curve[str(Fm)] = {str(Md): max(mean[m][Fm] for m in eligible if design_mach[m] == Md)
                          for Md in sorted({design_mach[m] for m in eligible})}
    confirm = sorted({*winners.values(), singles[.04]})
    return dict(status='EVALUATED', eligible=len(eligible), winners={str(k): v for k, v in winners.items()},
                winner_design_mach={str(k): design_mach[v] for k, v in winners.items()}, P1=bool(p1),
                switching_gain={str(c): g for c, g in gains.items()}, best_single={str(c): s for c, s in singles.items()},
                P2=bool(p2), premise_holds=bool(p1 and p2), best_LD_by_design_mach=curve, euler_confirmation_set=confirm)


def confirm(cfd_root, decision):
    """Euler + friction winners among the confirmation set vs the wedge winners (declared check)."""
    from collect_su2 import equal_lift_table, load_cases
    cases = load_cases(cfd_root)
    rows = equal_lift_table(cases, 'medium')
    ids = dict(zip(decision['euler_confirmation_ids'], decision['euler_confirmation_set']))
    out = {}
    for model in ('euler', 'wedge'):
        ld = {}
        for r in rows:
            if r['model'] == model:
                ld.setdefault(ids[r['geometry']], {}).setdefault(r['mach'], {})[r['CL_target']] = r['LD']
        mean = {m: {Fm: float(np.mean(list(v[Fm].values()))) for Fm in FLIGHT_MACHS if Fm in v} for m, v in ld.items()}
        out[model] = {str(Fm): max((m for m in mean if Fm in mean[m]), key=lambda m: mean[m][Fm]) for Fm in FLIGHT_MACHS}
        out[model+'_mean_LD'] = mean
    out['wedge_winners_declared'] = decision['winners']
    out['confirmed'] = all(out['euler'][k] == decision['winners'][k] for k in decision['winners'])
    out['cases'] = len(cases)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family', help='folder of the version-6 batch (<job_id>/native_geometry.mat)')
    ap.add_argument('--audit', required=True, help='family_audit.json of the version-6 batch (check mode map)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--confirm-cfd', help='after the Euler run of --out/confirm: its cfd folder; evaluates the declared check only')
    a = ap.parse_args(argv)
    if a.confirm_cfd:
        decision = json.loads((Path(a.out)/'optimum_decision.json').read_text())
        res = confirm(a.confirm_cfd, decision)
        (Path(a.out)/'euler_confirmation.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if not k.endswith('_mean_LD')}, indent=1))
        return
    from certified_flight.mission import Mission
    import family_jobs as fj
    mission = Mission(ROOT, 'configs/mission.yaml')
    rows = json.loads(Path(a.audit).read_text())['rows']
    cand = candidates(rows, fj.TARGET_VOLUME_M3)
    ld, md, table = {}, {}, []
    for r in cand:
        res = polar(Path(a.family)/r['job_id']/'native_geometry.mat', mission)
        ld[r['job_id']] = res
        md[r['job_id']] = r['Ma_design']
        for Fm, v in res.items():
            for c, x in v.items():
                table.append(dict(job_id=r['job_id'], Ma_design=r['Ma_design'], delta_deg=r['delta_deg'], ratio=r['ratio'],
                                  a_x=r['a_x'], volume_m3=r['volume_m3'], flight_mach=Fm, CL=c, LD=x))
    result = dict(rule=__doc__.split('Declared rule')[1].strip(), candidates=len(cand), **criteria(ld, md))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'optimum_table.json').write_text(json.dumps(table, indent=1))
    if result.get('euler_confirmation_set'):
        ids = []
        for jid in result['euler_confirmation_set']:
            gid = 'OPT_'+jid.split('OPTB_')[1].replace('.', 'p')
            (out/'confirm'/gid).mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(a.family)/jid/'native_geometry.mat', out/'confirm'/gid/'native_geometry.mat')
            ids.append(gid)
        (out/'confirm'/'select.json').write_text(json.dumps(dict(cfd_geometries=ids, source='evaluate_optimum.py'), indent=1))
        result['euler_confirmation_ids'] = ids
    (out/'optimum_decision.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'rule'}, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'experiments'/'aero_fidelity'/'family'))
    main()
