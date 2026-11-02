"""Collect the SU2 Euler cases of make_su2_cases.py and apply the declared decision criteria.

    python scripts/cfd/collect_su2.py runs/cfd --out experiments/aero_fidelity/results/decision.json

For every case (geometry, Mach, alpha): Euler CL and pressure CD (history.csv, last iteration), the
tangent-wedge CL and CD of the same geometry file (src/certified_flight/friction.py, which reproduces the
project's P2 snapshot), and skin friction of the same geometry at the case free stream. Viscous drag is added
identically to both inviscid models, so differences between them come from the pressure field only.

Equal-lift comparison: for each (geometry, Mach, model) a quadratic CD(CL) is fitted through the alphas and
evaluated at the target lift coefficients; the total L/D = CL/(CD_pressure + CD_friction) is ranked across
geometries.

Declared decision criterion (experiments/aero_fidelity/README.md, fixed before any result):
  PREMISE_HOLDS if, for at least two of the three CL targets, the Euler+friction L/D is highest for the
  family member with the lowest design Mach at the lowest flight Mach AND for the member with the highest
  design Mach at the highest flight Mach, each leading the worst family member by at least 3 %.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight import friction as F  # noqa: E402

CL_TARGETS = (0.03, 0.04, 0.05)
MARGIN = 0.03


def last_history(path):
    rows = list(csv.reader(open(path, newline='')))
    head = [h.strip().strip('"') for h in rows[0]]
    d = dict(zip(head, [float(x) for x in rows[-1]]))
    cl = d.get('CL', d.get('"CL"'))
    cd = d.get('CD', d.get('"CD"'))
    if cl is None or cd is None:
        raise KeyError(f'no CL/CD in {path}: {head}')
    return cl, cd, int(d.get('Inner_Iter', len(rows)-1)), d.get('rms[Rho]', d.get('"rms[Rho]"'))


def design_mach(geometry_id):
    """Design Mach from the family naming FAM_MD<Mach>[...]; None for other ids."""
    if geometry_id.startswith('FAM_MD'):
        return float(geometry_id[6:].split('_')[0])
    return None


def fit_equal_lift(cl, cd, targets):
    """Quadratic CD(CL) through the cases; values at targets inside (or within 2 % of) the CL range."""
    cl, cd = np.asarray(cl, float), np.asarray(cd, float)
    if len(cl) < 3:
        return {}
    co = np.polyfit(cl, cd, 2)
    lo, hi = cl.min(), cl.max()
    return {t: float(np.polyval(co, t)) for t in targets if lo*.98 <= t <= hi*1.02}


def load_cases(root):
    cases = []
    for meta_path in sorted(Path(root).glob('*/*/M*_a*/case.json')):
        meta = json.loads(meta_path.read_text())
        hist = meta_path.parent/'history.csv'
        if not hist.exists():
            continue
        cl_e, cd_e, it, rms = last_history(hist)
        mat = Path(meta['geometry_mat'])
        p = F.load_panels(mat)
        al = np.deg2rad(meta['alpha_deg'])
        fs = meta['freestream']
        cl_w, cd_w = F.pressure_coefficients(p, meta['mach'], al)
        clf, cdf = F.friction_coefficients(p, meta['mach'], al, fs['T_K'], fs['rho_kgm3'])
        cases.append(dict(geometry=meta['geometry'], level=meta['level'], mach=meta['mach'], alpha_deg=meta['alpha_deg'],
                          iterations=it, rms_rho=rms, euler_CL=cl_e, euler_CD=cd_e, wedge_CL=cl_w, wedge_CD=cd_w,
                          friction_CL=clf, friction_CD=cdf))
    return cases


def equal_lift_table(cases, level):
    rows = []
    for g in sorted({c['geometry'] for c in cases if c['level'] == level}):
        for M in sorted({c['mach'] for c in cases if c['geometry'] == g and c['level'] == level}):
            sel = sorted([c for c in cases if c['geometry'] == g and c['mach'] == M and c['level'] == level],
                         key=lambda c: c['alpha_deg'])
            for model in ('euler', 'wedge'):
                cl = [c[f'{model}_CL']+c['friction_CL'] for c in sel]
                cd = [c[f'{model}_CD']+c['friction_CD'] for c in sel]
                for t, v in fit_equal_lift(cl, cd, CL_TARGETS).items():
                    rows.append(dict(geometry=g, mach=M, model=model, CL_target=t, CD_total=v, LD=t/v))
    return rows


def decide(rows):
    """Apply the declared criterion to the Euler rows (see module docstring)."""
    out = dict(per_condition={}, premise_holds=None)
    euler = [r for r in rows if r['model'] == 'euler']
    machs = sorted({r['mach'] for r in euler})
    if len(machs) < 2:
        out['reason'] = 'fewer than two flight Mach numbers'
        return out
    geoms = sorted({r['geometry'] for r in euler}, key=lambda g: design_mach(g) or 0)
    lo_g, hi_g = geoms[0], geoms[-1]
    ok_targets = 0
    for t in CL_TARGETS:
        res = {}
        for M, g_expect in ((machs[0], lo_g), (machs[-1], hi_g)):
            sel = {r['geometry']: r['LD'] for r in euler if r['mach'] == M and abs(r['CL_target']-t) < 1e-12}
            if len(sel) < len(geoms):
                res[M] = None
                continue
            best = max(sel, key=sel.get)
            worst = min(sel, key=sel.get)
            res[M] = dict(best=best, expected=g_expect, best_is_expected=best == g_expect,
                          lead_over_worst=sel[best]/sel[worst]-1, LD=sel)
        met = all(v is not None and v['best_is_expected'] and v['lead_over_worst'] >= MARGIN for v in res.values())
        out['per_condition'][str(t)] = dict(met=met, at_flight_mach={str(k): v for k, v in res.items()})
        ok_targets += met
    out['targets_met'] = ok_targets
    out['premise_holds'] = ok_targets >= 2
    return out


def switching_gain(rows, model):
    """Ideal per-condition switching versus the best single family member (mean over conditions)."""
    sel = [r for r in rows if r['model'] == model]
    conds = sorted({(r['mach'], r['CL_target']) for r in sel})
    geoms = sorted({r['geometry'] for r in sel})
    table = {g: np.array([next((r['LD'] for r in sel if r['geometry'] == g and (r['mach'], r['CL_target']) == c), np.nan)
                          for c in conds]) for g in geoms}
    ok = [g for g in geoms if np.isfinite(table[g]).all()]
    if not ok:
        return None
    best_fixed = max(ok, key=lambda g: table[g].mean())
    ideal = np.max([table[g] for g in ok], axis=0).mean()
    return dict(best_fixed=best_fixed, gain_pct=100*(ideal/table[best_fixed].mean()-1),
                winners={f'M{c[0]:g}_CL{c[1]:g}': max(ok, key=lambda g: table[g][i]) for i, c in enumerate(conds)})


def grid_convergence(cases):
    """Euler CL/CD change between mesh levels for the (geometry, Mach, alpha) cases that exist at several levels."""
    order = ['coarse', 'medium', 'fine']
    out = []
    keys = sorted({(c['geometry'], c['mach'], c['alpha_deg']) for c in cases})
    for k in keys:
        by = {c['level']: c for c in cases if (c['geometry'], c['mach'], c['alpha_deg']) == k}
        lv = [l for l in order if l in by]
        if len(lv) >= 2:
            a, b = by[lv[-2]], by[lv[-1]]
            out.append(dict(geometry=k[0], mach=k[1], alpha_deg=k[2], levels=lv[-2:],
                            rel_change_CL=(b['euler_CL']-a['euler_CL'])/max(abs(b['euler_CL']), 1e-12),
                            rel_change_CD=(b['euler_CD']-a['euler_CD'])/max(abs(b['euler_CD']), 1e-12)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--out')
    ap.add_argument('--level', default='medium', help='mesh level used for the decision (default medium)')
    a = ap.parse_args(argv)
    cases = load_cases(a.root)
    rows = equal_lift_table(cases, a.level)
    report = dict(level=a.level, cases=len(cases), decision=decide(rows),
                  switching_gain_euler=switching_gain(rows, 'euler'), switching_gain_wedge=switching_gain(rows, 'wedge'),
                  grid_convergence=grid_convergence(cases), equal_lift=rows, raw_cases=cases)
    print(f'{len(cases)} converged-history cases; decision level {a.level}')
    print('| geometry | M | alpha | Euler CL | wedge CL | Euler CDp | wedge CDp | CDf |')
    print('|---|---|---|---|---|---|---|---|')
    for c in cases:
        if c['level'] == a.level:
            print(f"| {c['geometry']} | {c['mach']:g} | {c['alpha_deg']:g} | {c['euler_CL']:.4f} | {c['wedge_CL']:.4f} | "
                  f"{c['euler_CD']:.5f} | {c['wedge_CD']:.5f} | {c['friction_CD']:.5f} |")
    d = report['decision']
    print('premise holds:', d['premise_holds'], '| CL targets met:', d.get('targets_met'))
    for name in ('euler', 'wedge'):
        g = report[f'switching_gain_{name}']
        print(f'ideal switching gain ({name}):', None if g is None else f"{g['gain_pct']:.2f} % over {g['best_fixed']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1, default=float)+'\n')


if __name__ == '__main__':
    main()
