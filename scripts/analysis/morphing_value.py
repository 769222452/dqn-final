"""How much can geometry selection (morphing) gain, with and without skin friction?

    python scripts/analysis/morphing_value.py --out experiments/aero_fidelity/morphing_value.json

Two benchmarks per flight Mach (snapshot grid 4–8.25):
- the snapshot's own benchmark: equal-lift CD at fixed target CL values, i.e. the best geometry per
  (Mach, CL) against the best single geometry over all conditions (ideal, zero switching cost);
- free CL: every geometry at its own L/D-maximising CL, in equilibrium glide. Lift equals weight minus
  centrifugal relief, and the altitude is the one where q = W_eff/(S·CL); friction is taken at that
  altitude.
Each benchmark is computed with pressure drag only (snapshot) and with pressure + skin friction
(data/aero_friction/friction_table.npz).
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight.aero import AeroDatabase  # noqa: E402
from certified_flight.friction import SREF  # noqa: E402
from certified_flight.mission import Mission  # noqa: E402

MASS = 1500.
R_E = 6.371e6
G0 = 9.80665
CL_TARGETS = (0.026150265744858465, 0.02850948319855705, 0.0332974123812019, 0.0398, 0.0512894735198207,
              0.05350025264833205)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out')
    ap.add_argument('--machs', nargs='*', type=float, default=[4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.])
    a = ap.parse_args(argv)
    db = AeroDatabase(ROOT/'data/geometry_subset')
    atm = Mission(ROOT, 'configs/mission.yaml').model(None).atm
    fz = np.load(ROOT/'data/aero_friction/friction_table.npz')
    inv = {r['geometry_id']: r for r in csv.DictReader(open(ROOT/'data/geometry_subset/candidate_inventory.csv',
                                                             encoding='utf-8-sig'))}
    md = np.array([float(inv[g]['Ma_design']) for g in db.ids])
    fric = [RegularGridInterpolator((fz['mach_grid'], fz['alpha_rad_grid'], fz['altitude_m_grid']), fz['CD_friction'][i],
                                    bounds_error=False, fill_value=None) for i in range(len(db.ids))]
    alph = np.deg2rad(np.linspace(-1, 12, 131))

    def altitude(M, cl):
        def f(h):
            a_ = float(atm.speed_of_sound(h))
            V = M*a_
            W = MASS*(G0-V*V/(R_E+h))
            return .5*float(atm.density(h))*V*V*SREF*cl-W
        try:
            return brentq(f, 5e3, 80e3)
        except ValueError:
            return np.nan

    out = dict(machs=a.machs, free_cl={}, fixed_cl={})
    for with_f in (False, True):
        key = 'pressure+friction' if with_f else 'pressure'
        LD = np.full((len(a.machs), len(db.ids)), np.nan)
        CLs = np.full_like(LD, np.nan)
        fixed = np.full((len(a.machs), len(CL_TARGETS), len(db.ids)), np.inf)
        for m, M in enumerate(a.machs):
            y = PchipInterpolator(db.A, db.curves(M), axis=1)(alph)          # [geom, alpha, 2]
            for i in range(len(db.ids)):
                cl, cdp = y[i, :, 0], y[i, :, 1]
                ok = np.flatnonzero((cl > .01) & np.r_[True, np.diff(cl) > 0])
                best = -1.
                hs = np.array([altitude(M, c) for c in cl[ok]])
                cdf = np.array([float(fric[i]([M, alph[j], h])[0]) if with_f and np.isfinite(h) else 0.
                                for j, h in zip(ok, hs)])
                ld = np.where(np.isfinite(hs), cl[ok]/(cdp[ok]+cdf), -1)
                k = int(np.argmax(ld))
                LD[m, i], CLs[m, i] = ld[k], cl[ok][k]
                for t, c in enumerate(CL_TARGETS):
                    if cl[ok].min() <= c <= cl[ok].max():
                        aa = float(np.interp(c, cl[ok], alph[ok]))
                        h = altitude(M, c)
                        cd = float(np.interp(aa, alph, cdp))+(float(fric[i]([M, aa, h])[0]) if with_f else 0.)
                        fixed[m, t, i] = cd
                del best
        kf = int(np.nanargmax(np.nanmean(LD, axis=0)))
        win = [db.ids[int(np.nanargmax(LD[m]))] for m in range(len(a.machs))]
        gain = float(np.mean(np.nanmax(LD, axis=1))/np.mean(LD[:, kf])-1)
        out['free_cl'][key] = dict(best_fixed=db.ids[kf], best_fixed_mean_LD=float(np.mean(LD[:, kf])),
                                   ideal_mean_LD=float(np.mean(np.nanmax(LD, axis=1))), ideal_gain_pct=100*gain,
                                   winners=win, winner_design_mach=[float(md[db.index[w]]) for w in win],
                                   best_LD_per_mach=np.nanmax(LD, axis=1).round(3).tolist(),
                                   best_CL_per_mach=[float(CLs[m, db.index[w]]) for m, w in enumerate(win)],
                                   corr_design_mach_LD=[float(np.corrcoef(md[np.isfinite(LD[m])], LD[m][np.isfinite(LD[m])])[0, 1])
                                                        for m in range(len(a.machs))])
        feas = np.all(np.isfinite(fixed), axis=(0, 1))
        mean_cd = np.where(feas, fixed.mean(axis=(0, 1)), np.inf)
        kb = int(np.argmin(mean_cd))
        ideal = float(np.min(fixed, axis=2).mean())
        wins = np.argmin(fixed, axis=2).ravel()
        out['fixed_cl'][key] = dict(best_fixed=db.ids[kb], ideal_gain_pct=100*(1-ideal/mean_cd[kb]),
                                    winner_count=int(len(set(wins.tolist()))),
                                    winners=sorted({db.ids[w] for w in wins.tolist()}))
        f = out['free_cl'][key]
        print(f"[{key}] free CL: best fixed {f['best_fixed']} mean L/D {f['best_fixed_mean_LD']:.3f}; ideal morphing gain "
              f"{f['ideal_gain_pct']:.2f} %; winners by Mach {[w[:14] for w in win]}")
        print(f"[{key}] fixed-CL benchmark (snapshot definition): ideal gain {out['fixed_cl'][key]['ideal_gain_pct']:.2f} % "
              f"over {out['fixed_cl'][key]['best_fixed']}, {out['fixed_cl'][key]['winner_count']} distinct winners")
        print(f"[{key}] best L/D per Mach {f['best_LD_per_mach']} at CL {np.round(f['best_CL_per_mach'], 4).tolist()}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
