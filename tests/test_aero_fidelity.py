"""Skin-friction module and CFD geometry export (experiments/aero_fidelity)."""
import csv
import sys

import numpy as np
import pytest

from certified_flight import friction as F
from certified_flight.aero import AeroDatabase
from certified_flight.build import ROOT

sys.path.insert(0, str(ROOT/'scripts'/'cfd'))
import make_su2_cases as cfd  # noqa: E402

SNAP = ROOT/'data/geometry_subset'


def test_oblique_shock_and_prandtl_meyer_reference_values():
    pr, Tr, rr, M2, det = F.oblique(np.array([5.]), np.array([np.deg2rad(10.)]))
    assert pr[0] == pytest.approx(3.0454, rel=2e-3)          # NACA 1135 chart value, beta = 19.38 deg
    assert M2[0] == pytest.approx(3.9990, rel=2e-3)
    assert not det[0]
    pr, Tr, rr, M2, vac = F.expansion(np.array([2.]), np.array([np.deg2rad(10.)]))
    assert M2[0] == pytest.approx(2.3848, rel=1e-3)
    assert pr[0] == pytest.approx(((1+.2*4)/(1+.2*M2[0]**2))**3.5, rel=1e-9)
    _, _, _, _, det = F.oblique(np.array([2.]), np.array([np.deg2rad(30.)]))
    assert det[0]                                              # beyond theta_max(M=2) = 22.97 deg


def test_flat_plate_laminar_friction_matches_blasius():
    """Aligned flat plate, alpha = 0: the panel sum equals the exact laminar average 1.328/sqrt(Re*_L)."""
    L, b, n = 2., 1., 200
    x = np.linspace(0, L, n+1)
    s1, s2 = x[:-1], x[1:]
    p = F.Panels(centre=np.c_[.5*(s1+s2), np.zeros(n), np.zeros(n)], normal=np.tile([0., 0., -1.], (n, 1)),
                 area=(s2-s1)*b, s1=s1, s2=s2, surface=np.zeros(n, int))
    M, T, rho = 6., 220., .02
    cl, cd = F.friction_coefficients(p, M, 0., T, rho, mode='laminar')
    V = M*np.sqrt(F.GAMMA*F.R_AIR*T)
    Ts = T*(1+.032*M*M+.58*(F.TW_K/T-1))
    rs = rho*T/Ts
    cf = 1.328/np.sqrt(rs*V*L/F.sutherland(Ts))*rs/rho
    assert cd == pytest.approx(2*cf*L*b/F.SREF, rel=1e-4)      # mirrored half body -> factor 2 (s1 clamp at 1e-9)
    assert abs(cl) < 1e-12


@pytest.mark.parametrize('gid', ['G42', 'S2_MD12_dd296f618e'])
def test_panel_pressure_reproduces_the_snapshot(gid):
    db = AeroDatabase(SNAP)
    p = F.load_panels(SNAP/'geometries'/gid/'native_geometry.mat')
    i = db.index[gid]
    for M in (5., 7.):
        im = list(db.M).index(M)
        for ad in (4., 6.):
            ia = list(np.round(np.rad2deg(db.A), 6)).index(ad)
            cl, cd = F.pressure_coefficients(p, M, np.deg2rad(ad))
            assert cl == pytest.approx(db.values[i, im, ia, 0], abs=3e-4)
            assert cd == pytest.approx(db.values[i, im, ia, 1], rel=1e-2)


def test_cfd_surfaces_are_watertight_with_inventory_volumes():
    inv = {r['geometry_id']: float(r['volume_ratio_G42'])
           for r in csv.DictReader(open(SNAP/'candidate_inventory.csv', encoding='utf-8-sig'))}

    def volume(g):
        V, T = cfd.closed_surface(SNAP/'geometries'/g/'native_geometry.mat')
        c = cfd.check_closed(T)
        assert c['open_edges'] == 0 and c['nonmanifold_edges'] == 0
        return np.sum(np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])))/6
    g42 = volume('G42')
    for g in ('S2_MD12_dd296f618e', 'V2M_MD4_001'):
        assert volume(g)/g42 == pytest.approx(inv[g], abs=1e-4)


# ---------------------------------------------------------------------------------- collect_su2 (decision logic)
sys.path.insert(0, str(ROOT/'scripts'/'cfd'))
import collect_su2 as collect  # noqa: E402


def test_equal_lift_fit_recovers_a_quadratic_polar():
    cl = np.array([.02, .04, .06])
    cd = .002+.1*cl**2
    got = collect.fit_equal_lift(cl, cd, (.03, .05, .09))
    assert got[.03] == pytest.approx(.002+.1*.03**2, rel=1e-9)
    assert got[.05] == pytest.approx(.002+.1*.05**2, rel=1e-9)
    assert .09 not in got                                       # outside the fitted CL range


def _rows(ld):
    """ld[(geometry, mach)] = L/D at every CL target."""
    out = []
    for (g, M), v in ld.items():
        for t in collect.CL_TARGETS:
            out.append(dict(geometry=g, mach=M, model='euler', CL_target=t, CD_total=t/v, LD=v))
    return out


def test_decision_requires_design_mach_matching_with_margin():
    geoms = ('FAM_MD4', 'FAM_MD6', 'FAM_MD8')
    match = {(g, M): 8. for g in geoms for M in (4., 8.)}
    match[('FAM_MD4', 4.)] = 9.
    match[('FAM_MD8', 8.)] = 9.
    assert collect.decide(_rows(match))['premise_holds'] is True
    flat = dict(match)
    flat[('FAM_MD4', 4.)] = 8.1                                # best only by 1.2 % over the worst member
    flat[('FAM_MD8', 8.)] = 8.1
    assert collect.decide(_rows(flat))['premise_holds'] is False
    wrong = dict(match)
    wrong[('FAM_MD8', 4.)] = 9.5                                # the high-design-Mach member wins at Mach 4
    assert collect.decide(_rows(wrong))['premise_holds'] is False


def test_switching_gain_is_zero_when_one_member_dominates():
    rows = _rows({('FAM_MD4', 4.): 8., ('FAM_MD8', 4.): 9., ('FAM_MD4', 8.): 8., ('FAM_MD8', 8.): 9.})
    assert collect.switching_gain(rows, 'euler')['gain_pct'] == pytest.approx(0., abs=1e-9)
    rows = _rows({('FAM_MD4', 4.): 9., ('FAM_MD8', 4.): 8., ('FAM_MD4', 8.): 8., ('FAM_MD8', 8.): 9.})
    assert collect.switching_gain(rows, 'euler')['gain_pct'] == pytest.approx(100*(9/8.5-1), rel=1e-9)
