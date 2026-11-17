"""Model, scenarios and aerodynamic tables against the original project and basic invariants."""
import json
from pathlib import Path

import numpy as np
import pytest

from target_rl import aero, config, dynamics, scenario as SC

FIX = json.loads((Path(__file__).parent/'fixtures'/'reference_model.json').read_text())


def test_point_mass_matches_original_model():
    m = dynamics.Model()
    for r in FIX['rhs']:
        xd, q = m.rhs(r['x'], r['cl'], r['cd'])
        assert np.allclose(xd, r['rhs'], rtol=1e-12, atol=1e-12)
        for k in ('rho', 'a', 'mach', 'g', 'q_dyn', 'heat_flux', 'n_load'):
            assert q[k] == pytest.approx(r[k], rel=1e-12)


def test_atmosphere_matches_original_model():
    atm = dynamics.Model().atm
    for r in FIX['atmosphere']:
        assert atm.density(r['h']) == pytest.approx(r['rho'], rel=1e-12)
        assert atm.speed_of_sound(r['h']) == pytest.approx(r['a'], rel=1e-12)
        assert atm.gravity(r['h']) == pytest.approx(r['g'], rel=1e-12)


def test_configuration_matches_original_mission():
    cfg = config.load()
    mis = FIX['mission']
    assert cfg['vehicle']['mass_kg'] == mis['mass'] and cfg['vehicle']['reference_area_m2'] == mis['S']
    assert cfg['mission']['initial_state'] == mis['x0']
    for k in ('dynamic_pressure_pa_max', 'heat_proxy_wm2_max', 'load_factor_g_max'):
        assert cfg['mission']['limits'][k] == mis['limits'][k]
    for k in ('altitude_m', 'velocity_mps', 'gamma_rad'):
        assert cfg['mission']['limits'][k] == mis['limits'][k]
    assert cfg['vehicle']['heat_C'] == mis['heat']['C'] and cfg['vehicle']['heat_R_n'] == mis['heat']['R_n']


def test_scenarios_match_original_generator():
    for r in FIX['scenarios']:
        sc = SC.scenario(r['level'], r['seed'])
        assert (sc.dh, sc.dv, sc.k_lift, sc.k_drag) == (r['dh'], r['dv'], r['kL'], r['kD'])
        s, k = SC.density_field(sc)
        assert [k[i] for i in (0, 1, 7, 100, 640)] == r['field_k']
        f = SC.Field(sc)
        assert f(s[7]) == pytest.approx(k[7]) and f(2.5e3) == pytest.approx(.5*(k[0]+k[1]))


def test_train_and_test_seeds_are_disjoint():
    s = config.load()['seeds']
    a, b, c = (set(config.seed_range(s[k])) for k in ('train', 'test', 'random_targets'))
    assert not (a & b) and not (a & c) and not (b & c)


def test_default_library_is_the_298_snapshot():
    assert config.load()['aero']['active'] == 'L298'


@pytest.mark.parametrize('name, count', [('L298', 298), ('F8D2000', 2000)])
def test_drag_tables(name, count, monkeypatch):
    monkeypatch.setenv('TARGET_RL_LIBRARY', name)
    assert config.runs_dir().name == name
    t = aero.tables()
    assert t['sha256'] == config.library()['sha256'] and t['n_geometries'] == count
    assert t['fixed'] == config.library()['fixed_geometry']
    ok = np.isfinite(t['cd_fix'])
    assert np.all(t['cd_min'][ok] <= t['cd_fix'][ok]+1e-15) and np.all(t['cd_fix'][ok] <= t['cd_max'][ok]+1e-15)
    d = aero.Drag(True)
    for M in (5.1, 6.6, 7.9):
        for cl in (.02, .035, .06):
            lo, hi = d.cd(M, cl, 0.), d.cd(M, cl, 1.)
            assert 0 < lo < hi and d.cd(M, cl, .5) == pytest.approx(.5*(lo+hi))
    f = aero.Drag(False)
    assert d.cd(7., .03, 0.) <= f.cd(7., .03) <= 1.2*d.cd(7., .03, 0.)
