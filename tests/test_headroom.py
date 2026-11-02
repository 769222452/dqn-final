"""Headroom study (src/certified_flight/headroom.py): model identities, estimator and planner calibration."""
from pathlib import Path

import numpy as np
import pytest

from certified_flight import headroom as H
from certified_flight.aero import AeroDatabase
from certified_flight.build import runtime_config
from certified_flight.mission import Mission

ROOT = Path(__file__).resolve().parents[1]
BOUND_NOMINAL_KM = 2511.03          # experiments/improve/bounds/dh0_dv0.json (relaxed_bound, 0.1 s check)


@pytest.fixture(scope='module')
def world():
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    z = np.load(ROOT/'experiments/headroom/polar_table.npz')
    surface = H.PolarSurface(z['knots'], z['cd'])
    return mission, surface, (float(db.M[0]), float(db.M[-1])), z


def test_polar_surface_matches_table_at_nodes(world):
    _, surface, _, z = world
    for i in (0, 5, 17):
        for j in (10, 25, 40):
            assert float(surface(z['knots'][i], z['cl'][j])) == pytest.approx(z['cd'][i, j], rel=1e-9)


def test_polar_is_convex_in_cl(world):
    z = world[3]
    second = np.diff(z['cd'], 2, axis=1)
    assert second.min() > -1e-6                    # a lower convex hull has no concave kink


def test_density_and_aero_scalings(world):
    mission, surface, _, _ = world
    F, _ = H.dynamics(mission, surface)
    x = np.array([27000., 2300., -.003, 0.])
    _, q1 = F(x, .026, [1., 1., 1.])
    _, q2 = F(x, .026, [1.2, 1., 1.])
    _, q3 = F(x, .026, [1., 1.1, .9])
    q1, q2, q3 = (np.asarray(q).ravel() for q in (q1, q2, q3))
    assert q2[0] == pytest.approx(1.2*q1[0], rel=1e-12)                 # dynamic pressure ~ density
    assert q2[1] == pytest.approx(np.sqrt(1.2)*q1[1], rel=1e-12)        # heat flux ~ sqrt(density)
    assert q3[4] == pytest.approx(1.1*q1[4], rel=1e-12)                 # CL scaled
    assert q3[5] == pytest.approx(.9*q1[5], rel=1e-12)                  # CD scaled


def test_density_field_statistics():
    s, k = H.density_field(.1, 200e3, seed=3)
    s2, k2 = H.density_field(.1, 200e3, seed=3)
    assert np.array_equal(k, k2) and s[0] == 0 and np.all(k > 0)
    z = np.log(k)
    long = np.concatenate([np.log(H.density_field(.1, 200e3, seed=i)[1]) for i in range(20)])
    assert abs(np.std(long)-.1) < .01
    near = np.mean([np.corrcoef(np.log(H.density_field(.1, 200e3, seed=i)[1])[:-40], np.log(H.density_field(.1, 200e3, seed=i)[1])[40:])[0, 1]
                    for i in range(20)])
    assert near == pytest.approx(np.exp(-1), abs=.15)                   # 40 steps of 5 km = 200 km
    assert z.shape == s.shape


def test_load_estimator_recovers_aero_biases(world):
    mission, surface, support, _ = world
    field = (np.array([0., 4e6]), np.array([1.05, 1.05]))
    truth = H.Truth(mission, surface, support, field, k_lift=1.07, k_drag=.94)
    est = H.LoadEstimator(truth, prior=.1)
    x = np.array(mission.data['initial_state']['x_h_V_gamma_s'], float)
    for i in range(120):
        c = .026+.004*np.sin(i/10)
        xn, _ = truth.step(x, c)
        est.add(x, xn, c, truth.k_rho(x[3]))
        x = xn
    k_lift, k_drag = est.values
    assert k_lift == pytest.approx(1.07, abs=.01)
    assert k_drag == pytest.approx(.94, abs=.01)


def test_nominal_plan_reproduces_the_relaxed_bound_and_flies_feasibly(world):
    mission, surface, support, _ = world
    x0 = np.array(mission.data['initial_state']['x_h_V_gamma_s'], float)
    planner = H.Planner(mission, surface, support)
    plan = planner.plan(x0, 1200., back=H.Planner.BASE_BACKOFF)
    assert plan['ok']
    assert abs(plan['range_km']-BOUND_NOMINAL_KM) < 1.0                 # the fast grid costs < 1 km vs the 480-step bound
    truth = H.Truth(mission, surface, support, (np.array([0., 4e6]), np.ones(2)))
    flown = H.fly(truth, x0, H.open_loop_controller(plan))
    assert flown['feasible']
    assert flown['range_km'] == pytest.approx(plan['range_km'], abs=.2)
