"""Opt-in time-consistent CL objective and boundary smoothing (AoA sawtooth fix).

Defaults must leave the V16 objective untouched; the new mode must weight the commanded CL
identically at the head and the tail of the plan and must cost the jump from the measured alpha.
"""
import json

import numpy as np
import pytest
import yaml

from certified_flight.aero import AeroDatabase
from certified_flight.build import ROOT, runtime_config
from certified_flight.mission import Mission
from certified_flight.ocp import TwoStageOCP
from certified_flight.plant import CompiledPlant


@pytest.fixture(scope='module')
def parts():
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    plant = CompiledPlant(mission, db)
    traj = yaml.safe_load(mission.path(physical['trajectory']).read_text(encoding='utf-8'))
    return mission, db, plant, traj, physical['ocp']


def terms_at(parts, extra, u):
    mission, db, plant, traj, cfg = parts
    ocp = TwoStageOCP(mission, db, traj, dict(cfg, **extra), plant)
    dts = ocp.grid(1200)
    b = ocp.build(dts)
    g = db.index['G42']
    x0 = np.array(mission.data['initial_state']['x_h_V_gamma_s'])
    n = len(dts)
    v = np.r_[np.tile(x0/ocp.xscale, n+1), np.asarray(u)/ocp.uscale]
    p = np.r_[x0, .05, .025, 0., 0., 1., 0., db.parameters[g], db.parameters[g], 1.]
    return dict(zip(b['term_names'], np.asarray(b['terms'](v, p)).ravel())), dts


def test_default_objective_is_v16(parts):
    assert parts[4]['tracking_mode'] == 'mean_3s'
    base, dts = terms_at(parts, {}, np.full(len(TwoStageOCP(*parts[:2], parts[3], parts[4], parts[2]).grid(1200)), .05))
    assert 'control_curvature' not in base
    jumped, _ = terms_at(parts, {}, np.full(len(dts), .06))
    assert jumped['control_rate'] == 0.     # V16: the jump from the measured alpha is free


def test_boundary_jump_is_costed_when_enabled(parts):
    n = len(TwoStageOCP(*parts[:2], parts[3], parts[4], parts[2]).grid(1200))
    terms, _ = terms_at(parts, dict(boundary_smoothing=True), np.full(n, .06))
    assert terms['control_rate'] == pytest.approx(((.06-.05)/.1)**2)


def test_pointwise_horizon_weights_head_and_tail_alike(parts):
    mission, db, plant, traj, cfg = parts
    ocp = TwoStageOCP(mission, db, traj, cfg, plant)
    dts = ocp.grid(1200)
    times = np.r_[0., np.cumsum(dts)]
    extra = dict(tracking_mode='pointwise_horizon')
    base, _ = terms_at(parts, extra, np.full(len(dts), .05))
    head = np.full(len(dts), .05)
    head[0] += .01
    tail = np.full(len(dts), .05)
    late = int(np.flatnonzero(times[:-1] >= 30.)[0])
    tail[late] += .01
    # With a constant state guess the same alpha change has the same CL effect, so the
    # objective change per second of affected horizon must be the same early and late.
    d_head = (terms_at(parts, extra, head)[0]['target']-base['target'])/dts[0]
    d_tail = (terms_at(parts, extra, tail)[0]['target']-base['target'])/dts[late]
    assert abs(d_head) > 1e-9
    # Equal up to the in-interval state evolution (1 RK substep in the head, 2 in the tail).
    assert d_tail == pytest.approx(d_head, rel=1e-3)
    # V16 mean_3s ignores the same late change entirely.
    v16_base, _ = terms_at(parts, {}, np.full(len(dts), .05))
    assert terms_at(parts, {}, tail)[0]['target'] == pytest.approx(v16_base['target'], abs=1e-15)


def test_invalid_weights_rejected(parts):
    with pytest.raises(ValueError):
        terms_at(parts, dict(horizon_tracking_weight=-1.), np.zeros(1))
