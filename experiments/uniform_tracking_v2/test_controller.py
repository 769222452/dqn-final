"""Tests local to the experimental controller; original files are unchanged."""
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]/'src'))

import numpy as np
import pytest

from continuous_tracking.experiment import build
from controller import UniformTrackingOCP
from check import memory


@pytest.fixture(scope='module')
def system():
    original, _, _ = build()
    config = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    config['compile_nlp'] = False
    return UniformTrackingOCP(original.m, original.db, original.traj, original.cfg, original.plant, config)


def nominal(ocp):
    m = ocp.m.data
    g = ocp.db.index['G42']
    return dict(x=m['initial_state']['x_h_V_gamma_s'], alpha_prev=m['initial_state']['alpha_rad'],
                t=0., source=g, target=g, fraction=1., rate=0., CL_des=.025,
                alpha_rate_prev=0., alpha_interval_prev_s=.5)


def point(ocp, b):
    c = nominal(ocp)
    n = len(b['times'])-1
    v = np.r_[np.tile(np.array(c['x'])/ocp.xscale, n+1),
              np.full(n, c['alpha_prev']/ocp.uscale), np.zeros(b['n_slack'])]
    p = np.r_[c['x'], c['alpha_prev'], c['CL_des'], 1., 0.,
              ocp.db.parameters[c['source']], ocp.db.parameters[c['target']], 0., .5]
    return v, p


def values(b, v, p):
    return dict(zip(b['term_names'], np.asarray(b['terms'](v, p)).ravel()))


def test_reject_other_protocol(system):
    config = dict(system.experiment, protocol='continuous-alpha-full-horizon-tracking/1')
    with pytest.raises(ValueError, match='protocol'):
        UniformTrackingOCP(system.m, system.db, system.traj, system.cfg, system.plant, config)


@pytest.mark.parametrize('key', ['alpha_rate_prev', 'alpha_interval_prev_s'])
def test_executed_memory_required(system, key):
    c = nominal(system)
    del c[key]
    with pytest.raises(ValueError, match='memory'):
        system.solve_context(c)


def test_memory_uses_last_executed_interval():
    rows = [{'times':[0., .5, 1.], 'alpha_nodes_deg':[4., 4.02, 4.03]}]
    assert memory(rows, 0)['alpha_rate_prev'] == 0.
    assert memory(rows, 1)['alpha_rate_prev'] == pytest.approx(np.deg2rad(.02))
    assert memory(rows, 1)['alpha_interval_prev_s'] == .5


def test_uniform_soft_constraints_on_both_sides_of_three_seconds(system):
    b = system.build(np.full(24, .5))
    v, p = point(system, b)
    # Identical local state/control produces identical error and bounds at
    # 0, 2.5, 3, 6 and 11.5 s. A first-3s objective must not reappear here.
    err = np.asarray(b['errors'](v, p))
    np.testing.assert_array_equal(err[[0,5,6,12,23]], np.tile(err[0], (5,1)))
    assert 'current_tracking_cost' not in b['term_names']
    g0 = np.asarray(b['raw_solver'].get_function('nlp_g')(v,p)).ravel()
    base = values(b,v,p)
    for index in (0,5,6,12,23):
        modified = v.copy()
        modified[b['slack_start']+index] = .01
        g = np.asarray(b['raw_solver'].get_function('nlp_g')(modified,p)).ravel()
        changed = np.flatnonzero(abs(g-g0)>1e-12)
        # Only the ten signed CL inequalities change; all physical and
        # dynamic equalities remain bitwise identical.
        assert len(changed) == 10
        np.testing.assert_allclose((g-g0)[changed], -.01, rtol=0, atol=1e-15)
        assert np.all(np.isneginf(b['lbg'][changed]))
        assert np.all(b['ubg'][changed] == b['band_scaled'])
        assert values(b,modified,p)['slack']-base['slack'] == pytest.approx(.01*.5/3)
    assert np.all(b['lbw'][b['slack_start']:] == 0.)


def test_actual_previous_slope_enters_cost_only(system):
    b = system.build(np.full(6,.5))
    v,p = point(system,b)
    base = values(b,v,p)
    assert base['control_rate'] == 0. and base['rate_change'] == 0.
    previous = p.copy(); previous[-2] = .004
    expected = (.004*.5/system.uscale)**2
    assert values(b,v,previous)['rate_change'] == pytest.approx(expected)
    previous[-1] = 1.5
    assert values(b,v,previous)['rate_change'] == pytest.approx(expected/2)
    np.testing.assert_array_equal(b['raw_solver'].get_function('nlp_g')(v,p),
                                  b['raw_solver'].get_function('nlp_g')(v,previous))


def test_tracking_cost_covers_future_and_cannot_cancel_signed_error(system):
    b = system.build(np.full(24,.5))
    v,p = point(system,b)
    base = values(b,v,p)
    n = 24
    v[4*(n+1)+15] += .01/system.uscale  # node at t=8 s
    assert values(b,v,p)['target'] != pytest.approx(base['target'],abs=1e-6)
    assert values(b,v,p)['target'] >= 0.


def test_tail_real_solve_and_independent_audit(system):
    c = nominal(system)
    c['t'] = 1197.
    c['CL_des'] = float(system.plant.base.quantities(c['x'], c['alpha_prev'],
                           c['source'], c['target'],1.)[4])
    plan = system.solve_context(c)
    assert plan.success, plan.summary()
    assert plan.times[-1] == 3. and plan.alpha_nodes[0] == c['alpha_prev']
    assert plan.stages['optimizer_calls'] == 1
    assert not plan.stages['slack_required_for_safety_proven']
    audit,_ = system.plant.audit(c, plan.alpha_nodes, plan.times, c['CL_des'])
    assert audit['valid'], audit
    assert plan.terms['maximum_CL_slack'] >= -1e-12
