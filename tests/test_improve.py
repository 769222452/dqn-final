"""Opt-in improvements of experiments/improve: aligned objective, coordinated blocks, physical bound.

Defaults are covered by the existing tests (V16 reference bit-exact, aoa-fix contracts unchanged).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from certified_flight.aero import AeroDatabase
from certified_flight.bound import envelope, hull_cd, lower_hull, polar_samples
from certified_flight.build import ROOT, runtime_config
from certified_flight.contracts import mechanics_digest
from certified_flight.mission import Mission
from certified_flight.objectives import objective_ocp, objective_runtime_config
from certified_flight.ocp import TwoStageOCP
from certified_flight.plant import CompiledPlant
from rollout_improvement.policy import ChainPolicy
from rollout_improvement.rollouts import block_alternatives
from rollout_improvement.training import block_override_sets


@pytest.fixture(scope='module')
def parts():
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    return physical, mission, db


def test_aligned_only_removes_heat_weight(parts):
    physical, mission, db = parts
    consistent = objective_ocp(physical['ocp'], 'consistent')
    aligned = objective_ocp(physical['ocp'], 'aligned')
    assert {k: v for k, v in aligned.items() if k != 'heat_weight'} == consistent
    assert aligned['heat_weight'] == 0.
    c = objective_runtime_config(runtime_config(ROOT), 'consistent', surrogate='data/surrogate/controller.npz')
    a = objective_runtime_config(runtime_config(ROOT), 'aligned', surrogate='data/surrogate/controller.npz')
    assert mechanics_digest(a['physical']) != mechanics_digest(c['physical'])
    assert a['safety_options'] == c['safety_options']
    plant = CompiledPlant(mission, db)
    traj = yaml.safe_load(mission.path(physical['trajectory']).read_text(encoding='utf-8'))
    with pytest.raises(ValueError):
        TwoStageOCP(mission, db, traj, dict(aligned, heat_weight=-1.), plant).build(np.array([.5]*6))


def test_block_alternatives_shift_the_nominal_schedule():
    sched = {int(12*k): 5 for k in range(100)}
    alts = block_alternatives(0., 0, sched, (1, 2, 4))
    assert {(B, d) for B, d, _ in alts} == {(2, -1), (2, 1), (4, -1), (4, 1)}
    for B, d, acts in alts:
        assert sorted(acts) == [12*j for j in range(B)] and set(acts.values()) == {5+d}
    tail = block_alternatives(1176., 0, sched, (8,))      # clipped at the last decision (t = 1188 s)
    assert all(sorted(a) == [1176, 1188] for _, _, a in tail)
    assert block_alternatives(0., 0, {**sched, 12: 18}, (2,)) == [(2, -1, {0: 4, 12: 17})]   # 19 > MAX_ACTION


def test_chain_policy_block_override_then_schedule():
    env = SimpleNamespace(t=0.)
    sched = [(12.*k, 5) for k in range(100)]
    pol = ChainPolicy([], env, override={24: 7, 36: 7}, schedule=sched)
    seen = []
    for k in range(6):
        env.t = 12.*k
        seen.append(pol.decide())
    assert seen == [5, 5, 7, 7, 5, 5]
    single = ChainPolicy([], env, override=(24., 7), schedule=sched)
    out = []
    for k in range(4):
        env.t = 12.*k
        out.append(single.decide())
    assert out == [5, 5, 7, 5]


def test_block_override_sets_are_greedy_and_disjoint():
    sched = [(12.*k, 5) for k in range(100)]
    labels = [dict(features=[0]*6, t=0., block=2, delta=1, actions={'0': 6, '12': 6}, advantage=3e-5),
              dict(features=[0]*6, t=12., block=2, delta=1, actions={'12': 6, '24': 6}, advantage=5e-5),
              dict(features=[0]*6, t=48., block=2, delta=-1, actions={'48': 4, '60': 4}, advantage=2e-5),
              dict(features=[0]*6, t=96., block=2, delta=-1, actions={'96': 4, '108': 4}, advantage=-1e-4)]
    sets = block_override_sets(labels, sched, 1e-5)
    assert [k for k, _, _ in sets] == [1, 2]
    k, entries, gain = sets[-1]
    assert sorted(e['t'] for e in entries) == [12., 24., 48., 60.] and gain == pytest.approx(7e-5)
    assert all(e['base_action'] == 5 for e in entries)


def test_lower_hull_and_envelope_bound_the_polar(parts):
    _, _, db = parts
    x = np.array([0., 1., 2., 3., 1.5])
    y = np.array([1., 0., 1., 4., 2.])
    assert lower_hull(x, y).tolist() == [[0., 1.], [1., 0.], [2., 1.], [3., 4.]]
    knots, lines, _ = envelope(db, knots=[6., 6.25, 6.5])
    for M in (6.05, 6.25, 6.4):
        cl, cd = polar_samples(db, M, alpha_step_deg=.25)
        sel = (cl > -.005) & (cl < .13)
        bound = np.array([hull_cd(knots, lines, M, c) for c in cl[sel]])
        assert np.all(bound <= cd[sel]+1e-12)
