"""Surrogate refit pipeline of the AoA study: conventions, contracts, configuration and round trip.

- the fitting code uses the V16 kernel-ridge convention (checked against data/surrogate/controller.npz);
- V16 contracts, configuration and qualification are unchanged; the consistent objective gets its own
  contract and is refused by the V16 qualification;
- the teacher's post context reproduces FlightEnvironment.step for the same planned Mach;
- a model fitted by scripts/aoa_fit_surrogate.py loads through SurrogateController.load.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from certified_flight.build import ROOT, runtime_config
from certified_flight.contracts import mechanics_digest
from certified_flight.corrector import PostCorrector
from certified_flight.distill import RIDGE, Teacher, fit_head, read_jsonl
from certified_flight.environment import FlightEnvironment
from certified_flight.lookahead import LookaheadGuard
from certified_flight.objectives import contract, objective_runtime_config, objective_safety
from certified_flight.surrogate import SurrogateController, kernel

CONTEXTS = ROOT/'experiments/aoa_fix/surrogate/contexts.jsonl'


@pytest.fixture(scope='module')
def v16():
    teacher = Teacher('v16')
    model = SurrogateController.load(ROOT/'data/surrogate/controller.npz', teacher.db, teacher.mission,
                                     teacher.physical, {'mission_sha256': teacher.mission.config_path,
                                                        'trajectory_sha256': teacher.mission.path(
                                                            teacher.physical['trajectory'])})
    return teacher, model


def test_fit_matches_v16_kernel_ridge_convention():
    z = np.load(ROOT/'data/surrogate/controller.npz')
    for head in ('pre', 'post'):
        X, w, m, length = z[head+'_x'], z[head+'_weights'], z[head+'_mean'], float(z[head+'_length'])
        Y = m+(kernel(X, X, length)+RIDGE*np.eye(len(X)))@w     # V16 training targets under the convention
        mean, weights = fit_head(X, Y, length)
        np.testing.assert_allclose(mean, m, rtol=0, atol=1e-10)
        np.testing.assert_allclose(weights, w, rtol=1e-6, atol=1e-6*np.abs(w).max())


def test_contracts():
    config = runtime_config(ROOT)
    assert contract(config['physical']) == 'rl298-consistent/2:ec1c13606315569b'     # V16 digest unchanged
    assert objective_runtime_config(config, 'v16') == config
    consistent = objective_runtime_config(config, 'consistent', surrogate='data/surrogate/controller.npz')
    assert mechanics_digest(consistent['physical']) != mechanics_digest(config['physical'])
    other = deepcopy(consistent['physical'])
    other['ocp']['horizon_tracking_weight'] = 5.
    assert mechanics_digest(other) != mechanics_digest(consistent['physical'])
    assert consistent['safety_options']['boundary_smoothing'] is True
    assert consistent['safety_options']['lookahead_qualification_report'] == 'data/lookahead_surrogate/AB_metrics.json'
    with pytest.raises(ValueError):
        objective_runtime_config(config, 'consistent')     # the V16 surrogate has the v16 teacher


def test_v16_qualification_unchanged_and_refuses_boundary_smoothing(v16):
    teacher, model = v16
    safety = runtime_config(ROOT)['safety_options']
    assert LookaheadGuard.qualify(teacher.plant, model, safety)['report'] == 'data/lookahead/AB_metrics.json'
    with pytest.raises(ValueError, match='boundary_smoothing'):
        LookaheadGuard.qualify(teacher.plant, model, objective_safety(safety, 'consistent'))


def test_teacher_post_context_matches_environment(v16):
    teacher, model = v16
    env = FlightEnvironment(teacher.plant, model, teacher.physical, runtime_config(ROOT)['safety_options'],
                            post_corrector=PostCorrector(teacher.plant, runtime_config(ROOT)['safety_options']))
    seen = []
    for action in (5, 12, 5, 5):      # includes a geometry change and a decision during the 8 s blend
        _, _, _, info = env.step(action)
        seen.append((info['transition_started'], info['post_context']['rate']))
        pre = {k: v for k, v in info['pre_context'].items()}
        post, _ = teacher.post_context(pre, info['planned_mach'])
        for key in ('source', 'target', 'rate', 'CL_des', 'alpha_prev', 't'):
            assert post[key] == info['post_context'][key], key
        assert post['fraction'] == pytest.approx(info['post_context']['fraction'], abs=1e-12)
    assert any(started for started, _ in seen) and any(rate > 0 for _, rate in seen)


def test_fit_script_round_trip(v16, tmp_path):
    """Labels from the V16 surrogate itself -> fit -> load; the refit reproduces it on its training contexts."""
    teacher, model = v16
    rows = read_jsonl(CONTEXTS)[:60]
    labels = []
    for k, r in enumerate(rows):
        pre = dict(r['pre_context'], control_contract=model.control_contract)
        mach, _ = model.predict(pre, 'pre')
        post, _ = teacher.post_context(pre, mach[0])
        alpha, _ = model.predict(post, 'post')
        labels.append(dict(id=r['id'], split='train' if k % 5 else 'validation', ok=True, pre_context=r['pre_context'],
                           pre_mach=float(mach[0]), post_context={a: b for a, b in post.items() if a != 'control_contract'},
                           post_alpha=alpha.tolist()))
    path = tmp_path/'labels.jsonl'
    path.write_text(''.join(json.dumps(x)+'\n' for x in labels), encoding='utf-8')
    sys.path.insert(0, str(ROOT/'scripts'))
    import aoa_fit_surrogate
    out = tmp_path/'surrogate_test'/'controller.npz'
    aoa_fit_surrogate.main(['--objective', 'v16', '--labels', str(path), '--out', str(out)])
    refit = SurrogateController.load(out, teacher.db, teacher.mission, teacher.physical,
                                     {'mission_sha256': teacher.mission.config_path,
                                      'trajectory_sha256': teacher.mission.path(teacher.physical['trajectory'])})
    assert refit.control_contract == model.control_contract
    for x in labels:
        if x['split'] != 'train':
            continue
        post = dict(x['post_context'], control_contract=model.control_contract)
        alpha, _ = refit.predict(post, 'post')
        assert np.max(np.abs(np.rad2deg(alpha-np.asarray(x['post_alpha'])))) < .05


def test_build_flight_environment_hook(monkeypatch, tmp_path):
    """AOA_OBJECTIVE selects an objective for unchanged training/evaluation code; it needs its own surrogate."""
    from certified_flight import build_flight
    monkeypatch.setenv('AOA_OBJECTIVE', 'consistent')
    monkeypatch.delenv('AOA_SURROGATE', raising=False)
    with pytest.raises(ValueError, match='refitted surrogate'):
        build_flight(tmp_path/'run', 'hook')
    monkeypatch.setenv('AOA_OBJECTIVE', 'v16')
    env, certifier = build_flight(tmp_path/'run', 'hook')
    assert env.cfg == runtime_config(ROOT)['physical']
