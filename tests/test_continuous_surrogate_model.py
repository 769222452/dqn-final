"""Protocol and data-isolation tests for freshly fitted continuous heads."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import casadi as ca

import continuous_surrogate.model as module
from continuous_surrogate.model import ContinuousSurrogate
from continuous_surrogate.teacher import PROTOCOL, DEFAULT_TEACHER_CONFIG, FullHorizonTeacher
from certified_flight.surrogate import OracleRequired


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    class Features:
        def __init__(self, db, mission):
            pass

        def vector(self, c):
            return np.asarray(c['feature'], float)

        def physical(self, c):
            return np.asarray(c['physical'], float)

    monkeypatch.setattr(module, 'Features', Features)
    (tmp_path/'manifest.json').write_text('{}', encoding='utf-8')
    db = SimpleNamespace(A=np.array([-.02, .21]), M=np.array([3.5, 8.25]), directory=tmp_path)
    mission = SimpleNamespace(data={'limits': {'alpha_rad': [-.02, .21], 'alpha_rate_radps': .08}})
    baseline = lambda c, head: np.array([5.]) if head == 'pre' else np.full(6, .05)
    records = []
    for i in range(8):
        context = dict(protocol=PROTOCOL, alpha_prev=.05, feature=np.full(25, i*.01).tolist(),
                       physical=[40000+i, 2400+i, 0., .05, 900.])
        for head in ('pre', 'post'):
            records.append(dict(protocol=PROTOCOL, accepted=True, head=head, context=context,
                y=[5.001] if head == 'pre' else [.051]*6, role='train', group='train'+str(i),
                label_provenance='FRESH_CONTINUOUS_REAL_OCP_AND_FULL_DUAL_GRID_AUDIT'))
    return db, mission, baseline, records


def test_whole_horizon_teacher_contract_is_explicit():
    assert DEFAULT_TEACHER_CONFIG['objective_priority'] == 'whole_horizon_tracking_then_economics'
    assert DEFAULT_TEACHER_CONFIG['reference_prediction'] == 'hold_current_CL'
    assert DEFAULT_TEACHER_CONFIG['tracking_optimality_absolute_allowance'] == 1e-8


def test_teacher_primary_really_penalizes_tracking_after_three_seconds(monkeypatch):
    """Inspect the symbolic NLP objective using a simple CL=alpha test plant."""
    captured = []

    def solver(name, plugin, nlp, opts):
        captured.append(nlp)
        return object()

    monkeypatch.setattr(ca, 'nlpsol', solver)
    limits = dict(altitude_m=[0., 80000.], velocity_mps=[0., 6000.], gamma_rad=[-1., 1.],
                  alpha_rad=[-.02, .21], alpha_rate_radps=.08,
                  dynamic_pressure_pa_max=80000., heat_proxy_wm2_max=2e6, load_factor_g_max=3.)

    def quantities(state, alpha, fraction, c0, c1):
        q = ca.vertcat(1000., 1000., 1., 5., alpha, .001, 1., 1.)
        return state*0., q

    def rk(state, alpha, slope, fraction, rate, c0, c1, dt, tau):
        samples = [ca.vertcat(state, quantities(state, alpha+slope*d, fraction, c0, c1)[1])
                   for d in (0., dt/2, dt/2, dt, dt)]
        return state, ca.horzcat(*samples)

    teacher = FullHorizonTeacher.__new__(FullHorizonTeacher)
    teacher.cache = {}
    teacher.m = SimpleNamespace(data={'limits': limits})
    teacher.db = SimpleNamespace(ncoef=1, A=np.array([-.02, .21]), M=np.array([3.5, 8.25]))
    teacher.plant = SimpleNamespace(rk=rk, base=SimpleNamespace(fn=quantities))
    teacher.uscale, teacher.xscale = .1, np.array([1e4, 1e3, .1, 1e5])
    teacher.experiment = dict(DEFAULT_TEACHER_CONFIG, compile_nlp=False)
    teacher.traj = {'aero_target_tracking': {'CL_scale': .1, 'weight_CL': 1.},
                   'objective': {'reference': {'energy_ref': 1e9, 'heat_load_ref': 5e7},
                                 'terminal': {}, 'w_drag': 1., 'w_heat': .5, 'w_terminal': 5.,
                                 'w_control_rate': .1, 'w_range': 1.},
                   'solver': {'scaling': {'s': 1e5}, 'ipopt': {}}}
    b = teacher.build(np.full(12, .5))
    nlp = captured[0]
    f = ca.Function('primary_test', [nlp['x'], nlp['p']], [nlp['f']])
    states = np.tile(np.array([40000., 2400., 0., 0.])/teacher.xscale, 13)
    controls = np.full(12, .025/teacher.uscale)
    parameters = np.r_[[40000., 2400., 0., 0.], .025, .025, 1., 0., 0., 0., 0.]
    original = np.r_[states, controls]
    changed = original.copy()
    changed[len(states)+9] += .01/teacher.uscale
    assert float(f(changed, parameters)) > float(f(original, parameters))+1e-6
    terms0 = dict(zip(b['term_names'], np.asarray(b['terms'](original, parameters)).ravel()))
    terms1 = dict(zip(b['term_names'], np.asarray(b['terms'](changed, parameters)).ravel()))
    assert terms1['current_tracking_cost'] == terms0['current_tracking_cost']
    assert terms1['target'] > terms0['target']


def test_fresh_fit_does_not_use_holdout_labels(inputs):
    db, mission, baseline, records = inputs
    clean = ContinuousSurrogate.fit(records, db, mission, baseline)
    heldout = deepcopy(records[0])
    heldout.update(role='test', group='blind', y=[7.])
    fitted = ContinuousSurrogate.fit(records+[heldout], db, mission, baseline)
    for head in ('pre', 'post'):
        np.testing.assert_array_equal(clean.heads[head].weights, fitted.heads[head].weights)
        assert fitted.metadata['training_counts'][head] == 8
    assert not fitted.metadata['full_qualification_passed']


def test_trajectory_groups_cannot_leak(inputs):
    db, mission, baseline, records = inputs
    holdout = deepcopy(records[0])
    holdout['role'] = 'test'
    with pytest.raises(ValueError, match='crosses'):
        ContinuousSurrogate.fit(records+[holdout], db, mission, baseline)


def test_old_labels_are_rejected_and_failure_does_not_train(inputs):
    db, mission, baseline, records = inputs
    old = deepcopy(records)
    old[0]['protocol'] = 'rl298-consistent/2'
    with pytest.raises(ValueError, match='incompatible'):
        ContinuousSurrogate.fit(old, db, mission, baseline)
    failed = deepcopy(records)
    failed[0]['accepted'] = False
    failed[0].pop('y')
    with pytest.raises(ValueError, match='Insufficient'):
        ContinuousSurrogate.fit(failed, db, mission, baseline)
    context = deepcopy(records[0]['context'])
    context.pop('protocol')
    context['control_contract'] = 'rl298-consistent/2:old'
    fitted = ContinuousSurrogate.fit(records, db, mission, baseline)
    with pytest.raises(ValueError, match='control contract'):
        fitted.predict(context, 'pre')


def test_future_node_projection_is_continuous_and_rate_bounded(inputs):
    db, mission, baseline, _ = inputs
    model = ContinuousSurrogate(db, mission, baseline)
    future = model.project([1., -1., 1., -1., 1., -1.], .05)
    full = np.r_[.05, future]
    assert full[0] == .05
    assert np.max(np.abs(np.diff(full))/.5) <= .08+1e-15
    assert np.min(full) >= -.02 and np.max(full) <= .21


def test_supported_inference_ood_and_save_load(inputs, tmp_path):
    db, mission, baseline, records = inputs
    model = ContinuousSurrogate.fit(records, db, mission, baseline)
    context = records[0]['context']
    prediction, diag = model.predict(context, 'post')
    assert prediction.shape == (6,) and diag['supported']
    assert not diag['full_qualification_passed']
    far = deepcopy(context)
    far['feature'] = [2.]*25
    with pytest.raises(OracleRequired, match='OUT_OF_DISTRIBUTION'):
        model.predict(far, 'post')
    path = tmp_path/'new_controller.npz'
    model.save(path)
    loaded = ContinuousSurrogate.load(path, db, mission, baseline)
    np.testing.assert_array_equal(prediction, loaded.predict(context, 'post')[0])
    with pytest.raises(FileExistsError):
        model.save(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    metadata['output_schema'] = 'six-half-second-zoh-commands'
    path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='output_schema'):
        ContinuousSurrogate.load(path, db, mission, baseline)
