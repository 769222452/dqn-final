"""Compact-support residuals must not import a distant recovery trajectory."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import continuous_surrogate.model as original
from certified_flight.surrogate import OracleRequired
from continuous_surrogate.model import ContinuousSurrogate
from continuous_surrogate.teacher import PROTOCOL
from continuous_recovery.local_model import LocalResidualSurrogate, MODEL_SCHEMA


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    class Features:
        def __init__(self, db, mission):
            pass

        def vector(self, c):
            return np.asarray(c['feature'], float)

        def physical(self, c):
            return np.asarray(c['physical'], float)

    monkeypatch.setattr(original, 'Features', Features)
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
                y=[5.+i*.001] if head == 'pre' else [.06]*6, role='train', group='train'+str(i),
                label_provenance='FRESH_CONTINUOUS_REAL_OCP_AND_FULL_DUAL_GRID_AUDIT'))
    return db, mission, baseline, records


def test_pre_fit_and_predictions_keep_original_algorithm(inputs):
    db, mission, baseline, records = inputs
    old = ContinuousSurrogate.fit(records, db, mission, baseline)
    new = LocalResidualSurrogate.fit(records, db, mission, baseline)
    for field in original.REGRESSOR_FIELDS:
        np.testing.assert_array_equal(getattr(old.heads['pre'], field), getattr(new.heads['pre'], field))
    np.testing.assert_array_equal(old.predict(records[0]['context'], 'pre')[0],
                                  new.predict(records[0]['context'], 'pre')[0])
    assert new.metadata['pre_model'] == 'unchanged_continuous_rbf'


def test_local_weights_are_nonnegative_and_sum_to_one_including_zero_prior(inputs):
    db, mission, baseline, records = inputs
    model = LocalResidualSurrogate.fit(records, db, mission, baseline)
    output, diag = model.predict(records[0]['context'], 'post')
    weights = [n['weight'] for n in diag['local_neighbours']]
    assert all(w > 0 for w in weights)
    assert diag['local_baseline_weight'] > 0
    assert sum(weights)+diag['local_baseline_weight'] == pytest.approx(1.)
    assert diag['local_weight_sum'] == pytest.approx(1.)
    assert all(n['distance'] < .25 for n in diag['local_neighbours'])
    assert {n['group'] for n in diag['local_neighbours']} == {f'train{i}' for i in range(8)}
    np.testing.assert_allclose(output, .05+.01*sum(weights))
    assert model.metadata['surrogate_config']['local_baseline_prior_weight'] == 1.
    assert model.metadata['post_no_global_mean']
    assert model.config is model.cfg


def test_far_recovery_label_cannot_change_local_post_prediction(inputs):
    db, mission, baseline, records = inputs
    before = LocalResidualSurrogate.fit(records, db, mission, baseline)
    distant = deepcopy(records[1])
    distant.update(group='far_recovery', y=[.19]*6)
    distant['context']['feature'] = [1.]*25
    after = LocalResidualSurrogate.fit(records+[distant], db, mission, baseline)
    value0, diag0 = before.predict(records[0]['context'], 'post')
    value1, diag1 = after.predict(records[0]['context'], 'post')
    np.testing.assert_array_equal(value0, value1)
    assert diag0['local_neighbours'] == diag1['local_neighbours']
    assert after.metadata['training_counts']['post'] == 9  # retained, never deleted
    assert 'far_recovery' in after.heads['post'].groups


def test_support_edge_tapers_to_zero_residual_and_exact_edge_is_unknown(inputs):
    db, mission, baseline, records = inputs
    for row in records:
        row['context']['feature'] = [0.]*25
    model = LocalResidualSurrogate.fit(records, db, mission, baseline)
    c = deepcopy(records[0]['context'])
    c['feature'] = [.249]*25
    near, d0 = model.predict(c, 'post')
    c['feature'] = [.2499]*25
    nearer, d1 = model.predict(c, 'post')
    assert np.max(abs(near-.05)) < 1e-8
    assert np.max(abs(nearer-.05)) < np.max(abs(near-.05))
    assert d1['local_baseline_weight'] > d0['local_baseline_weight']
    c['feature'] = [.25]*25
    with pytest.raises(OracleRequired) as exc:
        model.predict(c, 'post')
    assert exc.value.reason == 'OUT_OF_DISTRIBUTION'
    assert exc.value.diagnostics['local_baseline_weight'] == 1.
    value, diag = model._predict(c, 'post', False)
    np.testing.assert_array_equal(value, np.full(6, .05))
    assert not diag['supported']


def test_physical_box_remains_required_even_for_near_neighbour(inputs):
    db, mission, baseline, records = inputs
    model = LocalResidualSurrogate.fit(records, db, mission, baseline)
    c = deepcopy(records[0]['context'])
    c['physical'][0] = 50000.
    with pytest.raises(OracleRequired) as exc:
        model.predict(c, 'post')
    assert exc.value.diagnostics['nearest_distance'] == 0.
    assert exc.value.diagnostics['physical_bounds_excess'][0] > 0


def test_holdout_and_rejected_labels_cannot_fit_the_local_head(inputs):
    db, mission, baseline, records = inputs
    held = deepcopy(records[1])
    held.update(role='test', group='development_test', y=[.2]*6)
    rejected = deepcopy(records[1])
    rejected.update(accepted=False, group='rejected')
    rejected.pop('y')
    model0 = LocalResidualSurrogate.fit(records, db, mission, baseline)
    model1 = LocalResidualSurrogate.fit(records+[held, rejected], db, mission, baseline)
    np.testing.assert_array_equal(model0.heads['post'].residuals, model1.heads['post'].residuals)
    report = model1.evaluate([held])
    assert report['model_schema'] == MODEL_SCHEMA
    assert not report['full_qualification_passed']
    assert report['records'][0]['role'] == 'test'
    assert len(report['records'][0]['local_neighbours']) == 8


def test_group_partition_and_old_label_contract_still_rejected(inputs):
    db, mission, baseline, records = inputs
    held = deepcopy(records[1])
    held['role'] = 'test'
    with pytest.raises(ValueError, match='crosses'):
        LocalResidualSurrogate.fit(records+[held], db, mission, baseline)
    records[0]['protocol'] = 'old-zoh'
    with pytest.raises(ValueError, match='incompatible'):
        LocalResidualSurrogate.fit(records, db, mission, baseline)


def test_local_save_load_round_trip_keeps_evidence_and_hashes(inputs, tmp_path):
    db, mission, baseline, records = inputs
    model = LocalResidualSurrogate.fit(records, db, mission, baseline)
    path = tmp_path/'local.npz'
    model.save(path)
    loaded = LocalResidualSurrogate.load(path, db, mission, baseline)
    for head in ('pre', 'post'):
        value0, diag0 = model.predict(records[0]['context'], head)
        value1, diag1 = loaded.predict(records[0]['context'], head)
        np.testing.assert_array_equal(value0, value1)
        assert diag0 == diag1
    assert len(loaded.metadata['model_sha256']) == 64
    assert len(loaded.metadata['labels_sha256']) == 64
    assert loaded.metadata['model_schema'] == MODEL_SCHEMA
    with pytest.raises(FileExistsError):
        model.save(path)
    with np.load(path, allow_pickle=False) as arrays:
        assert 'post_residuals' in arrays and 'post_x' in arrays
        assert 'post_weights' not in arrays and 'post_mean' not in arrays


def test_local_schema_rejects_old_continuous_weights_and_reverse(inputs, tmp_path):
    db, mission, baseline, records = inputs
    old_path, new_path = tmp_path/'old.npz', tmp_path/'local.npz'
    ContinuousSurrogate.fit(records, db, mission, baseline).save(old_path)
    LocalResidualSurrogate.fit(records, db, mission, baseline).save(new_path)
    with pytest.raises(ValueError, match='model_schema'):
        LocalResidualSurrogate.load(old_path, db, mission, baseline)
    with pytest.raises(ValueError, match='model_schema'):
        ContinuousSurrogate.load(new_path, db, mission, baseline)


def test_saved_data_hash_and_configuration_are_checked(inputs, tmp_path):
    db, mission, baseline, records = inputs
    model = LocalResidualSurrogate.fit(records, db, mission, baseline)
    path = tmp_path/'local.npz'
    model.save(path)
    with pytest.raises(ValueError, match='configuration'):
        LocalResidualSurrogate.load(path, db, mission, baseline, {'ridge': .002})
    with path.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        LocalResidualSurrogate.load(path, db, mission, baseline)


@pytest.mark.parametrize('config', [{'max_neighbor_distance': .251},
                                    {'local_baseline_prior_weight': 0.},
                                    {'local_baseline_prior_weight': .1}])
def test_support_cannot_be_loosened_and_prior_not_tuned(inputs, config):
    db, mission, baseline, _ = inputs
    with pytest.raises(ValueError):
        LocalResidualSurrogate(db, mission, baseline, config)
