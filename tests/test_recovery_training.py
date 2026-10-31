"""New-controller training bookkeeping, with all flights and fitting mocked."""
import json
from pathlib import Path

import pytest

from continuous_recovery import training


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    identity = dict(sha256='stable-runtime', sources={}, inputs={'physics': 'original'},
                    software={'numpy': 'fixed'})
    monkeypatch.setattr(training, 'experiment_identity', lambda: identity)
    config = tmp_path/'controller.json'
    config.write_text(json.dumps({'controller_protocol': training.CONTROLLER_PROTOCOL}))
    model = tmp_path/'model.npz'
    model.write_bytes(b'fixture model: flights are mocked')
    metadata = dict(protocol=training.TEACHER_PROTOCOL, model_schema=training.MODEL_SCHEMA,
                    feature_schema=training.FEATURE_SCHEMA, output_schema=training.OUTPUT_SCHEMA,
                    model_sha256=training.integrity.sha256(model))
    training.save_json(model.with_suffix('.json'), metadata)
    return dict(out=tmp_path/'run', model=model, config=config, log=lambda _: None)


def test_fresh_warm_start_imports_only_chain_and_no_legacy_learning(setup, tmp_path):
    source = tmp_path/'old_policy.json'
    chain = [training.constant_stage(.02)]
    training.save_json(source, dict(chain=chain, G=999., iteration=99, schedule=[[0., 18]],
        labels_current=[{'forbidden': True}], label_archive=[{'forbidden': True}],
        history=[{'accepted': True}], stopped={'category': 'local_search_stationary'}))
    st = training.train(**setup, initial_policy=source, iterations=0)
    assert st['chain'] == chain and st['G'] is st['schedule'] is st['nominal'] is None
    assert st['iteration'] == st['policy_version'] == 0
    assert st['labels_current'] == st['label_archive'] == st['history'] == []
    assert st['stopped']['category'] == 'budget_limit'
    snap = training.load_json(Path(st['runtime_spec']['initial_policy']))
    assert snap == {'chain': chain}
    provenance = training.load_json(setup['out']/'provenance.json')
    assert provenance['initial_policy']['imported_fields'] == ['chain']
    assert provenance['initial_policy']['migrated_returns'] == 0


def test_default_constant_and_exact_resume_without_original_policy(setup):
    first = training.train(**setup, iterations=0)
    second = training.train(**setup, iterations=0)
    assert first['chain'] == [training.constant_stage(.025)]
    assert first['experiment_binding'] == second['experiment_binding']
    assert second['settings'][-1]['workers'] == 1
    assert second['settings'][-1]['keep_evidence'] is True
    assert second['runtime_spec']['training_protocol'] == training.TRAINING_PROTOCOL


@pytest.mark.parametrize('changed', ['config', 'model', 'metadata', 'runtime', 'settings'])
def test_changed_runtime_or_inputs_reject_resume_without_mutation(setup, monkeypatch, changed):
    training.train(**setup, iterations=0)
    before = (setup['out']/'state.json').read_bytes()
    options = dict(setup)
    if changed == 'config':
        cfg = training.load_json(setup['config'])
        training.save_json(setup['config'], dict(cfg, new_option=True))
    elif changed in ('model', 'metadata'):
        meta_path = setup['model'].with_suffix('.json')
        meta = training.load_json(meta_path)
        if changed == 'model':
            setup['model'].write_bytes(b'new model')
            meta['model_sha256'] = training.integrity.sha256(setup['model'])
        else:
            meta['new_metadata'] = True
        training.save_json(meta_path, meta)
    elif changed == 'runtime':
        monkeypatch.setattr(training, 'experiment_identity', lambda: dict(sha256='new-runtime', sources={}))
    else:
        options['label_decay'] = .25
    with pytest.raises(ValueError, match='resume experiment mismatch'):
        training.train(**options, iterations=0)
    assert (setup['out']/'state.json').read_bytes() == before


@pytest.mark.parametrize('field', ['config', 'model', 'initial_policy'])
def test_corrupt_input_snapshot_refuses_resume(setup, field):
    st = training.train(**setup, iterations=0)
    before = (setup['out']/'state.json').read_bytes()
    Path(st['runtime_spec'][field]).write_text('corrupt')
    with pytest.raises(ValueError, match='snapshot changed'):
        training.train(**setup, iterations=0)
    assert (setup['out']/'state.json').read_bytes() == before


def test_legacy_checkpoint_rejected_without_migration(setup):
    setup['out'].mkdir()
    training.save_json(setup['out']/'state.json', dict(seed=17, chain=[], G=2., iteration=10))
    before = (setup['out']/'state.json').read_bytes()
    with pytest.raises(ValueError, match='no compatible recovery'):
        training.train(**setup, iterations=0)
    assert (setup['out']/'state.json').read_bytes() == before


def test_runtime_spec_cannot_bypass_checkpoint_binding(setup):
    st = training.train(**setup, iterations=0)
    st['runtime_spec']['source_sha256'] = 'another source'
    training.save_json(setup['out']/'state.json', st)
    with pytest.raises(ValueError, match='runtime specification disagrees'):
        training.train(**setup, iterations=0)


@pytest.mark.parametrize('bad_policy', ['missing', 'malformed'])
def test_bad_initial_policy_creates_no_output(setup, tmp_path, bad_policy):
    path = tmp_path/'policy.json'
    if bad_policy == 'malformed':
        training.save_json(path, {'chain': [{'kind': 'constant', 'action': 19}]})
    with pytest.raises((FileNotFoundError, ValueError)):
        training.train(**setup, iterations=0, initial_policy=path)
    assert not setup['out'].exists()


def test_iterations_is_explicit_and_zero_never_dispatches(setup, monkeypatch):
    monkeypatch.setattr(training, 'run_jobs', lambda *_args, **_kwargs: pytest.fail('unexpected flight'))
    training.train(**setup, iterations=0)
    with pytest.raises(SystemExit):
        training.main(['--out', str(setup['out']), '--model', str(setup['model'])])


def complete(G=1.):
    return dict(success=True, completed_1200s=True, time_s=1200., G=G, range_km=2160.,
                stop='SUCCESS', switches=2, schedule=[[0., 5]], decision_prefix={'0.0': {'bound': True}})


@pytest.fixture
def fake_training(monkeypatch):
    calls = []

    def jobs(jobs, workers, costs=None):
        results = []
        for kind, args in jobs:
            calls.append((kind, args))
            folder = Path(args['folder'])
            folder.mkdir(parents=True)
            (folder/'proof.json').write_text('mock evidence')
            if kind == 'candidate':
                results.append(dict(complete(), chain_sha256=training.digest_json(args['chain']),
                    runtime_spec_sha256=training.digest_json(args['runtime_spec'])))
            else:
                assert training.complete_candidate(args['nominal'])
                t = args['times'][0]
                f = [0., 0., 0., 0., t/1200., 0.]
                row = dict(t=t, features=f, action=6, base_action=5, valid=True,
                           G_rest=1., advantage=0., stop='SUCCESS')
                results.append(dict(rows=[row], prefix={str(t): [0., 5, f]}))
        return results

    def fit(labels, seed, weights, inputs, hidden):
        return [[([[0.]*19 for _ in inputs], [0.]*19)]], {'mock': True}

    monkeypatch.setattr(training, 'run_jobs', jobs)
    monkeypatch.setattr(training, 'fit_ensemble', fit)
    return calls, jobs


def test_candidate_success_requires_full1200():
    assert training.complete_candidate(complete())
    assert not training.complete_candidate(dict(complete(), time_s=12.))
    assert not training.complete_candidate(dict(complete(), completed_1200s=False))
    assert not training.complete_candidate(dict(complete(), success=False))
    assert not training.complete_candidate(dict(complete(), G=float('nan')))


def test_new_controller_nominal_and_labels_resume_with_same_identity(setup, fake_training):
    first = training.train(**setup, iterations=1, states=1)
    resumed = training.train(**setup, iterations=0, states=1)
    assert resumed['nominal'] == first['nominal']
    assert resumed['G'] == first['G'] == 1.
    assert resumed['iteration'] == 1
    assert resumed['labels_current'] == first['labels_current']


def test_failed_candidate_is_never_accepted_or_clean_stop(setup, fake_training, monkeypatch):
    _, jobs = fake_training
    def failed(batch, workers, costs=None):
        results = jobs(batch, workers, costs)
        for (kind, args), result in zip(batch, results):
            if kind == 'candidate' and Path(args['folder']).name != 'initial':
                result.update(success=False, completed_1200s=False, time_s=600., G=999., stop='UNKNOWN')
        return results
    monkeypatch.setattr(training, 'run_jobs', failed)
    st = training.train(**setup, iterations=2, states=1, stride=12., patience=1)
    assert st['G'] == 1. and st['policy_version'] == 0 and st['iteration'] == 2
    assert all(not x['accepted'] and x['candidate_invalid'] > 0 for x in st['history'])
    assert st['stopped']['category'] == 'budget_limit'
    assert (setup['out']/'iter_000/candidate_m0.25/proof.json').exists()


def test_initial_truncation_keeps_empty_learning_state_and_evidence(setup, monkeypatch):
    monkeypatch.setattr(training, 'run_jobs', lambda *_args, **_kwargs:
        [dict(complete(), completed_1200s=False, success=False, time_s=300.)])
    with pytest.raises(RuntimeError, match='did not complete 1200'):
        training.train(**setup, iterations=1)
    st = training.load_json(setup['out']/'state.json')
    assert st['iteration'] == 0 and st['G'] is None and not st['history']
    assert (setup['out']/'iter_000/initial_result.json').exists()


def test_candidate_infrastructure_error_keeps_iteration_uncommitted(setup, fake_training, monkeypatch):
    _, jobs = fake_training
    def failed(batch, workers, costs=None):
        results = jobs(batch, workers, costs)
        if Path(batch[0][1]['folder']).name.startswith('candidate_'):
            results[0] = {'error': 'disk full'}
            results[1] = complete(G=999.)
        return results
    monkeypatch.setattr(training, 'run_jobs', failed)
    with pytest.raises(RuntimeError, match='incomplete'):
        training.train(**setup, iterations=1, states=1)
    st = training.load_json(setup['out']/'state.json')
    assert st['iteration'] == 0 and st['G'] == 1. and not st['labels_current'] and not st['history']
    assert 'stopped' not in st
    assert (setup['out']/'iter_000/candidate_results.json').exists()


def test_prefix_mismatch_keeps_evidence_and_no_labels(setup, fake_training, monkeypatch):
    _, jobs = fake_training
    def failed(batch, workers, costs=None):
        if batch[0][0] == 'branch':
            return [{'error': 'nominal prefix binding mismatch'}]
        return jobs(batch, workers, costs)
    monkeypatch.setattr(training, 'run_jobs', failed)
    with pytest.raises(RuntimeError, match='prefix binding mismatch'):
        training.train(**setup, iterations=1, states=1)
    st = training.load_json(setup['out']/'state.json')
    assert st['iteration'] == 0 and not st['label_archive']
    assert (setup['out']/'iter_000/branch_results.json').exists()
