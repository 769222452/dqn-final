"""Training bookkeeping regressions; flights and network fitting are mocked."""
import json
from pathlib import Path

import pytest

from rollout_improvement import training


@pytest.fixture
def stable_runtime(monkeypatch):
    monkeypatch.setattr(training, 'experiment_identity', lambda: {'source': 'abc', 'inputs': '123'})


def label(action=7, *, alpha=0., t=0., advantage=1e-4):
    return dict(features=[0., 0., 0., alpha, t, 0.], action=action, base_action=8,
                advantage=advantage, t=t*1200., v=0)


def test_exact_seen_split_uses_only_actual_model_inputs(monkeypatch):
    inputs = [0, 1, 2, 4]
    old = label(alpha=.1)
    seen = label(alpha=.9)  # different physical history, identical observation for this model
    unseen = label(action=9, alpha=.9)
    calls = []

    def check(members, rows, min_advantage, requested_inputs):
        calls.append([r['action'] for r in rows])
        assert requested_inputs == inputs
        return dict(pairs=len(rows), r2_clipped=.25) if rows else None

    monkeypatch.setattr(training, 'prediction_check', check)
    st = dict(check_members=['mock'], check_inputs=inputs, chain=[], check_chain_len=0,
              check_pair_signatures=[training.pair_signature(old, inputs)])
    check_new, split = training.previous_prediction_checks(st, [seen, unseen, label(action=8)], 1e-5)
    assert split['seen_pairs'] == split['unseen_pairs'] == 1
    assert calls == [[7], [9]]
    assert check_new['pairs'] == 1 and split['seen']['pairs'] == 1
    assert training.pair_signature(old, None) != training.pair_signature(seen, None)


def test_repeated_pairs_have_no_unseen_score(monkeypatch):
    old = label()
    monkeypatch.setattr(training, 'prediction_check',
                        lambda members, rows, *_: {'pairs': len(rows)} if rows else None)
    st = dict(check_members=['mock'], check_inputs=None, chain=[], check_chain_len=0,
              check_pair_signatures=[training.pair_signature(old, None)])
    check_new, split = training.previous_prediction_checks(st, [dict(old, advantage=-1e-4)], 1e-5)
    assert check_new is None and split['unseen_pairs'] == 0
    assert split['seen_pairs'] == 1 and split['seen']['pairs'] == 1


def test_missing_fit_history_is_unclassified_not_unseen():
    check_new, split = training.previous_prediction_checks(dict(check_members=['mock']), [label()], 1e-5)
    assert check_new is None and split['unclassified_pairs'] == 1 and split['unseen_pairs'] == 0


def test_new_run_is_bound_before_any_flight(tmp_path, stable_runtime):
    st = training.train(tmp_path/'run', iterations=0, log=lambda _: None)
    saved = json.loads((tmp_path/'run/state.json').read_text())
    assert saved == st
    assert saved['training_protocol'] == training.TRAINING_PROTOCOL
    assert saved['experiment_binding']['payload']['settings']['seed'] == 17
    assert saved['stopped']['category'] == 'budget_limit'
    assert saved['stopped']['convergence_proven'] is False
    assert saved['settings'][-1]['keep_evidence'] is True


@pytest.mark.parametrize('changed', [
    {'seed': 19}, {'states': 20}, {'stride': 24.}, {'window': 24.}, {'candidates': 5},
    {'patience': 2}, {'min_advantage': 2e-5}, {'continuation': 'closed-loop'}, {'inputs': 'all'},
    {'overrides': True}, {'initial_cl': .02}, {'label_decay': .25}, {'label_keep': 3}, {'hidden': (32, 16)},
])
def test_incompatible_resume_rejected_before_output_mutation(tmp_path, stable_runtime, changed):
    out = tmp_path/'run'
    training.train(out, iterations=0, log=lambda _: None)
    before = (out/'state.json').read_bytes()
    leftover = out/'iter_000/proof.json'
    leftover.parent.mkdir()
    leftover.write_text('original evidence')
    with pytest.raises(ValueError, match='resume experiment mismatch'):
        training.train(out, iterations=1, log=lambda _: None, **changed)
    assert (out/'state.json').read_bytes() == before
    assert leftover.read_text() == 'original evidence'
    assert not (out/'interrupted_attempts').exists()


def test_runtime_change_rejects_resume(tmp_path, stable_runtime, monkeypatch):
    training.train(tmp_path, iterations=0, log=lambda _: None)
    before = (tmp_path/'state.json').read_bytes()
    monkeypatch.setattr(training, 'experiment_identity', lambda: {'source': 'changed', 'inputs': '123'})
    with pytest.raises(ValueError, match='runtime source/input fingerprint'):
        training.train(tmp_path, iterations=0, log=lambda _: None)
    assert (tmp_path/'state.json').read_bytes() == before


def test_unfingerprinted_checkpoint_cannot_be_silently_migrated(tmp_path, stable_runtime):
    path = tmp_path/'state.json'
    path.write_text(json.dumps(dict(seed=17, iteration=10, G=1., chain=[], label_archive=[])))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='no compatible experiment fingerprint'):
        training.train(tmp_path, iterations=0, log=lambda _: None)
    assert path.read_bytes() == before


def test_corrupt_binding_is_rejected(tmp_path, stable_runtime):
    st = training.train(tmp_path, iterations=0, log=lambda _: None)
    st['experiment_binding']['payload']['settings']['seed'] = 88
    training.save_json(tmp_path/'state.json', st)
    with pytest.raises(ValueError, match='fingerprint is invalid'):
        training.train(tmp_path, iterations=0, log=lambda _: None)


def test_nonempty_unbound_directory_is_preserved(tmp_path, stable_runtime):
    (tmp_path/'proof.json').write_text('evidence')
    with pytest.raises(ValueError, match='nonempty without state.json'):
        training.train(tmp_path, iterations=0, log=lambda _: None)
    assert (tmp_path/'proof.json').read_text() == 'evidence'
    assert not (tmp_path/'state.json').exists()


def test_worker_count_and_invocation_budget_are_recorded_separately(tmp_path, stable_runtime):
    first = training.train(tmp_path, iterations=0, workers=1, log=lambda _: None)
    second = training.train(tmp_path, iterations=0, workers=2, keep_evidence=False, log=lambda _: None)
    assert first['experiment_binding'] == second['experiment_binding']
    assert [s['workers'] for s in second['settings']] == [1, 2]
    assert second['settings'][-1]['keep_evidence'] is False


def test_stop_categories_do_not_claim_convergence():
    clean = dict(accepted=False, positive_pairs=0, invalid=0)
    busy = dict(accepted=False, positive_pairs=1, invalid=1)
    stationary = training.stop_status([clean]*8, 4)
    stalled = training.stop_status([busy]*12, 4)
    assert stationary['category'] == 'local_search_stationary'
    assert stalled['category'] == 'stalled'
    assert stationary['convergence_proven'] is stalled['convergence_proven'] is False
    assert training.stop_status([busy]*11, 4) is None


@pytest.fixture
def fake_training(monkeypatch, stable_runtime):
    fit_labels = []

    def jobs(jobs, workers, costs=None):
        out = []
        for kind, args in jobs:
            folder = Path(args[0])
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'proof.json').write_text('mock evidence')
            if kind == 'candidate':
                out.append(dict(success=True, G=1., range_km=2000., stop='DONE', switches=2,
                                schedule=[[0., 8]], geometry_switches=2, cl_action_changes=0))
            else:
                t = args[2][0]
                row = label(t=t/1200.)
                invalid = dict(row, action=9, valid=False, stop='NEXT_STATE_NOT_CERTIFIED',
                               G_rest=0., outcome='RUNNING')
                good = dict(row, valid=True, G_rest=1., stop='DONE', outcome='SUCCESS')
                out.append(dict(rows=[good, invalid], prefix={str(t): [0., 8, row['features']]}))
        return out

    def fit(labels, seed, weights, inputs, hidden):
        fit_labels.append([dict(row) for row in labels])
        return [[([[0.]*19 for _ in inputs], [0.]*19)]], {}

    monkeypatch.setattr(training, 'run_jobs', jobs)
    monkeypatch.setattr(training, 'fit_ensemble', fit)
    return fit_labels


def test_explicit_continue_runs_only_requested_budget_and_keeps_failure_evidence(tmp_path, fake_training):
    options = dict(states=1, stride=12., patience=1, workers=1, log=lambda _: None)
    st = training.train(tmp_path, iterations=5, **options)
    assert st['iteration'] == 3 and st['stopped']['category'] == 'stalled'
    assert 'converged' not in st
    assert all(row['invalid_stops'] == {'NEXT_STATE_NOT_CERTIFIED': 1} for row in st['history'])
    assert all(x['action'] != 9 for labels in fake_training for x in labels)
    assert (tmp_path/'iter_000/branch_t0000/proof.json').exists()
    assert (tmp_path/'iter_000/candidate_m0.25/proof.json').exists()
    before = (tmp_path/'state.json').read_bytes()
    unchanged = training.train(tmp_path, iterations=10, **options)
    assert unchanged['iteration'] == 3 and (tmp_path/'state.json').read_bytes() == before
    resumed = training.train(tmp_path, iterations=1, use_stop_rule=False, **options)
    assert resumed['iteration'] == 4 and resumed['stopped']['category'] == 'budget_limit'
    assert resumed['stop_history'][-1]['category'] == 'stalled'
    assert resumed['stop_history'][-1]['ignored_rule'] is True
    assert resumed['history'][-1]['prediction_check'] is None
    assert resumed['history'][-1]['prediction_diagnostics']['seen_pairs'] == 1


def test_interrupted_attempt_is_archived_without_deleting_evidence(tmp_path):
    idir = tmp_path/'iter_000'
    idir.mkdir()
    (idir/'proof.json').write_text('proof 1')
    training.prepare_iteration_directory(idir, True, lambda _: None)
    assert not idir.exists()
    assert (tmp_path/'interrupted_attempts/iter_000_attempt_001/proof.json').read_text() == 'proof 1'
    idir.mkdir()
    (idir/'proof.json').write_text('proof 2')
    training.prepare_iteration_directory(idir, True, lambda _: None)
    assert (tmp_path/'interrupted_attempts/iter_000_attempt_002/proof.json').read_text() == 'proof 2'


@pytest.mark.parametrize('all_failed', [False, True])
def test_candidate_infrastructure_error_does_not_advance_or_select_subset(
        tmp_path, fake_training, monkeypatch, all_failed):
    options = dict(states=1, stride=12., patience=1, workers=1, log=lambda _: None)
    completed = training.train(tmp_path, iterations=1, **options)
    assert completed['iteration'] == 1
    dispatch = training.run_jobs

    def failed_candidates(jobs, workers, costs=None):
        results = dispatch(jobs, workers, costs)
        if all(kind == 'candidate' for kind, _ in jobs):
            results[0] = dict(error='OSError: full disk while writing candidate evidence')
            if all_failed:
                results = [dict(error=f'worker failure {i}') for i in range(len(results))]
            else:
                results[1]['G'] = 2.  # even a better survivor cannot justify partial-population acceptance
        return results

    monkeypatch.setattr(training, 'run_jobs', failed_candidates)
    with pytest.raises(RuntimeError, match='iteration 1 is incomplete'):
        training.train(tmp_path, iterations=5, **options)
    saved = json.loads((tmp_path/'state.json').read_text())
    assert saved['iteration'] == 1 and saved['history'] == completed['history']
    assert saved['G'] == completed['G'] and saved['chain'] == completed['chain']
    assert saved['label_archive'] == completed['label_archive']
    assert saved.get('stopped', {}).get('category') not in ('local_search_stationary', 'stalled')
    evidence = json.loads((tmp_path/'iter_001/candidate_results.json').read_text())
    assert evidence['complete'] is False and 'error' in evidence['results'][0]
    assert len(evidence['specs']) == len(evidence['results']) == 6
    assert (tmp_path/'iter_001/candidate_m0.25/proof.json').exists()
    assert (tmp_path/'iter_001/labels.json').exists()


def test_resolved_evidence_path_must_remain_in_run_directory(tmp_path, monkeypatch):
    run = tmp_path/'run'
    idir = run/'iter_000'
    idir.mkdir(parents=True)
    (idir/'proof.json').write_text('original')
    original_resolve = Path.resolve

    def escaped_resolve(path, *args, **kwargs):
        # Model a directory junction/symlink resolving outside the intended run.
        return tmp_path/'outside' if path == idir else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', escaped_resolve)
    with pytest.raises(ValueError, match='escapes its intended run directory'):
        training.prepare_iteration_directory(idir, False, lambda _: None)
    assert (idir/'proof.json').read_text() == 'original'
    with pytest.raises(ValueError, match='escapes its intended run directory'):
        training.prepare_iteration_directory(idir, True, lambda _: None)
    assert not (run/'interrupted_attempts').exists()


def test_prune_checks_all_targets_before_removing_any(tmp_path, monkeypatch):
    idir = tmp_path/'iter_000'
    normal, escaping = idir/'branch_t0000', idir/'candidate_m0.25'
    normal.mkdir(parents=True)
    escaping.mkdir()
    (normal/'proof.json').write_text('proof')
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, 'resolve', lambda path, *args, **kwargs:
                        tmp_path/'outside' if path == escaping else original_resolve(path, *args, **kwargs))
    with pytest.raises(ValueError, match='escapes its intended run directory'):
        training.prune(idir, None)
    assert (normal/'proof.json').read_text() == 'proof' and escaping.exists()


def test_explicit_prune_preserves_candidate_summary(tmp_path):
    idir = tmp_path/'iter_000'
    (idir/'candidate_m0.25').mkdir(parents=True)
    (idir/'branch_t0000').mkdir()
    summary = idir/'candidate_results.json'
    summary.write_text('{"complete": true}')
    training.prune(idir, None)
    assert summary.exists() and list(idir.iterdir()) == [summary]


@pytest.mark.parametrize('flags,expected', [([], True), (['--keep-evidence'], True),
                                          (['--discard-rejected-evidence'], False)])
def test_cli_requires_explicit_evidence_discard(monkeypatch, flags, expected):
    recorded = []
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: recorded.append(args))
    assert training.main(['--out', 'unused']+flags) == 0
    assert recorded[0][17] is expected
