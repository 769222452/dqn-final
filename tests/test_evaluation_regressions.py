"""Evaluation cache/provenance and incomplete-test regressions without flight solves."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rollout_improvement import evaluation as ev

RUNTIME = {'schema': 'test-runtime/1', 'sha256': 'verified-inputs-and-code'}


class FakePolicy:
    def __init__(self, chain, env, **kwargs):
        self.decisions = [(0., 8)]


def initial_stubs(monkeypatch, fail_first=False):
    calls = []

    def build(folder, kind, **kwargs):
        calls.append(Path(folder))
        Path(folder).mkdir(parents=True)
        (Path(folder)/'proof.json').write_text('{"retained": true}', encoding='utf-8')
        if fail_first and len(calls) == 1:
            raise OSError('test failed attempt')
        env = SimpleNamespace(t=1200., x=np.array([30000., 1500., 0., 2.4e6]), switches=1,
                              last_outcome='SUCCESS')
        return env, object()

    monkeypatch.setattr(ev, 'build_flight', build)
    monkeypatch.setattr(ev, 'ChainPolicy', FakePolicy)
    monkeypatch.setattr(ev, 'certified_fly', lambda *a, **kw: (1., 'DONE', [8, 8]))
    return calls


def test_cache_identifies_policy_state_schedule_runtime_and_schema(tmp_path, monkeypatch):
    calls = initial_stubs(monkeypatch)
    request = ('initial', [], (0., 0.), None, str(tmp_path/'case'), RUNTIME)
    result = ev._job(request)
    assert result['switches'] == 1 and result['cl_action_changes'] == 0
    assert result['geometry_switches'] == 1 and result['steps'] == 2 and result['time_s'] == 1200.
    assert result['final_state'][-1] == 2.4e6
    assert result['completed'] and result['outcome'] == 'SUCCESS'
    assert ev._job(request) == result and len(calls) == 1
    original = (tmp_path/'case'/'result.json').read_bytes()
    alternatives = [
        ('initial', [{'kind': 'constant', 'action': 18}], (0., 0.), None, request[4], RUNTIME),
        ('initial', [], (500., 100.), None, request[4], RUNTIME),
        ('initial', [], (0., 0.), {0: 18}, request[4], RUNTIME),
        (*request[:5], {'sha256': 'other-runtime'}),
    ]
    for changed in alternatives:
        with pytest.raises(ev.EvaluationCacheMismatch):
            ev._job(changed)
        assert (tmp_path/'case'/'result.json').read_bytes() == original
    assert (tmp_path/'case'/'run'/'proof.json').exists()


def test_unversioned_result_is_never_reused_or_overwritten(tmp_path):
    done = tmp_path/'result.json'
    done.write_text('{"range_km": 9999}', encoding='utf-8')
    original = done.read_bytes()
    with pytest.raises(ev.EvaluationCacheMismatch):
        ev._job(('initial', [], (0., 0.), None, str(tmp_path), RUNTIME))
    assert done.read_bytes() == original


def test_error_attempt_retries_and_preserves_prior_result_and_proofs(tmp_path, monkeypatch):
    calls = initial_stubs(monkeypatch, fail_first=True)
    request = ('initial', [], (0., 0.), None, str(tmp_path/'case'), RUNTIME)
    first = ev._job(request)
    assert 'test failed attempt' in first['error']
    second = ev._job(request)
    assert second['completed'] and second['attempt'] == 2 and len(calls) == 2
    archived = json.loads((tmp_path/'case'/'result.attempt0001.json').read_text())
    assert archived == first
    assert (tmp_path/'case'/'run'/'proof.json').exists()
    assert (tmp_path/'case'/'run_attempt0002'/'proof.json').exists()


def test_direct_legacy_job_computes_identity_once(tmp_path, monkeypatch):
    initial_stubs(monkeypatch)
    calls = []
    monkeypatch.setattr(ev, 'experiment_identity', lambda: calls.append(1) or RUNTIME)
    ev._job(('initial', [], (0., 0.), None, str(tmp_path)))
    assert calls == [1]


def test_explicit_evidence_discard_is_opt_in(tmp_path, monkeypatch):
    initial_stubs(monkeypatch)
    r = ev._job(('initial', [], (0., 0.), None, str(tmp_path), RUNTIME, True))
    assert not r['evidence_retained'] and not (tmp_path/'run').exists()
    assert (tmp_path/'result.json').exists()


def test_probe_never_continues_after_failed_prefix(tmp_path, monkeypatch):
    env = SimpleNamespace(t=3., x=np.array([30000., 1500., 0., 7200.]), switches=1,
                          last_outcome='INCOMPLETE')
    monkeypatch.setattr(ev, 'build_flight', lambda *a, **kw: (env, object()))
    monkeypatch.setattr(ev, 'ChainPolicy', FakePolicy)
    calls = []
    monkeypatch.setattr(ev, 'certified_fly', lambda *a, **kw: calls.append(kw) or
                        (.003, 'NEXT_STATE_NOT_CERTIFIED', [8]))
    monkeypatch.setattr(ev, 'chain_action', lambda *a: pytest.fail('must not request a probe action'))
    r = ev._job(('probe', [], (12., 1, 'closed'), {0: 8, 12: 8}, str(tmp_path), RUNTIME))
    assert len(calls) == 1 and r['failure_reason'] == 'PROBE_PREFIX_NOT_REACHED'
    assert r['stop_time_s'] == 3. and not r['completed'] and not r['valid']


class InlinePool:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, jobs, chunksize=1):
        return [function(job) for job in jobs]


def evaluation_stubs(monkeypatch, *, probe_error=False, initial_error=False, nominal_failed=False):
    monkeypatch.setattr(ev, 'test_set_points', lambda name: ([(0., 0.), (1., 0.)], (12.,)))
    monkeypatch.setattr(ev.mp, 'get_context', lambda name: SimpleNamespace(Pool=lambda n: InlinePool()))
    identity_calls, probe_calls = [], []
    monkeypatch.setattr(ev, 'experiment_identity', lambda: identity_calls.append(1) or RUNTIME)

    def job(args):
        kind, chain, p, schedule, out, runtime, discard = args
        assert runtime is RUNTIME
        if kind == 'initial':
            if initial_error and not chain and p == (1., 0.):
                return dict(kind=kind, p=p, error='initial failed', completed=False)
            success = not (nominal_failed and not chain and p == (0., 0.))
            return dict(kind=kind, dh=p[0], dV=p[1], G=1., range_km=2400., switches=1,
                        cl_action_changes=0, completed=success,
                        outcome='SUCCESS' if success else 'INCOMPLETE',
                        stop='DONE' if success else 'NEXT_STATE_NOT_CERTIFIED', schedule=[[0., 8], [12., 8]])
        probe_calls.append(p)
        if probe_error and p == (12., 1, 'open'):
            return dict(kind=kind, p=p, completed=False, error='probe failed')
        return dict(kind=kind, t=p[0], delta=p[1], loop=p[2], G=1.1, completed=True,
                    valid=True, stop='DONE', redecisions_changed=1)

    monkeypatch.setattr(ev, '_job', job)
    return identity_calls, probe_calls


def test_probe_error_is_counted_and_main_aggregate_withheld(tmp_path, monkeypatch):
    identities, probes = evaluation_stubs(monkeypatch, probe_error=True)
    s = ev.evaluate([], tmp_path, workers=1, log=lambda text: None)
    assert identities == [1] and len(probes) == 4
    assert not s['complete'] and s['coverage']['probes']['error'] == 1
    assert s['probe_requested_pairs'] == 2 and s['probe_valid_pairs'] == 1
    assert len(s['probe_missing_pairs']) == 1 and len(s['job_errors']) == 1
    assert s['probe_mean_A_closed'] is None and s['subset_statistics']['probe_mean_A_closed'] is not None
    assert 'NaN' not in (tmp_path/'summary.json').read_text()


def test_complete_summary_can_reuse_same_request_but_not_another_policy(tmp_path, monkeypatch):
    evaluation_stubs(monkeypatch)
    first = ev.evaluate([], tmp_path, workers=1, log=lambda text: None)
    assert first['complete'] and first['probe_valid_pairs'] == first['probe_requested_pairs'] == 2
    assert first['coverage']['initial'] == dict(requested=2, returned=2, complete=2,
                                               failed=0, skipped=0, error=0, missing=0)
    assert ev.evaluate([], tmp_path, workers=1, log=lambda text: None)['complete']
    original = (tmp_path/'summary.json').read_bytes()
    with pytest.raises(ev.EvaluationCacheMismatch):
        ev.evaluate([{'kind': 'constant', 'action': 18}], tmp_path, workers=1, log=lambda text: None)
    assert (tmp_path/'summary.json').read_bytes() == original


def test_initial_error_is_counted_and_gain_mean_withheld(tmp_path, monkeypatch):
    evaluation_stubs(monkeypatch, initial_error=True)
    s = ev.evaluate([], tmp_path, workers=1, log=lambda text: None)
    assert s['coverage']['initial']['error'] == 1 and len(s['initial_missing']) == 1
    assert s['perturbed_mean_gain_km'] is None and s['perturbed_gain_ci95_km'] is None


def test_failed_nominal_withholds_all_probes(tmp_path, monkeypatch):
    _, probes = evaluation_stubs(monkeypatch, nominal_failed=True)
    s = ev.evaluate([], tmp_path, workers=1, log=lambda text: None)
    assert probes == [] and s['coverage']['probes']['skipped'] == 4
    assert not s['complete'] and s['probe_mean_A_closed'] is None
    assert all(p['reasons'] == ['NOMINAL_MISSION_NOT_COMPLETED']*2 for p in s['probe_missing_pairs'])


def compare_module():
    spec = importlib.util.spec_from_file_location('compare_regression', Path(__file__).parents[1]/'scripts'/'compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_summary():
    return dict(test_set='test', reference_cl=.025, initial_conditions=[
        dict(dh=1., dV=0., range_km=2400., gain_km=1., stop='DONE', completed=True,
             reference_completed=False, reference_stop='CONSTRAINT_VIOLATION')],
        nominal_range_km=2400., nominal_gain_km=1., probes=[], probe_mean_A_closed=None,
        probe_mean_A_open=None, probe_closed_not_worse=0, complete=True,
        comparison_identity={'runtime': RUNTIME, 'schema': ev.EVALUATION_SCHEMA})


def test_comparison_checks_runtime_and_handles_no_reference_successes():
    module = compare_module()
    a, b = comparison_summary(), comparison_summary()
    r = module.compare({'a': a, 'b': b})
    conditional = r['policies']['a']['gain_where_reference_completed_km']
    assert conditional['n'] == 0 and conditional['mean'] is None and conditional['ci95'] is None
    b['comparison_identity'] = {'runtime': {'sha256': 'other'}}
    with pytest.raises(ValueError, match='identities'):
        module.compare({'a': a, 'b': b})
    b = comparison_summary()
    b['complete'] = False
    with pytest.raises(ValueError, match='incomplete'):
        module.compare({'a': a, 'b': b})


def test_empty_bootstrap_is_explicitly_unavailable():
    assert ev.paired_bootstrap_ci([]) is None
