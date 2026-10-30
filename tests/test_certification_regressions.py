"""Terminal/result/evidence regressions without OCP solves or flight simulation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from certified_flight.certification import AUDIT_REJECTIONS, Certifier, NO_SOLUTION, REJECT, UNKNOWN, classification
from certified_flight.oracle_evidence import oracle_sinks
from rollout_improvement import rollouts


def fake_env(t=0.):
    return SimpleNamespace(t=t, x=np.array([30000., 2000., 0., 500000.]), alpha=.05,
                           previous_cl_raw=.04, history_valid=True, done=False,
                           last_outcome='INCOMPLETE', switches=0)


class FakeCertifier:
    def __init__(self, env, outcome='SUCCESS', stop_after=2, unavailable_after=None):
        self.env = env
        self.outcome, self.stop_after = outcome, stop_after
        self.unavailable_after = unavailable_after
        self.commits = 0

    def pool(self, preference):
        mask = np.zeros(19, bool)
        if self.unavailable_after is None or self.commits < self.unavailable_after:
            mask[int(np.argmax(preference))] = True
        return {'mask': mask}

    def commit(self, pool, action):
        assert pool['mask'][action]
        self.commits += 1
        self.env.t += 3.
        self.env.switches += int(self.commits == 1)
        if self.commits == self.stop_after:
            self.env.done = True
            self.env.last_outcome = self.outcome
        return .1, self.env.done, {}

    def fork(self, folder):
        return FakeCertifier(deepcopy(self.env), self.outcome, self.stop_after, self.unavailable_after)


@pytest.mark.parametrize('outcome, expected', [
    ('SUCCESS', 'DONE'), ('CONSTRAINT_VIOLATION', 'CONSTRAINT_VIOLATION'),
    ('NUMERICAL_FAILURE', 'NUMERICAL_FAILURE'), ('EXTERNAL_INTERRUPT', 'EXTERNAL_INTERRUPT'),
])
def test_terminal_outcomes_do_not_alias_success(outcome, expected):
    env = fake_env(1197.)
    certifier = FakeCertifier(env, outcome=outcome, stop_after=1)
    policy = rollouts.ChainPolicy([], env)
    reward, stop, acts = rollouts.certified_fly(certifier, policy)
    assert (reward, stop, acts) == (.1, expected, [8])
    assert rollouts.mission_completed(env, stop) is (outcome == 'SUCCESS')
    # Calling on an already terminated flight cannot execute another action.
    assert rollouts.certified_fly(certifier, policy) == (0., expected, [])
    assert certifier.commits == 1


def test_mission_completion_requires_full_duration_and_success():
    env = fake_env(1200.)
    env.last_outcome = 'CONSTRAINT_VIOLATION'
    assert not rollouts.mission_completed(env, 'DONE')
    env.last_outcome = 'SUCCESS'
    assert rollouts.mission_completed(env, 'DONE')
    env.t = 3.
    assert not rollouts.mission_completed(env, 'DONE')
    env.t = 1203.
    assert not rollouts.mission_completed(env, 'DONE')


@pytest.mark.parametrize('outcome', ['SUCCESS', 'CONSTRAINT_VIOLATION'])
def test_candidate_counts_geometry_events_separately(monkeypatch, tmp_path, outcome):
    env = fake_env(1194.)
    certifier = FakeCertifier(env, outcome=outcome)
    monkeypatch.setattr(rollouts, 'build_flight', lambda *args, **kwargs: (env, certifier))
    result = rollouts.candidate_job((tmp_path, []))
    assert 'error' not in result
    assert result['switches'] == result['geometry_switches'] == 1
    assert result['cl_action_changes'] == 0
    assert result['action_counts'][8] == result['steps'] == 2
    assert result['outcome'] == outcome and result['time_s'] == 1200.
    assert result['completed'] is result['success'] is (outcome == 'SUCCESS')


def test_failed_override_keeps_executed_steps_and_final_state(monkeypatch, tmp_path):
    env = fake_env()
    certifier = FakeCertifier(env, stop_after=100, unavailable_after=1)
    monkeypatch.setattr(rollouts, 'build_flight', lambda *args, **kwargs: (env, certifier))
    result = rollouts.branch_job((tmp_path, [], [0.], 0, None))
    assert 'error' not in result
    assert result['steps'] == 2 and len(result['rows']) == 2
    for row in result['rows']:
        assert row['steps'] == 1 and row['executed'] == [row['action']]
        assert not row['valid'] and not row['completed']
        assert row['stop'] == 'NEXT_STATE_NOT_CERTIFIED'
        assert row['time_s'] == 3. and row['outcome'] == 'INCOMPLETE'
        assert row['G_rest'] == .1 and row['final_state'] == env.x.tolist()


def test_early_success_outcome_cannot_become_valid_branch(monkeypatch, tmp_path):
    env = fake_env()
    certifier = FakeCertifier(env, stop_after=1)
    monkeypatch.setattr(rollouts, 'build_flight', lambda *args, **kwargs: (env, certifier))
    result = rollouts.branch_job((tmp_path, [], [0.], 0, None))
    assert result['steps'] == 2
    assert all(row['stop'] == 'DONE' and not row['valid'] for row in result['rows'])


def test_fork_rebinds_oracle_events_and_artifacts(tmp_path):
    base = tmp_path/'base'
    event_sink, artifact_sink = oracle_sinks(base)
    plant, db = object(), object()
    env = SimpleNamespace(ocp=SimpleNamespace(plant=plant), db=db,
                          event_sink=event_sink, artifact_sink=artifact_sink)
    certifier = Certifier(env, base/'proofs', {})
    left = certifier.fork(tmp_path/'left')
    right = certifier.fork(tmp_path/'right')
    nested = left.fork(tmp_path/'nested')
    for name, current in [('base', certifier), ('left', left), ('right', right), ('nested', nested)]:
        marker = float(len(name))
        event = dict(attempt=1, flight=name)
        plan = SimpleNamespace(alpha=np.array([marker]), times=np.array([0., 1.]),
                               x=np.full((2, 4), marker), dual={})
        arrays = {0.1: dict(x=plan.x, samples=plan.x, alpha=plan.alpha, times=plan.times)}
        current.env.event_sink(event)
        current.env.artifact_sink(event, 'post', plan, arrays)
        assert current.env.ocp.plant is plant and current.env.db is db
    for name in ('base', 'left', 'right', 'nested'):
        folder = tmp_path/name/'oracle'/'a000001'
        assert json.loads((folder/'event.json').read_text())['flight'] == name
        with np.load(folder/'post'/'solver_plan.npz') as plan:
            assert plan['alpha'].tolist() == [float(len(name))]
        assert (folder/'post'/'audit_dt0.1.npz').is_file()


def test_fork_does_not_reuse_existing_proof_number(tmp_path):
    event_sink, artifact_sink = oracle_sinks(tmp_path/'base')
    env = SimpleNamespace(ocp=SimpleNamespace(plant=object()), db=object(),
                          event_sink=event_sink, artifact_sink=artifact_sink)
    certifier = Certifier(env, tmp_path/'base'/'proofs', {})
    target = tmp_path/'branch'/'proofs'
    target.mkdir(parents=True)
    (target/'p000005.json').write_text('{}', encoding='utf-8')
    assert certifier.fork(target.parent).probes == 5


@pytest.mark.parametrize('reason', sorted(AUDIT_REJECTIONS))
def test_only_explicit_failed_audits_are_rejected(reason):
    assert classification(reason) == REJECT


@pytest.mark.parametrize('reason', [
    'TRUE_OCP_OR_AUDIT_EXCEPTION', 'AUDIT_FILE_PERMISSION_ERROR',
    'RK_STAGE_UNKNOWN_EXCEPTION', 'SEARCH_BUDGET_EXHAUSTED', 'NOT_CHECKED',
])
def test_infrastructure_errors_remain_unknown(reason):
    assert classification(reason) == UNKNOWN


def test_solver_no_solution_remains_distinct_from_physical_rejection():
    assert classification('TRUE_OCP_SOLVE_FAILED') == NO_SOLUTION
