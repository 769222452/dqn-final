"""Fast deterministic controller doubles; no OCP or production flight is run."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from certified_flight import CL_ACTIONS
from continuous_recovery import training_rollouts as rollout
from continuous_surrogate.experiment import digest
from rollout_improvement.policy import ChainPolicy, constant_stage


class FakeEnv:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.x = np.array([40000., 2400., 0., 0.])
        self.t, self.alpha, self.total_reward = 0., .05, 0.
        self.history_valid, self.previous_cl_raw = False, None
        self.done, self.last_outcome, self.switches = False, 'INCOMPLETE', 0
        self.controller = dict(epoch=1, memory=dict(request_CL=.04, nodes=[.05, .06]), exit_passes=1)
        self.transition = dict(source=42, target=12, started=0., duration=8.)
        self.rows = []

    def snapshot(self):
        return dict(x=self.x.tolist(), t=self.t, alpha=self.alpha, reward=self.total_reward,
                    history_valid=self.history_valid, previous_cl_raw=self.previous_cl_raw,
                    done=self.done, outcome=self.last_outcome, switches=self.switches,
                    controller=deepcopy(self.controller), transition=deepcopy(self.transition),
                    identity=digest(self.spec))

    def snapshot_hash(self):
        return digest(self.snapshot())


class FakeCertifier:
    forks = []

    def __init__(self, folder, env):
        self.folder, self.env = Path(folder), env
        self.folder.mkdir(parents=True, exist_ok=True)
        self.policy_state = None
        self.records = []

    def bind_policy_state(self, state):
        self.policy_state = deepcopy(state)

    def pool(self, preference):
        requested = int(np.argmax(preference))
        mask = np.zeros(19, bool)
        failed = (self.env.spec.get('fail_request') == requested
                  and self.env.t >= self.env.spec.get('fail_time', np.inf))
        if not failed:
            mask[requested] = True
        return dict(mask=mask, requested=requested)

    def commit(self, pool, action):
        assert pool['mask'][action]
        assert self.policy_state is not None
        env = self.env
        raw = CL_ACTIONS[action]
        env.x[3] += 1000.+100.*raw
        env.x[0] += raw*.01
        env.t += 3.
        reward = .001+raw*.0001
        env.total_reward += reward
        env.alpha = raw
        env.history_valid, env.previous_cl_raw = True, raw
        env.controller['memory']['request_CL'] = raw
        env.controller['memory']['nodes'][0] = raw
        env.controller['epoch'] += 1
        env.done = env.t >= env.spec.get('end_time', 1200.)
        env.last_outcome = 'SUCCESS' if env.done else 'INCOMPLETE'
        return reward, env.done, {}

    def fork(self, folder):
        clone = FakeCertifier(folder, deepcopy(self.env))
        self.forks.append(clone)
        return clone


@pytest.fixture
def backend(monkeypatch):
    def build(folder, runtime_spec, initial_offset=None):
        env = FakeEnv(runtime_spec)
        if initial_offset:
            env.x[:3] += np.array(initial_offset)
        return env, FakeCertifier(folder, env)
    FakeCertifier.forks = []
    monkeypatch.setattr(rollout, 'build_training_flight', build)
    return build


def candidate(tmp_path, spec=None, **extras):
    args = dict(folder=tmp_path/'nominal', chain=[constant_stage(.04)], runtime_spec=spec or {})
    args.update(extras)
    return rollout.candidate_job(args)


def test_full_candidate_and_partial_run_never_claim_completion(tmp_path, backend):
    complete = candidate(tmp_path)
    assert complete.get('error') is None
    assert complete['success'] and complete['completed_1200s']
    assert complete['steps'] == 400 and complete['decisions'] == 100
    assert complete['action_counts'][CL_ACTIONS.index(.04)] == 400
    assert len(complete['decision_prefix']) == 100
    assert complete['decision_prefix']['12']['policy_state']['decisions'] == [[0., 8]]
    partial = candidate(tmp_path, folder=tmp_path/'short', max_decisions=4)
    assert not partial['success'] and not partial['completed_1200s']
    assert partial['time_s'] == 12 and partial['stop'] == 'MAX_DECISIONS'
    premature = candidate(tmp_path, dict(end_time=300.), folder=tmp_path/'premature')
    assert premature['outcome'] == 'SUCCESS' and not premature['completed']


def test_all_scheduler_memory_and_recovery_state_owned_by_fork(tmp_path, backend):
    env, certifier = backend(tmp_path/'base', {})
    policy = ChainPolicy([constant_stage(.04)], env)
    rollout.certified_fly(certifier, policy, until=12.)
    before = rollout.snapshot_policy(policy)
    branch = certifier.fork(tmp_path/'branch')
    clone = rollout.clone_policy(policy, branch.env, override=(12., 9))
    assert clone.action == policy.action and clone.decision_t == policy.decision_t
    assert digest(clone.decisions) == digest(policy.decisions) and clone.decisions is not policy.decisions
    assert branch.env.snapshot_hash() == env.snapshot_hash()
    rollout.certified_fly(branch, clone, until=15.)
    assert clone.action == 9 and clone.decisions[-1] == (12., 9)
    assert rollout.snapshot_policy(policy) == before
    assert env.t == 12. and env.controller['memory']['request_CL'] == .04
    assert branch.env.controller['memory']['request_CL'] == .045
    assert env.history_valid and env.previous_cl_raw == .04


def test_unchanged_action_branch_is_bitwise_nominal(tmp_path, backend):
    env, certifier = backend(tmp_path/'base', {})
    policy = ChainPolicy([constant_stage(.04)], env)
    rollout.certified_fly(certifier, policy, until=12.)
    fork = certifier.fork(tmp_path/'unchanged')
    clone = rollout.clone_policy(policy, fork.env, override=(12., 8))
    nominal_return, nominal_stop, nominal_actions = rollout.certified_fly(certifier, policy)
    branch_return, branch_stop, branch_actions = rollout.certified_fly(fork, clone)
    assert nominal_return == branch_return
    assert nominal_stop == branch_stop == 'DONE'
    assert nominal_actions == branch_actions
    assert env.snapshot_hash() == fork.env.snapshot_hash()
    assert digest(policy.decisions) == digest(clone.decisions)


@pytest.mark.parametrize('open_loop', [True, False])
def test_branches_use_bound_nominal_prefix_and_full_continuation(tmp_path, backend, open_loop):
    nominal = candidate(tmp_path)
    args = dict(folder=tmp_path/'branches', chain=[constant_stage(.04)], times=[12., 24.],
                sweep=0, schedule=nominal['schedule'] if open_loop else None,
                runtime_spec={}, nominal=nominal)
    result = rollout.branch_job(args)
    assert result.get('error') is None
    assert len(result['rows']) == 4
    for row in result['rows']:
        assert row['valid'] and row['prefix_verified'] and row['time_s'] == 1200.
        assert row['executed'] == [row['requested_action']]*4
        assert row['G_base_rest'] == nominal['G']-result['prefix'][str(row['t'])][0]
        assert row['advantage'] == row['G_rest']-row['G_base_rest']
        assert row['actual_executed_actions'][4:] == [8]*int((1200.-row['t']-12.)/3.)
        assert len(row['final_policy_state']['decisions']) == 100


def test_prefix_mismatch_aborts_before_any_branch_and_saves_evidence(tmp_path, backend):
    nominal = candidate(tmp_path)
    nominal['decision_prefix']['12']['state_sha256'] = 'different-controller-memory'
    result = rollout.branch_job(dict(folder=tmp_path/'branches', chain=[constant_stage(.04)],
        times=[12.], sweep=0, schedule=None, runtime_spec={}, nominal=nominal))
    assert 'NOMINAL_PREFIX_MISMATCH' in result['error']
    assert result['rows'] == [] and FakeCertifier.forks == []
    assert (tmp_path/'branches'/'prefix_mismatch_t0012.json').is_file()


def test_incomplete_branch_is_unknown_not_a_negative_training_label(tmp_path, backend):
    spec = dict(fail_request=7, fail_time=12.)
    nominal = candidate(tmp_path, spec)
    result = rollout.branch_job(dict(folder=tmp_path/'branches', chain=[constant_stage(.04)],
        times=[12.], sweep=0, schedule=None, runtime_spec=spec, nominal=nominal))
    assert result.get('error') is None
    refused, passed = result['rows']
    assert refused['requested_action'] == 7
    assert not refused['valid'] and refused['advantage'] is None
    assert refused['executed'] == [] and refused['time_s'] == 12.
    assert refused['failure_semantics'] == 'UNKNOWN_NOT_CERTIFIED'
    assert not refused['universal_infeasibility_proven']
    assert refused['stop'] == 'NEXT_STATE_NOT_CERTIFIED'
    assert passed['valid'] and result['invalid_branches'] == 1


def test_old_nominal_reward_and_short_nominal_are_rejected(tmp_path, backend):
    nominal = candidate(tmp_path)
    nominal['runtime_spec_sha256'] = 'old-controller'
    result = rollout.branch_job(dict(folder=tmp_path/'branches', chain=[constant_stage(.04)],
        times=[0.], sweep=0, schedule=None, runtime_spec={}, nominal=nominal))
    assert 'another controller' in result['error']
    short = candidate(tmp_path, folder=tmp_path/'short', max_decisions=4)
    result = rollout.branch_job(dict(folder=tmp_path/'branches_short', chain=[constant_stage(.04)],
        times=[0.], sweep=0, schedule=None, runtime_spec={}, nominal=short))
    assert 'complete successful nominal' in result['error']


def test_dispatch_rejects_unknown_job():
    with pytest.raises(ValueError, match='Unknown'):
        rollout.dispatch(('not-a-job', {}))


def test_diagnostics_separate_refused_trials_from_executed_controller():
    first = dict(t0=0., t1=3., CL_rmse=.003, boundary_jump_deg=-.01, max_alpha_rate_degps=2.,
        margins=dict(q=.2, heat=.3), action=8, requested_action=8,
        selected_source='TRACKING_PRIORITY_NOMINAL',
        details=dict(pre=dict(supported=True), post=dict(supported=False), ocp_called=0))
    second = dict(t0=3., t1=6., CL_rmse=.004, boundary_jump_deg=.02, max_alpha_rate_degps=1.,
        margins=dict(q=.1, heat=.4), action=7, requested_action=8,
        selected_source='REUSED_AUDITED_OCP_TAIL',
        details=dict(pre=dict(supported=False), post=dict(supported=True), ocp_called=1))
    refused = dict(status='UNKNOWN_NOT_CERTIFIED', reason='SEARCH_BUDGET_EXHAUSTED', accepted=False,
        details=dict(pre=dict(supported=False), post=dict(supported=False), ocp_called=1))
    attempts = [dict(status='VERIFIED_PASS', reason=None, accepted=True, details=first['details']),
                refused, dict(status='VERIFIED_PASS', reason=None, accepted=True, details=second['details'])]
    result = rollout.candidate_diagnostics(SimpleNamespace(rows=[first, second]),
                                           SimpleNamespace(records=attempts))
    assert result['CL_tracking_RMSE'] == pytest.approx(np.sqrt((.003**2+.004**2)/2))
    assert result['max_boundary_jump_deg'] == .02 and result['max_alpha_rate_degps'] == 2.
    assert result['minimum_normalized_margins'] == dict(q=.1, heat=.3)
    assert result['safety_substitution_steps'] == result['recovery_tail_steps'] == 1
    assert result['certification_status_counts'] == dict(VERIFIED_PASS=2, UNKNOWN_NOT_CERTIFIED=1)
    assert result['certification_refusal_reasons'] == dict(SEARCH_BUDGET_EXHAUSTED=1)
    assert result['executed_pre_ood_count'] == result['executed_post_ood_count'] == 1
    assert result['attempted_pre_ood_count'] == result['attempted_post_ood_count'] == 2
    assert result['executed_ocp_calls'] == 1 and result['attempted_ocp_calls'] == 2
    assert result['executed_ocp_ratio'] == .5 and result['attempted_ocp_ratio'] == 2/3
    broken = deepcopy(first)
    del broken['margins']
    with pytest.raises(KeyError):
        rollout.candidate_diagnostics(SimpleNamespace(rows=[broken]), SimpleNamespace(records=[]))
