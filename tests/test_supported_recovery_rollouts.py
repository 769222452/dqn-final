"""Deterministic backend doubles: no OCP, model preparation or real training."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from certified_flight import CL_ACTIONS
from continuous_surrogate.experiment import digest
from rollout_improvement.policy import constant_stage
from supported_recovery import rollouts
from supported_recovery.support import build_support_label, continuation_identity


class FakeEnv:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.identity = digest(spec)
        self._state = dict(x=[40000., 2400., 0., 0.], t=0., alpha=.05, reward=0., switches=0,
            history_valid=False, previous_cl_raw=None, previous_cl_cmd=None, previous_cl_actual=None,
            transition=dict(source=42, target=12, started=0., duration=8.))
        self.controller_memory = dict(protocol='continuous-recovery-controller/1', identity=self.identity,
            memory=dict(request_CL=.04, controls=dict(alpha_nodes_rad=[.05, .06])), epoch=1, exit_passes=1)
        self.rows = []

    def snapshot(self):
        return deepcopy(dict(protocol='continuous-recovery-training-flight/1', identity=self.identity,
            state=self._state, controller_memory=self.controller_memory,
            certification_conditions=dict(guard=dict(budget=1), dead_end_retry={})))

    def snapshot_hash(self):
        return digest(self.snapshot())

    def load_state_dict(self, snapshot):
        if snapshot['identity'] != self.identity or snapshot['controller_memory']['identity'] != self.identity:
            raise ValueError('Incompatible training flight identity')
        if snapshot['certification_conditions'] != self.snapshot()['certification_conditions']:
            raise ValueError('Incompatible training certification conditions')
        self._state = deepcopy(snapshot['state'])
        self.controller_memory = deepcopy(snapshot['controller_memory'])

    @property
    def x(self):
        return np.array(self._state['x'])

    @property
    def t(self):
        return self._state['t']

    @property
    def alpha(self):
        return self._state['alpha']

    @property
    def total_reward(self):
        return self._state['reward']

    @property
    def switches(self):
        return self._state['switches']

    @property
    def history_valid(self):
        return self._state['history_valid']

    @property
    def previous_cl_raw(self):
        return self._state['previous_cl_raw']

    @property
    def done(self):
        return self.t >= self.spec.get('end_time', 1200.)

    @property
    def last_outcome(self):
        return 'SUCCESS' if self.done else 'INCOMPLETE'


class FakeCertifier:
    def __init__(self, folder, env):
        self.folder, self.env = Path(folder), env
        self.folder.mkdir(parents=True, exist_ok=True)
        self.policy_state, self.records, self.commits = None, [], 0
        self.token = object()

    def bind_policy_state(self, state):
        self.policy_state = deepcopy(state)

    def pool(self, preference):
        requested = int(np.argmax(preference))
        mask = np.zeros(19, bool)
        if self.env.spec.get('fail_request') != requested:
            mask[self.env.spec.get('substitute', {}).get(str(requested), requested)] = True
        return dict(mask=mask, token=self.token, state_hash=self.env.snapshot_hash())

    def commit(self, pool, action):
        assert pool['token'] is self.token and pool['state_hash'] == self.env.snapshot_hash()
        assert pool['mask'][action] and self.policy_state is not None
        self.commits += 1
        env, raw = self.env, CL_ACTIONS[action]
        reward = .001+raw*.0001+(env.x[0]-40000.)*1e-8
        env._state['x'][3] += 1000.+100.*raw
        env._state.update(t=env.t+3., reward=env.total_reward+reward, alpha=raw,
            history_valid=True, previous_cl_raw=raw, previous_cl_cmd=raw, previous_cl_actual=raw)
        env.controller_memory['epoch'] += 1
        env.controller_memory['memory']['request_CL'] = raw
        env.controller_memory['memory']['controls']['alpha_nodes_rad'][0] = raw
        return reward, env.done, {}


@pytest.fixture
def backend(monkeypatch):
    instances = []

    def build(folder, runtime_spec, initial_offset=None):
        env = FakeEnv(runtime_spec)
        if initial_offset is not None:
            env._state['x'][0] += initial_offset[0]
            env._state['x'][1] += initial_offset[1]
        certifier = FakeCertifier(folder, env)
        instances.append((env, certifier))
        return env, certifier

    build.instances = instances
    monkeypatch.setattr(rollouts, 'build_training_flight', build)
    return build


def origin(backend, folder, *, spec=None, chain=None, t=12., altitude=40000.):
    spec = {} if spec is None else spec
    chain = [constant_stage(.04)] if chain is None else chain
    env, certifier = backend(folder, spec)
    env._state.update(t=t, reward=.125)
    env._state['x'][0] = altitude
    policy = rollouts.make_policy(chain, env)
    binding = rollouts.decision_binding(certifier, policy, spec)
    return binding


def pair_args(tmp_path, backend, *, spec=None, **kwargs):
    spec = {} if spec is None else spec
    args = dict(folder=tmp_path/'pair', runtime_spec=spec, reference_chain=[constant_stage(.04)],
        binding=origin(backend, tmp_path/'source', spec=spec), base_action=8, action=9)
    args.update(kwargs)
    return args


@pytest.mark.parametrize('mode', ['open-loop', 'closed-loop'])
def test_pair_recomputes_both_returns_from_off_nominal_state(tmp_path, backend, mode):
    binding = origin(backend, tmp_path/'observed', altitude=41000., chain=[constant_stage(.045)])
    binding['G_prefix'] = -999999.  # Neither cached prefix nor nominal score is trusted.
    args = pair_args(tmp_path, backend, binding=binding, continuation=mode, G=1e9, G_base_rest=-1e9)
    if mode == 'open-loop':
        args['schedule'] = [[float(t), 8] for t in range(0, 1200, 12)]
    result = rollouts.paired_job(args)
    assert not result.get('error'), result.get('error')
    assert result['valid'] and result['completed'] and result['baseline_recomputed']
    assert result['G_base_rest'] == pytest.approx(396*(.001004+.00001))
    assert result['advantage'] == pytest.approx(4*(.045-.04)*.0001)
    assert result['advantage'] == result['G_candidate_rest']-result['G_base_rest']
    assert result['base']['initial_snapshot_sha256'] == result['candidate']['initial_snapshot_sha256']
    assert result['candidate']['first_12s_requested_actions'] == [9]*4
    assert result['candidate']['actual_executed_actions'][4:] == [8]*392
    assert result['support_label']['valid'] and result['support_label']['advantage'] == result['advantage']
    assert result['binding']['chain_sha256'] != result['reference_prefix_sha256']
    assert (tmp_path/'pair'/'base'/'restored_origin.json').is_file()
    assert (tmp_path/'pair'/'candidate'/'result.json').is_file()


def test_equal_action_has_independent_identical_results_and_memory(tmp_path, backend):
    args = pair_args(tmp_path, backend, action=8)
    original = deepcopy(args['binding'])
    result = rollouts.paired_job(args)
    assert not result.get('error'), result.get('error')
    assert result['valid'] and result['advantage'] == 0.
    assert result['base']['final_snapshot_sha256'] == result['candidate']['final_snapshot_sha256']
    left, right = backend.instances[-2:]
    assert left[0] is not right[0] and left[1].token is not right[1].token
    left[0].controller_memory['memory']['controls']['alpha_nodes_rad'][0] = 99
    assert right[0].controller_memory['memory']['controls']['alpha_nodes_rad'][0] == .04
    assert args['binding'] == original


def test_refused_arm_is_unknown_and_never_committed(tmp_path, backend):
    args = pair_args(tmp_path, backend, spec=dict(fail_request=9))
    result = rollouts.paired_job(args)
    assert not result.get('error'), result.get('error')
    assert result['base']['completed']
    assert not result['valid'] and result['advantage'] is None
    assert result['candidate']['steps'] == backend.instances[-1][1].commits == 0
    assert result['candidate']['stop'] == 'NEXT_STATE_NOT_CERTIFIED'
    assert result['failure_semantics'] == 'UNKNOWN_NOT_CERTIFIED'
    assert result['support_label']['valid'] is False


def test_safety_substitution_is_evidence_for_request_not_candidate_physical_cl(tmp_path, backend):
    args = pair_args(tmp_path, backend, spec=dict(substitute={'9': 8}))
    result = rollouts.paired_job(args)
    assert not result.get('error'), result.get('error')
    assert result['valid'] and result['advantage'] == 0.
    assert result['action_semantics'] == 'certified_policy_request'
    assert result['action'] == result['support_label']['candidate_action'] == 9
    assert result['candidate']['first_12s_requested_actions'] == [9]*4
    assert result['candidate']['first_12s_actual_executed_actions'] == [8]*4
    assert result['candidate']['first_12s_substitution_count'] == 4


@pytest.mark.parametrize('time', [12., 1188.])
def test_bounded_pair_never_produces_training_label_even_if_it_reaches_end(tmp_path, backend, time):
    binding = origin(backend, tmp_path/'observed', t=time)
    result = rollouts.paired_job(pair_args(tmp_path, backend, binding=binding, max_decisions=4))
    assert not result.get('error'), result.get('error')
    assert not result['valid'] and result['advantage'] is None and result['bounded_smoke']
    assert result['failure_semantics'] == 'BOUNDED_CHECK_NOT_TRAINING_LABEL'
    assert result['completed'] is (time == 1188.)


@pytest.mark.parametrize('mutation,error', [
    ('runtime', 'runtime specification'), ('state', 'snapshot hash'),
    ('scheduler', 'policy snapshot hash'), ('base', 'frozen reference'),
    ('boundary', 'twelve-second boundary'), ('identity', 'flight identity'),
    ('features', 'network features'),
])
def test_origin_tamper_or_wrong_reference_is_rejected(tmp_path, backend, mutation, error):
    args = pair_args(tmp_path, backend)
    if mutation == 'runtime':
        args['binding']['runtime_spec_sha256'] = 'old'
    elif mutation == 'state':
        args['binding']['state']['controller_memory']['epoch'] += 1
    elif mutation == 'scheduler':
        args['binding']['policy_state']['action'] = 2
    elif mutation == 'base':
        args['base_action'] = 7
    elif mutation == 'boundary':
        args['binding']['t'] = 15.
    elif mutation == 'features':
        args['binding']['features'][0] += 1.
    else:
        args['binding']['state']['identity'] = 'different'
        args['binding']['state_sha256'] = digest(args['binding']['state'])
    result = rollouts.paired_job(args)
    assert error in result['error']
    assert result['valid'] is False and result['advantage'] is None
    assert all(certifier.commits == 0 for _, certifier in backend.instances)


def test_open_loop_requires_complete_explicit_tail(tmp_path, backend):
    result = rollouts.paired_job(pair_args(tmp_path, backend, continuation='open-loop', schedule=[[12., 8]]))
    assert 'every remaining decision' in result['error']
    assert result['advantage'] is None


def test_candidate_logs_actual_decisions_once_and_preserves_source_binding(tmp_path, backend):
    reference = [constant_stage(.04)]
    env, _ = backend(tmp_path/'label_source', {})
    continuation = continuation_identity(reference)
    support = build_support_label(env, reference_prefix_sha256=digest(reference),
        continuation_sha256=continuation['sha256'], base_action=8, candidate_action=9, advantage=1e-5)
    bias = np.zeros(19)
    bias[9] = 2.
    stage = dict(kind='supported_model', members=[[(np.zeros((4, 19)).tolist(), bias.tolist())]],
        margin=.5, inputs=[0, 1, 2, 4], support_mode='exact', support_records=[support],
        reference_prefix_sha256=digest(reference), runtime_identity=env.identity,
        continuation=continuation, continuation_sha256=continuation['sha256'],
        action_semantics='certified_policy_request')
    result = rollouts.candidate_job(dict(folder=tmp_path/'candidate', chain=reference+[stage], runtime_spec={}))
    assert not result.get('error'), result.get('error')
    assert result['success'] and result['time_s'] == 1200.
    assert result['requested_actions'][:4] == [9]*4
    assert result['requested_actions'][4:] == [8]*396
    assert len(result['support_logs']) == len(result['decision_prefix']) == 100
    assert result['decision_prefix']['0']['policy_state']['action'] is None
    assert result['decision_prefix']['12']['policy_state']['decisions'] == [[0., 9]]
    assert result['support_logs'][0]['stages'][0]['selected_action'] == 9
    assert result['support_logs'][1]['stages'][0]['selected_action'] == 8
    assert sum(len(row['executions']) for row in result['support_logs']) == 400
    assert result['decision_prefix']['12']['runtime_spec_sha256'] == digest({})


def test_read_only_binding_rejects_selection_that_mutates_controller(tmp_path, backend, monkeypatch):
    select = rollouts.policy_action

    def mutate(chain, env, record=False):
        env.controller_memory['epoch'] += 1
        return select(chain, env, record=record)

    monkeypatch.setattr(rollouts, 'policy_action', mutate)
    result = rollouts.candidate_job(dict(folder=tmp_path/'candidate', chain=[constant_stage(.04)], runtime_spec={}))
    assert 'Read-only policy selection changed' in result['error']
    assert backend.instances[-1][1].commits == 0


def test_partial_candidate_never_claims_full_horizon(tmp_path, backend):
    result = rollouts.candidate_job(dict(folder=tmp_path/'candidate', chain=[constant_stage(.04)],
                                         runtime_spec={}, max_decisions=4))
    assert not result.get('error'), result.get('error')
    assert not result['success'] and result['time_s'] == 12.
    assert result['stop'] == 'MAX_DECISIONS' and result['bounded_smoke']


def test_dispatch_rejects_unknown_job():
    with pytest.raises(ValueError, match='Unknown'):
        rollouts.dispatch(('unknown', {}))
