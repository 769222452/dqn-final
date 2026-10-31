"""Stateful, transactionally certified flights for recovery-policy training.

This backend owns all trajectory and recovery memory.  A fork copies that memory,
but never copies an executable certificate.  Surrogate weights and physics are
loaded without fitting; every branch action receives fresh current/future audits.
"""
from copy import copy, deepcopy
from pathlib import Path
import time
import uuid

import numpy as np

from certified_flight import CL_ACTIONS, integrity
from certified_flight.certification import PASS
from certified_flight.transition import Transition
from continuous_surrogate.experiment import baseline, digest
from continuous_tracking.experiment import clean, write_json
from .experiment import initial_state
from .execution import seal_execution, commit_certified
from .learned_policy import certify_ranked, _clone_controller

PROTOCOL = 'continuous-recovery-training-flight/1'
STATE_FIELDS = frozenset(('x', 'alpha', 't', 'transition', 'reward', 'switches',
                         'history_valid', 'previous_cl_raw', 'previous_cl_cmd', 'previous_cl_actual'))
_INHERIT = object()


def snapshot_policy(policy):
    """The exact held 12-second decision, including branch override/schedule."""
    return clean(dict(protocol='recovery-training-policy-state/1', chain_sha256=digest(policy.chain),
        action=policy.action, decision_t=policy.decision_t, decisions=deepcopy(policy.decisions),
        override=deepcopy(policy.override), schedule=deepcopy(policy.schedule)))


def restore_policy(chain, env, snapshot, override=_INHERIT, schedule=_INHERIT):
    """Restore a scheduler without silently re-deciding in the middle of a block."""
    from rollout_improvement.policy import ChainPolicy
    policy = ChainPolicy(chain, env,
        override=snapshot['override'] if override is _INHERIT else override,
        schedule=snapshot['schedule'] if schedule is _INHERIT else schedule)
    if (snapshot.get('protocol') != 'recovery-training-policy-state/1'
            or snapshot.get('chain_sha256') != digest(policy.chain)):
        raise ValueError('Incompatible training policy snapshot')
    action, tick, decisions = snapshot['action'], snapshot['decision_t'], snapshot['decisions']
    if action is None:
        if tick is not None or decisions:
            raise ValueError('Inconsistent undecided policy snapshot')
    elif (type(action) is not int or not 0 <= action < len(CL_ACTIONS)
            or tick is None or not np.isfinite(tick) or tick < 0 or tick > env.t
            or abs(tick/12.-round(tick/12.)) > 1e-9 or not decisions
            or list(decisions[-1]) != [tick, action]):
        raise ValueError('Invalid held policy decision')
    previous = -1.
    for t, a in decisions:
        if (type(a) is not int or not 0 <= a < len(CL_ACTIONS) or not np.isfinite(t)
                or t <= previous or t > env.t or abs(t/12.-round(t/12.)) > 1e-9):
            raise ValueError('Invalid policy decision history')
        previous = t
    policy.action, policy.decision_t = action, tick
    policy.decisions = deepcopy(decisions)
    return policy


def _checked_state(state, system):
    state = deepcopy(state)
    if set(state) != STATE_FIELDS:
        raise ValueError('Training flight checkpoint has incompatible state fields')
    x = np.asarray(state['x'], float)
    if x.shape != (4,) or not np.isfinite(np.r_[x, state['alpha'], state['t'], state['reward']]).all():
        raise ValueError('Training flight state must be finite')
    if (not 0. <= state['t'] <= system['mission'].data['duration_s']
            or abs(state['t']/3.-round(state['t']/3.)) > 1e-9):
        raise ValueError('Training flight time must lie on the mission decision grid')
    if type(state['switches']) is not int or state['switches'] < 0:
        raise ValueError('Invalid geometry switch count')
    z = Transition(**state['transition'])
    if (type(z.source) is not int or type(z.target) is not int
            or not 0 <= z.source < len(system['db'].ids)
            or not 0 <= z.target < len(system['db'].ids)
            or z.duration != 8. or not np.isfinite(z.started) or z.started > state['t']):
        raise ValueError('Invalid original eight-second geometry transition')
    if type(state['history_valid']) is not bool:
        raise ValueError('History-valid flag must be boolean')
    history = [state[k] for k in ('previous_cl_raw', 'previous_cl_cmd', 'previous_cl_actual')]
    if state['history_valid']:
        if (any(v is None for v in history) or not np.isfinite(history).all()
                or history[0] not in CL_ACTIONS):
            raise ValueError('Invalid committed absolute-CL history')
    elif history != [None, None, None]:
        raise ValueError('Uninitialized CL history must be empty')
    return state


def _fork_controller(controller):
    """Own mutable manager/model/solver-cache containers, share compiled kernels.

    The plant's map cache contains only immutable compiled integration functions;
    solver cache entries contain immutable problem descriptions and compiled NLPs.
    No trajectories, warm-start multipliers, PASS records or safety outcomes are
    taken from either cache.
    """
    def solver_data(value):
        # CasADi/compiled callable objects remain shared. Their problem data,
        # lists and numeric arrays do not share writable containers.
        if isinstance(value, dict):
            return {k:solver_data(v) for k, v in value.items()}
        if isinstance(value, list):
            return [solver_data(v) for v in value]
        if isinstance(value, tuple):
            return tuple(solver_data(v) for v in value)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    child = _clone_controller(controller)
    plant = copy(controller.system['plant'])
    if hasattr(plant, 'maps'):
        plant.maps = dict(plant.maps)
    child.system['plant'] = child.plant = plant
    if hasattr(child.guard, 'plant'):
        child.guard.plant = plant
    if hasattr(child.guard, 'reachability'):
        child.guard.reachability = copy(child.guard.reachability)
        child.guard.reachability.plant = plant
    if 'reachability' in child.system:
        child.system['reachability'] = copy(child.system['reachability'])
        child.system['reachability'].plant = plant
    for key in ('config', 'original', 'identity'):
        if key in child.system:
            child.system[key] = deepcopy(child.system[key])
    for key in ('cfg', 'limits', 'wn'):
        if hasattr(child.guard, key):
            setattr(child.guard, key, deepcopy(getattr(child.guard, key)))
    teacher = controller.system.get('teacher')
    if teacher is not None:
        teacher_copy = copy(teacher)
        # The cache dictionary is mutable, cached solver descriptions are not.
        teacher_copy.cache = solver_data(teacher.cache)
        teacher_copy.plant = plant
        for key in ('cfg', 'traj', 'experiment', 'xscale'):
            if hasattr(teacher_copy, key):
                setattr(teacher_copy, key, deepcopy(getattr(teacher_copy, key)))
        child.system['teacher'] = teacher_copy
    if hasattr(controller.model, '__dict__'):
        memo = {id(controller.db): controller.db,
                id(controller.system['mission']): controller.system['mission']}
        child.model = deepcopy(controller.model, memo)
        if hasattr(child.model, 'baseline_callable'):
            child.model.baseline_callable = baseline(child.system)
    child.physical = baseline(child.system)
    return child


class TrainingFlight:
    """Minimal ChainPolicy environment with a complete recovery-state checkpoint."""

    def __init__(self, system, model, identity, identity_payload, state=None, controller=None):
        from .controller import RecoveryController
        self.system, self.model = system, model
        self.identity, self.identity_payload = identity, deepcopy(identity_payload)
        if state is None:
            state, _ = initial_state(system)
            state.update(history_valid=False, previous_cl_raw=None,
                         previous_cl_cmd=None, previous_cl_actual=None)
        self._state = _checked_state(state, system)
        self.initial_state = deepcopy(self._state)
        self.controller = controller or RecoveryController(system, model, identity)
        self.cl_actions, self.n_actions = list(CL_ACTIONS), len(CL_ACTIONS)
        self.rows = []
        self.certifier = None

    @property
    def state(self):
        return deepcopy(self._state)

    @property
    def x(self):
        return np.asarray(self._state['x'], float).copy()

    @property
    def alpha(self):
        return self._state['alpha']

    @property
    def t(self):
        return self._state['t']

    @property
    def switches(self):
        return self._state['switches']

    @property
    def total_reward(self):
        return self._state['reward']

    @property
    def history_valid(self):
        return self._state['history_valid']

    @property
    def previous_cl_raw(self):
        return self._state['previous_cl_raw']

    @property
    def previous_cl_cmd(self):
        return self._state['previous_cl_cmd']

    @property
    def previous_cl_actual(self):
        return self._state['previous_cl_actual']

    @property
    def transition(self):
        return Transition(**self._state['transition'])

    @property
    def last_outcome(self):
        return self.system['mission'].outcome(self.t, self.x)

    @property
    def done(self):
        return self.last_outcome != 'INCOMPLETE'

    def snapshot(self):
        return dict(protocol=PROTOCOL, identity=self.identity, state=self.state,
                    controller_memory=self.controller.state_dict(),
                    certification_conditions=self.certification_conditions())

    def certification_conditions(self):
        guard = self.controller.guard
        return clean(dict(guard={k:deepcopy(getattr(guard, k)) for k in
            ('budget', 'max_evaluations', 'correction_budget', 'horizon', 'cfg') if hasattr(guard, k)},
            dead_end_retry=deepcopy(self.system.get('original', {}).get('dead_end_retry'))))

    state_dict = snapshot

    def snapshot_hash(self):
        return digest(self.snapshot())

    def load_state_dict(self, snapshot):
        if snapshot.get('protocol') != PROTOCOL or snapshot.get('identity') != self.identity:
            raise ValueError('Incompatible training flight identity or protocol')
        if snapshot.get('certification_conditions') != self.certification_conditions():
            raise ValueError('Incompatible training certification conditions')
        state = _checked_state(snapshot['state'], self.system)
        candidate = _fork_controller(self.controller)
        candidate.load_state_dict(snapshot['controller_memory'])
        # Validation finishes before either trajectory or controller changes.
        self._state, self.controller = state, candidate
        if self.certifier is not None:
            self.certifier._pending = None
            self.certifier.policy_state = None


class TrainingCertifier:
    """Fresh first-PASS pools; execute their sealed trajectory exactly once."""

    def __init__(self, env, folder):
        self.env, self.folder = env, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.token = uuid.uuid4().hex
        self._pending, self.policy_state = None, None
        self.pool_records, self.records = [], []
        self.retry_options = deepcopy(env.system.get('original', {}).get('dead_end_retry'))
        self.env.certifier = self
        write_json(self.folder/'identity.json', dict(sha256=env.identity, payload=env.identity_payload))
        write_json(self.folder/'initial_state.json', env.snapshot())
        self._checkpoint()

    def bind_policy_state(self, state):
        # digest validates a finite JSON snapshot; no policy evaluation occurs here.
        digest(state)
        self.policy_state = deepcopy(state)

    def _checkpoint(self):
        write_json(self.folder/'checkpoint.json', dict(self.env.snapshot(),
            policy_state=deepcopy(self.policy_state), certificates_reused=False,
            execution_commit_mode='SEALED_CURRENT_CERTIFICATE'))

    @staticmethod
    def _execution_witness(answer, pool, before, policy, record_hash):
        return dict(manager_before=before['controller_memory'],
            manager_after=answer['controller_state'], guard=answer['selected'].diagnostics,
            policy_state=policy, requested_action=pool['requested_action'],
            selected_action=answer['action'], selected_raw_CL=float(CL_ACTIONS[answer['action']]),
            certified_action_mask=pool['mask'].tolist(), candidate_evidence_hash=record_hash,
            selected_details_hash=digest(answer['details']))

    def pool(self, preference):
        if self.env.done:
            raise ValueError('Cannot certify a completed mission')
        key = self.env.snapshot_hash()
        index = len(self.pool_records)
        folder = self.folder/f'decision_{index:04d}'
        folder.mkdir(parents=False, exist_ok=False)
        before = self.env.snapshot()
        policy = deepcopy(self.policy_state)
        write_json(folder/'before.json', dict(before, policy_state=policy))
        answer = certify_ranked(self.env.controller, self.env.state, preference,
                                folder/'candidates', retry_options=self.retry_options)
        if self.env.snapshot_hash() != key:
            raise ValueError('Certification changed the live flight or recovery memory')
        record = clean({k: v for k, v in answer.items()
                        if k not in {'selected', 'z', 'manager_after'}})
        record.update(key=key, pool_id=index, policy_state=policy,
                      controller_identity=self.env.identity)
        write_json(folder/'action_pool.json', record)
        self.pool_records.append(record)
        self.records.extend(deepcopy(answer['attempts']))
        public = dict(key=key, pool_id=index, certifier_token=self.token,
            action=answer['action'], requested_action=answer['requested_action'],
            mask=np.asarray(answer['mask'], bool), preference=np.asarray(preference, float).copy(),
            checked=list(answer['checked']), statuses=list(answer['statuses']),
            reasons=deepcopy(answer['reasons']), wall_s=answer['wall_s'],
            certification_exhausted=bool(answer['action'] is None),
            evidence_path=str(folder/'action_pool.json'))
        # Freeze evidence at certification time, before exposing the pool. Never
        # create a new binding from potentially changed inputs at commit time.
        record_hash = digest(record)
        evidence, seal_wall = None, 0.
        if answer['action'] is not None:
            seal_started = time.perf_counter()
            witness = self._execution_witness(answer, public, before, policy, record_hash)
            evidence = seal_execution(self.env.system, self.env.state, answer['z'],
                self.env.identity, float(answer['cmd']), witness, answer['selected'])
            seal_wall = time.perf_counter()-seal_started
        self._pending = dict(public_hash=digest(public), public=public, answer=answer,
            record_hash=record_hash, before=before, policy=policy, folder=folder,
            evidence=evidence, seal_wall_s=seal_wall)
        if answer['action'] is None:
            write_json(self.folder/'stop_state.json', dict(reason='NEXT_STATE_NOT_CERTIFIED',
                state=before, policy_state=policy, last_certification=public['evidence_path'],
                statuses=answer['statuses'], reasons=answer['reasons'],
                universal_infeasibility_proven=False, state_committed=False))
        return public

    def commit(self, pool, action):
        commit_started = time.perf_counter()
        pending = self._pending
        if (pending is None or pool.get('certifier_token') != self.token
                or digest(pool) != pending['public_hash']
                or pool.get('key') != self.env.snapshot_hash()
                or digest(self.policy_state) != digest(pending['policy'])):
            raise ValueError('Stale or incompatible training action pool')
        if (type(action) not in (int, np.int64, np.int32)
                or not 0 <= int(action) < len(CL_ACTIONS)):
            raise ValueError('Invalid absolute CL action')
        action = int(action)
        answer = pending['answer']
        selected = answer['selected']
        if (action != answer['action'] or not pool['mask'][action]
                or pool['statuses'][action] != PASS or selected is None or not selected.accepted):
            raise ValueError('Action lacks a current PASS certificate')
        manager_after = answer['manager_after']
        if digest(manager_after.state_dict()) != digest(answer['controller_state']):
            raise ValueError('Certified recovery memory changed before commit')
        nodes, times = np.asarray(selected.alpha_nodes, float), np.asarray(selected.times, float)
        if nodes.shape != (7,) or not np.array_equal(times, np.arange(7)*.5):
            raise ValueError('Certified action must retain the original execution grid')
        state, z, cmd = self.env.state, answer['z'], float(answer['cmd'])
        witness = self._execution_witness(answer, pool, pending['before'],
                                          pending['policy'], pending['record_hash'])
        evidence = pending['evidence']
        if evidence is None:
            raise ValueError('Action lacks sealed execution evidence')
        proof = evidence.proof
        folder = pending['folder']
        write_json(folder/'certificate.json', dict(proof, status=PASS,
            proposed_transition=vars(z), future_witness=witness))
        execution_started = time.perf_counter()
        next_state, audit, arrays = commit_certified(self.env.system, state, z, nodes,
            proof, self.env.identity, cmd, witness, evidence)
        execution_wall = time.perf_counter()-execution_started
        dense = audit['grids']['0.1']
        detail = answer['details']
        next_state['switches'] += int(bool(detail.get('switched', False)))
        next_state.update(history_valid=True, previous_cl_raw=float(CL_ACTIONS[action]),
            previous_cl_cmd=cmd, previous_cl_actual=float(dense['CL_mean_first3']))
        next_state = _checked_state(next_state, self.env.system)
        # Preserve the exact per-step reward arithmetic used by mission.reward.
        outcome = self.env.system['mission'].outcome(next_state['t'], np.asarray(next_state['x']))
        reward = float(self.env.system['mission'].reward(next_state['x'][3]-state['x'][3], outcome)['reward'])
        margins = {k:min(dense['margins'][k], audit['grids']['0.5']['margins'][k])
                   for k in dense['margins']}
        samples = arrays['0.1']['samples']
        weights = np.asarray([1., 2., 2., 1.])/6.
        energy = (samples[10]*samples[1]).reshape(-1, 5)[:, :4] @ weights
        info = clean(dict(action=action, requested_action=pool['requested_action'],
            t0=state['t'], t1=next_state['t'], CL_raw=CL_ACTIONS[action], CL_cmd=cmd,
            CL_mean=dense['CL_mean_first3'], CL_rmse=dense['CL_rmse_first3'],
            state_end=next_state['x'], switched=bool(detail.get('switched', False)),
            source=self.env.system['db'].ids[z.source], target=self.env.system['db'].ids[z.target],
            reward=reward, outcome=outcome, safety_status=selected.status,
            selected_source=detail.get('selected_source'), details=detail,
            execution_audit=audit, margins=margins, alpha_nodes_deg=np.rad2deg(nodes).tolist(),
            max_alpha_rate_degps=audit['maximum_alpha_rate_degps'],
            boundary_jump_deg=float(np.rad2deg(nodes[0]-state['alpha'])),
            drag_work_J=float(np.sum(energy)*.1),
            wall_s=float(pool['wall_s'])+pending['seal_wall_s']+time.perf_counter()-commit_started,
            certificate_path=str(folder/'certificate.json'),
            candidate_evidence_path=str(folder/'action_pool.json'),
            execution_protocol=proof['protocol'], certified_trajectory_reused=True,
            actual_execution_reintegrated=False, execution_integration_calls=0,
            execution_seal_wall_s=pending['seal_wall_s'], execution_commit_wall_s=execution_wall,
            old_certificates_reused=False))
        # All numerical, state and evidence checks happen before the live commit.
        np.savez_compressed(folder/'executed.npz', alpha_nodes=nodes, times=times,
            **{f'{key}_{field}': arr[field] for key, arr in arrays.items() for field in ('x', 'samples')})
        write_json(folder/'execution.json', info)
        self.env._state, self.env.controller = next_state, manager_after
        self.env.rows.append(info)
        self._pending = None
        self._checkpoint()
        return reward, self.env.done, info

    def fork(self, folder):
        controller = _fork_controller(self.env.controller)
        branch = TrainingFlight(controller.system, controller.model, self.env.identity,
            self.env.identity_payload, state=self.env.state, controller=controller)
        branch.initial_state = deepcopy(self.env.initial_state)
        child = TrainingCertifier(branch, folder)
        child.policy_state = deepcopy(self.policy_state)
        child._checkpoint()
        write_json(child.folder/'branch_origin.json', dict(parent_snapshot_sha256=self.env.snapshot_hash(),
            inherited_policy_state=child.policy_state, certificate_cache_copied=False))
        if branch.snapshot_hash() != self.env.snapshot_hash():
            raise ValueError('Branch state differs from its exact nominal origin')
        return child


def build_training_flight(folder, runtime_spec, initial_offset=None):
    """Build the new backend without training/refitting any controller component."""
    from rollout_improvement.identity import experiment_identity
    from .local_model import LocalResidualSurrogate
    from .preparation import build
    spec = deepcopy(runtime_spec)
    config, model_path = (Path(spec[key]).resolve(strict=True) for key in ('config', 'model'))
    references = dict(config_sha256=integrity.sha256(config),
        model_sha256=integrity.sha256(model_path),
        model_metadata_sha256=integrity.sha256(model_path.with_suffix('.json')))
    if ('initial_policy' in spec) != ('initial_policy_sha256' in spec):
        raise ValueError('Initial-policy reference requires both path and SHA256')
    if 'initial_policy' in spec:
        references['initial_policy_sha256'] = integrity.sha256(Path(spec['initial_policy']).resolve(strict=True))
    for key, actual in references.items():
        if key in spec and spec[key] != actual:
            raise ValueError('Training runtime '+key+' mismatch')
    source = experiment_identity()
    if spec.get('source_sha256', source['sha256']) != source['sha256']:
        raise ValueError('Training runtime source_sha256 mismatch')
    system = build(config)
    model = LocalResidualSurrogate.load(model_path, system['db'], system['mission'], baseline(system))
    system['model_hash'] = references['model_sha256']
    identity_payload = dict(protocol=PROTOCOL, references=references, source=source,
        config=system['config'], model_metadata=model.metadata,
        training_protocol=spec.get('training_protocol'), runtime_identity=spec.get('runtime_identity'),
        physical=system['original']['physical'], safety_options=system['original']['safety_options'],
        retry_options=system['original']['dead_end_retry'], policy_driven=True,
        fixed_config_flight_CL_used=False, CL_actions=list(CL_ACTIONS), decision_s=12., execution_s=3.)
    env = TrainingFlight(system, model, digest(identity_payload), identity_payload)
    if initial_offset is not None:
        offset = np.asarray(initial_offset, float)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError('Initial offset must be finite altitude/speed pair')
        env._state['x'][0] += float(offset[0])
        env._state['x'][1] += float(offset[1])
        env.initial_state = env.state
    certifier = TrainingCertifier(env, folder)
    return env, certifier
