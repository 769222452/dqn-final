"""Frozen original improvement policy with transactional recovery certification.

The upper policy is re-evaluated on measured states every twelve seconds. No
saved action schedule, optimization, replay or parameter update is used here.
Each lower-controller trial owns its recovery memory and temporary guard budget.
"""
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from certified_flight import CL_ACTIONS
from certified_flight.certification import PASS, UNKNOWN, classification
from continuous_surrogate.experiment import baseline, digest, write_json
from continuous_surrogate.safety import SafetyResult
from rollout_improvement.features import features
from rollout_improvement.policy import ChainPolicy, DECISION_S

PROTOCOL = 'continuous-recovery-frozen-policy/1'


def _validate_chain(chain):
    if not isinstance(chain, list):
        raise ValueError('Checkpoint must contain a JSON policy chain')
    for stage in chain:
        if not isinstance(stage, dict):
            raise ValueError('Every policy stage must be a dictionary')
        kind = stage.get('kind', 'model')
        if kind == 'constant':
            action = stage['action']
            if type(action) is not int or not 0 <= action < len(CL_ACTIONS):
                raise ValueError('Invalid constant absolute-CL action')
            if 'cl' in stage and stage['cl'] != CL_ACTIONS[action]:
                raise ValueError('Constant action and CL disagree')
        elif kind == 'override':
            for entry in stage['entries']:
                for name in ('action', 'base_action'):
                    if type(entry[name]) is not int or not 0 <= entry[name] < len(CL_ACTIONS):
                        raise ValueError('Invalid override action')
                if not np.isfinite(entry['t']):
                    raise ValueError('Nonfinite override time')
        elif kind == 'model':
            inputs = stage.get('inputs')
            if inputs is not None and (not isinstance(inputs, list) or not inputs
                    or len(set(inputs)) != len(inputs)
                    or any(type(i) is not int or not 0 <= i < 6 for i in inputs)):
                raise ValueError('Invalid original six-feature input subset')
            if not np.isfinite(stage['margin']) or not np.isfinite(stage['window']):
                raise ValueError('Nonfinite policy margin/window')
            if stage['window'] < 0 or not np.isfinite(stage['times']).all():
                raise ValueError('Invalid policy activation times')
            if not isinstance(stage['members'], list) or not stage['members']:
                raise ValueError('An ensemble must have at least one member')
            for member in stage['members']:
                width = 6 if inputs is None else len(inputs)
                if not member:
                    raise ValueError('An ensemble member must have layers')
                for W, b in member:
                    W, b = np.asarray(W, float), np.asarray(b, float)
                    if (W.ndim != 2 or b.ndim != 1 or W.shape != (width, len(b))
                            or not np.isfinite(W).all() or not np.isfinite(b).all()):
                        raise ValueError('Incompatible or nonfinite frozen network layer')
                    width = len(b)
                if width != len(CL_ACTIONS):
                    raise ValueError('Frozen network output must preserve all nineteen actions')
        else:
            raise ValueError('Unknown frozen policy stage: '+str(kind))


class FrozenChainPolicy:
    """Adapter around the unchanged ChainPolicy; history belongs to committed flight."""

    def __init__(self, checkpoint_path):
        path = Path(checkpoint_path).resolve()
        content = path.read_bytes()
        checkpoint = json.loads(content.decode('utf-8-sig'))
        chain = checkpoint.get('chain') if isinstance(checkpoint, dict) else checkpoint
        _validate_chain(chain)
        self.metadata = dict(protocol=PROTOCOL, checkpoint_path=str(path),
            checkpoint_sha256=hashlib.sha256(content).hexdigest(), chain_sha256=digest(chain),
            chain=deepcopy(chain), actions=CL_ACTIONS.copy(), decision_s=DECISION_S,
            feature_names=['h', 'V', 'gamma', 'alpha', 't', 'prev'],
            frozen_weights=True, state_feedback=True, schedule_replay=False,
            checkpoint_protocol=checkpoint.get('protocol') if isinstance(checkpoint, dict) else None)
        self.identity = digest(self.metadata)
        self.env = SimpleNamespace(x=np.zeros(4), alpha=0., t=0.,
                                   history_valid=False, previous_cl_raw=None)
        self.policy = ChainPolicy(deepcopy(chain), self.env)
        self.last_observed_t = None

    def choose(self, state, history_valid=False, previous_cl_raw=None):
        x = np.asarray(state['x'], float)
        now, alpha = float(state['t']), float(state['alpha'])
        if x.shape != (4,) or not np.isfinite(np.r_[x, now, alpha]).all():
            raise ValueError('Policy requires a finite measured flight state')
        if now < 0. or (self.last_observed_t is not None and now < self.last_observed_t):
            raise ValueError('Policy time must not move backwards')
        if previous_cl_raw is not None and float(previous_cl_raw) not in CL_ACTIONS:
            raise ValueError('Previous executed raw CL must be an original absolute action')
        self.env.x, self.env.alpha, self.env.t = x.copy(), alpha, now
        self.env.history_valid = bool(history_valid)
        self.env.previous_cl_raw = previous_cl_raw
        n_before = len(self.policy.decisions)
        preference = self.policy.preference()
        self.last_observed_t = now
        return dict(action=int(self.policy.action), CL_raw=CL_ACTIONS[self.policy.action],
            preference=preference.tolist(), features=features(self.env).tolist(),
            policy_decision_time_s=float(self.policy.decision_t),
            new_policy_decision=len(self.policy.decisions) != n_before)

    def state_dict(self):
        return dict(protocol=PROTOCOL, identity=self.identity, action=self.policy.action,
                    decision_t=self.policy.decision_t, decisions=deepcopy(self.policy.decisions),
                    last_observed_t=self.last_observed_t)

    def load_state_dict(self, state):
        if state.get('protocol') != PROTOCOL or state.get('identity') != self.identity:
            raise ValueError('Incompatible frozen-policy checkpoint')
        action, tick, last = state['action'], state['decision_t'], state['last_observed_t']
        decisions = deepcopy(state['decisions'])
        if action is None:
            if tick is not None or last is not None or decisions:
                raise ValueError('Inconsistent uninitialized policy state')
        elif (type(action) is not int or not 0 <= action < len(CL_ACTIONS)
              or tick is None or last is None or not np.isfinite([tick, last]).all()
              or tick < 0 or last < tick or last >= tick+DECISION_S+1e-9
              or abs(tick/DECISION_S-round(tick/DECISION_S)) > 1e-9):
            raise ValueError('Invalid policy scheduler state')
        for t, a in decisions:
            if (type(a) is not int or not 0 <= a < len(CL_ACTIONS)
                    or not np.isfinite(t) or t < 0):
                raise ValueError('Invalid policy decision history')
        if action is not None and (not decisions or list(decisions[-1]) != [tick, action]):
            raise ValueError('Policy scheduler and decision history disagree')
        self.policy.action, self.policy.decision_t = action, tick
        self.policy.decisions, self.last_observed_t = decisions, last


def _clone_controller(controller):
    """Compiled dynamics are immutable/shared; control and guard state are owned."""
    candidate = copy(controller)
    candidate.memory = deepcopy(controller.memory)
    candidate.identity = deepcopy(controller.identity)
    candidate.system = dict(controller.system)
    candidate.guard = copy(controller.guard)
    candidate.system['safety'] = candidate.guard
    candidate.physical = baseline(candidate.system)
    return candidate


def certify_ranked(controller, state, preference, folder, retry_options=None):
    """Original first-PASS preference search, without mutating the live manager.

    Only explicit final SEARCH_BUDGET_EXHAUSTED refusals are retry-eligible.
    Recovery OCP failures and other UNKNOWN outcomes do not become physical
    infeasibility claims. Returned manager_after is installed only after commit.
    """
    q = np.asarray(preference, float)
    if q.shape != (len(CL_ACTIONS),) or not np.isfinite(q).all():
        raise ValueError('Preference must be finite with nineteen entries')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    original_manager, original_state = digest(controller.state_dict()), digest(state)
    order = [int(a) for a in np.argsort(-q)]
    checked, attempts = [], []
    statuses, reasons = [UNKNOWN]*len(CL_ACTIONS), ['NOT_CHECKED']*len(CL_ACTIONS)
    mask = [False]*len(CL_ACTIONS)
    answer = dict(action=None, requested_action=order[0], requested_CL_raw=CL_ACTIONS[order[0]],
                  selected=None, z=None, cmd=None, details=None,
                  manager_after=None, controller_state=None, retry=None)

    def probe(action, retried=False):
        candidate = _clone_controller(controller)
        event = None
        raw = CL_ACTIONS[action]
        if candidate.memory is not None and candidate.memory['request_CL'] != raw:
            event = dict(event='RECOVERY_INVALIDATED_RAW_CL_CHANGE',
                         old_CL_raw=candidate.memory['request_CL'], new_CL_raw=raw,
                         old_epoch=candidate.epoch, old_exit_passes=candidate.exit_passes)
            candidate.memory, candidate.exit_passes = None, 0
        attempt_dir = folder/(('retry_' if retried else 'action_')+f'{action:02d}')
        attempt_dir.mkdir(parents=False, exist_ok=False)
        old_budget = candidate.guard.budget, candidate.guard.max_evaluations
        start = time.perf_counter()
        try:
            if retried:
                candidate.guard.budget = float(retry_options['wall_s'])
                candidate.guard.max_evaluations = int(retry_options['max_evaluations'])
            selected, z, cmd, detail, manager_state = candidate.decide(deepcopy(state), raw, attempt_dir)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, RuntimeError,
                FloatingPointError, OverflowError) as exc:
            detail = dict(events=[], error=repr(exc), selected_source=None,
                          candidate_context='UNCOMMITTED_TRIAL')
            selected = SafetyResult(False, 'CANDIDATE_NUMERICAL_OR_DOMAIN_FAILURE',
                None, None, None, None, dict(proof_status=UNKNOWN, error=repr(exc)))
            z, cmd, manager_state = None, None, candidate.state_dict()
        finally:
            candidate.guard.budget, candidate.guard.max_evaluations = old_budget
        if event is not None:
            detail.setdefault('events', []).insert(0, event)
        status = PASS if selected.accepted else selected.diagnostics.get('proof_status', classification(selected.status))
        if status == PASS and not selected.accepted:
            raise ValueError('Refused candidate cannot carry a PASS status')
        statuses[action], reasons[action] = status, None if selected.accepted else selected.status
        if action not in checked:
            checked.append(action)
        entry = dict(action=action, CL_raw=raw, accepted=bool(selected.accepted), status=status, reason=reasons[action],
                     wall_s=time.perf_counter()-start, retry=retried, details=deepcopy(detail),
                     diagnostics=deepcopy(selected.diagnostics), evidence_path=str(attempt_dir),
                     certificate_reused=False, live_state_unchanged=True)
        attempts.append(entry)
        write_json(attempt_dir/'candidate.json', entry)
        if digest(controller.state_dict()) != original_manager or digest(state) != original_state:
            raise RuntimeError('Candidate certification mutated live flight/controller state')
        if selected.accepted:
            mask[action] = True
            answer.update(action=action, selected=selected, z=z, cmd=cmd, details=detail,
                          manager_after=candidate, controller_state=manager_state)
        return selected.accepted

    for action in order:
        if probe(action):
            break
    eligible = [a for a in checked if statuses[a] == UNKNOWN
                and reasons[a] == 'SEARCH_BUDGET_EXHAUSTED']
    if answer['action'] is None and retry_options and retry_options['enabled'] and eligible:
        if (not np.isfinite(retry_options['wall_s'])
                or retry_options['wall_s'] < controller.guard.budget
                or type(retry_options['max_evaluations']) is not int
                or retry_options['max_evaluations'] < controller.guard.max_evaluations):
            raise ValueError('Dead-end retry must preserve or increase the configured budget')
        tick = time.perf_counter()
        retry = dict(actions_eligible=eligible, original_wall_s=controller.guard.budget,
                     original_max_evaluations=controller.guard.max_evaluations,
                     retry_wall_s=retry_options['wall_s'],
                     retry_max_evaluations=retry_options['max_evaluations'])
        for action in order:
            if action in eligible and probe(action, retried=True):
                break
        retry.update(recovered=answer['action'] is not None, wall_s=time.perf_counter()-tick)
        answer['retry'] = retry
    answer.update(checked=checked, statuses=statuses, reasons=reasons, mask=mask,
                  attempts=attempts, preference=q.tolist(), wall_s=time.perf_counter()-started,
                  certification_exhausted=answer['action'] is None, universal_infeasibility_proven=False,
                  failure_semantics='RUNTIME_CERTIFICATION_EXHAUSTED' if answer['action'] is None else None)
    write_json(folder/'pool.json', {k:v for k, v in answer.items()
               if k not in {'selected', 'z', 'manager_after', 'controller_state'}})
    return answer
