"""Original frozen chain plus exact-support-gated improvement stages."""
from copy import deepcopy

import numpy as np

from certified_flight import CL_ACTIONS
from rollout_improvement.ensemble import lower_confidence_bound
from rollout_improvement.features import features
from rollout_improvement.policy import (ChainPolicy, DECISION_S, INITIAL_ACTION,
    chain_action, constant_stage, neighbourhood)
from .support import (canonical_hash, prefix_chain_hash, support_context,
                      support_decision, validate_support_mode)

_INHERIT = object()


def policy_action(chain, env, record=False):
    """Pure decision query; record=True returns (action, evidence), never appends logs."""
    f = features(env)
    if not any(s.get('kind') == 'supported_model' for s in chain):
        action = int(chain_action(chain, f))
        evidence = dict(t=float(env.t), baseline_action=action, requested_action=action,
                        executed_action=None, stages=[], executions=[], protocol='supported-recovery-decision/1')
        return (action, evidence) if record else action
    context = support_context(env)
    action, legacy, records, baseline = INITIAL_ACTION, [], [], None
    for index, stage in enumerate(chain):
        if stage.get('kind', 'model') != 'supported_model':
            if stage.get('kind', 'model') not in ('model', 'constant', 'override'):
                raise ValueError('Unknown supported-policy stage kind')
            legacy.append(stage)
            continue
        if legacy:
            action = int(chain_action([constant_stage(CL_ACTIONS[action])] + legacy, f))
            legacy = []
        if baseline is None:
            baseline = action
        incoming = action
        candidates = neighbourhood(incoming)
        prefix = prefix_chain_hash(chain[:index])
        candidate_support = [support_decision(stage, context,
            reference_prefix_sha256=prefix, base_action=incoming, candidate_action=a) for a in candidates]
        scores, mean, std = lower_confidence_bound(stage['members'], f, stage.get('inputs'))
        if not np.isfinite(np.r_[scores, mean, std]).all():
            raise ValueError('Non-finite supported-model predictions')
        for row in candidate_support:
            row['score'] = float(scores[row['action']])
        ungated = max(candidates, key=lambda a: (scores[a], -abs(a-incoming)))
        eligible = [r['action'] for r in candidate_support if r['supported']]
        choices = sorted(set(eligible + [incoming]))
        proposed = max(choices, key=lambda a: (scores[a], -abs(a-incoming)))
        if proposed != incoming and scores[proposed] > stage['margin'] and scores[proposed] > scores[incoming]:
            action = int(proposed)
        reason = ('SUPPORTED_LCB_IMPROVEMENT' if action != incoming else
                  'NO_SUPPORTED_ALTERNATIVE' if not any(a != incoming for a in eligible) else
                  'LCB_MARGIN_OR_BASE_NOT_EXCEEDED')
        records.append(dict(stage_index=index, reference_prefix_sha256=prefix,
            context_sha256=canonical_hash(context), incoming_action=incoming,
            base_action=incoming, proposed_action=int(ungated), selected_action=action,
            candidates=candidate_support,
            candidate_support=candidate_support, ungated_proposed_action=int(ungated),
            supported_proposed_action=int(proposed), requested_action=action,
            reason=reason, exact_support=next((r['exact_support'] for r in candidate_support if r['action'] == action), False),
            lcb={str(a): float(scores[a]) for a in candidates}, margin=float(stage['margin'])))
    if legacy:
        action = int(chain_action([constant_stage(CL_ACTIONS[action])] + legacy, f))
    evidence = dict(protocol='supported-recovery-decision/1', t=float(env.t),
                    context_sha256=canonical_hash(context), baseline_action=baseline,
                    requested_action=action, executed_action=None, stages=records, executions=[])
    return (action, evidence) if record else action


class SupportPolicy(ChainPolicy):
    """The same twelve-second scheduler, with explicit support and execution logs."""

    def __init__(self, chain, env, override=None, schedule=None):
        # ChainPolicy prepares original stages exactly as before. New stages own
        # immutable arrays separately, preserving saved-chain hash arithmetic.
        super().__init__(deepcopy(chain), env, override=override, schedule=schedule)
        for stage in self.chain:
            if stage.get('kind') == 'supported_model':
                validate_support_mode(stage.get('support_mode', 'exact'))
                members = []
                for member in stage['members']:
                    params = []
                    for weight, bias in member:
                        w, b = np.array(weight, copy=True), np.array(bias, copy=True)
                        w.setflags(write=False)
                        b.setflags(write=False)
                        params.append((w, b))
                    members.append(params)
                stage['members'] = members
        self.logs = []

    def decide(self):
        t = float(self.env.t)
        if self.decision_t is None or t >= self.decision_t + DECISION_S - 1e-9:
            self.decision_t = DECISION_S*np.floor((t+1e-9)/DECISION_S)
            if self.override is not None and abs(self.decision_t-self.override[0]) < 1e-9:
                self.action = self.override[1]
                log = dict(t=t, requested_action=self.action, reason='BRANCH_OVERRIDE', stages=[],
                           executed_action=None, executions=[], protocol='supported-recovery-decision/1')
            elif self.schedule is not None:
                self.action = self.schedule[int(round(self.decision_t))]
                log = dict(t=t, requested_action=self.action, reason='CONTINUATION_SCHEDULE', stages=[],
                           executed_action=None, executions=[], protocol='supported-recovery-decision/1')
            else:
                self.action, log = policy_action(self.chain, self.env, record=True)
            self.decisions.append((self.decision_t, self.action))
            self.logs.append(log)
        return self.action

    def record_execution(self, action, *, t=None):
        """Called only after successful certification commit, never by inference."""
        if type(action) is not int or not 0 <= action < len(CL_ACTIONS) or self.action is None:
            raise ValueError('A committed action and an existing policy decision are required')
        time = float(self.env.t if t is None else t)
        if not self.logs or self.logs[-1]['t'] < self.decision_t:
            self.logs.append(dict(protocol='supported-recovery-decision/1', t=float(self.decision_t),
                requested_action=self.action, reason='RESTORED_HELD_DECISION', stages=[],
                executed_action=None, executions=[]))
        self.logs[-1]['executed_action'] = action
        self.logs[-1]['executions'].append(dict(t=time, requested_action=self.action,
                                               executed_action=action))


def make_policy(chain, env, override=None, schedule=None):
    return SupportPolicy(chain, env, override=override, schedule=schedule)


def restore_policy(chain, env, snapshot, override=_INHERIT, schedule=_INHERIT):
    """Use original snapshot validation, then restore our own scheduler class."""
    from continuous_recovery.training_backend import restore_policy as validate_restore
    options = {}
    if override is not _INHERIT:
        options['override'] = override
    if schedule is not _INHERIT:
        options['schedule'] = schedule
    checked = validate_restore(chain, env, snapshot, **options)
    policy = make_policy(chain, env, override=checked.override, schedule=checked.schedule)
    policy.action, policy.decision_t = checked.action, checked.decision_t
    policy.decisions = deepcopy(checked.decisions)
    return policy
