"""Policies as chains of improvement stages.

A policy maps the state at each 12 s decision to one of the 19 absolute lift-coefficient targets. It starts
from the initial action (CL 0.04, or a constant stage) and applies the accepted improvement stages in order;
each stage sees the action produced by the previous ones:

* ``constant``: replace the action by a fixed one.
* ``model``: advantage-ensemble stage. Within +-NEIGHBOURHOOD classes of the current action, pick the action
  with the largest lower confidence bound; switch only if that bound exceeds the stage margin and the bound
  of the current action. Active only within +-window seconds of the decision times that carried labels.
* ``override``: at a labelled decision time, replace the labelled base action by the measured better one.
  Stages that carry a state tolerance ``tol`` additionally require the state to match the labelled one.
"""
import numpy as np

from certified_flight import CL_ACTIONS

from .ensemble import lower_confidence_bound
from .features import features

DECISION_S = 12.
INITIAL_ACTION = CL_ACTIONS.index(.04)
NEIGHBOURHOOD = 2
MAX_ACTION = len(CL_ACTIONS) - 1


def preference(target_cl):
    """Ranking of the 19 actions by distance to the target CL (ties: lower index first); higher is preferred.

    The certifier probes actions in this order, so the executed action is the certifiable action closest to
    the policy's choice.
    """
    return -np.abs(np.asarray(CL_ACTIONS)-float(target_cl)) - 1e-9*np.arange(len(CL_ACTIONS))


def constant_stage(cl):
    return dict(kind='constant', action=CL_ACTIONS.index(cl), cl=float(cl))


def neighbourhood(a):
    return list(range(max(0, a-NEIGHBOURHOOD), min(MAX_ACTION, a+NEIGHBOURHOOD)+1))


def _override(stage, f, a):
    t = float(f[4])*1200
    gated = 'tol' in stage and stage.get('match') != 't'
    for e in stage['entries']:
        if abs(e['t']-t) < 1e-3 and a == e['base_action'] and (
                not gated or float(np.max(np.abs(np.asarray(e['features'][:4])-np.asarray(f[:4])))) <= stage['tol']):
            return e['action']
    return a


def _in_window(stage, t):
    return bool(np.any(np.abs(np.asarray(stage['times'])-t) <= stage['window']+1e-9))


def chain_action(chain, f):
    """Action of the policy `chain` for the feature vector f."""
    a = INITIAL_ACTION
    t = float(f[4])*1200
    for stage in chain:
        kind = stage.get('kind', 'model')
        if kind == 'constant':
            a = stage['action']
        elif kind == 'override':
            a = _override(stage, f, a)
        elif _in_window(stage, t):
            lcb, _, _ = lower_confidence_bound(stage['members'], f, stage.get('inputs'))
            cand = max(neighbourhood(a), key=lambda j: (lcb[j], -abs(j-a)))
            if cand != a and lcb[cand] > stage['margin'] and lcb[cand] > lcb[a]:
                a = cand
    return a


class ChainPolicy:
    """Decision rule for certified flights; decides at multiples of DECISION_S and holds the action in between.

    override: optional (decision time, action) imposed at one decision, or a dict {decision time: action}
              imposed at several decisions (coordinated block alternatives, opt-in --blocks).
    schedule: optional {decision time: action}; every other decision then replays this fixed sequence
              (open-loop continuation) instead of re-deciding from the state.
    """

    def __init__(self, chain, env, override=None, schedule=None):
        self.chain, self.env, self.override = chain, env, override
        self.overrides = ({int(round(float(t))): int(a) for t, a in override.items()} if isinstance(override, dict)
                          else None)
        self.schedule = None if schedule is None else {
            int(round(float(t))): int(a) for t, a in (schedule.items() if isinstance(schedule, dict) else schedule)}
        self.action = None
        self.decision_t = None
        self.decisions = []

    def decide(self):
        t = float(self.env.t)
        if self.decision_t is None or t >= self.decision_t+DECISION_S-1e-9:
            self.decision_t = DECISION_S*np.floor((t+1e-9)/DECISION_S)
            if self.overrides is not None and int(round(self.decision_t)) in self.overrides:
                self.action = self.overrides[int(round(self.decision_t))]
            elif self.overrides is None and self.override is not None and abs(self.decision_t-self.override[0]) < 1e-9:
                self.action = self.override[1]
            elif self.schedule is not None:
                self.action = self.schedule[int(round(self.decision_t))]
            else:
                self.action = chain_action(self.chain, features(self.env))
            self.decisions.append((self.decision_t, self.action))
        return self.action

    def preference(self):
        return preference(CL_ACTIONS[self.decide()])
