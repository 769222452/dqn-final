"""Certified flights, branch labels and candidate evaluation.

Every 3 s step of every flight is executed only after its certificate passed (see certified_flight). The
simulator is deterministic, so advantages are measured exactly by branching from a copy of the same state:

  A(s, a) = G_rest(a) - G_rest(pi(s)),

where G_rest(a) is the return from s when action a is held for one 12 s decision and the flight is then
continued to the end of the mission. The unchanged branch is the nominal trajectory itself, so
G_rest(pi(s)) = G_pi - G_prefix(s) costs no extra flight. The continuation after the branch decision is either

* open loop (proposed): replay the nominal decision sequence of pi, so the label measures the effect of the
  action itself and stays stationary while pi changes elsewhere; or
* closed loop (baseline): pi re-decides from the perturbed states, so the label also contains pi's reaction.
"""
import time
import traceback
from pathlib import Path

import numpy as np

from certified_flight import CL_ACTIONS, OracleRequired, build_flight

from .features import features
from .policy import DECISION_S, INITIAL_ACTION, MAX_ACTION, ChainPolicy, chain_action

MISSION_S = 1200.
RADIUS_BLOCK_S = 96.   # the alternative radius (1 or 2 classes) alternates between consecutive 96 s blocks


def certified_fly(certifier, policy, until=None):
    """Fly `policy` with certification until `until` (s) or the end. Returns (G, stop reason, actions)."""
    env = certifier.env
    G = 0.
    stop = None
    acts = []
    while until is None or env.t < until-1e-9:
        pref = policy.preference()
        pool = certifier.pool(pref)
        if not pool['mask'].any():
            stop = 'NEXT_STATE_NOT_CERTIFIED'
            break
        a = int(np.argmax(np.where(pool['mask'], pref, -np.inf)))
        try:
            r, done, _ = certifier.commit(pool, a)
        except OracleRequired as exc:
            stop = 'EXECUTION_REFUSED:'+exc.reason
            break
        G += float(r)
        acts.append(a)
        if done:
            stop = 'DONE'
            break
    return G, stop, acts


def alternatives(a0, t, sweep=0):
    """Branch actions at decision time t: pi(s)+-1 or pi(s)+-2, alternating between 96 s blocks of the mission
    and between sweeps, so that two sweeps test all four neighbours of every state."""
    k = 1 if (int((t+1e-9)//RADIUS_BLOCK_S)+sweep) % 2 == 0 else 2
    return [a for a in (a0-k, a0+k) if 0 <= a <= MAX_ACTION]


def branch_job(args):
    """Fly pi to each decision time and measure the advantages of the alternatives there.

    args: (folder, chain, times, sweep, schedule); schedule = pi's nominal decisions for open-loop
    continuation or None for closed-loop continuation.
    """
    folder, chain, times, sweep, schedule = args
    sched = None if schedule is None else {int(round(float(t))): int(a) for t, a in schedule}
    try:
        folder = Path(folder)
        env, certifier = build_flight(folder/'base', 'branch')
        pi = ChainPolicy(chain, env)
        rows = []
        prefix = {}
        started = time.perf_counter()
        G_acc = 0.
        steps = 0
        for t_b in times:
            G_pre, stop, pre_acts = certified_fly(certifier, pi, until=t_b)
            G_acc += G_pre
            steps += len(pre_acts)
            if stop is not None:
                return dict(error=f'nominal flight stopped ({stop}) before t={t_b}', rows=rows)
            f = features(env).tolist()
            a0 = chain_action(chain, np.asarray(f))
            prefix[t_b] = (G_acc, a0, f)
            if sched is not None and sched.get(int(round(t_b))) != a0:
                return dict(error=f'schedule mismatch at t={t_b}: {sched.get(int(round(t_b)))} vs {a0}', rows=rows)
            for a in alternatives(a0, t_b, sweep):
                tick = time.perf_counter()
                branch = certifier.fork(folder/f't{int(t_b):04d}_a{a:02d}')
                pol = ChainPolicy(chain, branch.env, override=(t_b, a), schedule=sched)
                g1, s1, acts = certified_fly(branch, pol, until=t_b+DECISION_S)
                if s1 is not None and s1 != 'DONE':
                    rows.append(dict(t=t_b, action=a, base_action=a0, features=f, valid=False, stop=s1))
                    continue
                g2, s2, acts2 = (0., 'DONE', []) if s1 == 'DONE' else certified_fly(branch, pol)
                steps += len(acts)+len(acts2)
                rows.append(dict(t=t_b, action=a, base_action=a0, features=f, executed=acts[:4], G_rest=g1+g2,
                                 valid=s2 == 'DONE', stop=s2, wall_s=time.perf_counter()-tick))
        return dict(rows=rows, prefix={str(t): list(x) for t, x in prefix.items()}, steps=steps,
                    wall_s=time.perf_counter()-started)
    except BaseException:
        return dict(error=traceback.format_exc(), rows=[])


def candidate_job(args):
    """Fly a candidate policy over the full mission. args: (folder, chain[, initial_offset])."""
    folder, chain = args[:2]
    offset = args[2] if len(args) > 2 else None
    try:
        env, certifier = build_flight(Path(folder)/'run', 'candidate', initial_offset=offset)
        pi = ChainPolicy(chain, env)
        tick = time.perf_counter()
        G, stop, acts = certified_fly(certifier, pi)
        counts = np.bincount(acts, minlength=len(CL_ACTIONS))
        return dict(steps=len(acts), G=G, stop=stop, success=bool(stop == 'DONE' and env.t >= MISSION_S-1e-6),
                    range_km=float(env.x[3]/1000),
                    decisions_changed=int(sum(a != INITIAL_ACTION for _, a in pi.decisions)), decisions=len(pi.decisions),
                    switches=int(sum(a != b for a, b in zip(acts, acts[1:]))), action_counts=counts.tolist(),
                    mean_CL=float(counts @ np.asarray(CL_ACTIONS)/max(counts.sum(), 1)), wall_s=time.perf_counter()-tick,
                    schedule=[[float(t), int(x)] for t, x in pi.decisions])
    except BaseException:
        return dict(error=traceback.format_exc())


def dispatch(job):
    kind, args = job
    return branch_job(args) if kind == 'branch' else candidate_job(args)
