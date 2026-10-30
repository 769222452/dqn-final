"""Exact-state action certificates, the certified action pool and certified execution.

A certificate is the result of executing one CL action on a sandbox copy of the
environment at the exact current state (surrogate, physics fallback or true OCP,
followed by the corrector, lookahead guard and audits). Certificates are cached
only under the SHA-256 of the exact state and certifier tag; there is no reuse
across nearby states. An action may be executed only with a VERIFIED_PASS
certificate for the current state, and execution re-integrates the plant.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .corrector import audit_controls, compact_audit
from .diagnostics import interval_moments
from .oracle_evidence import oracle_sinks
from .serialization import append_jsonl, write_json
from .state import STATE_FIELDS, capture_state
from .surrogate import OracleRequired
from .transition import Transition

CL_ACTIONS = [-.01, 0., .01, .015, .02, .025, .03, .035, .04, .045, .05, .055, .06, .07, .08, .09, .10, .11, .12]

PASS = 'VERIFIED_PASS'
REJECT = 'VERIFIED_REJECT'
UNKNOWN = 'UNKNOWN_NOT_CERTIFIED'
NO_SOLUTION = 'OCP_NO_SOLUTION_FOUND'
AUDIT_REJECTIONS = frozenset({
    'DENSE_AUDIT_REJECTED', 'FUTURE_DENSE_AUDIT_REJECTED',
    'FAST_EXECUTION_AUDIT_FAILURE', 'FAST_PLANT_RK_STAGE_FAILURE',
    'TRUE_OCP_PLAN_AUDIT_REJECTED', 'PRECOMMIT_AUDIT_REJECTED',
    'CERTIFICATE_EXECUTION_AUDIT_REJECTED',
})


def classification(reason):
    """Certificate status of a refusal reason."""
    if reason == 'TRUE_OCP_SOLVE_FAILED':
        return NO_SOLUTION
    # A named audit exception may be an I/O or solver infrastructure error, not
    # evidence that a trajectory violated a physical constraint. Reject only
    # refusal codes emitted after an actual independent audit failed.
    if reason in AUDIT_REJECTIONS:
        return REJECT
    return UNKNOWN


def state_key(env, tag):
    """SHA-256 of the double-precision state, transition, history and certifier tag."""
    return hashlib.sha256(json.dumps(dict(state=capture_state(env), context=tag),
                                     sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@contextmanager
def exact_integrations(plant, enabled=True):
    """Memoise plant.integrate on exact argument bytes within one certification.

    Returns independent copies because callers add audit flags to the results.
    """
    original = plant.integrate
    cache = {}
    stats = dict(hits=0, misses=0)

    def integrate(x, u, tt, source, target, w0, rate, dt):
        key = tuple((np.asarray(v, dtype=np.float64).shape, np.asarray(v, dtype=np.float64).tobytes())
                    for v in (x, u, tt))+(int(source), int(target), float(w0), float(rate), float(dt))
        if key in cache:
            stats['hits'] += 1
            return deepcopy(cache[key])
        stats['misses'] += 1
        r = original(x, u, tt, source, target, w0, rate, dt)
        if len(cache) < 256:
            cache[key] = deepcopy(r)
        return r

    if enabled:
        plant.integrate = integrate
    try:
        yield stats
    finally:
        plant.integrate = original


@dataclass
class Certificate:
    key: str
    action: int
    status: str
    before: dict
    after: dict | None
    info: dict | None
    failure: dict | None
    proof_id: str
    wall_s: float


class Certifier:
    """Certifies actions of a CertifiedFlightEnvironment and executes certified actions.

    Writes proofs/pNNNNNN.json per certificate and, next to the proof folder,
    pools.jsonl (one row per pool), physics_events.jsonl (physics-fallback
    attempts) and retry_events.jsonl (bounded retries).
    """

    def __init__(self, env, folder, tag, memo=True, retry_options=None):
        self.env = env
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.tag = deepcopy(tag)
        self.memo = memo
        self.cache = {}
        self.probes = max([int(p.stem[1:]) for p in self.folder.glob('p*.json')]+[0])
        self.cache_hits = 0
        self.records = []
        self.oracle_events = []
        self.pool_records = []
        self.retry_options = dict(retry_options or {'enabled': False})
        self.retry_records = []

    def key(self):
        return state_key(self.env, self.tag)

    def certify(self, action):
        """Certificate of one action at the current exact state (cached per state key)."""
        events = self.env.physics_events
        start = len(events)
        certificate = self._certify(action)
        for event in events[start:]:
            event.update(proof_id=certificate.proof_id, certificate_status=certificate.status,
                         certificate_only=True, committed_to_real_environment=False,
                         certified=bool(certificate.status == PASS and certificate.info
                                        and certificate.info.get('channel') == 'PHYSICS_BASELINE_ON_OOD'))
            append_jsonl(self.folder.parent/'physics_events.jsonl', event)
        return certificate

    def _certify(self, action):
        key = self.key()
        pair = (key, int(action))
        if pair in self.cache:
            self.cache_hits += 1
            return self.cache[pair]
        before = capture_state(self.env)
        self.probes += 1
        index = self.probes
        sandbox = self.env._candidate(self.env.controller)
        sandbox.oracle_events = []
        sandbox.attempt_counter = index-1
        sink = self.env.event_sink
        if sink:
            sandbox.event_sink = lambda event: sink(dict(event, certificate_only=True,
                                                         committed_to_real_environment=False))
        start = time.perf_counter()
        info = after = failure = None
        with exact_integrations(self.env.ocp.plant, self.memo) as memo:
            try:
                _, _, info = sandbox.step(int(action))
                after = capture_state(sandbox)
                status = PASS
            except OracleRequired as exc:
                failure = dict(reason=exc.reason, head=exc.head, context=deepcopy(exc.context),
                               diagnostics=deepcopy(exc.diagnostics))
                status = classification(exc.reason)
        assert capture_state(self.env) == before
        self.oracle_events.extend(sandbox.oracle_events)
        proof_id = f'p{index:06d}'
        wall = time.perf_counter()-start
        c = Certificate(key, int(action), status, before, after, info, failure, proof_id, wall)
        record = dict(key=key, action=int(action), CL=self.env.cl_actions[int(action)], status=status,
                      before=before, after=after, info=info, failure=failure, wall_s=wall, exact_integrations=memo,
                      real_state_unchanged=True, oracle_events=sandbox.oracle_events)
        write_json(self.folder/(proof_id+'.json'), record)
        self.records.append(dict(key=key, action=int(action), status=status, wall_s=wall, memo=memo,
                                 reason=None if failure is None else failure['reason'], proof_id=proof_id))
        # A budget timeout is unknown, not permanently blacklisted; only an explicit
        # retry (see pool) deletes the cached certificate and searches again.
        self.cache[pair] = c
        return c

    def pool(self, preference, deadline=None):
        """Certified action pool at the current state.

        Actions are probed in the order np.argsort(-preference) until one passes;
        if none passes, every action is probed. If the pool is empty and some
        actions only ran out of lookahead search budget, those actions are
        certified once more with the larger retry budget. A realtime deadline
        (perf_counter time) stops probing and is never given a retry.
        Returns a dict with 'mask' (bool per action), 'certificates', 'checked'.
        """
        pool = self._search(np.asarray(preference, dtype=float), deadline)
        options = self.retry_options
        eligible = [a for a in pool['checked']
                    if (c := self.cache[(pool['key'], a)]).status == UNKNOWN
                    and (c.failure or {}).get('reason') == 'SEARCH_BUDGET_EXHAUSTED']
        if not pool['mask'].any() and options['enabled'] and eligible and deadline is None:
            old_pool = pool
            guard = self.env.lookahead_guard
            original = guard.budget, guard.max_evaluations
            event = dict(initial_pool_id=pool['pool_id'], key=pool['key'],
                         time_s=self.env.t, actions_eligible=eligible,
                         superseded_proofs={str(a): self.cache[(pool['key'], a)].proof_id for a in eligible},
                         original_wall_s=original[0], original_max_evaluations=original[1],
                         retry_wall_s=options['wall_s'], retry_max_evaluations=options['max_evaluations'])
            for action in eligible:
                del self.cache[(pool['key'], action)]  # old proof files and records are kept
            tick = time.perf_counter()
            try:
                guard.budget = float(options['wall_s'])
                guard.max_evaluations = int(options['max_evaluations'])
                pool = self._search(np.asarray(preference, dtype=float), deadline)
            finally:
                guard.budget, guard.max_evaluations = original
            if pool['key'] != old_pool['key']:
                raise RuntimeError('Retry changed the physical state or certificate context')
            event.update(retry_pool_id=pool['pool_id'], recovered=bool(pool['mask'].any()),
                         wall_s=time.perf_counter()-tick, pass_actions=np.flatnonzero(pool['mask']).tolist())
            self.retry_records.append(event)
            append_jsonl(self.folder.parent/'retry_events.jsonl', event)
            pool['wall_s'] += old_pool['wall_s']
            pool['retry'] = event
        pool['certification_exhausted'] = not bool(pool['mask'].any())
        pool['failure_semantics'] = 'RUNTIME_CERTIFICATION_EXHAUSTED' if pool['certification_exhausted'] else None
        return pool

    def _search(self, q, deadline=None):
        start = time.perf_counter()
        env = self.env
        if q.shape != (env.n_actions,) or not np.isfinite(q).all():
            raise ValueError(f'preference must be a finite array of shape ({env.n_actions},)')
        mask = np.zeros(env.n_actions, bool)
        certs = {}
        checked = []
        statuses = [UNKNOWN]*env.n_actions
        reasons = ['NOT_CHECKED']*env.n_actions
        proof_ids = [None]*env.n_actions

        def late():
            return deadline is not None and time.perf_counter() >= deadline

        def probe(a):
            a = int(a)
            if a in checked or late():
                return
            c = self.certify(a)
            checked.append(a)
            statuses[a] = c.status
            reasons[a] = (c.failure or {}).get('reason')
            proof_ids[a] = c.proof_id
            if c.status == PASS:
                mask[a] = True
                certs[a] = c

        order = [int(a) for a in np.argsort(-q)]
        for a in order:
            probe(a)
            if mask.any() or late():
                break
        if not mask.any():
            for a in order:
                probe(a)
        timed_out = deadline is not None and time.perf_counter() > deadline
        row = dict(pool_id=len(self.pool_records)+1, key=self.key(), time_s=env.t,
                   checked=checked, statuses=statuses, reasons=reasons, proof_ids=proof_ids,
                   mask=mask.tolist(), preference=q.tolist(), wall_s=time.perf_counter()-start,
                   deadline_exceeded=timed_out)
        self.pool_records.append(row)
        append_jsonl(self.folder.parent/'pools.jsonl', row)
        return dict(key=row['key'], mask=mask, certificates=certs, preference=q,
                    checked=checked, wall_s=time.perf_counter()-start, deadline_exceeded=timed_out,
                    deadline=deadline, pool_id=row['pool_id'])

    def commit(self, pool, action):
        """Execute a certified action on the real environment; returns (reward, done, info).

        The plan is re-integrated and must reproduce the certified endpoint, reward
        and bookkeeping exactly. Raises OracleRequired if the execution audit or a
        realtime deadline fails; the state is unchanged in that case.
        """
        reward, done, info, rollout = self._execute(pool, action)
        # Absolute-CL audit of the executed chain (after the state was installed).
        plant = self.env.ocp.plant
        assert np.array_equal(rollout['x'][-1], self.env.x)
        power = (rollout['samples'][1]*rollout['samples'][10]).reshape(-1, 5)[:, :4]
        info['drag_dissipation_J_kg'] = float(.5*np.sum(power@np.array([1., 2., 2., 1.])/6)
                                              / plant.mission.data['model']['mass_kg'])
        raw = CL_ACTIONS[int(action)]
        cmd = info['CL_cmd']
        assert info['CL_raw'] == raw
        assert info['pre_context']['CL_raw'] == raw and info['post_context']['CL_raw'] == raw
        assert info['pre_context']['CL_des'] == cmd and info['post_context']['CL_des'] == cmd
        assert info['generator']['CL_des'] == cmd
        for trial in info.get('lookahead', {}).get('trials', []):
            for forecast in trial.get('forecasts', []):
                for segment in forecast.get('control_sequence', []):
                    assert segment['context']['CL_raw'] == raw
        info['absolute_CL_chain_verified'] = True
        return reward, done, info

    def _execute(self, pool, action):
        action = int(action)
        if pool['key'] != self.key() or action not in pool['certificates'] or not pool['mask'][action]:
            raise ValueError('Stale or uncertified action')
        if pool.get('deadline_exceeded'):
            raise ValueError('Realtime certification deadline exceeded')
        c = pool['certificates'][action]
        if c.key != self.key() or c.status != PASS:
            raise ValueError('Invalid proof')
        env = self.env
        info = deepcopy(c.info)
        # Fresh independent integration at execution time. A cached state is
        # NEVER installed as a substitute for actual physics.
        au = audit_controls(env.ocp.plant, info['post_context'], info['plant']['alpha'])
        if not au['valid']:
            raise OracleRequired('CERTIFICATE_EXECUTION_AUDIT_REJECTED', info['post_context'], 'post',
                                 compact_audit(au))
        actual = au['actual']
        new_x = actual['x'][-1].copy()
        if not np.array_equal(new_x, np.asarray(c.after['x'])):
            raise RuntimeError('Certified endpoint no longer matches exact dynamics')
        t1 = env.t+3
        outcome = env.m.outcome(t1, new_x)
        reward = env.m.reward(float(new_x[3]-env.x[3]), outcome)
        reward['original_reward'] = reward['reward']
        reward['switch_cost'] = float(env.control.get('switch_cost', 0.)) if info.get('transition_event') else 0.
        reward['reward'] -= reward['switch_cost']
        if reward != info['reward']:
            raise RuntimeError('Certificate reward mismatch')
        z = Transition(info['post_context']['source'], info['post_context']['target'], info['transition_start_s'],
                       env.cfg['transition']['duration_s'])
        last = env.last_transition_completed
        if z.source != z.target and not z.active(t1):
            last = z.started+z.duration
        z.settle(t1)
        moments = interval_moments(actual, horizon=3., target=info['CL_cmd'])
        # All checks finish before physical/history/cumulative reward commit.
        real_env = env
        env = real_env._candidate(real_env.controller)
        env.x = new_x
        env.alpha = float(actual['alpha'][-1])
        env.t = t1
        env.transition = z
        env.last_transition_completed = last
        env.switches += int(info['transition_started'])
        env.previous_cl_raw = info['CL_raw']
        env.previous_cl_cmd = info['CL_cmd']
        env.previous_cl_actual = moments['mean']
        env.history_valid = True
        env.done = bool(reward['terminated'] or reward['truncated'])
        env.last_outcome = outcome
        env.total_reward += reward['reward']
        env.decision_index += 1
        if capture_state(env) != c.after:
            raise RuntimeError('Execution bookkeeping drift')
        if pool.get('deadline') is not None and time.perf_counter() > pool['deadline']:
            raise OracleRequired('STRICT_EXECUTION_DEADLINE_EXCEEDED', info['post_context'], 'post',
                                 dict(valid_control=True, committed=False))
        for name in STATE_FIELDS:
            setattr(real_env, name, deepcopy(getattr(env, name)))
        info.update(certificate_key=c.key, certificate_id=c.proof_id,
                    actual_execution_reintegrated=True, execution_audit=compact_audit(au))
        # No stale control certificates survive under a different exact state.
        self.cache = {k: v for k, v in self.cache.items() if k[0] == self.key()}
        return reward['reward'], real_env.done, info, actual

    def fork(self, folder):
        """Independent copy sharing only the plant and the aerodynamic database; proofs go to folder/'proofs'."""
        env = self.env
        branch = deepcopy(self, {id(env.ocp.plant): env.ocp.plant, id(env.db): env.db})
        branch.folder = Path(folder)/'proofs'
        branch.folder.mkdir(parents=True, exist_ok=True)
        branch.probes = max([branch.probes]+[int(p.stem[1:]) for p in branch.folder.glob('p*.json')])
        event_sink, artifact_sink = oracle_sinks(folder)
        if env.event_sink is not None:
            branch.env.event_sink = event_sink
        if env.artifact_sink is not None:
            branch.env.artifact_sink = artifact_sink
        return branch
