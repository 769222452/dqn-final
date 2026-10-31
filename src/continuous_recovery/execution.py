"""Single-use execution of an already audited, exactly bound three-second plan.

The fresh certificate retains both independent .1 s and .5 s audits.  Submission
checks the same physical/control/mission inputs and consumes an immutable copy of
their results; it never integrates the same plan a second time.  This is a local
transaction object, not a checkpoint format or a cross-state PASS cache.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

PROTOCOL = 'continuous-recovery-execution/1'
_GRID = np.arange(7)*.5
_SYSTEM_KEYS = ('config', 'original', 'identity', 'model_hash', 'controller_protocol')
_OWNER_KEYS = ('mission', 'plant', 'db')


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported execution evidence type: {type(value).__name__}')


def _pack(value):
    return json.dumps(value, default=_json_default, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


def _pack_arrays(value):
    """Small JSON metadata plus exact ndarray bytes; no hex expansion or copies
    of the large witness.  The sealed payload is an internal immutable buffer.
    """
    buffers = []
    offset = 0

    def encode(array):
        nonlocal offset
        if not isinstance(array, np.ndarray):
            return _json_default(array)
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError('Execution evidence must contain finite numeric arrays')
        a = np.ascontiguousarray(array)
        raw = a.tobytes()
        description = {'__execution_array__': True, 'dtype': a.dtype.str,
                       'shape': list(a.shape), 'offset': offset, 'size': len(raw)}
        buffers.append(raw)
        offset += len(raw)
        return description

    header = json.dumps(value, default=encode, allow_nan=False,
                        sort_keys=True, separators=(',', ':')).encode('utf-8')
    return b''.join((len(header).to_bytes(8, 'little'), header, *buffers))


def _unpack_arrays(payload):
    end = 8+int.from_bytes(payload[:8], 'little')
    raw = memoryview(payload)[end:]

    def decode(value):
        if value.get('__execution_array__') is True:
            start = value['offset']
            # Copies own their data; no writable view points into the seal.
            return np.frombuffer(raw[start:start+value['size']], dtype=value['dtype']).reshape(value['shape']).copy()
        return value

    return json.loads(payload[8:end], object_hook=decode)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _evidence_sha(binding, payload):
    h = hashlib.sha256()
    for component in (*binding, payload):
        h.update(len(component).to_bytes(8, 'little'))
        h.update(component)
    return h.hexdigest()


def _system_binding(system):
    # Source/model identities and complete experiment/safety/mission data are
    # immutable during a transaction.  Compiled map caches are deliberately not
    # included: adding a compiled function is not a change to the physical RHS.
    result = {key: system.get(key) for key in _SYSTEM_KEYS}
    result['mission_data'] = system['mission'].data
    return result


def _binding(system, state, transition, identity, cl, witness, nodes, times):
    # Serialize every component exactly once.  Keep these immutable bytes so
    # commit can compare directly, without decoding and re-encoding the witness.
    return tuple(_pack(v) for v in (state, vars(transition), identity, float(cl), witness,
                                    np.asarray(nodes, float), np.asarray(times, float),
                                    _system_binding(system)))


def _is_true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _validate_current_audit(state, cl, checked):
    """Validate the complete accepted current segment without solving dynamics."""
    if (not _is_true(checked.accepted)
            or checked.diagnostics.get('proof_status', 'VERIFIED_PASS') != 'VERIFIED_PASS'):
        raise ValueError('Execution requires a current PASS certificate')
    nodes, times = np.asarray(checked.alpha_nodes, float), np.asarray(checked.times, float)
    if (nodes.shape != (7,) or not np.isfinite(nodes).all()
            or not np.array_equal(times, _GRID) or nodes[0] != float(state['alpha'])
            or not np.isfinite(float(cl))):
        raise ValueError('Execution evidence must use the exact continuous three-second plan')
    x0 = np.asarray(state['x'], float)
    if x0.shape != (4,) or not np.isfinite(x0).all():
        raise ValueError('Execution state must contain four finite physical values')
    audit, arrays = checked.audit, checked.arrays
    if (not isinstance(audit, dict) or not _is_true(audit.get('valid'))
            or not _is_true(audit.get('continuous_at_entry'))
            or not _is_true(audit.get('control_valid'))
            or not isinstance(arrays, dict)
            or set(arrays) != {'0.1', '0.5'}
            or set(audit.get('grids', {})) != {'0.1', '0.5'}):
        raise ValueError('Execution requires both complete independent grid audits')
    if 'CL_target' in audit and float(audit['CL_target']) != float(cl):
        raise ValueError('Execution audit CL target mismatch')
    for grid in ('0.1', '0.5'):
        dt, r, summary = float(grid), arrays[grid], audit['grids'][grid]
        n = round(3./dt)
        if (not _is_true(summary.get('valid')) or not _is_true(r.get('valid'))
                or r.get('dt') != dt):
            raise ValueError('Execution grid is missing its successful audit')
        shapes = {'x': (n, 4), 'samples': (12, 5*n), 'times': (n,),
                  'alpha': (n,), 'alpha_end': (n,), 'alpha_rate': (n,)}
        numeric = {}
        for key, shape in shapes.items():
            a = np.asarray(r.get(key), float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'Invalid execution grid array: {grid}/{key}')
            numeric[key] = a
        tt = np.arange(n)*dt
        index = np.clip(np.searchsorted(times, tt+1e-9, side='right')-1, 0, len(nodes)-2)
        if (not np.array_equal(numeric['times'], tt)
                or not np.array_equal(numeric['alpha'], np.interp(tt, times, nodes))
                or not np.array_equal(numeric['alpha_end'], np.interp(tt+dt, times, nodes))
                or not np.array_equal(numeric['alpha_rate'], (np.diff(nodes)/np.diff(times))[index])
                or not np.array_equal(numeric['samples'][:4, 0], x0)
                or not np.array_equal(numeric['samples'][:4, 4::5].T, numeric['x'])
                or not np.array_equal(numeric['x'][-1], summary.get('end_state'))):
            raise ValueError('Execution arrays disagree with state, controls, grid or audit endpoint')
    # Serialization in seal_execution rejects all remaining non-finite values;
    # do not traverse and encode the exact same audit/arrays a second time here.
    return nodes, times


@dataclass(frozen=True, slots=True, init=False, repr=False)
class CertifiedExecution:
    """An in-memory single-use capability, created only by ``seal_execution``.

    All evidence data live in immutable bytes.  Public properties and a commit
    return independent copies, so consumers cannot mutate sealed arrays/proofs.
    Python's reflective ``object.__setattr__`` is not a security boundary; this
    contract prevents accidental mutation and stale/mismatched transactions.
    """
    _payload: bytes
    _binding: tuple
    _proof: bytes
    _owners: tuple
    _consumed: bool

    def __init__(self):
        raise TypeError('Use seal_execution to create certified execution evidence')

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol):
        raise TypeError('Executable evidence cannot be checkpointed; obtain a fresh certificate')

    @property
    def proof(self):
        return json.loads(self._proof)

    @property
    def consumed(self):
        return self._consumed


def seal_execution(system, state, transition, identity, cl, witness, checked):
    """Freeze a fresh accepted current audit together with its exact inputs."""
    nodes, times = _validate_current_audit(state, cl, checked)
    if float(state['t'])+3. > float(system['mission'].data['duration_s']):
        raise ValueError('Execution would extend beyond the mission end')
    binding = _binding(system, state, transition, identity, cl, witness, nodes, times)
    # guard.audit adds CL_target to the plant result; old commit_audited returned
    # the plant result.  Retain the original output shape and arithmetic exactly.
    audit = dict(checked.audit)
    audit.pop('CL_target', None)
    payload = _pack_arrays(dict(audit=audit, arrays=checked.arrays))
    proof = dict(protocol=PROTOCOL, state_hash=_sha(binding[0]), nodes_hash=_sha(binding[5]),
        identity_hash=identity, transition_hash=_sha(binding[1]), CL_cmd=float(cl),
        times_s=times.tolist(), endpoint=np.asarray(checked.arrays['0.5']['x'])[-1].tolist(),
        witness_hash=_sha(binding[4]), system_hash=_sha(binding[7]),
        execution_evidence_sha256=_evidence_sha(binding, payload), execution_reintegrated=False,
        audit_grids_s=[.1, .5])
    evidence = object.__new__(CertifiedExecution)
    object.__setattr__(evidence, '_payload', payload)
    object.__setattr__(evidence, '_binding', binding)
    object.__setattr__(evidence, '_proof', _pack(proof))
    object.__setattr__(evidence, '_owners', tuple(system.get(key) for key in _OWNER_KEYS))
    object.__setattr__(evidence, '_consumed', False)
    return evidence


def commit_certified(system, state, proposed_transition, nodes, proof, identity, cl, witness, evidence):
    """Validate exact bindings and advance using frozen dual-grid evidence only."""
    if not isinstance(evidence, CertifiedExecution) or evidence.consumed:
        raise ValueError('Missing or already consumed certified execution evidence')
    if (any(system.get(key) is not owner for key, owner in zip(_OWNER_KEYS, evidence._owners))
            or _pack(proof) != evidence._proof
            or _evidence_sha(evidence._binding, evidence._payload) != proof.get('execution_evidence_sha256')):
        raise ValueError('Stale or incompatible recovery-controller execution certificate')
    binding = _binding(system, state, proposed_transition, identity, cl, witness, nodes, _GRID)
    if binding != evidence._binding:
        raise ValueError('Stale or incompatible recovery-controller execution inputs')
    saved = _unpack_arrays(evidence._payload)
    audit, arrays = saved['audit'], saved['arrays']
    x = arrays['0.5']['x'][-1].copy()
    if not np.array_equal(x, proof['endpoint']):
        raise ValueError('Certified versus committed endpoint mismatch')
    # Identical arithmetic/order to the reference commit_audited implementation.
    now = state['t']+3.
    z = copy(proposed_transition)
    z.settle(now)
    outcome = system['mission'].outcome(now, x)
    reward = system['mission'].reward(float(x[3]-state['x'][3]), outcome)['reward']
    next_state = dict(state, x=x.tolist(), alpha=float(nodes[-1]), t=now,
                      transition=vars(z).copy(), reward=state['reward']+reward)
    object.__setattr__(evidence, '_consumed', True)
    return next_state, audit, arrays
