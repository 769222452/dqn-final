"""Control contract: identifies the control mechanics a surrogate was trained for.

The contract string stored in the surrogate metadata has the form ``<scheme>:<digest>``.
The digest is the first 16 hex characters of the SHA-256 of the mechanics below;
the surrogate is only used when it matches the configured mechanics.
"""
import hashlib
import json


# Optional OCP objective settings and their defaults. A setting enters the digest only when it differs
# from its default, so contracts of configurations without it (V16) are unchanged.
OPTIONAL_OBJECTIVE_KEYS = {'boundary_smoothing': False, 'horizon_tracking_weight': 1., 'second_difference_weight': 0.,
                           'heat_weight': .5}   # .5 = configs/trajectory.yaml objective.w_heat (V16)


def control_mechanics(config):
    c = config.get('control', {})
    o = config.get('ocp', {})
    out = {'pre_transition_model': c.get('pre_transition_model', 'frozen'),
           'reachability': bool(c.get('reachability', False)),
           'tracking_mode': o.get('tracking_mode', 'instantaneous'),
           'tracking_weight_multiplier': float(o.get('tracking_weight_multiplier', 1.))}
    for key, default in OPTIONAL_OBJECTIVE_KEYS.items():
        value = o.get(key, default)
        if value != default:
            out[key] = bool(value) if isinstance(default, bool) else float(value)
    return out


def mechanics_digest(config):
    return hashlib.sha256(json.dumps(control_mechanics(config), sort_keys=True).encode()).hexdigest()[:16]


def matching_contract(contract, config):
    """Return ``contract`` if its digest matches the configured mechanics, otherwise raise."""
    if not isinstance(contract, str) or contract.rpartition(':')[2] != mechanics_digest(config):
        raise ValueError('Surrogate control contract does not match the configured control mechanics')
    return contract
