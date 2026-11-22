"""Policies as chains of improvement stages (ported from rollout_improvement.policy, two-dimensional actions).

A policy maps the state at each 12 s decision to an action (CL level, morph level). It starts from the initial
policy pi0 (first stage) and applies the accepted model stages in order. pi0 is either one constant action
(kind 'constant', round 1) or a target lookup table (kind 'lookup', round 2): the constant action that was best
for the training target nearest to this flight's target in (range/100 km, speed/100 m/s). A model stage looks at the
actions within the neighbourhood of the current one (CL +-2 levels, w +-1 level), picks the one with the largest
lower confidence bound of the predicted advantage, and switches only if that bound exceeds the stage margin and
the bound of the current action. It is active within +-window seconds of the decision times that carried labels.
"""
import numpy as np

from . import config, flight as F
from .ensemble import as_arrays, lower_confidence_bound

BASE = ('h', 'V', 'gamma', 't', 'range_to_go', 'speed_error')
EXTRA = ('rho', 'q', 'kL', 'kD', 'rho_prev')
TARGET = ('target_range', 'target_speed')      # read by the lookup stage only; never a network input
GUIDE = ('pred_range_error', 'pred_speed_error', 'w_plan', 'r_plan')   # guided flights only (guided.py); study R:
#        predicted range / speed errors, current w, current CL target x 10
NAMES = BASE+EXTRA+TARGET+GUIDE
RADIUS_BLOCK_S = 96.
_CACHE = {}


def input_index(names):
    return [NAMES.index(n) for n in names]


def neighbourhood(a, morph=True):
    k_cl, k_w = config.load()['training']['neighbourhood']
    n_cl = len(F.world(morph)['cl_levels'])
    n_w = len(F.world(morph)['w_levels'])
    i, j = F.decode(a, morph)
    return [F.encode(ii, jj, morph) for ii in range(max(0, i-k_cl), min(n_cl-1, i+k_cl)+1)
            for jj in range(max(0, j-k_w), min(n_w-1, j+k_w)+1)]


def alternatives(a0, t, sweep, morph=True):
    """Branch actions at decision time t: CL +-k (k = 1 or 2, alternating between 96 s blocks and between
    sweeps) at the same w, and w +-1 at the same CL."""
    n_cl = len(F.world(morph)['cl_levels'])
    n_w = len(F.world(morph)['w_levels'])
    i, j = F.decode(a0, morph)
    k = 1 if (int((t+1e-9)//RADIUS_BLOCK_S)+sweep) % 2 == 0 else 2
    out = [F.encode(ii, j, morph) for ii in (i-k, i+k) if 0 <= ii < n_cl]
    out += [F.encode(i, jj, morph) for jj in (j-1, j+1) if 0 <= jj < n_w]
    return out


def _members(stage):
    key = (stage.get('run'), stage.get('iteration'), stage.get('margin'))
    if key not in _CACHE:
        if len(_CACHE) > 256:
            _CACHE.clear()
        _CACHE[key] = as_arrays(stage['members'])
    return _CACHE[key]


def horizon():
    """Seconds per unit of the time feature (mission.duration_s; 1200 s in rounds 1-2 and v2)."""
    return float(config.load()['mission']['duration_s'])


def initial_action(stage, f):
    """Action of the first stage for the feature vector f."""
    if stage.get('kind', 'constant') in ('constant', 'plan'):    # 'plan': study A, offset 0 = follow the plan
        return int(stage['action'])
    if stage['kind'] == 'pfb':     # study R start rule: step the CL target on the predicted range error; brake late
        eD = float(f[NAMES.index('pred_range_error')])*100.
        eV = float(f[NAMES.index('pred_speed_error')])*100.
        t = float(f[3])*horizon()
        i_cl = 4 if eD > stage['d2'] else 3 if eD > stage['d1'] else 0 if eD < -stage['d2'] else 1 if eD < -stage['d1'] else 2
        i_w = 0
        if stage.get('brake_on') == 'range':     # v3 (end at the target speed): brake while the predicted overshoot is large
            cl_t = float(f[NAMES.index('r_plan')])/10.     # current CL target (feature x 10)
            if stage['morph'] and eD > stage['d_brake'] and cl_t >= stage.get('cl_brake', -1.):   # brake once CL is high
                step = stage.get('brake_step_km')       # graded: one w level per brake_step_km of predicted overshoot
                i_w = int(stage['i_w_brake']) if not step else min(int(stage['i_w_brake']), int(eD//step))
        elif stage['morph'] and eV > stage['v_brake'] and t >= stage['t_brake']:
            i_w = int(stage.get('i_w_brake', 3))
        return F.encode(i_cl, i_w, stage['morph'])
    D, V = float(f[NAMES.index('target_range')])*1000., float(f[NAMES.index('target_speed')])*1000.
    T = np.asarray(stage['targets'], float)
    k = int(np.argmin(((T[:, 0]-D)/100.)**2+((T[:, 1]-V)/100.)**2))
    return int(stage['actions'][k])


def chain_action(chain, f, morph=True):
    """Action of the policy `chain` for the full feature vector f (NAMES order)."""
    a = initial_action(chain[0], f)
    t = float(f[3])*horizon()
    for stage in chain[1:]:
        if not np.any(np.abs(np.asarray(stage['times'])-t) <= stage['window']+1e-9):
            continue
        lcb, mean, _ = lower_confidence_bound(_members(stage), f, stage['inputs'])
        if stage.get('score') == 'mean':
            lcb = mean
        cand = max(neighbourhood(a, morph), key=lambda j: (lcb[j], -abs(j-a)))
        if cand != a and lcb[cand] > stage['margin'] and lcb[cand] > lcb[a]:
            a = cand
    return a
