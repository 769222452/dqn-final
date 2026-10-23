"""Surrogate controller, exact-physics baselines and the oracle-request exception.

Two kernel-ridge heads predict residuals on exact physical baselines: the planned
Mach of the preliminary OCP ('pre') and the six half-second alpha commands of the
final OCP ('post'). Aerodynamic tables and physics stay exact. Distance/range gates
are empirical coverage checks; a query outside coverage raises OracleRequired.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .atmosphere import FastAtmosphere
from .contracts import matching_contract

FEATURE_SCHEMA = 'flight-aero-fingerprint/1'
REGRESSOR_FIELDS = ('x', 'mean', 'weights', 'length', 'lower', 'upper')
POST_ROUTING = 'adaptation_coverage_first'
OOD_REASONS = frozenset({'OUT_OF_DISTRIBUTION', 'UNSUPPORTED_SURROGATE_OUTPUT'})


class OracleRequired(RuntimeError):
    """A controller, corrector, guard or audit refused; carries the reason and its context."""

    def __init__(self, reason, context, head, diagnostics=None):
        super().__init__(reason)
        self.reason = reason
        self.context = context
        self.head = head
        self.diagnostics = diagnostics or {}


class SurrogateUnavailable(OracleRequired):
    """Only model inference/support errors qualify for a fallback takeover."""


class Features:
    """Surrogate inputs and the exact physical baselines the surrogate corrects."""

    def __init__(self, db, mission):
        self.db, self.m = db, mission
        self.atm = FastAtmosphere(mission.model(None).atm)
        self.angles = np.deg2rad([-1., 2., 6., 12.])
        self.coefficient_scale = np.array([.1, .03])

    def mach(self, c):
        return float(c['x'][1]/self.atm.speed_of_sound(c['x'][0]))

    def physical(self, c):
        return np.array([c['x'][0], c['x'][1], c['x'][2], c['alpha_prev'], 1200-c['t']], float)

    def vector(self, c):
        """Scaled flight state and command plus CL/CD fingerprints of both endpoint geometries."""
        if c['CD_des'] is not None:
            raise ValueError('The surrogate is CL-only')
        M = self.mach(c)
        self.db.check(M, c['alpha_prev'])
        x = c['x']
        base = [(x[0]-40000)/10000, (x[1]-2400)/800, x[2]/.1, c['alpha_prev']/.1,
                (1200-c['t'])/1200, c['CL_des']/.1, c['fraction'], c['rate']*8, M/8]
        y = self.db.curves(M, [c['source'], c['target']])
        fingerprints = PchipInterpolator(self.db.A, y, axis=1)(self.angles)/self.coefficient_scale
        return np.r_[base, fingerprints.ravel()]

    def project(self, alpha, previous):
        """Clip a command sequence to the alpha bounds and the half-second rate limit."""
        out = []
        step = self.m.data['limits']['alpha_rate_radps']*self.m.data['plant_dt_s']
        for a in alpha:
            previous = float(np.clip(a, max(self.db.A[0], previous-step), min(self.db.A[-1], previous+step)))
            out.append(previous)
        return np.array(out)

    def alpha_reference(self, c):
        """Equal-lift alpha on the exact endpoint PCHIPs at the same Mach with the known 8 s blend."""
        M = self.mach(c)
        y = self.db.curves(M, [c['source'], c['target']])
        pc = [PchipInterpolator(self.db.A, yy[:, 0], extrapolate=False) for yy in y]
        fraction = np.minimum(1, c['fraction']+c['rate']*np.arange(6)*.5)
        lo = np.full(6, self.db.A[0])
        hi = np.full(6, self.db.A[-1])
        for _ in range(32):
            a = (lo+hi)/2
            lift = (1-fraction)*pc[0](a)+fraction*pc[1](a)
            lo = np.where(lift < c['CL_des'], a, lo)
            hi = np.where(lift >= c['CL_des'], a, hi)
        return self.project((lo+hi)/2, c['alpha_prev'])

    def baseline(self, c, head):
        return np.array([self.mach(c)]) if head == 'pre' else self.alpha_reference(c)


class PhysicalBaseline:
    """Unfitted exact-physics controller: current Mach (pre) and equal-lift alpha (post)."""

    def __init__(self, db, mission, control_contract):
        self.db = db
        self.features = Features(db, mission)
        self.control_contract = control_contract
        self.metadata = {'control_contract': control_contract, 'kind': 'PHYSICS_BASELINE'}

    def predict(self, c, head):
        if c.get('control_contract') != self.control_contract:
            raise ValueError('Physical baseline control contract mismatch')
        self.db.check(self.features.mach(c), c['alpha_prev'])
        pred = self.features.baseline(c, head)
        if not np.isfinite(pred).all():
            raise OracleRequired('NONFINITE_BASELINE_OUTPUT', c, head)
        return pred, {'supported': True, 'raw_output': pred.tolist(),
                      'support_basis': 'No learned coverage model; physical domain and dense execution audit only.'}


def kernel(x, y, length):
    delta = np.maximum(0, (x*x).sum(1)[:, None]+(y*y).sum(1)[None]-2*x@y.T)/x.shape[1]
    return np.exp(-delta/(2*length*length))


class Regressor:
    """RBF kernel-ridge head with its training support (inputs and physical bounds)."""

    def predict(self, x):
        return self.mean+kernel(np.atleast_2d(x), self.x, self.length)@self.weights

    def coverage(self, x, physical, cfg):
        """Supported if the nearest training input is close enough and the physical state lies in the padded box."""
        nearest = float(np.min(np.sqrt(np.mean((self.x-x)**2, axis=1))))
        pad = np.maximum(np.array([1000, 50, .015, .02, 30.]), cfg['support_padding_fraction']*(self.upper-self.lower))
        outside = np.maximum(self.lower-pad-physical, 0)+np.maximum(physical-self.upper-pad, 0)
        return {'supported': bool(nearest <= cfg['max_neighbor_distance'] and not np.any(outside > 0)),
                'nearest_distance': nearest, 'physical_bounds_excess': outside.tolist()}

    @classmethod
    def from_arrays(cls, z, prefix):
        r = cls()
        for name in REGRESSOR_FIELDS:
            setattr(r, name, z[prefix+name].copy())
        r.length = float(r.length)
        return r


class SurrogateController:
    """Pre and post heads; the post head routes between an anchor and an expanded expert.

    The anchor expert is used inside its own coverage unless the expanded expert covers
    the query and the adaptation samples are strictly closer than the anchor samples.
    """

    def __init__(self, db, mission, config, control_contract):
        self.db, self.m, self.cfg = db, mission, config
        self.features = Features(db, mission)
        self.heads = {}
        self.post_anchor = None
        self.post_adaptation_x = None
        self.control_contract = control_contract
        self.metadata = {'control_contract': control_contract}

    def predict(self, c, head):
        if c.get('control_contract') != self.control_contract:
            raise ValueError('Controller input control contract mismatch')
        x = self.features.vector(c)
        reg = self.heads[head]
        coverage = reg.coverage(x, self.features.physical(c), self.cfg)
        if head == 'post':
            expanded_coverage = coverage
            anchor_coverage = self.post_anchor.coverage(x, self.features.physical(c), self.cfg)
            use_anchor = anchor_coverage['supported']
            adaptation_distance = float(np.min(np.sqrt(np.mean((self.post_adaptation_x-x)**2, axis=1))))
            if expanded_coverage['supported'] and adaptation_distance+1e-12 < anchor_coverage['nearest_distance']:
                use_anchor = False
            if use_anchor:
                reg = self.post_anchor
                coverage = anchor_coverage
            routing = dict(post_routing=POST_ROUTING, post_expert='anchor' if use_anchor else 'expanded',
                           anchor_coverage=anchor_coverage, expanded_coverage=expanded_coverage,
                           adaptation_nearest_distance=adaptation_distance)
            coverage = dict(coverage, **routing)
        if not coverage['supported']:
            raise OracleRequired('OUT_OF_DISTRIBUTION', c, head, coverage)
        raw = self.features.baseline(c, head)+reg.predict(x)[0]
        if not np.isfinite(raw).all():
            raise OracleRequired('NONFINITE_CONTROLLER_OUTPUT', c, head, coverage)
        if head == 'pre':
            if not self.db.M[0] <= raw[0] <= self.db.M[-1]:
                raise OracleRequired('PREDICTED_MACH_DOMAIN_EXIT', c, head)
            output = raw
        else:
            output = self.features.project(raw, c['alpha_prev'])
        return output, dict(coverage, raw_output=raw.tolist(), projection_max_rad=float(np.max(abs(output-raw))))

    @classmethod
    def load(cls, path, db, mission, config, coupled_files):
        """Load controller.npz and its metadata (controller.json) with all consistency checks.

        coupled_files maps metadata hash keys (e.g. 'mission_sha256') to the files they protect.
        """
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        contract = matching_contract(meta.get('control_contract'), config)
        if meta['feature_schema'].split('-', 1)[-1] != FEATURE_SCHEMA:
            raise ValueError('Controller feature schema mismatch')
        if meta['model_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Controller hash mismatch')
        if meta['snapshot_sha256'] != hashlib.sha256((db.directory/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Aerodynamic snapshot mismatch')
        for key, file in coupled_files.items():
            if meta[key] != hashlib.sha256(Path(file).read_bytes()).hexdigest():
                raise ValueError('Controller '+key+' mismatch')
        obj = cls(db, mission, meta['surrogate_config'], contract)
        obj.metadata = meta
        with np.load(path, allow_pickle=False) as z:
            anchor_fields = {name for name in z.files if name.startswith('post_anchor_')}
            if meta.get('post_routing') != POST_ROUTING or \
               anchor_fields != {'post_anchor_'+name for name in REGRESSOR_FIELDS}:
                raise ValueError('Post routing metadata and anchor arrays disagree')
            if 'post_adaptation_x' not in z.files:
                raise ValueError('Post routing and adaptation support disagree')
            for head in ('pre', 'post'):
                obj.heads[head] = Regressor.from_arrays(z, head+'_')
            obj.post_anchor = Regressor.from_arrays(z, 'post_anchor_')
            a = z['post_adaptation_x'].copy()
            if a.ndim != 2 or not len(a) or a.shape[1] != obj.heads['post'].x.shape[1] or not np.isfinite(a).all():
                raise ValueError('Invalid adaptation training support')
            obj.post_adaptation_x = a
        return obj


class CheckedSurrogate:
    """Normalizes controller failures to SurrogateUnavailable and validates the output shape."""

    def __init__(self, controller):
        self.controller = controller

    def predict(self, context, head):
        try:
            value, diagnostics = self.controller.predict(context, head)
        except OracleRequired as exc:
            raise SurrogateUnavailable(exc.reason, context, head, deepcopy(exc.diagnostics)) from exc
        except (FloatingPointError, OverflowError, RuntimeError) as exc:
            raise SurrogateUnavailable('SURROGATE_PREDICTION_FAILED', context, head,
                                       {'error': repr(exc)}) from exc
        value = np.asarray(value, float)
        if value.shape != ((1,) if head == 'pre' else (6,)) or not np.isfinite(value).all():
            raise SurrogateUnavailable('INVALID_SURROGATE_OUTPUT', context, head,
                                       {'output': value.tolist()})
        if not diagnostics.get('supported', True):
            raise SurrogateUnavailable('UNSUPPORTED_SURROGATE_OUTPUT', context, head, diagnostics)
        return value, diagnostics


class PhysicsOnRefusal:
    """Surrogate first; the exact-physics baseline only when the surrogate refuses as out of distribution.

    The full corrector/lookahead/audit chain still decides; this only adds a
    candidate before the expensive true OCP takeover.
    """

    def __init__(self, surrogate, baseline):
        self.surrogate, self.baseline = surrogate, baseline
        self.metadata = surrogate.metadata
        self.control_contract = surrogate.control_contract
        self.used = []

    def predict(self, c, head):
        try:
            output, diag = self.surrogate.predict(c, head)
            if diag.get('supported', True):
                return output, diag
            refusal = dict(reason='UNSUPPORTED_SURROGATE_OUTPUT', diagnostics=diag)
        except OracleRequired as exc:
            if exc.reason not in OOD_REASONS:
                raise
            refusal = dict(reason=exc.reason, diagnostics=exc.diagnostics)
        output, diag = self.baseline.predict(c, head)
        self.used.append(dict(head=head, t=c['t'], reason=refusal['reason']))
        return output, dict(diag, supported=True, support_basis='PHYSICS_BASELINE_ON_SURROGATE_OOD',
                            empirical_support=False, surrogate_refusal=refusal['reason'])
