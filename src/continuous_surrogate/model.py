"""Fresh residual kernel models with an incompatible continuous-output protocol.

The coverage gate is an empirical neighbourhood check, not qualification or a
safety guarantee. Every inference still requires the independent controller
audits. Only accepted *new* OCP labels in the training partition may fit weights.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from certified_flight.surrogate import Features, OracleRequired, Regressor, kernel, REGRESSOR_FIELDS
from .teacher import PROTOCOL, json_value

MODEL_SCHEMA = 'continuous-surrogate-rbf/1'
FEATURE_SCHEMA = 'continuous-flight-aero-fingerprint/1'
OUTPUT_SCHEMA = 'fixed-entry-six-future-linear-alpha-nodes/1'
DEFAULT_MODEL_CONFIG = {
    'ridge': 0.001,
    'pre_length_scale': 0.3,
    'post_length_scale': 1.2,
    'max_neighbor_distance': 0.25,
    'support_padding_fraction': 0.1,
    'minimum_training_labels_per_head': 8,
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


class ContinuousSurrogate:
    def __init__(self, db, mission, baseline_callable, config=None, metadata=None):
        self.db, self.m, self.baseline_callable = db, mission, baseline_callable
        self.features = Features(db, mission)
        self.cfg = dict(DEFAULT_MODEL_CONFIG)
        self.cfg.update(config or {})
        for name in ('ridge', 'pre_length_scale', 'post_length_scale', 'max_neighbor_distance'):
            if not np.isfinite(self.cfg[name]) or self.cfg[name] <= 0:
                raise ValueError('Invalid continuous surrogate '+name)
        if not np.isfinite(self.cfg['support_padding_fraction']) or self.cfg['support_padding_fraction'] < 0:
            raise ValueError('Invalid support padding')
        if self.cfg['minimum_training_labels_per_head'] < 1:
            raise ValueError('At least one training label is required')
        self.heads = {}
        self.control_contract = PROTOCOL
        self.metadata = dict(metadata or {})
        self.metadata.update(protocol=PROTOCOL, model_schema=MODEL_SCHEMA,
                             feature_schema=FEATURE_SCHEMA, output_schema=OUTPUT_SCHEMA,
                             output_times_s=[.5, 1., 1.5, 2., 2.5, 3.],
                             surrogate_config=deepcopy(self.cfg),
                             empirical_support_only=True, full_qualification_passed=False)

    @staticmethod
    def _context(context):
        c = deepcopy(context)
        c.setdefault('CD_des', None)
        if c.get('protocol', PROTOCOL) != PROTOCOL:
            raise ValueError('Continuous surrogate input protocol mismatch')
        if 'control_contract' in c and c['control_contract'] != PROTOCOL:
            raise ValueError('Continuous surrogate input control contract mismatch')
        return c

    def _baseline_with_diagnostics(self, c, head):
        value = self.baseline_callable(c, head)
        diagnostics = {}
        if isinstance(value, tuple):
            value, diagnostics = value
        value = np.asarray(value, float)
        if value.shape != ((1,) if head == 'pre' else (6,)) or not np.isfinite(value).all():
            raise ValueError('Invalid new continuous baseline output')
        return value, deepcopy(diagnostics)

    def baseline(self, c, head):
        return self._baseline_with_diagnostics(c, head)[0]

    def project(self, future_nodes, previous):
        """Project future nodes; the present alpha is fixed by the consumer."""
        limits = self.m.data['limits']
        low, high = max(self.db.A[0], limits['alpha_rad'][0]), min(self.db.A[-1], limits['alpha_rad'][1])
        rate_step = .5*limits['alpha_rate_radps']
        values = np.asarray(future_nodes, float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('Post requires six finite future alpha nodes')
        output = []
        for value in values:
            previous = float(np.clip(value, max(low, previous-rate_step), min(high, previous+rate_step)))
            output.append(previous)
        return np.asarray(output)

    @classmethod
    def fit(cls, records, db, mission, baseline_callable, config=None, metadata=None):
        records = list(records)
        obj = cls(db, mission, baseline_callable, config, metadata)
        partitions = {}
        for row in records:
            if row.get('protocol') != PROTOCOL:
                raise ValueError('Old or incompatible teacher labels cannot train this protocol')
            if row.get('label_provenance') != 'FRESH_CONTINUOUS_REAL_OCP_AND_FULL_DUAL_GRID_AUDIT':
                raise ValueError('Labels must have fresh continuous OCP provenance')
            role = row.get('role', 'train')
            if role not in ('train', 'val', 'validation', 'test'):
                raise ValueError('Unknown label partition')
            group = row.get('group')
            if group is not None:
                previous = partitions.setdefault(str(group), role)
                if previous != role:
                    raise ValueError('A trajectory group crosses training/validation/test partitions')
        train_counts = {}
        for head in ('pre', 'post'):
            chosen = [r for r in records if r.get('head') == head
                      and r.get('role', 'train') == 'train' and r.get('accepted') is True]
            if len(chosen) < obj.cfg['minimum_training_labels_per_head']:
                raise ValueError(f'Insufficient accepted training labels for {head}: {len(chosen)}')
            contexts = [obj._context(r['context']) for r in chosen]
            x = np.asarray([obj.features.vector(c) for c in contexts])
            y = np.asarray([r['y'] for r in chosen], float)
            expected = 1 if head == 'pre' else 6
            if y.shape != (len(chosen), expected) or not np.isfinite(y).all():
                raise ValueError('Invalid continuous teacher output dimensions')
            baselines = np.asarray([obj.baseline(c, head) for c in contexts])
            residual = y-baselines
            reg = Regressor()
            reg.x, reg.mean = x, residual.mean(axis=0)
            reg.length = float(obj.cfg[head+'_length_scale'])
            reg.weights = np.linalg.solve(kernel(x, x, reg.length)+obj.cfg['ridge']*np.eye(len(x)),
                                          residual-reg.mean)
            physical = np.asarray([obj.features.physical(c) for c in contexts])
            reg.lower, reg.upper = physical.min(axis=0), physical.max(axis=0)
            obj.heads[head] = reg
            train_counts[head] = len(chosen)
        obj.metadata.update(training_counts=train_counts,
                            labels_sha256=canonical_hash(records),
                            training_groups=sorted(k for k, v in partitions.items() if v == 'train'))
        return obj

    def _predict(self, context, head, require_coverage):
        if head not in self.heads:
            raise ValueError('Missing continuous surrogate head')
        c = self._context(context)
        x = self.features.vector(c)
        reg = self.heads[head]
        coverage = reg.coverage(x, self.features.physical(c), self.cfg)
        coverage.update(empirical_support_only=True, full_qualification_passed=False)
        if require_coverage and not coverage['supported']:
            raise OracleRequired('OUT_OF_DISTRIBUTION', c, head, coverage)
        baseline, baseline_diagnostics = self._baseline_with_diagnostics(c, head)
        raw = baseline+reg.predict(x)[0]
        if not np.isfinite(raw).all():
            raise OracleRequired('NONFINITE_CONTROLLER_OUTPUT', c, head, coverage)
        if head == 'pre':
            if not self.db.M[0] <= raw[0] <= self.db.M[-1]:
                raise OracleRequired('PREDICTED_MACH_DOMAIN_EXIT', c, head, coverage)
            value = raw
        else:
            value = self.project(raw, c['alpha_prev'])
        return value, dict(coverage, raw_output=raw.tolist(), output_schema=OUTPUT_SCHEMA,
                           projection_max_rad=float(np.max(abs(value-raw))), protocol=PROTOCOL,
                           baseline_diagnostics=baseline_diagnostics)

    def predict(self, context, head):
        return self._predict(context, head, True)

    def evaluate(self, records):
        """Report partitions separately; held-out records never update the fit."""
        results = []
        for row in records:
            if not row.get('accepted'):
                continue
            if row.get('protocol') != PROTOCOL:
                raise ValueError('Evaluation label protocol mismatch')
            y, diag = self._predict(row['context'], row['head'], False)
            truth = np.asarray(row['y'], float)
            raw = np.asarray(diag['raw_output'])
            base = self.baseline(self._context(row['context']), row['head'])
            results.append(dict(head=row['head'], role=row.get('role', 'train'), group=row.get('group'),
                                supported=diag['supported'], nearest_distance=diag['nearest_distance'],
                                baseline_rmse=float(np.sqrt(np.mean((base-truth)**2))),
                                raw_rmse=float(np.sqrt(np.mean((raw-truth)**2))),
                                projected_rmse=float(np.sqrt(np.mean((y-truth)**2))),
                                unit='Mach' if row['head'] == 'pre' else 'rad'))
        return {'protocol': PROTOCOL, 'full_qualification_passed': False, 'records': results}

    def save(self, path):
        path = Path(path)
        meta_path = path.with_suffix('.json')
        if path.exists() or meta_path.exists():
            raise FileExistsError('Refusing to overwrite continuous model artifacts')
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {head+'_'+field: np.asarray(getattr(reg, field))
                for head, reg in self.heads.items() for field in REGRESSOR_FIELDS}
        with path.open('xb') as handle:
            np.savez_compressed(handle, **data)
        self.metadata.update(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             mission_data_sha256=canonical_hash(self.m.data),
                             snapshot_sha256=hashlib.sha256((self.db.directory/'manifest.json').read_bytes()).hexdigest())
        with meta_path.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(json_value(self.metadata), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return self.metadata

    @classmethod
    def load(cls, path, db, mission, baseline_callable, config=None):
        path = Path(path)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        for key, expected in (('protocol', PROTOCOL), ('model_schema', MODEL_SCHEMA),
                              ('feature_schema', FEATURE_SCHEMA), ('output_schema', OUTPUT_SCHEMA)):
            if metadata.get(key) != expected:
                raise ValueError('Old or incompatible surrogate '+key)
        if metadata['model_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Continuous model hash mismatch')
        if metadata['mission_data_sha256'] != canonical_hash(mission.data):
            raise ValueError('Continuous model mission mismatch')
        if metadata['snapshot_sha256'] != hashlib.sha256((db.directory/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Continuous model aerodynamic snapshot mismatch')
        if config is not None and dict(DEFAULT_MODEL_CONFIG, **config) != metadata['surrogate_config']:
            raise ValueError('Continuous model configuration mismatch')
        obj = cls(db, mission, baseline_callable, metadata['surrogate_config'], metadata)
        with np.load(path, allow_pickle=False) as z:
            for head in ('pre', 'post'):
                reg = Regressor.from_arrays(z, head+'_')
                output_dim = 1 if head == 'pre' else 6
                if (reg.x.ndim != 2 or reg.x.shape[1] != 25
                        or reg.weights.shape != (len(reg.x), output_dim)
                        or reg.mean.shape != (output_dim,)
                        or reg.lower.shape != (5,) or reg.upper.shape != (5,)
                        or not all(np.isfinite(getattr(reg, k)).all() for k in REGRESSOR_FIELDS)):
                    raise ValueError('Malformed continuous surrogate '+head)
                obj.heads[head] = reg
        return obj
