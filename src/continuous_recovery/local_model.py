"""Local Post residuals for the continuous recovery experiment.

Pre retains the previous continuous RBF definition. Post uses compact-support,
nonnegative weights and a zero-residual physical-baseline prior: no fitted global
mean or distant recovery label contributes outside the original support radius.
The neighbourhood gate is empirical coverage, never a safety certificate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from certified_flight.surrogate import OracleRequired, Regressor, REGRESSOR_FIELDS, kernel
from continuous_surrogate.model import (
    ContinuousSurrogate, DEFAULT_MODEL_CONFIG, FEATURE_SCHEMA, OUTPUT_SCHEMA, canonical_hash,
)
from continuous_surrogate.teacher import PROTOCOL, json_value

MODEL_SCHEMA = 'continuous-recovery-local-post/1'
LOCAL_FIELDS = ('x', 'residuals', 'lower', 'upper')
LOCAL_CONFIG = dict(DEFAULT_MODEL_CONFIG, local_baseline_prior_weight=1.0)


class LocalPostRegressor:
    """Compact Wendland C2 kernel with an explicit zero-residual prior."""

    coverage = Regressor.coverage

    def predict_local(self, x, radius, prior_weight):
        distances = np.sqrt(np.mean((self.x-np.asarray(x, float))**2, axis=1))
        chosen = np.flatnonzero(distances < radius)
        r = distances[chosen]/radius
        unnormalized = (1-r)**4*(4*r+1)
        denominator = float(prior_weight+np.sum(unnormalized))
        weights = unnormalized/denominator
        baseline_weight = float(prior_weight/denominator)
        residual = weights@self.residuals[chosen]
        neighbours = [dict(training_index=int(i), group=self.groups[i], distance=float(distances[i]),
                           weight=float(w)) for i, w in zip(chosen, weights)]
        return residual, dict(local_neighbours=neighbours, local_baseline_weight=baseline_weight,
                              local_weight_sum=float(np.sum(weights)+baseline_weight),
                              local_kernel='wendland_c2', local_radius=float(radius),
                              local_positive_kernel_mass=float(np.sum(unnormalized)))


class LocalResidualSurrogate(ContinuousSurrogate):
    """Same continuous output contract, incompatible local-model artifact schema."""

    def __init__(self, db, mission, baseline_callable, config=None, metadata=None):
        cfg = dict(LOCAL_CONFIG, **(config or {}))
        super().__init__(db, mission, baseline_callable, cfg, metadata)
        if self.cfg['local_baseline_prior_weight'] != 1.0:
            raise ValueError('This experiment fixes the local zero-residual prior weight at 1.0')
        if self.cfg['max_neighbor_distance'] > DEFAULT_MODEL_CONFIG['max_neighbor_distance']:
            raise ValueError('Local support radius cannot widen the original OOD threshold')
        self.metadata.update(model_schema=MODEL_SCHEMA,
                             pre_model='unchanged_continuous_rbf',
                             post_model='local_nonnegative_residual_with_zero_prior',
                             post_no_global_mean=True,
                             surrogate_config=deepcopy(self.cfg))
        self.config = self.cfg

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
            if group is not None and partitions.setdefault(str(group), role) != role:
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
            if y.shape != (len(chosen), 1 if head == 'pre' else 6) or not np.isfinite(y).all():
                raise ValueError('Invalid continuous teacher output dimensions')
            residuals = y-np.asarray([obj.baseline(c, head) for c in contexts])
            physical = np.asarray([obj.features.physical(c) for c in contexts])
            if x.shape != (len(chosen), 25) or not np.isfinite(x).all() or not np.isfinite(physical).all():
                raise ValueError('Invalid continuous training features')
            if head == 'pre':
                reg = Regressor()
                reg.x, reg.mean = x, residuals.mean(axis=0)
                reg.length = float(obj.cfg['pre_length_scale'])
                reg.weights = np.linalg.solve(kernel(x, x, reg.length)+obj.cfg['ridge']*np.eye(len(x)),
                                              residuals-reg.mean)
            else:
                reg = LocalPostRegressor()
                reg.x, reg.residuals = x, residuals
                reg.groups = [None if r.get('group') is None else str(r['group']) for r in chosen]
            reg.lower, reg.upper = physical.min(axis=0), physical.max(axis=0)
            obj.heads[head] = reg
            train_counts[head] = len(chosen)
        obj.metadata.update(training_counts=train_counts, labels_sha256=canonical_hash(records),
                            post_training_groups=obj.heads['post'].groups,
                            training_groups=sorted(k for k, role in partitions.items() if role == 'train'))
        return obj

    def _predict(self, context, head, require_coverage):
        if head == 'pre':
            value, diagnostics = super()._predict(context, head, require_coverage)
            return value, dict(diagnostics, model_schema=MODEL_SCHEMA, pre_model='unchanged_continuous_rbf')
        if head != 'post' or head not in self.heads:
            raise ValueError('Missing continuous surrogate head')
        c = self._context(context)
        x, physical = self.features.vector(c), self.features.physical(c)
        reg = self.heads[head]
        coverage = reg.coverage(x, physical, self.cfg)
        residual, diagnostics = reg.predict_local(x, self.cfg['max_neighbor_distance'],
                                                 self.cfg['local_baseline_prior_weight'])
        # At the compact-support boundary every kernel is zero. This is UNKNOWN,
        # even though the old closed-radius empirical gate included equality.
        coverage['supported'] = bool(coverage['supported'] and diagnostics['local_positive_kernel_mass'] > 0)
        coverage.update(diagnostics, empirical_support_only=True, full_qualification_passed=False,
                        model_schema=MODEL_SCHEMA)
        if require_coverage and not coverage['supported']:
            raise OracleRequired('OUT_OF_DISTRIBUTION', c, head, coverage)
        baseline, baseline_diagnostics = self._baseline_with_diagnostics(c, head)
        raw = baseline+residual
        if not np.isfinite(raw).all():
            raise OracleRequired('NONFINITE_CONTROLLER_OUTPUT', c, head, coverage)
        output = self.project(raw, c['alpha_prev'])
        return output, dict(coverage, raw_output=raw.tolist(), local_residual=residual.tolist(),
                            baseline_diagnostics=baseline_diagnostics, output_schema=OUTPUT_SCHEMA,
                            projection_max_rad=float(np.max(abs(output-raw))), protocol=PROTOCOL)

    def evaluate(self, records):
        records = list(records)
        report = super().evaluate(records)
        accepted = [r for r in records if r.get('accepted')]
        for row, record in zip(report['records'], accepted):
            if record['head'] == 'post':
                _, diagnostics = self._predict(record['context'], 'post', False)
                row.update({k: diagnostics[k] for k in (
                    'local_neighbours', 'local_baseline_weight', 'local_weight_sum', 'local_radius')})
        report['model_schema'] = MODEL_SCHEMA
        return report

    def save(self, path):
        path = Path(path)
        meta_path = path.with_suffix('.json')
        if path.exists() or meta_path.exists():
            raise FileExistsError('Refusing to overwrite local model artifacts')
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'pre_'+field: np.asarray(getattr(self.heads['pre'], field)) for field in REGRESSOR_FIELDS}
        data.update({'post_'+field: np.asarray(getattr(self.heads['post'], field)) for field in LOCAL_FIELDS})
        with path.open('xb') as handle:
            np.savez_compressed(handle, **data)
        self.metadata.update(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             mission_data_sha256=canonical_hash(self.m.data),
                             snapshot_sha256=hashlib.sha256((self.db.directory/'manifest.json').read_bytes()).hexdigest(),
                             post_training_groups=self.heads['post'].groups)
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
            raise ValueError('Local model hash mismatch')
        if metadata['mission_data_sha256'] != canonical_hash(mission.data):
            raise ValueError('Local model mission mismatch')
        if metadata['snapshot_sha256'] != hashlib.sha256((db.directory/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Local model aerodynamic snapshot mismatch')
        if config is not None and dict(LOCAL_CONFIG, **config) != metadata['surrogate_config']:
            raise ValueError('Local model configuration mismatch')
        obj = cls(db, mission, baseline_callable, metadata['surrogate_config'], metadata)
        with np.load(path, allow_pickle=False) as z:
            pre = Regressor.from_arrays(z, 'pre_')
            if (pre.x.ndim != 2 or pre.x.shape[1] != 25 or len(pre.x) == 0
                    or pre.weights.shape != (len(pre.x), 1) or pre.mean.shape != (1,)
                    or pre.lower.shape != (5,) or pre.upper.shape != (5,) or pre.length <= 0
                    or np.any(pre.lower > pre.upper)
                    or not all(np.isfinite(getattr(pre, k)).all() for k in REGRESSOR_FIELDS)):
                raise ValueError('Malformed local-model Pre')
            post = LocalPostRegressor()
            for field in LOCAL_FIELDS:
                setattr(post, field, z['post_'+field].copy())
            post.groups = list(metadata['post_training_groups'])
            if (post.x.ndim != 2 or post.x.shape[1] != 25 or len(post.x) == 0
                    or post.residuals.shape != (len(post.x), 6)
                    or post.lower.shape != (5,) or post.upper.shape != (5,)
                    or len(post.groups) != len(post.x) or np.any(post.lower > post.upper)
                    or not all(np.isfinite(getattr(post, k)).all() for k in LOCAL_FIELDS)):
                raise ValueError('Malformed local-model Post')
        obj.heads.update(pre=pre, post=post)
        return obj
