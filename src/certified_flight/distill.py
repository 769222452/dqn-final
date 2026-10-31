"""Teacher labels and kernel-ridge fitting for a surrogate of a given inner-loop OCP objective.

Teacher: for a recorded pre context, the true pre OCP gives the planning Mach (mean state of the first
3 s, as OracleController), the min-drag generator picks the geometry for that Mach, the geometry
transition is commanded exactly as in FlightEnvironment.step, and the true post OCP (warm-started from
the pre plan) gives the six 0.5 s alpha commands. Both plans must pass audit_full_plan.

Fit: the V16 model class and conventions, verified against data/surrogate/controller.npz
(tests/test_aoa_surrogate.py): residuals on Features.baseline, RBF kernel on the 25 scaled features,
W = solve(K + ridge*I, Y - mean(Y)), length scale per head chosen on validation MSE, physical support box
from the training contexts.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .aero import AeroDatabase
from .build import ROOT, runtime_config
from .mission import Mission
from .objectives import contract, objective_ocp
from .ocp import OracleController, TwoStageOCP
from .plant import CompiledPlant
from .surrogate import FEATURE_SCHEMA, REGRESSOR_FIELDS, Features, OracleRequired, kernel
from .transition import Transition

LENGTH_SCALES = (.15, .3, .6, 1.2)
RIDGE = 1e-3


class Teacher:
    def __init__(self, objective, root=ROOT):
        config = runtime_config(root)
        self.physical = deepcopy(config['physical'])
        self.physical['ocp'] = objective_ocp(self.physical['ocp'], objective)
        self.mission = Mission(root, self.physical['mission'])
        self.db = AeroDatabase(Path(root)/self.physical['snapshot'])
        self.plant = CompiledPlant(self.mission, self.db)
        traj = yaml.safe_load(self.mission.path(self.physical['trajectory']).read_text(encoding='utf-8'))
        self.ocp = TwoStageOCP(self.mission, self.db, traj, self.physical['ocp'], self.plant)
        self.contract = contract(self.physical)
        self.duration = float(self.physical['transition']['duration_s'])

    def post_context(self, pre, mach):
        """Generator + transition command of FlightEnvironment.step for the planned Mach."""
        t = float(pre['t'])
        z = Transition(int(pre['source']), int(pre['target']), t-self.duration*float(pre['fraction']), self.duration)
        z.settle(t)
        choice = self.db.generate(float(mach), pre['CL_des'], current=z.target)
        if choice['geometry_index'] is not None:
            z.command(choice['geometry_index'], t)
        return dict(pre, source=z.source, target=z.target, fraction=z.fraction(t),
                    rate=1/z.duration if z.active(t) else 0., control_contract=self.contract), choice

    def label(self, pre_context):
        tick = time.perf_counter()
        pre = dict(pre_context, control_contract=self.contract)
        record = dict(solver_calls={'pre': 0, 'post': 0}, phases={})
        oracle = OracleController(self.ocp, record)
        out = dict(pre_context=pre_context)
        try:
            mach, _ = oracle.predict(pre, 'pre')
            post, choice = self.post_context(pre, mach[0])
            out.update(pre_mach=float(mach[0]), post_context={k: v for k, v in post.items() if k != 'control_contract'},
                       generator=dict(status=choice['status'], geometry_index=choice['geometry_index']))
            alpha, _ = oracle.predict(post, 'post')
            out.update(ok=True, post_alpha=np.asarray(alpha, float).tolist())
        except OracleRequired as exc:
            out.update(ok=False, reason=exc.reason, head=exc.head)
        out['solver'] = {h: dict(status=p['solver']['status'], iterations=p['solver']['iterations'],
                                 solve_s=p['solver']['solve_s'], audit_valid=p.get('full_plan_audit', {}).get('valid'))
                         for h, p in record['phases'].items()}
        out['wall_s'] = time.perf_counter()-tick
        return out


def fit_head(X, Y, length, ridge=RIDGE):
    mean = Y.mean(0)
    weights = np.linalg.solve(kernel(X, X, length)+ridge*np.eye(len(X)), Y-mean)
    return mean, weights


def predict_head(X_train, mean, weights, length, X):
    return mean+kernel(np.atleast_2d(X), X_train, length)@weights


def design(features, labels):
    """Feature matrices, residual targets and physical support of usable labels."""
    out = dict(pre=dict(X=[], Y=[]), post=dict(X=[], Y=[]), physical=[])
    for r in labels:
        pre, post = r['pre_context'], r['post_context']
        out['pre']['X'].append(features.vector(pre))
        out['pre']['Y'].append([r['pre_mach']-features.mach(pre)])
        out['post']['X'].append(features.vector(post))
        out['post']['Y'].append(np.asarray(r['post_alpha'])-features.alpha_reference(post))
        out['physical'].append(features.physical(pre))
    return {k: (np.asarray(v) if k == 'physical' else {kk: np.asarray(vv, float) for kk, vv in v.items()})
            for k, v in out.items()}


def fit(teacher, labels, objective, dataset_sha256, data_use, gate_config, length_scales=LENGTH_SCALES):
    """Fit both heads on split=='train', select length scales on split=='validation'.

    Returns (arrays for controller.npz, metadata without model_sha256)."""
    features = Features(teacher.db, teacher.mission)
    usable = [r for r in labels if r.get('ok')]
    train = design(features, [r for r in usable if r['split'] == 'train'])
    val = design(features, [r for r in usable if r['split'] == 'validation'])
    arrays, selection = {}, {}
    lower, upper = train['physical'].min(0), train['physical'].max(0)
    for head in ('pre', 'post'):
        X, Y = train[head]['X'], train[head]['Y']
        candidates = []
        for length in length_scales:
            mean, weights = fit_head(X, Y, length)
            err = predict_head(X, mean, weights, length, val[head]['X'])-val[head]['Y']
            candidates.append(dict(length_scale=length, validation_mse=float(np.mean(err**2))))
        best = min(candidates, key=lambda c: c['validation_mse'])['length_scale']
        mean, weights = fit_head(X, Y, best)
        for name, value in zip(REGRESSOR_FIELDS, (X, mean, weights, np.array(best), lower, upper)):
            arrays[head+'_'+name] = np.asarray(value, float)
        selection[head] = dict(candidates=candidates, selected_length_scale=best, train_count=int(len(X)),
                               validation_count=int(len(val[head]['X'])))
    mission_file = teacher.mission.config_path
    meta = dict(control_contract=teacher.contract, objective=objective, ocp=teacher.physical['ocp'],
                selection=selection, gate_config=gate_config, transition_duration_s=teacher.duration,
                data_use=data_use, feature_schema='rl298-'+FEATURE_SCHEMA,
                surrogate_config=dict(kind='residual_rbf_kernel_ridge', ridge=RIDGE, length_scales=list(length_scales),
                                      max_neighbor_distance=.25, support_padding_fraction=.1,
                                      planned_mach_output='delta_from_current_mach',
                                      alpha_output='residual_on_exact_equal_lift_rate_limited_schedule'),
                mission_sha256=sha256(mission_file),
                trajectory_sha256=sha256(teacher.mission.path(teacher.physical['trajectory'])),
                snapshot_sha256=sha256(teacher.db.directory/'manifest.json'), dataset_sha256=dataset_sha256,
                label_counts=dict(total=len(labels), usable=len(usable),
                                  **{s: sum(r['split'] == s for r in usable) for s in ('train', 'validation', 'test')}))
    return arrays, meta


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
