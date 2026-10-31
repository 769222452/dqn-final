"""Fit the V16-class surrogate (kernel-ridge residual heads) to teacher labels of an objective.

    python scripts/aoa_fit_surrogate.py --objective consistent \
        --labels runs/aoa/surrogate/labels.jsonl [runs/aoa/surrogate/dagger_labels.jsonl] \
        --out data/surrogate_consistent/controller.npz

Writes controller.npz + controller.json in the format SurrogateController.load checks (hashes of the
model, aerodynamic snapshot, mission and trajectory; control contract of the objective's mechanics) and
loads the result back through SurrogateController.load. Fitting on split 'train' only; the length scale
of each head is chosen on 'validation'; 'test' is left for scripts/aoa_qualify_surrogate.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from certified_flight.distill import Teacher, fit, read_jsonl, sha256  # noqa: E402
from certified_flight.objectives import OBJECTIVES  # noqa: E402
from certified_flight.serialization import clean  # noqa: E402
from certified_flight.surrogate import SurrogateController  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--objective', choices=sorted(OBJECTIVES), required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--out', required=True, help='controller.npz path (metadata goes next to it as .json)')
    a = p.parse_args(argv)
    labels, digest = [], hashlib.sha256()
    for f in a.labels:
        digest.update(Path(f).read_bytes())
        labels += read_jsonl(f)
    ids = [r['id'] for r in labels]
    if len(ids) != len(set(ids)):
        raise SystemExit('duplicate label ids')
    teacher = Teacher(a.objective)
    v16 = json.loads((ROOT/'data/surrogate/controller.json').read_text(encoding='utf-8'))
    arrays, meta = fit(teacher, labels, a.objective, digest.hexdigest(),
                       'AOA_FIX_TEACHER_LABELS_'+'+'.join(Path(f).name for f in a.labels), v16['gate_config'])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    meta.update(model_sha256=sha256(out), label_files=[str(f) for f in a.labels],
                scale_up_gate_passed=None, acceptance_passed=None)
    out.with_suffix('.json').write_text(json.dumps(clean(meta), indent=2)+'\n', encoding='utf-8')
    model = SurrogateController.load(out, teacher.db, teacher.mission, teacher.physical,
                                     {'mission_sha256': teacher.mission.config_path,
                                      'trajectory_sha256': teacher.mission.path(teacher.physical['trajectory'])})
    print(json.dumps(dict(model=str(out), model_sha256=meta['model_sha256'], contract=model.control_contract,
                          counts=meta['label_counts'],
                          length_scales={h: s['selected_length_scale'] for h, s in meta['selection'].items()},
                          validation_mse={h: min(c['validation_mse'] for c in s['candidates'])
                                          for h, s in meta['selection'].items()})))


if __name__ == '__main__':
    main()
