"""The augmentation design is fixed and preserves the old held-out partitions."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from continuous_recovery.preparation import (
    ADDED_GROUP, ADDED_TIMES, generate_labels, prepare_contexts,
)
from continuous_surrogate.teacher import PROTOCOL


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def sources(tmp_path):
    flight = tmp_path/'source_flight'
    for t in ADDED_TIMES:
        k = int(t/3)
        put(flight/f'decision_{k:04d}'/'before.json', dict(
            t=t, alpha=.05, x=[40000., 2400., 0., t*2400.],
            transition=dict(source=14, target=14, started=0., duration=8.), reward=0., switches=1))
        put(flight/f'decision_{k-1:04d}'/'certificate.json', {'proof': f'at_{t-3}'})
    step = tmp_path/'step_09.json'
    put(step, dict(accepted=True, t=615., context=dict(
        t=615., alpha_prev=.05, x=[40000., 2400., 0., 1476000.], source=50, target=50,
        fraction=1., rate=0., CL_des=.025, CL_raw=.025, CD_des=None,
        control_contract=PROTOCOL, proposal_origin='OLD_POST')))
    return flight, step


def test_eight_predeclared_contexts_have_sources_and_single_train_group(sources):
    flight, step = sources
    system = {'config': {'flight': {'CL': .025}}}
    contexts = prepare_contexts(system, flight, step)
    assert [c['context']['t'] for c in contexts] == [*ADDED_TIMES, 615.]
    assert all(c['group'] == ADDED_GROUP and c['role'] == 'train' for c in contexts)
    assert all(len(c['source']['sha256']) == 64 for c in contexts)
    assert all(c['context']['CL_des'] == .025 for c in contexts)
    assert 'proposal_origin' not in contexts[-1]['context']


def test_saved_source_time_mismatch_is_not_silently_remapped(sources):
    flight, step = sources
    path = flight/'decision_0040'/'before.json'
    changed = json.loads(path.read_text())
    changed['t'] = 121.
    put(path, changed)
    with pytest.raises(ValueError, match='time mismatch'):
        prepare_contexts({'config': {'flight': {'CL': .025}}}, flight, step)


def test_serial_generation_preserves_all_48_old_records_and_new_partitions(sources, tmp_path):
    flight, step = sources
    calls = []

    class Teacher:
        experiment = {'test_teacher_contract': 1}

        def solve_audited(self, context, head, output_dir):
            calls.append((context['t'], head))
            return dict(protocol=PROTOCOL, teacher_contract=self.experiment, accepted=True,
                        label_provenance='FRESH_CONTINUOUS_REAL_OCP_AND_FULL_DUAL_GRID_AUDIT',
                        context=deepcopy(context), head=head,
                        y=[5.] if head == 'pre' else [.05]*6), None

    class DB:
        def generate(self, mach, cl, current):
            assert mach == 5. and cl == .025
            return {'geometry_index': current}

    old = []
    for i in range(24):
        for head in ('pre', 'post'):
            old.append(dict(case=i, teacher_contract=Teacher.experiment, accepted=True,
                            head=head, role='train' if i < 16 else 'validation' if i < 20 else 'test',
                            group=f'old_group_{i}', untouched_marker=i))
    old_source = tmp_path/'old_labels.json'
    put(old_source, old)
    system = dict(config={'flight': {'CL': .025}}, teacher=Teacher(), db=DB())
    out = tmp_path/'new_labels'
    records = generate_labels(system, old_source, out, flight, step)
    assert records[:48] == old
    assert len(records) == 64
    assert calls == [(t, head) for t in [*ADDED_TIMES, 615.] for head in ('pre', 'post')]
    assert all(r['group'] == ADDED_GROUP and r['role'] == 'train' for r in records[48:])
    assert sorted({r['case'] for r in records[48:]}) == list(range(24, 32))
    summary = json.loads((out/'summary.json').read_text())
    assert summary['new_ocp_labels_generated'] == 16
    assert summary['accepted']['pre_train'] == 24
    assert summary['accepted']['pre_validation'] == 4
    assert summary['accepted']['post_test'] == 4
    assert json.loads(old_source.read_text()) == old
    with pytest.raises(FileExistsError):
        generate_labels(system, old_source, out, flight, step)
