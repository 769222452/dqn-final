import json

import pytest

from certified_flight import integrity
from rollout_improvement.identity import digest_json, experiment_identity


def fixture_root(tmp_path):
    for name in ('data', 'configs', 'src', 'scripts'):
        (tmp_path/name).mkdir()
    (tmp_path/'configs/runtime.json').write_text('{"horizon":30}', encoding='utf-8')
    (tmp_path/'src/control.py').write_text('GAIN = 1\n', encoding='utf-8')
    (tmp_path/'data/table.bin').write_bytes(b'aero')
    integrity.write_sums(tmp_path, ['configs/runtime.json', 'data/table.bin'])
    return tmp_path


def test_identity_stable_and_sensitive_to_sources(tmp_path):
    root = fixture_root(tmp_path)
    before = experiment_identity(root)
    assert before == experiment_identity(root)
    assert json.loads(json.dumps(before)) == before
    (root/'src/control.py').write_text('GAIN = 2\n', encoding='utf-8')
    assert experiment_identity(root)['sha256'] != before['sha256']


def test_identity_rejects_stale_input_manifest(tmp_path):
    root = fixture_root(tmp_path)
    (root/'data/table.bin').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Modified input'):
        experiment_identity(root)


def test_identity_changes_for_new_input_manifest(tmp_path):
    root = fixture_root(tmp_path)
    before = experiment_identity(root)
    (root/'data/table.bin').write_bytes(b'changed')
    integrity.write_sums(root, ['configs/runtime.json', 'data/table.bin'])
    assert experiment_identity(root)['sha256'] != before['sha256']


def test_json_digest_rejects_nonfinite_values():
    assert digest_json({'a':1,'b':2}) == digest_json({'b':2,'a':1})
    with pytest.raises(ValueError):
        digest_json({'a':float('nan')})
