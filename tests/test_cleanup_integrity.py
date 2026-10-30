"""Retired assets remain recoverable and cleanup cannot silently retune inputs."""
import hashlib
import json
from pathlib import Path
import zipfile

import yaml

from certified_flight import integrity
from certified_flight.build import ROOT


def archive_index():
    return json.loads((ROOT / 'archive/retired_assets_20261030.json').read_text(encoding='utf-8'))


def test_retired_assets_and_original_bindings_are_recoverable():
    index = archive_index()
    archive = ROOT / index['archive']
    assert integrity.sha256(archive) == index['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(index['entries']) | {'_archive_manifest.json'}
        assert json.loads(z.read('_archive_manifest.json'))['entries'] == index['entries']
        for rel, record in index['entries'].items():
            assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
            data = z.read(rel)
            assert len(data) == record['bytes']
            assert hashlib.sha256(data).hexdigest() == record['sha256']
            if record['disposition'] == 'retired':
                assert not (ROOT / rel).exists()
        old = json.loads(z.read('data/geometry_subset/manifest.json'))
        current = json.loads((ROOT / 'data/geometry_subset/manifest.json').read_text(encoding='utf-8'))
        unchanged = {item['path']: item for item in old['files']}
        for item in current['files']:
            if item['path'] != 'README_ZH.txt':
                assert item == unchanged[item['path']]
        assert len(list((ROOT / 'data/geometry_subset/geometries').glob('*/native_geometry.mat'))) == 298
        old_controller = json.loads(z.read('data/surrogate/controller.json'))
        assert integrity.sha256(ROOT / 'data/surrogate/controller.npz') == old_controller['model_sha256']


def test_cleanup_does_not_change_retained_configuration_values():
    def assert_subset(current, original):
        if isinstance(current, dict):
            for key, value in current.items():
                assert key in original
                assert_subset(value, original[key])
        else:
            assert type(current) is type(original) and current == original

    index = archive_index()
    with zipfile.ZipFile(ROOT / index['archive']) as z:
        for name in ('mission', 'vehicle', 'environment', 'trajectory'):
            rel = f'configs/{name}.yaml'
            assert_subset(yaml.safe_load((ROOT / rel).read_text(encoding='utf-8')),
                          yaml.safe_load(z.read(rel)))


def test_integrity_list_covers_remaining_inputs_physics_and_archive():
    entries = integrity.read_sums(ROOT)
    for folder in ('configs', 'data', 'src/morphing_waverider', 'archive'):
        for path in (ROOT / folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel != integrity.SUMS:
                assert rel in entries, rel
    integrity.verify(ROOT)
