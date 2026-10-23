"""SHA-256 integrity check of the runtime inputs listed in data/SHA256SUMS.

The list covers every data file, the configuration files and the sources of the
physics package morphing_waverider. Lines use the ``sha256sum`` format: ``<hex digest>  <path>``
with paths relative to the repository root and forward slashes.
"""
import hashlib
from pathlib import Path

SUMS = 'data/SHA256SUMS'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_sums(root):
    entries = {}
    for line in (Path(root) / SUMS).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, rel = line.split('  ', 1)
        entries[rel] = digest
    return entries


def verify(root):
    """Raise RuntimeError if a listed file is missing, outside the root or modified."""
    root = Path(root).resolve()
    entries = read_sums(root)
    if not entries:
        raise RuntimeError('Empty integrity list: ' + SUMS)
    for rel, digest in entries.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Missing input file: ' + rel)
        if sha256(path) != digest:
            raise RuntimeError('Modified input file: ' + rel)
    return len(entries)


def write_sums(root, paths):
    """Write data/SHA256SUMS for the given repository-relative paths (maintenance helper)."""
    root = Path(root).resolve()
    lines = [f'{sha256(root / rel)}  {Path(rel).as_posix()}' for rel in sorted(paths)]
    (root / SUMS).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
