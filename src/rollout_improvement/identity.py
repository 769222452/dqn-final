"""Content identities for resumable training and reusable evaluation results.

Resolve and verify actual inputs before trusting their manifest. Identities do
not include output paths or timestamps, and are computed once per invocation.
"""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

from certified_flight import integrity

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_SCHEMA = 'certified-experiment/1'


def digest_json(value):
    """Hash a finite JSON value with stable key order and float representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def experiment_identity(root=None):
    """Identity of verified input bytes, executing sources and numeric software.

    A changed file cannot reuse a cache merely because SHA256SUMS was left
    unchanged: verify() checks the actual contents before hashing the manifest.
    This fingerprint identifies an experiment, not a guarantee against changes
    in wall-clock search outcomes under a different machine load.
    """
    root = Path(ROOT if root is None else root).resolve()
    integrity.verify(root)
    inputs = integrity.read_sums(root)
    inputs[integrity.SUMS] = integrity.sha256(root/integrity.SUMS)
    # Include configs even if a newly added file has not yet entered SHA256SUMS.
    for path in sorted((root/'configs').rglob('*')):
        if path.is_file():
            inputs[path.relative_to(root).as_posix()] = integrity.sha256(path)
    sources = {}
    for folder in ('src', 'scripts'):
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh'):
                sources[path.relative_to(root).as_posix()] = integrity.sha256(path)
    software = dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                    packages={name: version(name) for name in ('numpy', 'scipy', 'casadi', 'PyYAML')})
    numeric = dict(codegen_enabled=os.environ.get('CERTIFIED_FLIGHT_CODEGEN', '1').strip().lower()
                   not in ('0', 'false', 'no', 'off'),
                   blas_threads={k: os.environ.get(k) for k in
                                 ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    payload = dict(schema=IDENTITY_SCHEMA, inputs=inputs, sources=sources, software=software, numeric=numeric)
    return dict(payload, sha256=digest_json(payload))
