"""Optional C code generation for numeric CasADi evaluations.

The generated C evaluates the identical CasADi expression graph that the MX
virtual machine would evaluate. It is compiled without fast-math and without
floating-point contraction so IEEE double results stay identical.

Set CERTIFIED_FLIGHT_CODEGEN=0 to use the MX virtual machine; CERTIFIED_FLIGHT_CC
selects the C compiler. Compiled libraries are cached in <repository>/.cache/codegen.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import casadi as ca

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'.cache'/'codegen'
FLAGS = ('-O1', '-fno-fast-math', '-ffp-contract=off', '-shared')
_BUNDLED = Path('C:/Program Files/CodeBlocks/MinGW/bin/gcc.exe')
_LOADED: dict[str, ca.Function] = {}
_LOCK = threading.Lock()
_FAILURES: list[str] = []


def enabled() -> bool:
    return os.environ.get('CERTIFIED_FLIGHT_CODEGEN', '1').strip().lower() not in ('0', 'false', 'no', 'off')


def compiler() -> str | None:
    found = os.environ.get('CERTIFIED_FLIGHT_CC') or shutil.which('gcc')
    if found is None and _BUNDLED.is_file():
        found = str(_BUNDLED)
    return found


def compile_function(fn: ca.Function) -> ca.Function:
    """Return a compiled equivalent of ``fn``, or ``fn`` itself when unavailable.

    The cache key is the SHA-256 of the generated C source plus compiler flags,
    so any change to the expression graph produces a new library.
    """
    if not enabled():
        return fn
    cc = compiler()
    if cc is None:
        _FAILURES.append(f'{fn.name()}: no C compiler')
        return fn
    name = fn.name()
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix='gen_', dir=CACHE))
        try:
            gen = ca.CodeGenerator(name+'.c')
            gen.add(fn)
            gen.generate(str(build)+os.sep)
            source = build/(name+'.c')
            digest = hashlib.sha256(source.read_bytes()+' '.join(FLAGS).encode()).hexdigest()[:24]
            library = CACHE/(f'{name}_{digest}'+('.dll' if os.name == 'nt' else '.so'))
            with _LOCK:
                if digest in _LOADED:
                    return _LOADED[digest]
                if not library.exists():
                    candidate = build/('lib'+('.dll' if os.name == 'nt' else '.so'))
                    command = [cc, *FLAGS, str(source), '-o', str(candidate)]
                    if os.name != 'nt':
                        command.insert(1, '-fPIC')
                    env = dict(os.environ, PATH=str(Path(cc).parent)+os.pathsep+os.environ.get('PATH', ''))
                    subprocess.run(command, env=env, check=True, capture_output=True, timeout=1800)
                    try:
                        os.replace(candidate, library)  # atomic; parallel workers may race
                    except OSError:
                        if not library.exists():
                            raise
                compiled = ca.external(name, str(library))
                _LOADED[digest] = compiled
                return compiled
        finally:
            shutil.rmtree(build, ignore_errors=True)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        _FAILURES.append(f'{name}: {exc!r}')
        return fn
