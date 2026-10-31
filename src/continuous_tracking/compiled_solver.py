"""Compile the existing CasADi NLP helpers, without fast-math or new solver settings.

Uses CasADi's documented generate_dependencies -> shared library -> nlpsol
workflow. Source and flags determine the cache key; no approximate derivatives.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import signal

import casadi as ca

from certified_flight import codegen


def compile_solver(solver, name, options):
    compiler=codegen.compiler()
    if not codegen.enabled() or compiler is None:
        return solver, {'compiled':False,'reason':'Compiler disabled or unavailable'}
    cache=codegen.ROOT/'.cache'/'continuous_nlp'
    cache.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='build_',dir=cache)).resolve()
    # CodeGenerator expects a basename. The process is single-threaded here;
    # restore the process directory even if generation fails.
    previous=Path.cwd()
    try:
        os.chdir(folder)
        solver.generate_dependencies('nlp.c')
    finally:
        os.chdir(previous)
    source=folder/'nlp.c'
    # This large sparse derivative graph makes gcc -O1 compilation very slow.
    # -O0 still removes the MX interpreter and preserves evaluation order.
    flags=('-O0','-fno-fast-math','-ffp-contract=off','-shared')
    digest=hashlib.sha256(source.read_bytes()+' '.join(flags).encode()+ca.__version__.encode()).hexdigest()
    library=cache/(digest+('.dll' if os.name=='nt' else '.so'))
    if not library.exists():
        candidate=folder/library.name
        command=[compiler,*flags,str(source),'-o',str(candidate)]
        if os.name!='nt':
            command.insert(1,'-fPIC')
        env=dict(os.environ,PATH=str(Path(compiler).parent)+os.pathsep+os.environ.get('PATH',''))
        with (folder/'compiler.log').open('wb') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=log,env=env,start_new_session=os.name!='nt')
            try:
                code=proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # Kill only this compiler's process tree; pipe reader threads
                # must not leave cc1 running after a parent gcc timeout.
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=log,stderr=log,timeout=10)
                else:
                    os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=10)
                raise RuntimeError('NLP compiler exceeded 120 s; see '+str(folder/'compiler.log'))
            if code:
                raise RuntimeError('NLP compiler failed; see '+str(folder/'compiler.log'))
        os.replace(candidate,library)
    compiled=ca.nlpsol(name,'ipopt',str(library),options)
    return compiled,dict(compiled=True,library=str(library),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         flags=list(flags))
