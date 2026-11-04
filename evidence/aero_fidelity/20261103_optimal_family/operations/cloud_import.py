"""Import a completed local MATLAB batch, checking every file before frozen audit."""
from pathlib import Path,PurePosixPath
import hashlib,json,sys,tarfile
OPS=Path('/hy-tmp/aero_opt_operations_20261103'); REPO=Path('/hy-tmp/dqn-aero-opt-f248c4f')
letter=sys.argv[1]; assert letter in ('a','b')
expected=sys.argv[2]; archive=OPS/f'family_{letter}_generated.tar.gz'
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(archive)==expected
dest=REPO/f'runs/aero_opt_{letter}'; assert not dest.exists(),'Refuse overwrite of existing family'
with tarfile.open(archive) as tf:
    members=tf.getmembers()
    for m in members:
        p=PurePosixPath(m.name)
        assert p.parts[0]=='family' and not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())
    dest.mkdir(parents=True)
    tf.extractall(dest,members=members,filter='data')
fam=dest/'family'; hashes=json.loads((fam/'TRANSFER_SHA256.json').read_text())
assert set(hashes)=={p.relative_to(fam).as_posix() for p in fam.rglob('*') if p.is_file() and p.name!='TRANSFER_SHA256.json'}
for name,h in hashes.items(): assert sha(fam/name)==h,name
receipt=dict(stage=letter,verified=True,archive_sha256=expected,files=len(hashes),job_count=len(json.loads((fam/'jobs.json').read_text())))
with (OPS/f'import_{letter}.json').open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
print(json.dumps(receipt))
