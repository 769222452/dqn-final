"""Write manifest + zip for GoCart_Batch_Win7 from an explicit allowlist (no results/logs/runtime leftovers)."""
import hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/v6_wedge_geometry_screening'
PKG = OUT / 'GoCart_Batch_Win7'

# README with a UTF-8 BOM so Windows 7 Notepad shows Chinese correctly
rd = PKG / 'README_使用说明.txt'
b = rd.read_bytes()
if not b.startswith(b'\xef\xbb\xbf'):
    rd.write_bytes(b'\xef\xbb\xbf' + b)
for sub, note in (('results', 'Created at run time: CFD_RESULTS.csv and runs\\. Nothing here is a CFD result yet.\r\n'),
                  ('logs', 'Run logs are written here (runner.log).\r\n')):
    (PKG / sub).mkdir(exist_ok=True)
    (PKG / sub / 'README.txt').write_text(note, encoding='ascii')

static = ['CHECK_GOCART_ENV.bat', 'RUN_ONE.bat', 'RUN_ALL.bat', 'STOP_AFTER_CURRENT.bat', 'cases.csv', 'cases_full_343.csv',
          'config.ini', 'README_使用说明.txt', 'scripts/gc_runner.vbs', 'scripts/launch.bat', 'scripts/adapter_gocart.bat',
          'scripts/exe_info.vbs', 'scripts/ADAPTER_CONTRACT.txt', 'case_template/PUT_VERIFIED_CASE_INPUTS_HERE.txt',
          'results/README.txt', 'logs/README.txt']
files = static + ['geometries/' + p.name for p in sorted((PKG / 'geometries').glob('*.step'))]
selftest = OUT / 'logs/gocart_batch_mock_selftest.json'
extra = {}
if selftest.exists():
    (PKG / 'logs/gocart_batch_mock_selftest.json').write_bytes(selftest.read_bytes())
    files.append('logs/gocart_batch_mock_selftest.json')
for f in files:
    assert (PKG / f).exists(), f
sha = {f: hashlib.sha256((PKG / f).read_bytes()).hexdigest() for f in files}
man = {'package': 'GoCart_Batch_Win7', 'geometries': 31, 'default_cases': 2, 'full_cases': 343,
       'CFD_STARTED': False, 'GOCART_ADAPTER_CONFIGURED': False, 'WINDOWS7_TESTED': False,
       'mock_tests_on_windows11_passed': json.loads(selftest.read_text())['tests_passed'] if selftest.exists() else None,
       'files_sha256': sha}
(PKG / 'package_manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
z = OUT / 'GoCart_Batch_Win7.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files + ['package_manifest.json']:
        zf.write(PKG / f, 'GoCart_Batch_Win7/' + f)
print(z, z.stat().st_size, len(files) + 1, 'files')
