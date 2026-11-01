"""Check mission support against the saved P2 tables and their existing validation evidence."""
from __future__ import annotations
from exp_common import *

def inspect_support():
    rows=[]
    for path in sorted((ROOT/'results/v6_matlab_no_new_cfd/db').glob('p_*.mat')):
        d=loadmat(path,variable_names=['Mach','Alpha','valid'])
        m=np.asarray(d['Mach']).ravel(); a=np.asarray(d['Alpha']).ravel()
        rows.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'M_min':float(m.min()),'M_max':float(m.max()),'alpha_min':float(a.min()),'alpha_max':float(a.max()),'computed_points':int(np.asarray(d['valid']).size),'valid_points':int(np.asarray(d['valid']).sum())})
    if not rows: raise RuntimeError('No existing P2 support tables found')
    support_m=[max(r['M_min'] for r in rows),min(r['M_max'] for r in rows)]
    support_a=[max(r['alpha_min'] for r in rows),min(r['alpha_max'] for r in rows)]
    tests=load(ROOT/'results/v6_matlab_no_new_cfd/tests/v7_test_results.json')
    if not all(bool(t['pass']) for t in tests): raise RuntimeError('Existing P2 analytic regression evidence contains failures')
    return {'P2_table_M_support':support_m,'P2_table_alpha_support':support_a,'tables':rows,'existing_analytic_tests_passed':len(tests),
            'test_source':'results/v6_matlab_no_new_cfd/tests/v7_test_results.json','test_source_sha256':sha(ROOT/'results/v6_matlab_no_new_cfd/tests/v7_test_results.json'),
            'scope':'Existing computed P2 table support plus analytic/numerical regression evidence; not full-domain 3D CFD validation. Every new geometry/state still requires model_valid.'}

def record_current_support():
    evidence=inspect_support(); domain=load(OUT/'mission/domain.json')
    assert min(domain['M_grid'])>=evidence['P2_table_M_support'][0] and max(domain['M_grid'])<=evidence['P2_table_M_support'][1]
    assert min(domain['alpha_grid'])>=evidence['P2_table_alpha_support'][0] and max(domain['alpha_grid'])<=evidence['P2_table_alpha_support'][1]
    domain['support_M']=[max(domain['support_M'][0],evidence['P2_table_M_support'][0]),min(domain['support_M'][1],evidence['P2_table_M_support'][1])]
    domain['support_alpha']=[max(domain['support_alpha'][0],evidence['P2_table_alpha_support'][0]),min(domain['support_alpha'][1],evidence['P2_table_alpha_support'][1])]
    domain['support_evidence']='mission/support_evidence.json: intersection of configured mission limits and existing P2 table support; candidate-specific model_valid remains mandatory'
    dump(OUT/'mission/support_evidence.json',evidence); dump(OUT/'mission/domain.json',domain)
    manifest=load(OUT/'baseline/additional_source_hashes.json')
    manifest.update({r['path']:r['sha256'] for r in evidence['tables']}); manifest[evidence['test_source']]=evidence['test_source_sha256']; dump(OUT/'baseline/additional_source_hashes.json',manifest)
    print(json.dumps({k:v for k,v in evidence.items() if k!='tables'},ensure_ascii=False))

if __name__=='__main__': record_current_support()
