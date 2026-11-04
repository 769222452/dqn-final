"""Final evidence consistency, image metadata and source protection check."""
import csv,json,sys,platform
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
import numpy as np
from prepare import ROOT,OUT,TOOLS,sha,dump,protected

def main():
    results=json.loads((OUT/'pareto_geometry_audit.json').read_text(encoding='utf-8'))['results'];assert len(results)==52;assert sorted(r['original_row_index'] for r in results)==list(range(1,53))
    budget=json.loads((OUT/'geometry_audit_budget.json').read_text(encoding='utf-8'));assert budget['geometry_calls']==52;assert len(set(c['design_index'] for c in budget['calls']))==52;assert budget['aerodynamic_calls']==budget['ocp_calls']==budget['training_calls']==0
    fm=json.loads((OUT/'figure_manifest.json').read_text(encoding='utf-8'));assert len(fm['files'])==151
    image_results=[]
    for name in fm['files']:
        p=Path(name)
        with Image.open(p) as im:
            assert im.format=='PNG';dpi=im.info.get('dpi');assert dpi and all(abs(d-300)<.1 for d in dpi);im.verify()
        image_results.append({'path':str(p),'sha256':sha(p),'dpi':dpi})
    ver=json.loads((OUT/'independent_intersection_verification.json').read_text(encoding='utf-8'));assert len(ver['results'])==52 and ver['all_confirmed']
    for r,v in zip(results,ver['results']):
        assert r['design_index']==v['design_index'];assert r['surface_intersection_count']==v['record_count']
        if r['classification']=='INVALID':assert v['proper_interior_crossing_pairs']>0
        if r['classification']=='VALID':assert r['surface_intersection_count']==r['self_intersection_count']==0 and r['single_valued_graph'] and r['defined_section_count']==6
        assert r['body_transform_invariant'] and r['opposite_diagonal_sensitivity']['crossing_presence_agrees']
        assert r['degenerate_face_count']==r['allowed_tip_degenerate_count']==98
        assert r['boundary_checks']['non_symmetry_interior_vertices_on_y0']==0
        m=loadmat(Path(r['case_dir'])/'native_geometry.mat');assert np.array_equal(np.ravel(m['inputs10']),r['inputs10'])
    before=json.loads((OUT/'protected_files_before.json').read_text(encoding='utf-8'));after=protected();assert before==after
    dump(OUT/'protected_files_after.json',after)
    # Supplemental expanded search without changing the frozen source manifest.
    roots=[Path(p) for p in ['D:/final-DQN','D:/final-DQN-v3','D:/final-DQN-physics-v4','D:/mat','D:/final-DQN-v5']];matches=[]
    for root in roots:
        for p in root.rglob('wave_rider_optimization_final.mat'):
            m=loadmat(p);matches.append({'path':str(p),'sha256':sha(p),'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in m.items() if not k.startswith('__')}})
    dump(OUT/'expanded_source_search.json',{'roots':list(map(str,roots)),'matches':matches});assert len(matches)==1
    report={'status':'PASS','native_design_count':52,'valid_count':26,'invalid_count':26,'png_count':151,'all_images_verified_png_300dpi':True,'independent_intersection_records_confirmed':sum(r['record_count'] for r in ver['results']),'protected_files_unchanged':len(before['files']),'rules_hash_unchanged':sha(OUT/'classification_rules.json')==json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))['rules_sha256'],'test_count':15,'python':sys.version,'platform':platform.platform(),'tool_hashes':{p.name:sha(p) for p in TOOLS.glob('*') if p.is_file()},'images':image_results}
    assert report['rules_hash_unchanged'];dump(OUT/'delivery_verification.json',report)
    allfiles=[]
    for root in [OUT,TOOLS]:
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='output_manifest.json':allfiles.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(OUT/'output_manifest.json',{'files':allfiles,'file_count':len(allfiles),'total_bytes':sum(p['bytes'] for p in allfiles),'self_excluded':'output_manifest.json'})
    print(json.dumps({k:v for k,v in report.items() if k not in ['images','tool_hashes','python','platform']},ensure_ascii=False))
if __name__=='__main__':main()
