"""Record source excerpts and static semantic evidence; never execute teacher optimizer."""
from pathlib import Path
import json,re
from prepare import ROOT,OUT,sha,dump

def evidence(path,needle,context=2):
    lines=path.read_text(encoding='utf-8-sig').splitlines();hits=[]
    for i,l in enumerate(lines):
        if needle in l:hits.append({'line':i+1,'excerpt':'\n'.join(f'{j+1}: {lines[j]}' for j in range(max(0,i-context),min(len(lines),i+context+1)))})
    return {'path':str(path),'sha256':sha(path),'needle':needle,'hits':hits}

def main():
    mat=Path('D:/mat/.code');active=ROOT/'teacher_matlab/active';original=ROOT/'teacher_matlab/original'
    queries=[(mat/'main_optimize.m','surrogate_assisted_ga_wave_rider_6'),(mat/'surrogate_assisted_ga_wave_rider_6.m','gamultiobj(surrogateFitness'),(mat/'surrogate_assisted_ga_wave_rider_6.m','~isnan(result.L_D)'),(mat/'waveRiderEval.m','if any(isnan(wav_x(:)))'),(mat/'waveRiderEval.m','local_wav_up_z(1,j)='),(mat/'waveRiderEval.m','Vol=2*volume'),(mat/'volume_simianti.m','abs(dot'),(mat/'vol_eff_sim.m','find_ellipse_a'),(active/'build_geometry_core.m','Vector_V='),(active/'build_geometry_core.m',"rat' * local_x_y_z_xingxian'"),(active/'build_geometry_core.m','local_wav_up_z(1,j)='),(active/'build_geometry_core.m','最外侧只有一个点'),(original/'generatewaverider.m','Generate_Upper_Surface_Bezier'),(original/'generatewaverider.m','-wav_z'),(active/'create_shock_functions.m','x_y_z_v = rat'),(active/'create_shock_functions.m','F_vec = cross'),(active/'build_geometry_core.m','vector_C=cross(F_val,Vector_V)'),(ROOT/'src/morphing_waverider/teacher_v5/geometry.py','np.array([1.,-1.,-1.])')]
    sources=[evidence(p,n) for p,n in queries]
    # Also preserve complete optimizer files as text evidence within permitted
    # output folder; no executable copies are added to any MATLAB search path.
    for name in ['main_optimize.m','surrogate_assisted_ga_wave_rider_6.m','waveRiderEval.m','volume.m','volume_wumianti.m','volume_simianti.m','vol_eff_sim.m']:
        p=mat/name;q=OUT/'source_evidence'/(name+'.txt');q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes())
    trace={'optimizer_gate':'ORIGINAL_OPTIMIZER_HAS_NO_GEOMETRY_VALIDITY_GATE','checks_found':{'upper_lower_intersection':False,'minimum_thickness':False,'self_intersection':False,'all_coordinate_finiteness':False,'partial_numeric_check':'waveRiderEval tests wav_x NaN/Inf/isreal; evaluatePoint accepts both objectives not NaN','positive_volume_is_sufficient':False},'physical_semantics':{'lower':'Shock-based streamline coordinates in rat frame, transformed back by rat transpose; then native per-streamline interp1 in x.','upper':'Ruled surface starting at the same leading edge; y=y_LE, z=z_LE+(x-x_LE)*tan(AOA_prog).','fixed_design_AOA_deg':-4,'body_transform':'diag(1,-1,-1), proper rigid rotation det=+1','thickness':'When each skin is a single-valued z(x,y), upper_body_z-lower_body_z = lower_program_z-upper_program_z. On folded projections it is not a globally defined thickness.','outer_normals':'Native traversal is checked locally; for a valid graph, outward upper/lower orientation is opposite. Graph projection reversal alone is not physical self-intersection.','alternative_upper':'Bezier/CST construction is commented out in original generatewaverider; it is not active evidence of intended skin.'},'static_frame_consistency_question':{'observation':'create_shock_functions differentiates rat-rotated coordinates, so F_val and ru/rv are rotated-frame quantities. build_geometry_core computes cross(F_val,Vector_V), retaining the program-frame freestream vector. rat*Vector_V is [1,0,0], not Vector_V at AOA=-4 deg.','status':'SOURCE_LEVEL_FRAME_MIXING_FOUND; causal contribution to intersections NOT established. No modified geometry run or AOA sweep authorized/performed.','scope':'Present in original geometry logic too; ask teacher whether this is intentional and which vector frame is required.'},'evidence':sources}
    dump(OUT/'optimizer_and_geometry_source_trace.json',trace)
    print('Source trace saved; no original optimizer or aero executed.')
if __name__=='__main__':main()
