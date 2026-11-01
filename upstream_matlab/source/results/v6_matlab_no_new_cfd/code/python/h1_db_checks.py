from pathlib import Path
import sys; sys.dont_write_bytecode=True
sys.path.insert(0, str((Path(__file__).resolve().parents[4] / 'results/v6_matlab_no_new_cfd/code/python'))); sys.path.insert(0, str((Path(__file__).resolve().parents[4] / 'tools/v6_aero_physics_validation')))
import numpy as np, pandas as pd, json
import aerodb7 as D
import common as C1, aeromodels as AM
OUT=D.OUT; h=pd.read_csv(OUT/"tests/heldout_direct_matlab.csv"); hin=pd.read_csv(OUT/"work/heldout_inputs.csv")
res=[]
for br,name in (("STRICT_ORIGINAL","STRICT"),("AREA_MORPH_EXPLORATORY","AREA")):
    db=D.AeroDB7(br,base=0.5,friction=True)
    for r,ri in zip(h.itertuples(),hin.itertuples()):
        if r.branch!=name: continue
        c=db.components(r.Mach,np.deg2rad(r.alpha_deg),r.xi,h=r.h_m)
        # Python independent strip model on the same coordinates
        P=C1.R.panels(C1.load_mat((Path(__file__).resolve().parents[4] / ri.mat_path).resolve())); rr,_=AM.all_models(P,r.Mach,r.alpha_deg,C1.SREF,None); cs=rr["C_strip_shock_expansion"]
        res.append(dict(branch=name,param=r.param,Mach=r.Mach,alpha_deg=r.alpha_deg,h_m=r.h_m,CL_direct=r.CL_p+r.CL_f,CL_db=c["CL_pressure_base"]+c["CL_friction"],CD_direct=r.CD_p+r.CD_f+r.CD_base05,CD_db=c["CD_pressure_base"]+c["CD_friction"],
            CDf_direct=r.CD_f,CDf_db=c["CD_friction"],CDp_direct=r.CD_p,CDp_pyC=cs[1],CLp_direct=r.CL_p,CLp_pyC=cs[0],CL_p_db=c["CL_pressure_base"]))
d=pd.DataFrame(res)
d["dCD_pct"]=(d.CD_db/d.CD_direct-1)*100; d["dCDf_pct"]=(d.CDf_db/d.CDf_direct-1)*100; d["dCL_abs"]=d.CL_db-d.CL_direct; d["dCL_pct_of_0.03"]=d.dCL_abs/0.03*100
d["MATLAB_vs_PythonC_dCD_pct"]=(d.CDp_direct/d.CDp_pyC-1)*100; d["MATLAB_vs_PythonC_dCL_abs"]=d.CLp_direct-d.CLp_pyC
d.to_csv(OUT/"tests/db_vs_direct_heldout.csv",index=False)
s={}
for b,x in d.groupby("branch"):
    s[b]=dict(n=len(x),CD_total_rms_pct=float(np.sqrt((x.dCD_pct**2).mean())),CD_total_max_abs_pct=float(x.dCD_pct.abs().max()),CD_fric_rms_pct=float(np.sqrt((x.dCDf_pct**2).mean())),CL_abs_rms=float(np.sqrt((x.dCL_abs**2).mean())),CL_abs_max=float(x.dCL_abs.abs().max()),
              matlab_vs_python_CD_max_abs_pct=float(x.MATLAB_vs_PythonC_dCD_pct.abs().max()),matlab_vs_python_CL_max_abs=float(x.MATLAB_vs_PythonC_dCL_abs.abs().max()))
print(json.dumps(s,indent=1)); json.dump(s,open(OUT/"tests/db_consistency_summary.json","w"),indent=1)
print(d.sort_values("dCD_pct",key=abs,ascending=False).head(6)[["branch","param","Mach","alpha_deg","h_m","CD_direct","CD_db","dCD_pct","dCL_abs"]].round(5).to_string())
