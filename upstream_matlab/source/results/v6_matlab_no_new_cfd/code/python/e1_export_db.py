"""Export the queryable database to long CSV (pressure grid 23x15, friction grid 13x7x5 altitudes) + provenance."""
from pathlib import Path
import sys; sys.dont_write_bytecode=True
sys.path.insert(0, str((Path(__file__).resolve().parents[4] / 'results/v6_matlab_no_new_cfd/code/python')))
import numpy as np, pandas as pd, aerodb7 as D
OUT=D.OUT; P=[]; F=[]
for b in ("STRICT_ORIGINAL","AREA_MORPH_EXPLORATORY"):
    Lp=D.load_branch(b,"p"); Lg=D.load_branch(b,"g")
    for mp,mg in zip(Lp,Lg):
        M=np.asarray(mp["Mach"]); A=np.asarray(mp["Alpha"]); Mi,Ai=np.meshgrid(M,A,indexing="ij")
        P.append(pd.DataFrame(dict(branch=b,geometry_id=str(mp["gid"]),param=float(mp["param"]),geometry_hash=str(mp["ghash"]),Mach=Mi.ravel(),alpha_deg=Ai.ravel(),CL_pressure=np.asarray(mp["CLp"]).ravel(),CD_pressure=np.asarray(mp["CDp"]).ravel(),
            CD_base_pb0p5=np.asarray(mp["CDb5"]).ravel(),CD_base_pb0p3=np.asarray(mp["CDb3"]).ravel(),Cm_pressure=np.asarray(mp["Cm"]).ravel(),valid=np.asarray(mp["valid"]).astype(bool).ravel())))
        M=np.asarray(mg["Mach"]); A=np.asarray(mg["Alpha"]); H=np.asarray(mg["h_m"])
        Mi,Ai,Hi=np.meshgrid(M,A,H,indexing="ij")
        F.append(pd.DataFrame(dict(branch=b,geometry_id=str(mg["gid"]),param=float(mg["param"]),Mach=Mi.ravel(),alpha_deg=Ai.ravel(),h_m=Hi.ravel(),CL_friction=np.asarray(mg["CLf"]).ravel(),CD_friction=np.asarray(mg["CDf"]).ravel())))
p=pd.concat(P); f=pd.concat(F)
for d in (p,f): d["source"]="offdesign_aero_v7 P2 (MATLAB R2023a), NO CFD"; d["version"]="offdesign_aero_v7-1.0"
p["domain"]="Mach 3.0-8.5, alpha -2..12 deg"; f["domain"]="Mach 3.0-8.5, alpha -2..12 deg, h 20-44 km; turbulent Eckert, Tw=1000 K"
p.to_csv(OUT/"DB_pressure.csv",index=False); f.to_csv(OUT/"DB_friction.csv",index=False)
print(len(p),len(f),"invalid pressure cells:",int((~p.valid).sum()), "min/max CD_p",p.CD_pressure.min(),p.CD_pressure.max())
