from pathlib import Path
import sys; sys.dont_write_bytecode=True
sys.path.insert(0, str((Path(__file__).resolve().parents[4] / 'results/v6_matlab_no_new_cfd/code/python')))
import numpy as np, pandas as pd
from scipy.io import savemat
import g1_geometry as G
ROOT=G.ROOT; OUT=G.OUT; rng=np.random.default_rng(7); atm=pd.read_csv(OUT/"work/atm_levels.csv")
from scipy.interpolate import interp1d
Tf=interp1d(atm.h_m,atm.T_K); rf=interp1d(atm.h_m,np.log(atm.rho))
(OUT/"geometry/heldout").mkdir(exist_ok=True, parents=True)
geo=[]
for lam in (0.9375,1.0625,0.9875):
    lo,up=G.lam_map(lam); p=OUT/f"geometry/heldout/AREA_lam{lam:.4f}.mat"; savemat(p,{"wav_x":lo[...,0],"wav_y":lo[...,1],"wav_z":lo[...,2],"wav_up_x":up[...,0],"wav_up_y":up[...,1],"wav_up_z":up[...,2]}); geo.append(("AREA",lam,(lam-0.9)/0.2,str(p)))
md=pd.read_csv(OUT/"GEOMETRY_METADATA.csv"); 
for x in (0.8125,0.9375): r=md[(md.branch=="STRICT_ORIGINAL")&(np.isclose(md.value,x))].iloc[0]; geo.append(("STRICT",x,x,str((ROOT / r.mat_path).resolve())))
rows=[]
for b,p,xi,mp in geo:
    for _ in range(14):
        M=rng.uniform(3.05,8.45); a=rng.uniform(-1.95,11.95); h=rng.uniform(20500,43500); rows.append((b,p,xi,mp,M,a,h,float(Tf(h)),float(np.exp(rf(h)))))
pd.DataFrame(rows,columns=["branch","param","xi","mat_path","Mach","alpha_deg","h_m","T_K","rho"]).to_csv(OUT/"work/heldout_inputs.csv",index=False); print(len(rows))

