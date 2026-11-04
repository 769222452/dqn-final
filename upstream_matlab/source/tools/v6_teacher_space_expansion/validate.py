"""Bounded regression checks using the frozen real pool; does not generate fake research results."""
from exp_common import *
from analysis import rank_fixed,winners
from evaluation import equal_lift,usable
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

def main():
    verify_sources(); p=old_pool(); d=pd.read_csv(OUT/'baseline/frozen49.csv',float_precision='round_trip')
    ranks=rank_fixed(d,p); frozen=pd.read_csv(OLD/'screening/top5_by_condition.csv',float_precision='round_trip')
    actual=ranks[ranks['rank'].le(5)].sort_values(['M_flight','alpha_deg','rank'])
    expected=frozen.rename(columns={'alpha_flight_deg':'alpha_deg'}).sort_values(['M_flight','alpha_deg','rank'])
    assert len(actual)==len(expected)==245
    assert actual.geometry_id.tolist()==expected.geometry_id.tolist()
    assert np.allclose(actual.L_over_D,expected.L_over_D,rtol=1e-13,atol=1e-13)
    cw=winners(ranks); assert cw[cw.alpha_deg.ge(5)].geometry_id.isin(AB).all()
    assert len(p)==453 and p.geometry_hash.nunique()==453
    assert int(gate(p).sum())==299
    results=[]
    for m,g in d[d.geometry_id.eq('G42')].groupby('M_flight'):
        g=g.sort_values('alpha_deg'); assert usable(g).all()
        cl=PchipInterpolator(g.alpha_deg,g.CL,extrapolate=False); cd=PchipInterpolator(g.alpha_deg,g.CD_pressure,extrapolate=False)
        target=float((g.CL.iloc[0]+g.CL.iloc[-1])/2)
        alpha=brentq(lambda a:float(cl(a))-target,g.alpha_deg.min(),g.alpha_deg.max(),xtol=1e-12)
        r=equal_lift(g,target); assert r['status']=='OK' and abs(r['alpha_at_target']-alpha)<1e-9 and abs(r['CD_at_target']-cd(alpha))<1e-12
        outside=equal_lift(g,float(g.CL.max()+1)); assert outside['status']!='OK'
        # Fault injection on a copy checks that invalid gaps are not bridged.
        broken=g.copy(); broken.loc[broken.index[2:5],'model_valid']=0
        hole_target=float(cl(7.0)); assert equal_lift(broken,hole_target)['status']!='OK'
        results.append({'M_flight':m,'alpha_brent':alpha,'alpha_PCHIP_roots':r['alpha_at_target'],'CD_difference':float(r['CD_at_target']-cd(alpha))})
    mission=pd.read_csv(OUT/'mission/samples.csv'); assert mission.groupby('trajectory_id').dt_s.sum().eq(1200).all()
    pre=load(OUT/'preflight/summary.json'); assert pre['teacher_attempts']==12 and pre['passed']
    assert len(list((OUT/'cases').glob('*/record.json')))>=12
    info={'status':'PASS','frozen_top5_rows_exact_identity':245,'old_candidates':453,'old_hard_gate_pass':299,'PCHIP_vs_independent_brent_checks':results,'invalid_gap_and_outside_range_rejection':True,'mission_time_weights_each_s':1200,'preflight_real':pre,'protected_sources_unchanged':True}
    dump(OUT/'validation.json',info); print(json.dumps(clean(info),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
