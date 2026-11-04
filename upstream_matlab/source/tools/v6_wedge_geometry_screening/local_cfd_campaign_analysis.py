"""Analyse the SU2 campaign: CL/CD table, equal-lift comparison, geometry ranking vs wedge model, morphing benefit."""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(r'D:\v6-mission-morph-core\results\v6_wedge_geometry_screening')
C = OUT / 'local_cfd_campaign'
rows = [json.loads(p.read_text()) for p in C.glob('*/M*/result.json')]
d = pd.DataFrame(rows)
d = d[d.get('error').isna()] if 'error' in d else d
d['LD'] = d.CL / d.CD
d['converged'] = d.rms_rho < -7
w = pd.read_csv(OUT / 'offdesign/cross_mach_performance.csv')
w = w.rename(columns={'M_flight': 'M', 'alpha_flight': 'alpha', 'CL': 'wedge_CL', 'CD_pressure': 'wedge_CD', 'L_over_D': 'wedge_LD'})
d = d.merge(w[['geometry_id', 'M', 'alpha', 'wedge_CL', 'wedge_CD', 'wedge_LD']], on=['geometry_id', 'M', 'alpha'], how='left')
d = d.rename(columns={'CL': 'CFD_CL', 'CD': 'CFD_CD', 'LD': 'CFD_LD'})
d.sort_values(['M', 'alpha', 'geometry_id']).to_csv(C / 'campaign_CL_CD_table.csv', index=False, float_format='%.6g')

# equal-lift: for each geometry & Mach fit log-quadratic CD(CL) through the 3 alphas, evaluate at target CLs
targets = [0.020, 0.030, 0.040]
el = []
for (g, M), s in d.groupby(['geometry_id', 'M']):
    s = s.sort_values('alpha')
    if len(s) < 3:
        continue
    for src, cl, cd in (('CFD', s.CFD_CL.values, s.CFD_CD.values), ('wedge', s.wedge_CL.values, s.wedge_CD.values)):
        if np.isnan(cl).any():
            continue
        co = np.polyfit(cl, cd, 2)
        for t in targets:
            if cl.min() * 0.98 <= t <= cl.max() * 1.02:
                cdt = np.polyval(co, t)
                el.append(dict(source=src, geometry_id=g, M=M, CL_target=t, CD=cdt, LD=t / cdt))
el = pd.DataFrame(el)
el.to_csv(C / 'campaign_equal_lift.csv', index=False, float_format='%.6g')

rep = []
cfd = el[el.source == 'CFD']
for (M, t), s in cfd.groupby(['M', 'CL_target']):
    s = s.sort_values('LD', ascending=False)
    wd = el[(el.source == 'wedge') & (el.M == M) & (el.CL_target == t)].sort_values('LD', ascending=False)
    rep.append(dict(M=M, CL_target=t, best_CFD=s.iloc[0].geometry_id, LD_best=s.iloc[0].LD, worst_CFD=s.iloc[-1].geometry_id, LD_worst=s.iloc[-1].LD,
                    spread_pct=100 * (s.iloc[0].LD / s.iloc[-1].LD - 1), best_wedge=wd.iloc[0].geometry_id if len(wd) else None,
                    G42_LD=s[s.geometry_id == 'G42'].LD.mean()))
rep = pd.DataFrame(rep)
rep.to_csv(C / 'campaign_ranking_by_condition.csv', index=False, float_format='%.5g')
# rank agreement CFD vs wedge
agree = []
for (M, t), s in el.groupby(['M', 'CL_target']):
    a = s[s.source == 'CFD'].set_index('geometry_id').LD; b = s[s.source == 'wedge'].set_index('geometry_id').LD
    k = a.index.intersection(b.index)
    if len(k) > 2:
        agree.append(dict(M=M, CL_target=t, spearman=a[k].corr(b[k], method='spearman'), mean_ratio_CFD_over_wedge=(a[k] / b[k]).mean()))
agree = pd.DataFrame(agree); agree.to_csv(C / 'campaign_cfd_vs_wedge_agreement.csv', index=False, float_format='%.4g')
# morphing benefit: fixed geometry (best average over the grid) vs per-condition best geometry
piv = cfd.pivot_table(index='geometry_id', columns=['M', 'CL_target'], values='LD')
piv = piv.dropna(axis=0)
fixed = piv.mean(axis=1).idxmax()
morph = piv.max(axis=0); fx = piv.loc[fixed]
ben = pd.DataFrame({'fixed_geom_LD': fx, 'morph_best_LD': morph, 'gain_pct': 100 * (morph / fx - 1), 'best_geom': piv.idxmax(axis=0)})
ben.to_csv(C / 'campaign_morphing_benefit.csv', float_format='%.5g')
print(d[['geometry_id', 'M', 'alpha', 'CFD_CL', 'CFD_CD', 'CFD_LD', 'wedge_LD', 'rms_rho', 'iters']].sort_values(['M', 'alpha', 'geometry_id']).to_string(float_format=lambda v: '%.5g' % v))
print(rep.to_string(float_format=lambda v: '%.4g' % v)); print(agree.to_string()); print('fixed geometry:', fixed); print(ben.to_string())
