"""Additional scientific figures from completed metric snapshots; no new physics calls."""
from __future__ import annotations
import argparse
from exp_common import *


def render(tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats_path=OUT/f'metrics/{tag}_summary.json'; stats=load(stats_path)
    domain_path=OUT/'mission/domain.json'; domain=load(domain_path)
    source=OUT/f'metrics/{tag}_mission_top1.csv'
    d=pd.read_csv(source,float_precision='round_trip')
    grid=pd.MultiIndex.from_product([domain['M_grid'],domain['alpha_grid']],names=['M_flight','alpha_deg'])
    d=d.drop_duplicates(['M_flight','alpha_deg']).set_index(['M_flight','alpha_deg']).reindex(grid).reset_index()
    gids=sorted(d.geometry_id.dropna().unique()); legend_rows=(len(gids)+2)//3
    fig=plt.figure(figsize=(14,7+.25*legend_rows),layout='constrained')
    layout=fig.add_gridspec(2,1,height_ratios=[7,.25*legend_rows])
    ax=fig.add_subplot(layout[0]); legend_ax=fig.add_subplot(layout[1]); legend_ax.axis('off')
    absent=d[d.geometry_id.isna()]
    missing_label='No eligible winner at the required CL floor' if stats['equal_and_task_comparison_complete'] else 'No eligible winner / incomplete evidence'
    if len(absent): ax.scatter(absent.M_flight,absent.alpha_deg,s=28,marker='x',color='#b4b4b4',label=missing_label)
    colors=plt.cm.turbo(np.linspace(.03,.97,max(1,len(gids))))
    for i,gid in enumerate(gids):
        g=d[d.geometry_id.eq(gid)]; color=colors[i]; code=f'{i+1:02d}'
        ax.scatter(g.M_flight,g.alpha_deg,s=150,marker='s',color=color,label=f'{code}  {gid}')
        ink='black' if np.dot(color[:3],[.299,.587,.114])>.55 else 'white'
        for r in g.itertuples(): ax.text(r.M_flight,r.alpha_deg,code,ha='center',va='center',fontsize=5.5,color=ink)
    ax.set(xlabel='Flight Mach',ylabel='Angle of attack [deg]')
    ax.set_title(f'Full task-domain P2 nominal winners — {tag}\nWeak ties and non-core points are included; CORE42: {stats["CORE42"]["unique_top1"]} nominal, {len(stats["core_effective_verified"])} verified winners.',fontsize=11,pad=12)
    ax.grid(alpha=.15); ax.set_axisbelow(True)
    legend_ax.legend(*ax.get_legend_handles_labels(),ncol=3,fontsize=7.5,loc='upper center',frameon=False,markerscale=.8,columnspacing=2)
    task_image=OUT/f'figures/{tag}_full_task_Mach_alpha.png'; fig.savefig(task_image,dpi=160); plt.close(fig)
    outputs=[task_image]; sources=[source,domain_path,stats_path]
    trace_path=OUT/f'metrics/{tag}_oracle_trace.csv'; oracle_path=OUT/f'metrics/{tag}_oracle.json'
    if trace_path.exists() and oracle_path.exists():
        t=pd.read_csv(trace_path,float_precision='round_trip'); oracle=load(oracle_path)
        if 'fixed_cumulative_Ns' in t and oracle.get('J_fixed_Ns',0)>0:
            saving=t.fixed_cumulative_Ns-t.oracle_cumulative_Ns
            fraction=100*saving/float(oracle['J_fixed_Ns'])
            fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
            axes[0].plot(t.time_s,saving,color='#007f6e',linewidth=2)
            axes[0].set(ylabel='Cumulative pressure impulse saving [N s]',title=f'Ideal selection on the fixed 1200 s reference — {tag}')
            axes[1].plot(t.time_s,fraction,color='#007f6e',linewidth=2)
            axes[1].set(xlabel='Time [s]',ylabel='Saving / full best-fixed impulse [%]')
            for ax in axes: ax.grid(alpha=.2)
            axes[1].annotate(f'Final: {fraction.iloc[-1]:.6f}%',xy=(t.time_s.iloc[-1],fraction.iloc[-1]),xytext=(-10,-25),textcoords='offset points',ha='right')
            fig.suptitle('Pressure-only P2 upper bound; not a flight-range gain or realizable morphing control',fontsize=11)
            gain_image=OUT/f'figures/{tag}_oracle_impulse_gain.png'; fig.savefig(gain_image,dpi=160); plt.close(fig)
            outputs.append(gain_image); sources += [trace_path,oracle_path]
    manifest={'utc':utc(),'tag':tag,'sources':{str(f.relative_to(OUT)):sha(f) for f in sources},
              'outputs':[str(f.relative_to(OUT)) for f in outputs],'task_grid_points':len(grid),
              'ranked_task_grid_points':int(d.geometry_id.notna().sum()),
              'gain_definition':'J_fixed cumulative minus J_oracle cumulative; percent normalized by full-trajectory J_fixed',
              'scope':'Plots of the saved completed metric snapshot; no coefficient changes, new Teacher calls, or P2 evaluations'}
    dump(OUT/f'figures/{tag}_additional_figures.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--tag')
    args=parser.parse_args(); print(json.dumps(render(args.tag or load(OUT/'metrics/latest.json')['tag']),indent=2))
