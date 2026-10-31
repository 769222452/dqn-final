"""Scientific diagnostics from saved runs, with raw controls and no smoothing."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def controls(result, continuous):
    t,a=[],[]
    for row in result['rows']:
        values=row['alpha_nodes_deg'] if continuous else row['alpha_deg']+[row['alpha_deg'][-1]]
        t.extend(row['t0']+np.arange(7)*.5)
        a.extend(values)
    return np.array(t),np.array(a)


def run(args):
    new=json.loads((args.new/'result.json').read_text(encoding='utf-8'))
    old=json.loads((args.reference/'result.json').read_text(encoding='utf-8'))
    args.out.mkdir(parents=True,exist_ok=False)
    for lang in ('zh','en'):
        zh=lang=='zh'
        plt.rcParams.update({'font.family':['SimSun','DejaVu Sans'] if zh else ['DejaVu Sans'],
            'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
            'axes.unicode_minus':False,'legend.frameon':False,'pdf.fonttype':42})
        fig,ax=plt.subplots(3,1,figsize=(8,7.5),constrained_layout=True)
        for data,cont,color,label in ((old,False,'#CC6677','原代理执行' if zh else 'Original surrogate'),
                                     (new,True,'#4477AA','连续迎角，当前CL跟踪优先' if zh else 'Continuous alpha, current-CL priority')):
            t,a=controls(data,cont)
            if cont:
                ax[0].plot(t,a,color=color,lw=1.2,label=label)
                ax[1].plot(t,a,color=color,lw=1.4,label=label)
            else:
                ax[0].step(t,a,where='post',color=color,lw=.7,label=label,alpha=.8)
                ax[1].step(t,a,where='post',color=color,lw=1,label=label)
            ax[2].plot([r['t0']+1.5 for r in data['rows']],[r['CL_mean'] for r in data['rows']],
                       color=color,lw=1.2,label=label)
        ax[0].set_title('(a) 全程原始执行迎角（不平滑）' if zh else '(a) Raw executed alpha (no smoothing)',loc='left')
        ax[1].set_title('(b) 前60秒局部' if zh else '(b) First 60 seconds',loc='left')
        ax[2].set_title('(c) 每3秒实际CL积分均值' if zh else '(c) Actual CL: integrated 3-second mean',loc='left')
        ax[2].axhline(new['CL'],color='#444444',lw=1,ls='--',label='CL指令' if zh else 'CL command')
        ax[0].set_xlim(0,max(new['time_s'],old['time_s']))
        ax[1].set_xlim(0,min(60,max(new['time_s'],old['time_s'])))
        for i in range(2):
            ax[i].set_ylabel('迎角 / °' if zh else 'Alpha / deg')
        ax[2].set_ylabel('升力系数' if zh else 'Lift coefficient')
        for a in ax:
            a.set_xlabel('时间 / s' if zh else 'Time / s')
            a.grid(axis='y',color='#eeeeee',lw=.6)
        ax[0].legend(loc='best',fontsize=9)
        ax[2].legend(loc='best',fontsize=9)
        status=(f"新分支执行至 {new['time_s']:g} s；原分支执行至 {old['time_s']:g} s" if zh
                else f"New: {new['time_s']:g} s; original: {old['time_s']:g} s")
        fig.suptitle(status,fontsize=11)
        for ext in ('png','pdf'):
            fig.savefig(args.out/f'control_comparison_{lang}.{ext}',dpi=220)
        plt.close(fig)
    (args.out/'sources.json').write_text(json.dumps({'new':str(args.new/'result.json'),
        'reference':str(args.reference/'result.json'),'raw_controls':True,'smoothing':False},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
