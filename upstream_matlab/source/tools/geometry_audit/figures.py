"""Chinese academic figures from saved audit evidence only, default colour cycle."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from prepare import OUT,PARAMS,dump
from mesh_checks import triangulate

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'figure.dpi':110,'mathtext.fontset':'stix','axes.linewidth':.8})
COLORS=plt.rcParams['axes.prop_cycle'].by_key()['color']

def save(fig,path,note=None):
    if note:fig.text(.5,.02,note,ha='center',va='bottom',fontsize=9)
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def axes3d(title,grids):
    fig=plt.figure(figsize=(9.0,4.9));ax=fig.add_subplot(111,projection='3d');ax.set_title(title,pad=10,fontsize=12)
    ax.set_xlabel(r'$x_{\mathrm{p}}$ / m',labelpad=9);ax.set_ylabel(r'$y_{\mathrm{p}}$ / m',labelpad=9);ax.set_zlabel(r'$z_{\mathrm{p}}$ / m',labelpad=10)
    points=np.concatenate([g.reshape(-1,3) for g in grids]);lo=points.min(0);hi=points.max(0)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo,zoom=1.10);ax.view_init(elev=28,azim=-65)
    ax.yaxis.set_major_locator(MaxNLocator(3));ax.zaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis='both',labelsize=9)
    return fig,ax

def section_plot(section,path,title):
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for key,label,c in [('lower_segments_program','下表面',COLORS[0]),('upper_segments_program','上表面',COLORS[1])]:
        if key in section:
            s=np.asarray(section[key]);segments=s[:,:,[1,2]].copy();segments[:,:,1]*=-1
            ax.add_collection(LineCollection(segments,colors=c,linewidths=1.7,label=label))
    ax.autoscale();ax.set_xlabel(r'$y_{\mathrm{p}}$ / m');ax.set_ylabel(r'$z_{\mathrm{b}}=-z_{\mathrm{p}}$ / m');ax.set_title(title);ax.grid(alpha=.22);ax.legend();ax.set_aspect('equal',adjustable='datalim')
    note='原生三角面与截平面相交；各分支独立保留。'
    if section['status']!='OK':note+=' 此截面为多值曲线。'
    fig.subplots_adjust(bottom=.17);save(fig,path,note)

def make_case(r):
    idx=r['design_index'];case=Path(r['case_dir']);out=OUT/'figures'/f'design_{idx:02d}';out.mkdir(parents=True,exist_ok=True)
    m=loadmat(case/'native_geometry.mat');grids=[np.stack([m[p+a] for a in 'xyz'],-1) for p in ['wav_','wav_up_']]
    title=f'Pareto 构型 {idx:02d}（原始行 {r["original_row_index"]}）'
    fig,ax=axes3d(title+'：原生上下表面',grids)
    for g,label,c in zip(grids,['下表面','上表面'],COLORS):
        t,_=triangulate(g);ax.add_collection3d(Poly3DCollection(t,facecolor=c,edgecolor=c,alpha=.19,linewidths=.13))
    ax.legend(handles=[Line2D([0],[0],color=COLORS[i],label=label,lw=2) for i,label in enumerate(['下表面','上表面'])],loc='upper left')
    save(fig,out/'01_native_surfaces.png','程序坐标；保留原生 50×50 网格；单位沿用 V5 的 1 m/程序单位假设。')
    sections=json.loads((case/'sections.json').read_text(encoding='utf-8'))['sections']
    section_plot(sections[5],out/'02_tail_section.png',title+'：尾截面 '+r'$x/L=1$')
    def severity(s):return s.get('min_thickness',s.get('minimum_branch_gap')) if s.get('min_thickness',s.get('minimum_branch_gap')) is not None else np.inf
    worst=min(sections[:6],key=severity)
    section_label='最大负间隔采样截面' if severity(worst)<-r['length_tolerance'] else '最小间隔采样截面'
    section_plot(worst,out/'03_worst_section.png',title+'：'+section_label+' '+rf'$x/L={worst["x"]/r["length_scale"]:.2f}$')
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    if worst['status']=='OK':
        ax.plot(worst['y'],worst['thickness'],linewidth=1.6,label='同坐标厚度');ax.set_ylabel(r'$t=z_{\mathrm{l,p}}-z_{\mathrm{u,p}}$ / m')
        note='仅在共同 y 支持域比较；断点取上下截面实际断点并集。'
    else:
        seg=np.asarray(worst['branch_vertical_gaps']);ax.add_collection(LineCollection(seg,colors=COLORS[0],linewidths=1.5,label='分支竖向间隔'));ax.autoscale();ax.set_ylabel(r'$\Delta z_{\mathrm{branch}}$ / m')
        note='多值截面没有唯一厚度；此图只展示各分支间隔，不等同于蒙皮厚度。'
    ax.axhline(0,color='0.35',linestyle='--',linewidth=1);ax.set_xlabel(r'$y_{\mathrm{p}}$ / m');ax.set_title(title+f'：截面间隔 '+rf'$x/L={worst["x"]/r["length_scale"]:.2f}$');ax.grid(alpha=.22);ax.legend();fig.subplots_adjust(bottom=.17);save(fig,out/'04_thickness_or_branch_gap.png',note)
    inter=json.loads((case/'intersections.json').read_text(encoding='utf-8'))
    fig,ax=axes3d(title+'：实际三角面相交位置',grids)
    # All native grid edges retained, made faint to expose computed lines.
    for g in grids:
        ax.add_collection3d(Line3DCollection(list(g)+list(g.transpose(1,0,2)),colors='0.75',linewidths=.25,alpha=.38))
    handles=[]
    for k,label,c in [('upper_lower','上下表面交叉',COLORS[0]),('lower_self','下表面自交',COLORS[1]),('upper_self','上表面自交',COLORS[2])]:
        records=inter[k]['records'];lines=[];points=[]
        for rec in records:
            p=np.asarray(rec['points_program'])
            if len(p)>=2:lines.append(p)
            else:points.extend(p)
        if lines:ax.add_collection3d(Line3DCollection(lines,colors=c,linewidths=2.0))
        if points:
            p=np.asarray(points);ax.scatter(p[:,0],p[:,1],p[:,2],s=8,color=c)
        if records:handles.append(Line2D([0],[0],color=c,lw=2,label=f'{label}（{len(records)} 对）'))
    if handles:ax.legend(handles=handles,loc='upper left')
    else:ax.text2D(.05,.85,'未检出非允许交叉',transform=ax.transAxes)
    save(fig,out/'05_actual_intersections.png','相交线由数值窄相位求得；公共前缘接触已排除；计数为三角面片对。')
    return [str(p) for p in sorted(out.glob('*.png'))]

def make_parameters(results):
    out=OUT/'figures';out.mkdir(exist_ok=True);paths=[]
    classes=['VALID','INVALID','AMBIGUOUS','GENERATION_FAILED'];labels=['合法','非法','待判定','生成失败'];markers=['o','x','s','^']
    for param in PARAMS:
        fig,ax=plt.subplots(figsize=(7.2,4.6))
        for c,lab,mark in zip(classes,labels,markers):
            rows=[r for r in results if r['classification']==c]
            if rows:ax.scatter([r['parameters'][param] for r in rows],[classes.index(c)]*len(rows),label=f'{lab}（n={len(rows)}）',marker=mark,s=33)
        ax.set_yticks(range(4),labels);ax.set_ylim(-.5,3.5);ax.set_xlabel((rf'${param[0]}_{param[1]}$ / (°)' if param in ['a1','a2','b1','b2'] else (r'$d_1$ / 1' if param=='d1' else r'$a_x$ / 1')));ax.set_ylabel('几何合法性分类');ax.set_title(f'Pareto 设计参数 {param} 与几何合法性');ax.grid(axis='x',alpha=.22);ax.legend(loc='best');fig.subplots_adjust(bottom=.18)
        p=out/f'parameter_{param}_vs_validity.png';save(fig,p,'同类别点可能重叠；未添加随机抖动。相关关系不代表因果。');paths.append(str(p))
    return paths

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.case:
        r=json.loads((OUT/'cases'/f'design_{args.case:02d}'/'audit.json').read_text(encoding='utf-8'));print(make_case(r));return
    results=json.loads((OUT/'pareto_geometry_audit.json').read_text(encoding='utf-8'))['results']
    candidate_file=OUT/'V5_NEXT_VALID_GEOMETRY_CANDIDATES.json'
    chosen=[c['design_index'] for c in json.loads(candidate_file.read_text(encoding='utf-8'))['candidates']] if candidate_file.exists() else [r['design_index'] for r in results if r['classification']=='VALID'][:3]
    selected=[r for r in results if r['classification']=='INVALID']+[r for r in results if r['design_index'] in chosen]+[r for r in results if r['classification']=='AMBIGUOUS'][:1]
    paths=[]
    for r in sorted(selected,key=lambda r:r['design_index']):
        existing=list((OUT/'figures'/f'design_{r["design_index"]:02d}').glob('*.png'))
        paths.extend(map(str,existing) if args.resume and len(existing)==5 else make_case(r));print(f'FIGURES design={r["design_index"]}',flush=True)
    paths+=make_parameters(results);dump(OUT/'figure_manifest.json',{'dpi':300,'matplotlib':matplotlib.__version__,'colour_policy':'Matplotlib default cycle; grey contextual mesh only','selection':[r['design_index'] for r in selected],'files':paths,'count':len(paths)})
if __name__=='__main__':main()
