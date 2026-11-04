"""Evidence-based Chinese delivery report; no physical computations."""
import json,csv
from pathlib import Path
from prepare import ROOT,OUT,TOOLS,dump,sha

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def link(name,label=None):return f'[{label or name}]({(OUT/name).as_posix()})'
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def main():
    s=read('analysis_summary.json');rs=read('pareto_geometry_audit.json')['results'];m=read('source_manifest.json');b=read('geometry_audit_budget.json');c=read('V5_NEXT_VALID_GEOMETRY_CANDIDATES.json')['candidates'];verification=read('independent_intersection_verification.json');figs=read('figure_manifest.json');old=s['old45'];oldv=next(v for v in verification['results'] if v['design_index']==45)
    invalid=[r for r in rs if r['classification']=='INVALID'];valid=[r for r in rs if r['classification']=='VALID'];checks=verification['results'];badchecks=[r for r in checks if r['record_count']]
    ranges=list(csv.DictReader((OUT/'observed_parameter_ranges.csv').open(encoding='utf-8-sig')))
    range_rows=[]
    for p in ['a1','a2','b1','b2','d1','a_x']:
        lo=next(r for r in ranges if r['classification']=='VALID' and r['parameter']==p);hi=next(r for r in ranges if r['classification']=='INVALID' and r['parameter']==p)
        range_rows.append([p,f'{float(lo["min"]):.6f}–{float(lo["max"]):.6f}',f'{float(hi["min"]):.6f}–{float(hi["max"]):.6f}'])
    correlations=[r for r in s['correlations'] if r['target']=='valid_binary']
    fields=mdtable(['字段','shape','dtype'],[[k,str(v['shape']),v['dtype']] for k,v in m['fields'].items()])
    params=mdtable(['列','参数','实际最小值','实际最大值','原优化边界'],[[p['column_1based'],p['name'],f'{p["min"]:.12g}',f'{p["max"]:.12g}',str(p['source_bounds'])] for p in m['parameter_columns']])
    candidate_table=mdtable(['排序号 / MAT原始行','代表性','d1','距原45归一化距离','原存 L/D / Vol_eff'],[[f'{x["design_index"]} / {x["original_row_index"]}',x['reason'],f'{x["parameters"]["d1"]:.9f}',f'{x["normalized_distance_from_old45"]:.6f}',f'{x["stored_L_D"]:.6f} / {x["stored_Vol_eff"]:.6f}'] for x in c])
    class_table=mdtable(['分类','数量','占比'],[[k,v,f'{v/52:.1%}'] for k,v in s['counts'].items()])
    corr_table=mdtable(['量','与 VALID=1 的 Spearman ρ'],[[r['feature'],f'{r["spearman_rho"]:+.3f}' if r['spearman_rho'] is not None else '无变化，无法估计'] for r in correlations])
    old_section=read('cases/design_45/sections.json')['sections'][5]
    blocks=[]
    blocks.append(f'''# V5 教师 Pareto 原生构型几何合法性审计

本次完成 **52/52 个构型的几何生成及独立网格审计**。结果为 **26 个 VALID、26 个 INVALID、0 个 AMBIGUOUS、0 个 GENERATION_FAILED**，属于 **CASE B：合法与非法离散样本混合**。当前 V5 默认第 45 号仍然非法；存在可供后续固定表面验证的合法候选，不支持“教师 52 个构型全部非法”或“DQN / 360 km 任务不可行”的推论。

范围：仅调用冻结 `build_geometry_core` 生成原生上下表面，再对保存网格做几何诊断。没有调用气动评价、OCP、DQN、180 s 任务、GA/GPR/NHVI；没有修改几何公式、旧代码、旧结果或旧预算。几何单位延用当前 V5 的 **1 m/程序单位研究假设**，不是新增的尺寸标定结论。

**A. 全部 52 个设计的分类**

{class_table}

VALID 排序号：{', '.join(map(str,s['valid_designs']))}。

INVALID 排序号：{', '.join(map(str,s['invalid_designs']))}。

排序号采用 `pareto_front[:,0]` 稳定升序，MAT 原始行号为 1 起算。每行六参数、完整十输入、状态、哈希、厚度支持情况、交叉计数与原因见 {link('pareto_geometry_audit.csv')} 和 {link('pareto_geometry_audit.json')}。CSV 中空白厚度不是零：有多值截面时，全六截面的唯一厚度统计不存在，JSON 保留可定义截面的统计和分支间隔。

**B. 原第 45 号是否确实非法？**

是。它对应 MAT 原始第 **22** 行（0 起算 21），六参数为 `{json.dumps(old['parameters'])}`。重新生成的六个坐标数组与旧 `geometry_002_e96314d4/geometry.mat` **逐元素完全相同**；重算旧 V5 哈希为 `d34175b6f33acb8aa9f2dcc2ad58f968a1e3dea0bf28f3ef9b6b079044629d08`。新审计使用另一套包含十输入的 mesh hash schema，两者不能直接混比。证据见 {link('old45_identity_verification.json')}。

- 非公共前缘的上下表面交叉：**{old['surface_intersection_count']} 对三角面**；其中独立验证中 **{oldv['proper_interior_crossing_pairs']} 对**具有两三角形内部的交叉线段。这是面片对计数，不是 198 条独立物理裂缝。
- 相交区：`x/L = 0.658042–1.000000`，`y_program = 0.897533–0.949751 m`，延伸到尾截面；不是公共前缘接触。
- 尾截面最小厚度 **{old['tail_min_thickness']:.12f} m**，负厚度断点 {old_section['negative_count']}/{old_section['breakpoint_count']}；最差点 `{old_section['worst_location']}`。断点比例依赖网格，另存了负间隔横向长度比例，不能把断点比例称作负体积比例。
- 自交计数 0，program→body 刚性变换不改变交叉，换另一条原生网格对角线仍检出 {old['opposite_diagonal_sensitivity']['count']} 对。

所以旧负厚度结论在此构型上得到真正三角面交叉的支持，不是简单同 index 相减、上下命名颠倒或坐标翻转造成的假象。

**C. 个别构型、参数区域，还是普遍生成/坐标问题？**

它不是个别点，也不是全部构型必然失败：正好一半非法。26 个非法样本的交叉都位于后部外翼区域，总体包络 `x/L={s['intersection_x_over_L_extent'][0]:.6f}–1`、`y_program={s['intersection_y_program_extent'][0]:.6f}–{s['intersection_y_program_extent'][1]:.6f} m`，每个非法构型均延伸到尾部。该范围包含中后段，不能只称为尾截面孤立数值误差。

所有 52 个上表面均通过单值 XY 图面证书；42 个下表面也通过。下表面投影折返的排序号为 {', '.join(map(str,s['lower_xy_fold_designs']))}，共 10 个；其中预定六截面出现多值的有 {52-s['all_six_sections_single_valued_count']} 个。投影折返本身不等于曲面自交：本次实际自交检测总数为 0。对于这些构型，禁止排序折返 y 节点后伪造唯一厚度；分类依据是上下表面的实际交叉。

六参数的观测支持范围如下。这些只是已扫描点的 min/max，**不是允许填满的几何合法 box**。

{mdtable(['参数','VALID 样本范围','INVALID 样本范围'],range_rows)}

{corr_table}

样本内 d1 与合法性负相关最明显；b1、b2 也有相关，而 a_x 单独的秩相关接近零。d1 的合法与非法范围仍有重叠（约 0.1124–0.1188），因此不能把某个 d1 阈值当通用边界。a/b 差值的单变量相关很弱，也不能据此排除角度组合效应。原存 L/D 越大倾向更多合法点、原存 Vol_eff 相反，但该 Pareto 样本存在优化筛选及参数相关，**这些都不是因果结论**。p 值仅作为未校正的描述量保存在表中，不作为独立抽样显著性证据。

靠近原参数边界既有合法也有非法点；全体最小边界距离与合法性的相关也不能解释为“越靠边越安全”。逐参数边界距离、原目标排名见 {link('parameter_vs_validity.csv')}；组合量、交叉计数和间隔的相关见 {link('correlation_diagnostics.csv')}。

**D. 上下表面的真实代码语义**

`wav_*` 是基于激波/流线构造的下表面：在 `rat` 旋转框架计算流线，再用 `rat'` 回到程序框架，按原程序的逐流线 x 插值形成 50×50 网格。`wav_up_*` 是同一前缘出发、沿程序来流方向延伸的直纹上表面：`y=y_LE`，`z=z_LE+(x-x_LE)*tan(AOA_prog)`；这里 `AOA_prog=-4°`。它没有在有效路径中用额外实体蒙皮/Bezier 曲面替代。

原 `generatewaverider` 的显示使用 `-wav_z`，当前 V5 明确采用 `body=diag(1,-1,-1)*program`（det=+1）。对单值且共同覆盖的区域，正厚度为 `z_upper_body-z_lower_body = z_lower_program-z_upper_program`。单值性必须逐设计核验，不能预设全 52 个都成立。多值处应使用实际三角面交叉、必要时分支距离；本报告的分支竖向间隔图不冒充蒙皮厚度。

两表面原生前缘最大连接误差 {s['maximum_leading_edge_gap']:.3e} m；镜像对称边 y 误差不超过 {s['maximum_symmetry_y_error']:.3e} m。原生数据仅给出半体蒙皮，y=0 各自镜像，不要求上下蒙皮在整条对称面重合。尾部没有提供基底封闭面，全部明确标为 **OPEN_BASE**；未人工补盖。上下表面均有源代码刻意折叠的尖点行，各产生 49 个零面积三角形，总计 98/构型，原样记录且仅在该明确边界上允许，不据此判 INVALID。其余内部退化、NaN/Inf/complex 均未发现；邻接法向翻转计数为 0。

存在一个独立的源码坐标一致性问题：`create_shock_functions` 返回 `rat` 旋转后的法向/导数，而几何核心部分曲率计算仍使用 `cross(F_val, Vector_V)` 中的程序来流向量；旋转后的来流应为 `rat*Vector_V=[1,0,0]`。这是可定位的坐标框架混用，**不是已经验证的交叉唯一原因**。本次没有改 AOA、替换向量或生成对照几何，不能越过证据宣称修哪一行即可解决。

**E. 原教师优化是否有充分几何合法性约束？**

**ORIGINAL_OPTIMIZER_HAS_NO_GEOMETRY_VALIDITY_GATE**。

实际追踪链为 `main_optimize → surrogate_assisted_ga_wave_rider_6 → evaluatePoint → waveRiderEval`。GA 调用提供六参数边界，没有几何非线性约束；真实 `evaluatePoint` 只检查两个返回目标不是 NaN。`waveRiderEval` 只对 `wav_x` 做部分 NaN/Inf/实数检查，不检查全坐标、最小厚度、上下交叉或自交。`volume_simianti` 使用 `abs(dot(cross(...),...)/6)`，上层累加正体积项：局部反向、交叉或重叠并不会必然产生负总体积。`vol_eff_sim` 的截面椭圆估算也不是碰撞门控。因此有限的 L/D、Vol_eff 或正体积不构成几何合法证明。

完整可定位的行号、SHA256 和源码片段在 {link('optimizer_and_geometry_source_trace.json')}，原优化器文本证据保存在 `source_evidence/`，从未运行这些优化/气动入口。

确定的失败链是：**当前直纹上表面与部分下表面流线不兼容 → 内部交叉 → 原优化缺少门控仍可保留这些点**。尚不能唯一归咎于上表面公式、下表面流线或设计 AOA 实现；存在 26 个同链合法点，反驳“整套方法在所有参数处都必然非法”。

**F. 三个后续固定表面验证候选**

{candidate_table}

候选文件：{link('V5_NEXT_VALID_GEOMETRY_CANDIDATES.json')}。完整十输入和原生 MAT 路径已经保存；没有执行候选气动验证。第 42 号与原选点的 d1 相差约 0.0403，且其他参数也不同，不能直接把它当作原单 d1 shape family 的一个已验证 xi。第 16 号覆盖另一参数区域。选择依据是几何和代表性，不是气动最优性认证。

**G. 还缺少什么教师信息？**

合法离散候选已经足够三个；缺的不是再猜一个点，而是确认物理蒙皮与连续 family 的定义。请确认：当前直纹 `wav_up_*` 是否就是实际外蒙皮；原文件中被注释的 `Generate_Upper_Surface_Bezier` 是否才是最终采用上表面、其有效版本/参数是什么；程序 z、AOA 符号与旋转后向量应怎样配套；基底/对称连接和真实长度单位如何定义。对于源码框架混用，请给出正式约定或已验证样例，不能由审计自行改公式。

详细最小问题清单见 {link('TEACHER_GEOMETRY_DEFINITION_BLOCKER.md')}。这不是“全部 52 个都失败”的声明，也不要求为了本次扫描补造物理数据。

**H. 当前 V5 是否可进入 TEACHER_AERO_NUMERICALLY_CHECKED？**

**BLOCKED**。

这是对**当前冻结的 V5 默认第 45 号及其现有 shape family**的结论：默认点真实交叉，不能继续该配置的正式气动/快模型/DQN 路线。本次没有修改默认构型或建立新的合法连续支持域。第 42、34、16 号为下一次固定表面验证提供了明确候选；未来可仅针对这些已验证离散几何安排数值检查，但本报告既没有执行该阶段，也没有证明候选之间、原 xi 区间内或任意填充 box 内的中间形状合法。READY 不能由扫描工具运行成功或存在有限旧目标值自动推出。
''')
    blocks.append(f'''**数据来源与映射**

实际唯一匹配 MAT（搜索范围：`D:/mat` 与 `D:/final-DQN-v5`）：`{m['source_path']}`。

SHA256：`{m['source_sha256']}`。

{fields}

{params}

固定参数从真实 `evaluatePoint` 解析并与 V5 design_spec 交叉核对：Ma={m['fixed_parameters']['Ma']:g}，c1=c2={m['fixed_parameters']['c1']:.4f}，AOA_program={m['fixed_parameters']['AOA']:g}°。十输入顺序 `[Ma,a1,a2,b1,b2,c1,c2,d1,AOA,a_x]`，52 行全保存在 `jobs.json` 和各 case 的 `parameters.json`。没有使用 48×5 数据。源清单见 {link('source_manifest.json')}。

**预先冻结的判据、检验与计算预算**

首次新几何调用之前冻结 `classification_rules.json`，SHA256 `{m['rules_sha256']}`。长度容差 `max(128*eps64*L, 1e-9*L)`，L=5 时为 5×10⁻⁹ m；面积容差 `1e-14*L²`=2.5×10⁻¹³ m²。窄相位平面符号使用机器舍入量级，避免将几何容差错误用于近共面点吸附。规则未根据最终分类数量调整；诊断工具在全面扫描前修正了近共面接触误判及多值截面识别，并加入回归测试。

上下曲面和各自曲面的 AABB 层级筛选后，使用三角形平面截线与区间交叠，共面时使用多边形裁剪。仅排除确实处于公共前缘的接触，或能证明仅落在共享 native 顶点/边上的自表面邻接接触；测试覆盖共享顶点仍有内部穿透、共面折回等反例。两种对角线剖分的上下交叉存在性在 52/52 个设计一致。每个相交记录及其 native 面片索引全部保存，反射后的另一半体由 y≥0、y=0 仅在对称边及镜像关系验证，无另行构造修复表面。

独立线段—三角形 3×3 求解法复核了全部 {sum(v['record_count'] for v in checks)} 个记录，未确认记录数为 {sum(v['unconfirmed_count'] for v in checks)}；端点最大差 {max(v['maximum_endpoint_difference_m'] for v in checks):.3e} m。每个非法设计都有至少 {min(v['proper_interior_crossing_pairs'] for v in badchecks)} 个双三角形内部交叉段。program→body 刚性变换测试全部通过。上述是浮点、原生离散网格上的结果，不是对未采样连续解析曲面或加工误差的数学完备证明。

预定截面 x/L=0.1、0.25、0.5、0.75、0.9、1.0；实际三角面与 x 平面求交，使用共同 y 支持域断点并集，另加真实相交位置的中位 x 截面。每截面的零交叉、首次负间隔、最差位置、断点比例和横向长度比例保存在各 case 的 `sections.json`。多值截面保留原线段与 branch gap，唯一厚度字段空置；不会以同 index 相减代替。

15 项独立诊断测试通过，包括随机三角形两种算法互验 300 对、AABB 对暴力枚举交点完整性检查、投影折回、多值截面、共面重叠、真实内部交叉与允许边界接触。测试输出见 {link('unit_test_output.txt')}。

几何生成前三个实际耗时：{', '.join(f'{v:.3f}' for v in b['pilot_actual_elapsed_s'])} s；52 次几何调用合计 {b['geometry_elapsed_s_total']:.3f} s，两次 MATLAB 会话总 wall time {sum(x['wall_s'] for x in b['sessions']):.3f} s。单构型硬超时 120 s，连续 3 次超时停止；本次无超时、0 worker。网格主审计合计 {s['audit_elapsed_s_total']:.3f} s，独立校验/绘图另计，非新增生成调用。气动/OCP/训练调用均为 0。详见 {link('geometry_audit_budget.json')}。

**图片及交付目录**

共 **{figs['count']} 张 PNG，300 dpi**：26 个 INVALID 各 5 张，候选 42/34/16 各 5 张，加六张参数—合法性散点图。没有 AMBIGUOUS 样本，因此不伪造该类图片。所有图使用 matplotlib 默认颜色序列、中文图题/图例、变量与单位分明；每图单独输出。对于多值截面，第四张明确标为“分支竖向间隔”，没有伪造唯一厚度曲线。

每个重点构型：`01_native_surfaces.png`、`02_tail_section.png`、`03_worst_section.png`、`04_thickness_or_branch_gap.png`、`05_actual_intersections.png`。第三张的“最严重”指**六个预定采样截面**中最大负间隔，不宣称已经求连续曲面的全局极小点；若不存在交叉则显示合法截面的间隔。清单见 {link('figure_manifest.json')}。

- 总结果：{OUT.as_posix()}/
- 原生网格及逐设计证据：{(OUT/'cases').as_posix()}/design_01…design_52/
- 图片：{(OUT/'figures').as_posix()}/
- 独立工具和测试：{TOOLS.as_posix()}/
- 源证据：{(OUT/'source_evidence').as_posix()}/

保护文件前后共 **{s['protected_files']['protected_file_count_before']} 个**，SHA256 全部一致。`D:/final-DQN-v4` 实际不存在，已保护真实目录 `D:/final-DQN-physics-v4`，同时保护 D:/final-DQN、V3、V5 旧文件与 D:/mat。只排除 .git、环境/缓存等瞬态目录，以及本次指定的新工具/输出目录。详见 {link('protected_file_verification.json')} 与前后完整哈希表。

复核已有结果可运行工具目录中的 `test_mesh_checks.py`、`verify_intersections.py`，不需要新几何调用。`generate.py` 已按每 case 的完成标记跳过已有生成结果；原始 MATLAB 源码与 V5 bridge、reward、constraint、Generator、训练入口均未修改。
''')
    (OUT/'V5_PARETO_GEOMETRY_AUDIT_ZH.md').write_text('\n\n'.join(blocks),encoding='utf-8')
    blocker='''# 教师几何定义与当前 V5 默认构型阻断项

状态：**BLOCKED（当前 V5 默认第45号及原shape family）**。

本次 52 个原生构型中 26 个合法、26 个非法。不是“全部构型失败”；三个合法候选为排序号 42、34、16。它们只是离散固定几何，不能直接覆盖原默认 family 或证明中间变构路径。

已证实：第45号的尾部最小厚度为 −0.040794908839 m，198 对三角面实际相交；坐标刚性变换与两种网格对角线剖分均不能消除。上下表面共用前缘，交叉位于后部大 y 区域。原优化器缺少交叉/厚度/自交门控，正体积或有限目标不能过滤这些点。

尚未被本次扫描唯一归因：究竟应改上蒙皮定义、下表面流线、AOA/曲率框架，还是需要原项目另一个已验证的构型生成版本。本任务没有授权修改公式来做因果对照，所以以下问题需源定义证据，而非由程序员猜测。

最小教师确认问题：

1. **哪一个是实际上蒙皮？** 有效代码从前缘沿来流直线延伸 `wav_up_z=z_LE+(x-x_LE)*tan(AOA)`。这是物理外蒙皮，还是仅用于体积/受力评价的数学面？若是后者，请给出真实蒙皮的文件、版本、参数和一组已确认原生坐标。
2. **被注释的上表面函数是否属于正式定义？** `generatewaverider` 里 `Generate_Upper_Surface_Bezier(...)` 及 z 反号整段未生效。若正式项目用这一条链，请给出实际启用版本、Bezier/CST 参数和前缘/尾部接合约定；不能直接取消注释冒充修复。
3. **旋转前后各向量如何定义？** `create_shock_functions` 用 rat 旋转后的坐标计算 F、ru、rv；调用侧又以原程序 `Vector_V` 参与叉乘。请确认应使用哪个坐标框架。AOA=-4° 的物理正方向、program z 正方向、`rat` / `rat'` 用法需一致。代码框架混用可定位，但尚不能证明是全部穿透的唯一成因。
4. **最低边界与尺度定义是什么？** 当前只提供半体上下蒙皮和共同前缘；对称面镜像，基底 OPEN_BASE。请确认基底是否另有几何、尾部/侧边期望连接、1 程序单位是否真为 1 m，以及是否存在非零最小厚度要求。没有资料时保持现有声明，不补盖、不裁剪、不取绝对值。

下一阶段几何范围只能来自 26 个已验证样本及未来逐点验证的新样本；参数 min/max 只是统计摘要，不是合法盒子。若另行选择候选做固定表面气动数值检查，应保持几何固定并继续记录身份、坐标/力系约定和独立一致性验证；本次未进行该检查，也未放行 DQN。
'''
    (OUT/'TEACHER_GEOMETRY_DEFINITION_BLOCKER.md').write_text(blocker,encoding='utf-8')
    print('Chinese report and blocker written.')
if __name__=='__main__':main()
