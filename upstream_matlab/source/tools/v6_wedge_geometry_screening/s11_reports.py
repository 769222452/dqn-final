"""Stage 11: REPORT_ZH.md and CFD_HANDOFF_README_ZH.md (numbers read from the result tables)."""
import json
from common import OUT, SREF, read_csv

f = json.loads((OUT / "work/report_facts.json").read_text(encoding="utf-8")); acc = json.loads((OUT / "acceptance_status.json").read_text(encoding="utf-8"))
top = read_csv(OUT / "screening/top5_by_condition.csv"); sel = read_csv(OUT / "screening/selected_unique_geometries.csv"); gm = {r["geometry_id"]: r for r in read_csv(OUT / "geometry/geometry_manifest.csv")}
short = lambda g: g.replace("S2_", "").replace("V2M_", "V2M_").replace("_", "·", 1) if g.startswith("S2_") else g
conds = sorted({r["condition_id"] for r in top})
tbl = ["| 工况 (M, α°) | TOP1 | TOP2 | TOP3 | TOP4 | TOP5 | TOP1 L/D |", "|---|---|---|---|---|---|---|"]
for c in conds:
    r = sorted([x for x in top if x["condition_id"] == c], key=lambda x: int(x["rank"]))
    tbl.append(f"| {float(r[0]['M_flight']):.1f}, {float(r[0]['alpha_flight_deg']):.0f} | " + " | ".join(short(x["geometry_id"]) for x in r) + f" | {float(r[0]['L_over_D']):.2f} |")
n = f["nearby_pct_change_median_and_maxabs"]
cfd = [s for s in sel if int(s["n_top5_conditions"]) >= 22]
rep = f"""# V6 切楔模型构型筛选与 UG/NX 交付报告

> 所有气动数值均来自 **TANGENT_WEDGE_SHOCK_EXPANSION**（现有 `offdesign_aero_v7`，P2 条带激波—膨胀，纯压力项，无摩擦、无底阻）。它们是低阶模型预测，**未经 CFD 验证**，不能当作真实气动最优。
> 本任务在 `D:\\v6-mission-morph-core` 内完成；`D:\\v6-mission-morph` 未被修改（175,365 个文件 SHA-256 前后一致）。未运行 CFD、DQN 训练、OCP 或 1200 秒任务。

## 1. 验收状态

| 项目 | 状态 |
|---|---|
| TEACHER_GEOMETRY_GENERATION | **PARTIAL**（复用 453 个已有教师几何；本阶段新生成 0 个） |
| GEOMETRY_VALIDITY_GATE | PASS（453/453 审计证据为 VALID；读取证据，未重跑审计） |
| TANGENT_WEDGE_AERO | PASS（26/26 解析自检；28,154 次调用 0 个无效） |
| TOP5_SCREENING | PASS（49/49 个工况各有 5 个合格构型） |
| NEARBY_OFFDESIGN_EVALUATION | PASS |
| CROSS_MACH_COMPARISON | PASS |
| CAD_STEP_EXPORT | PASS（31/31 个 STEP 实体）；Parasolid、NX .prt：**BLOCKED** |
| CAD_REIMPORT_VALIDATION | PASS（仅 OpenCascade 重新导入，不是 NX） |
| NX_BAT_PREPARED | YES |
| NX_IMPORT_TESTED | **NO**（本机未安装 NX，状态 `NX_BATCH_NOT_EXECUTED`） |
| CFD_COMPLETED / DQN_TRAINED | NO / NO |

## 2. 回答 11 个问题

1. **已有合法构型**：候选清单 457 行，唯一原生哈希 453 个（"454"为旧说法），覆盖 46 个设计 Mach；453 个审计分类均为 VALID，全部参与筛选。其中 299 个满足冻结的 EXPLORATORY 体积/厚度筛选。
2. **新生成合法构型**：**0 个**。已有池覆盖设计 Mach 4–15，本阶段未启动教师 MATLAB 新生成。设计 Mach 9–14 只有少量构型，是覆盖薄弱处；若要补充需另行立项。
3. **49 个工况的 Top5**：见第 3 节完整表；明细在 `screening/top5_by_condition.csv`。
4. **是否集中于某些设计 Mach**：是。Top5 共 245 个席位，设计 Mach 分布 {f['top5_design_mach_bins']}；**没有一个设计 Mach<8 的构型进入 Top5**，Top1 全部来自设计 Mach≥12。出现最多的是 S2_MD13.0774（44 次）、S2_MD14.004、S2_MD15_029eb6f5fd、S2_MD8_b38c94c34e（各 42 次）。Top1 只涉及 {f['top1_unique']} 个几何，S2_MD14.004_4b1ceeb42b 在 27 个工况拿第一，S2_MD15_029eb6f5fd 在 15 个工况拿第一。Top1 相对 Top5 的 L/D 优势中位数仅约 {f['top1_over_top5_spread_median_pct']:.1f}%，Top5 内部差距不大。Top1 的 L/D 范围 {f['top1_LD_range'][0]:.1f}–{f['top1_LD_range'][1]:.1f}。
   注意：α=4° 的工况中 Top1 的 CL 只略高于升力下限 0.02615，下限在这里是起约束作用的。
5. **附近 Mach 与迎角的变化**（中位数 / 最大绝对值，相对中心工况，共 1715 行真实调用）：
   - M−0.25：CL {n['M-']['dCL_med']:+.1f}% / {n['M-']['dCL_maxabs']:.1f}%，CD {n['M-']['dCD_med']:+.1f}% / {n['M-']['dCD_maxabs']:.1f}%，L/D 仅 {n['M-']['dLD_med']:+.2f}%。
   - M+0.25：CL {n['M+']['dCL_med']:+.1f}%，CD {n['M+']['dCD_med']:+.1f}%，L/D {n['M+']['dLD_med']:+.2f}%。
   - α−1°：CL {n['ALPHA-']['dCL_med']:+.1f}%（最大 {n['ALPHA-']['dCL_maxabs']:.0f}%），CD {n['ALPHA-']['dCD_med']:+.1f}%，L/D {n['ALPHA-']['dLD_med']:+.1f}%。
   - α+1°：CL {n['ALPHA+']['dCL_med']:+.1f}%（最大 {n['ALPHA+']['dCL_maxabs']:.0f}%），CD {n['ALPHA+']['dCD_med']:+.1f}%（最大 {n['ALPHA+']['dCD_maxabs']:.0f}%），L/D {n['ALPHA+']['dLD_med']:+.1f}%。
   结论：迎角对 CL/CD 的影响远大于 ±0.25 Mach。这些 delta 包含飞行 Mach 本身的变化，不是设计 Mach 失配损失。
6. **不同设计 Mach 构型是否有优势交叉**：在 M5–8 的共同评价中几乎没有。对 31 个入选构型在 α=5/7/9° 下共 1395 个两两比较，仅 6 对的优劣顺序随飞行 Mach 改变。在检查的 α=4° 和 7° 上，S2_MD14.004_4b1ceeb42b 在各 Mach 都领先；α=10° 由 S2_MD15_029eb6f5fd 领先。当前模型没有显示明显的"设计 Mach 与飞行 Mach 匹配"优势：Top5 的设计 Mach（8–15）都高于飞行 Mach（5–8）。
7. **固定迎角 L/D 与等升力 CD 排名是否一致**：不完全一致。84 个比较中 Spearman 相关系数中位数 0.85、最小 0.39，第一名相同的比例 {f['ranking_consistency']['same_top1_fraction']*100:.0f}%。按固定迎角分：α=4°、6° 约 90% 一致，α=8° 约 62%，α=10° 为 0%（高迎角下各构型 CL 差别大，固定迎角比较等于比较不同升力）。651 个等升力查询中 630 个成功，21 个目标 CL 超出 α=2–12° 的计算范围。
8. **最值得做 CFD 的构型**（按 Top5 出现次数和角色）：S2_MD13.0774_e7af639292、S2_MD14.004_4b1ceeb42b、S2_MD15_029eb6f5fd、S2_MD8_b38c94c34e、S2_MD11.5_bf14cea04c、S2_MD9.75_4b3a631248，外加基准 G42 和多样性对照 V2_MD8_005。G42 从未进入 Top5（在合格者中排第 6–31 名，有 {f['G42_flagged_not_eligible_conditions']} 个工况低于升力下限）。多样性对照 V2_MD8_005 是按冻结规则（最大最小距离）选出的，L/D 很低（约 3.4），它是几何多样性对照，不是性能候选。
9. **文件是否真实存在、是否通过导入检查**：31 个 STEP 全部真实存在（共 34 MB）。每个都由 OpenCascade 内核生成 B-rep 实体、写成毫米单位，再重新读入检查：实体有效，自由边界 0，体积与原生网格相差最大 0.71%，原生节点到曲面的偏差最大 4.3e-10 mm。**Parasolid（.x_t）和 NX .prt 均未生成**（没有 Parasolid 内核，NX 未安装），`parasolid_path`、`prt_path` 列为空。
10. **.bat 状态**：`run_ug_import.bat`、`run_ug_import_all.bat` 和 NXOpen 日志脚本已准备。我在隔离副本里用桩程序验证了退出码逻辑（用法错误 2、不在清单 2、STEP 缺失 4、找不到 NX 3、桩 NX 未确认导入 5）。**没有任何功能经过真实 NX 验证**，NXOpen 脚本按标准日志写法编写，需要在有许可证的 NX 上首次试跑。
11. **源文件与旧结果**：保持不变。源工程 175,365 个文件重新逐个计算 SHA-256，新增、删除、内容变化均为 0。

## 3. 49 个工况的 Top5

{chr(10).join(tbl)}

## 4. 方法与冻结规则

- 几何：每个构型固定，评价点只改变飞行 Mach 和迎角；没有把设计 Mach 15 的构型按飞行 Mach 重新生成，也没有用迎角替换设计程序迎角。
- 升力下限 CL≥0.02615，来自 `results/fixed_geometry_baselines/comparison.json` 中可行固定构型 CL 的 5% 分位；体积/厚度使用项目已有 EXPLORATORY 筛选（体积≥0.85×G42，厚度≥0.5×G42），它是研究假设，不是工程要求。
- 参考面积 {SREF:.9f} m²，参考长度 5 m，与 `configs/mission_1200s.yaml` 和 v7 数据库一致。
- 本筛选只覆盖第一批 49 个主要工况（M 5–8，α 4–10°），不代表完整任务域 M3.5–9.25、α −2–12° 已覆盖。
- 邻域：中心点 ±0.25 Mach、±1° 迎角，每个点真实调用模型；跨 Mach 共同评价 13 个 Mach × 7 个迎角 × 31 个构型；等升力使用 α=2–12°（步长 0.5°）的 PCHIP 反插值，不外推。
- 切楔模型的无黏纯压力阻力 CD_pressure 不含摩擦和底阻，高设计 Mach 的细长构型在加入摩擦/底阻后排名可能改变，需要 CFD 或更高保真度验证。

## 5. 局限和未完成

- 没有新生成教师几何（TEACHER_GEOMETRY_GENERATION=PARTIAL）。
- Parasolid 和 NX .prt 阻塞；NX 导入没有经过验证。
- CAD 体的尾部封闭面是为 CAD 加的（`CAD_ONLY_BASE_CAP`），原生几何的尾部是开口；气动模型单独处理底面，不使用该封闭面。对称面 y 值由 7e-17 m 置为 0。
- 10 张图已生成，但我这次无法查看图片，没有目视检查版面。
- 单位：程序长度按 1 m 解释（审计注明物理尺度未确认），STEP 写成毫米。

## 6. 主要文件

- 排名：[top5_by_condition.csv](screening/top5_by_condition.csv)、[baseline_diversity.csv](screening/baseline_diversity.csv)、[all_candidate_scores.csv](screening/all_candidate_scores.csv)、[selected_unique_geometries.csv](screening/selected_unique_geometries.csv)
- 离设计：[nearby_performance.csv](offdesign/nearby_performance.csv)、[cross_mach_performance.csv](offdesign/cross_mach_performance.csv)、[equal_lift_comparison.csv](offdesign/equal_lift_comparison.csv)、[ranking_consistency.csv](offdesign/ranking_consistency.csv)
- 几何与 UG：[geometry_manifest.csv](geometry/geometry_manifest.csv)、[import_manifest.csv](ug_batch/import_manifest.csv)、[UG_CFD_HANDOFF](UG_CFD_HANDOFF)
- 状态：[acceptance_status.json](acceptance_status.json)、[identity_manifest.json](identity_manifest.json)、[budget_ledger.json](budget_ledger.json)、[screening_config.json](config/screening_config.json)
- 交付说明：[CFD_HANDOFF_README_ZH.md](CFD_HANDOFF_README_ZH.md)
- 代码：`tools/v6_wedge_geometry_screening/`（`wedge_eval.m` 只是对现有 v7 模型的批处理封装，模型本身未改）
"""
(OUT / "REPORT_ZH.md").write_text(rep, encoding="utf-8")

ho = f"""# CFD 交接说明（切楔筛选入选构型）

本文件供接收 UG/NX 几何并做后续 CFD 的人员使用。**这里的所有气动数值是切楔低阶模型预测，不是 CFD 结果。** 本阶段没有运行任何 CFD。

## 1. 文件

- 几何包：[UG_CFD_HANDOFF](UG_CFD_HANDOFF)，共 {len(sel)} 个构型文件夹；每个构型只存一份几何，对应多个飞行工况：
  - `geometry.step`：实体（毫米），由 OpenCascade 生成并重新导入检查；
  - `geometry_native.mat`：教师 MATLAB 原生 50×50 网格（半模型，源证据）；
  - `geometry_info.json`：参数、哈希、参考量、CAD 检查结果；
  - `related_flight_cases.csv`：该构型需要计算的飞行 Mach、迎角及角色（TOP1–5/BASELINE/DIVERSITY），并附切楔预测值作对比参考。
- 清单：[geometry_manifest.csv](geometry/geometry_manifest.csv)、[import_manifest.csv](ug_batch/import_manifest.csv)。
- 没有 Parasolid（.x_t）和 NX .prt 文件，也没有任何 CFD 求解脚本。

## 2. 批处理（.bat）

`ug_batch/run_ug_import.bat <GEOMETRY_ID>` 导入单个构型，`run_ug_import_all.bat` 导入全部。仅负责把 STEP 交给 NX 并保存 .prt，不是 CFD 脚本。

运行前设置环境变量 `UGII_BASE_DIR`（NX 安装目录，需包含 `NXBIN\\run_journal.exe`），或设置 `NX_RUN_JOURNAL` 指向 `run_journal.exe`。返回码：0 成功，2 用法错误/不在清单，3 找不到 NX，4 STEP 缺失，5 NX 运行但导入未确认。结果记入 `logs/nx_import_results.csv`。

**状态：`NX_BATCH_NOT_EXECUTED`。** 准备本包的机器没有安装 NX，所以 NXOpen 日志（`nx_journal/nx_import_step.py`）没有真实运行过，首次使用请先试跑 1 个构型。

## 3. 坐标、迎角与单位

- 机体坐标：x 向下游（机头 x=0，底部 x≈5000 mm），y 展向（全模型关于 y=0 对称），z 向上。
- 迎角 α>0 为抬头：来流方向 V=(cosα, 0, sinα)（来流从下方吹向机体）。
- 单位：STEP 为毫米。程序长度按 1 m 解释后乘 1000（审计注明物理尺度未确认）。
- 全模型由原生半模型加 y 镜像得到。原生几何尾部开口，STEP 中加了一个仅用于 CAD 的尾部封闭面（`CAD_ONLY_BASE_CAP`），建 CFD 域时请自行决定尾部处理。

## 4. 参考量

- 参考面积 Sref = {SREF:.9f} m²（对所有构型相同，不随形状缩放），参考长度 5 m。切楔数据的 CD 是无黏纯压力阻力，不含摩擦和底阻。

## 5. 需要统一的约定（由接收方确定并记录）

气体模型（比热比 1.4 量热完全气体/真实气体）、来流静压静温或高度、壁面温度/边界层状态（层流、湍流、转捩）、尾部处理（封闭/真空底阻）、网格收敛判据。

## 6. 建议优先计算的构型

{chr(10).join(f"- `{s['geometry_id']}`（设计 Mach {float(s['Ma_design']):.3f}，{s['roles']}，Top5 出现 {s['n_top5_conditions']} 次）" for s in cfd)}
- 对照：`G42`（基准），`V2_MD8_005`（多样性对照）。

每个构型的具体 Mach 和迎角见各自的 `related_flight_cases.csv`。

## 7. 返回结果时请记录

CL、CD（压力项、摩擦项、总值分开）、俯仰力矩与力矩参考点、残差下降量和收敛判据、迭代数、网格数量与首层网格高度、网格无关性检查、尾部处理方式、求解器与版本、气体和来流条件。
"""
(OUT / "CFD_HANDOFF_README_ZH.md").write_text(ho, encoding="utf-8"); print("reports written")
