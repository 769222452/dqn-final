# 真实 Teacher 构型空间扩展

根目录 `D:\v6-mission-morph-core`。本模块复用真实 `matlab/active/build_geometry_core.m`、冻结原生几何审计、当前闭合体积定义以及原始 `wedge_eval.m` 的 P2 压力气动。所有输出仅写入 `results/v6_teacher_space_expansion/`。

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\run.py --resume
```

默认保护上限：总 Teacher 尝试5000次（含3个再现检查），48小时，每批100个新候选，MATLAB并发1。每个候选记录输入、完整MAT、原生审计、失败原因、闭合体积和网格厚度、坐标哈希、原审计哈希和真实49点P2结果。任务扩展网格由成功1200秒轨迹及已有支持域自动确定。数据缓存依赖真实坐标身份及冻结源代码。

`STOP_REQUESTED` 文件会让程序在当前原子任务完成后安全退出。删除该文件后使用上面的命令恢复。操作系统文件锁防止重复控制器，复用活着的已有MATLAB进程，已完成生成不重复执行。程序异常不算物理失败；日志与请求均保留。生成过程中被强行终止的已开始调用记录为中断失败，不静默重做。

Windows上的单请求超时使用 `QueryUnbiasedInterruptTime`，排除系统睡眠和休眠时间；48小时总预算仍按原墙钟截止时间执行。终止所属MATLAB后先观察进程退出，再允许恢复，防止终止请求与新进程启动之间的竞争。相关控制流程回归检查为 `validate_runtime_timing.py`；它不调用MATLAB，也不生成气动数据。

批次候选提交时同时保存提案后的随机数状态；恢复已提交候选时补齐该状态。同一批次的适应性指标只应用一次，因此在赢家加密阶段中断不会重复增加停滞计数或重复调整下一批搜索方向。

已完成的初始基线汇总在续跑时保留，不用后续候选覆盖。长时间加密前先写出该批的 `metrics_pending_verification.json`；只有完成复核和批次收尾后才写 `completed.json`。尚未原子提交的 `.tmp.csv` 不进入排名。

`mission_all_scores.csv` 和 `mission_top1.csv` 的正式导出范围是 `mission/domain.json` 指定的共同任务网格。旧表中的额外工况、尚未覆盖全池的加密点不属于该正式排行榜；其原始P2值保留在 `aero/`，完整局部比较证据由 `verification/` 提供。已单独导出的 `supplemental_grid_diagnostics.csv` 会记录每点实际比较构型数量及全池/部分池状态。

用户明确授权下一阶段时，可增加总预算：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\run.py --resume --extend-hours 48 --max-teacher-attempts 10000
```

`--batch-size`、`--max-wall-time-hours`、`--max-teacher-attempts` 可覆盖配置。不要并行启动第二个MATLAB任务。

核心实现：

- `initialize.py`：453个真实旧构型、原生输入/哈希、闭合体积再核对；成功轨迹与任务分布。
- `preflight.py`：G42/A/B真实再现检查、12次新Teacher预检、逐候选流水线。
- `runtime.py` / `exp_worker.m`：单个真实MATLAB进程、原子结果、断点与进程所有权。
- `search.py`：Sobol、局部演化、固定工况/等CL定向和气动曲线互补性推荐。代理只参与选点。
- `evaluation.py`：统一P2网格、无外插的PCHIP等升力求解、固定参考轨迹理想压力阻力冲量。
- `analysis.py`：原始CL/CD排名、CORE42集中度、相邻点优势和等CL独立评价。
- `verification.py`：全可行候选池参与的局部Mach/迎角/CL加密；逐对验证相邻优势工况，保留各轮计划、全部竞争者结果和具体邻域排名。通过的构型只在保存的相邻区域获得有效赢家资格，未检查的区域不作稳定性结论。每次正式复核同时诊断所有本轮新增名义赢家。
- `reporting.py`：逐批中文报告、GoCart候选清单与图形。

回归检查使用原有真实数据，不调用新Teacher，不产生伪造气动结果：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\validate.py
```

已提交P2文件的输入身份、MAT内容和重复工况一致性检查，以及指定批次的真实生成失败诊断：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\audit_committed_p2.py
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\diagnose_batch.py --batch 1
```

这些命令读取已保存的真实数据，不调用Teacher或P2。进行中的任务可能在审计快照之后继续提交新文件，因此一次快照不证明整轮搜索已经完成。

可独立复核指定批次已保存的CORE42和等CL加密邻域排名、完整1200秒压力阻力积分，以及加密前后共同网格文件是否一致：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\audit_saved_snapshot.py --tag batch_0006 --previous-tag batch_0005_verified
```

审计逐工况重新排序全部可行竞争者，并分别检查原始相邻工况的0.5%门槛与加密邻域的排名保持。CORE42直接读取已提交的原始CL/CD，等CL读取保存的加密插值表；原始文件和输入均记录哈希。若该池尚未完成加密，只审计已保存积分，不声称完成邻域复核；结束加密后再次运行可补齐证据。轨迹输入与原积分器实际采用的CSV解析方式逐项核对，另一种精度解析方式造成的末位差异单独记录。输出为 `validation_<tag>_saved_snapshot.json`；它不替代最终交付审计。

`diagnose_batch.py` 另外核对逐候选冻结49点的有效数量，并分别统计该批新增合格构型在冻结网格、已保存任务网格上的P2有效率和CL门槛通过数量。几何合格率、模型有效率和固定迎角排名资格分别报告；未保存的任务网格统计标为待统计。批次统计包含原始输入文件哈希，可在该批任务网格汇总保存后再次生成完整统计。

用两个已完成共同网格计算的指标标签，分别归因某批新增构型的冻结工况、CORE42、等CL、扩展域和参考轨迹贡献：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\diagnose_contribution.py --tag batch_0004 --previous-tag batch_0003_verified
```

结果写入该批次的 `aerodynamic_contribution.csv/.json` 和 `new_task_grid_winning_points.csv`，包含输入哈希。任务工况统计权重来自全部输入轨迹，单条参考轨迹的oracle选用时间单独列出；名义工况改善不自动计入正式稳定区域。

轨迹积分缓存目前由整个P2表生成键，但积分只使用共同任务网格。增加局部加密点后，可运行以下独立工具，为完全相同的积分输入保存等价缓存键，减少重复插值：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\promote_oracle_cache.py --apply
```

省略 `--apply` 只预览。该工具要求来源缓存匹配当前完整代码、参考轨迹、工作域和原始P2内容哈希，并对每个复用构型逐项核对共同任务网格；找不到精确来源时跳过。首次运行通过原积分函数复算代表性构型，包含可用的不可行时刻，确认数组逐项一致。复用记录包含来源、目标及输入哈希，保存在 `operational/oracle_cache_promotions/`。可以在P2加密计算期间重复运行，无需重启求解器；实际新增或改变的任务网格数据仍须重新计算积分。首轮320构型的全池一致性证据见 `validation_oracle_cache_promotion_full_pool.json`。

结束交付前，重新生成最终标签对应的图表与报告，再运行独立交付审计：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\audit_delivery.py
```

该程序从已保存的原始P2复算冻结排名、核对等CL排名、全1200秒积分、候选身份、源文件、预算终止条件和交付文件，写入 `delivery_audit.json`。运行中也可读取快照，但不会把仍活着的求解器或尚未达到保护上限的未达标状态判为已结束。通过自动检查后仍需检查最终图形、报告的11项回答，并对停止后的完整文件集运行P2身份审计；本程序不会自行声明目标完成。

`reports/FINAL_REPORT_ZH.md` 在运行时也会更新，必须先看状态。只有 `TARGET_REACHED_VERIFIED` 才代表正式目标通过；`TARGET_NOT_REACHED` 表示预算耗尽，`RUNNING_*` 表示研究仍在运行。近似并列不会计作已证明的显著互补优势。未进行本轮CFD、DQN、OCP或连续变形验证。

搜索期的加密调度优先复核原始相邻点达到0.5%门槛的区域，每个工况仍包含全部可行竞争者。尚不具备正式数量门槛时，弱优势名义赢家的专项加密可暂缓，其身份保存为 `deferred_nominal_core_ids` / `deferred_nominal_equal_ids`，不能计入有效赢家或继承旧池的稳定结论。达到正式数量门槛后，恢复全部新增名义赢家的局部诊断；最终成功还要求这两组诊断全部完成。该调度仅改变计算顺序，不改变系数、几何门槛、相邻区域门槛、完整竞争池、排名或总预算。第7批调整前的计划与进度会单独归档，已算P2保留复用。

局部加密期间，可用以下命令读取当前计划与已原子提交的真实P2工况，并更新 `verification_progress.json`：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\observe_verification.py --seconds 900
```

省略 `--seconds` 只取一次快照。观察器按保存的计划重建工况集合，核对完整候选池身份，并检查实际进程的创建时间；它不会启动、停止或重启求解器。观察时限结束仅表示观察器退出，不表示计算结束。工况覆盖全部完成后仍须等待邻域排名证据；候选池变化时不会把旧计划当作新池的复核结果。`audit_progress.py` 同样会区分最新指标快照与当前候选池，避免用旧池积分证明扩展后的全池结果。

长时间运行的控制器可能已加载较早的报告排版代码。可在不重启求解器的情况下，使用当前已保存的指标快照更新报告解释段落：

```powershell
.\.venv\Scripts\python.exe -B tools\v6_teacher_space_expansion\refresh_snapshot_report.py
```

该工具保留统计计数和时间，更新匹配候选池的复核工况表、固定迎角/等CL状态标签、诊断段落，以及已保存的批次扩展域改善证据。历史改善逐项核对所属批次的原始排名，并明确标注当时的竞争池与统计权重。核心近似并列诊断从原始CL/CD分解第一、第二名差距，保存逐点系数比值并核对恒等式；它不代替等升力分析或改变排名。工具使用LF换行，并检查刷新期间控制器是否更新了报告；不会将下一批新增候选混入上一批的统计结果。结束交付前仍须基于最终停止状态重新完整生成报告。
