# MATLAB 构型生成与气动数据库源码

本目录补充原项目的上游源码：乘波体构型生成、几何审计、多马赫数气动建库、切楔气动计算和候选构型扩展。来自 `D:\v6-mission-morph-core`，整理日期为 2026-11-01。

这是历史源码及部分参考输入的逐字节快照，**没有重新生成构型、重建数据库或改变训练算法**。共 203 个来源文件，其中 52 个 MATLAB `.m` 文件、126 个 Python 辅助脚本。训练项目原有的 `src/`、`configs/` 和 `data/` 保持不变。

## 从哪里看起

以下路径均相对于本目录。

| 用途 | 代码入口 |
|---|---|
| 原始构型生成程序 | [generatewaverider.m](source/matlab/original/generatewaverider.m) |
| 实际使用的参数化构型生成核心 | [build_geometry_core.m](source/matlab/active/build_geometry_core.m) |
| 几何生成 / 气动计算的 JSON 批处理入口 | [v5_batch.m](source/matlab/active/v5_batch.m) |
| 原始多目标优化及参考结果 | [source_teacher](source/data/source_teacher/) |
| 多马赫数构型数据库的搜索、审计和组装 | [v5_multimach_geometry_db_v2/code](source/results/v5_multimach_geometry_db_v2/code/) |
| 多马赫数气动数据库的队列、MATLAB worker 和组装 | [v5_multimach_aero_database_v2/code](source/results/v5_multimach_aero_database_v2/code/) |
| V7 数据库构建 | [build_db_v7.m](source/results/v6_matlab_no_new_cfd/code/matlab/build_db_v7.m) |
| V7 数据库网格加密 | [build_db_refine_v7.m](source/results/v6_matlab_no_new_cfd/code/matlab/build_db_refine_v7.m) |
| 切楔 / 激波膨胀气动模型及辅助函数 | [offdesign_aero_v7.m](source/results/v6_matlab_no_new_cfd/code/matlab/offdesign_aero_v7.m) |
| 指定构型、飞行 Mach 和攻角的批量计算 | [wedge_eval.m](source/tools/v6_wedge_geometry_screening/wedge_eval.m) |
| 几何审计、任务筛选和候选空间扩展 | [tools](source/tools/) |
| 原始版本与实际版本之间的历史差异 | [patches](source/matlab/patches/) / [active_source_amendments.json](source/matlab/active_source_amendments.json) |

`original` 和 `active` 包含同名函数，**不要对整个快照执行 `addpath(genpath(...))`**。需要使用实际版本时，只加入 `source/matlab/active`；气动 V7 模型另加入相应的 `code/matlab` 目录。保留原有目录层级，使依赖 `mfilename` 或 `__file__` 的相对路径关系不变。

## 依赖与复现边界

- 原工具记录的 MATLAB 安装位置为 `D:/matlab/bin/matlab.exe`，部分脚本还引用旧项目和优化器的绝对路径；这些历史路径未被改写。换机器运行前必须核对路径。
- 构型生成使用 Optimization Toolbox、Symbolic Math Toolbox 等功能；原始优化器还使用 `fitrgp`、`gamultiobj`、`parfor`，涉及 Statistics and Machine Learning、Global Optimization、Parallel Computing 工具箱。
- Python 辅助脚本按功能使用 NumPy、SciPy、pandas、Matplotlib、psutil 等依赖。它们是历史建库工具，并非当前训练环境的依赖锁文件。
- 附带了原优化结果 MAT、G42 参考几何、几何元数据、审计规则、参考配置和路径解析辅助代码。**没有打包所有历史候选几何、CFD 结果、计算账本及大型中间数据库**；有些筛选、报告或任务仿真脚本仍依赖这些外部历史产物及旧版 Python 模块。
- V5 多马赫数数据库、V7 建库实验与最后 298 构型压力气动表属于不同阶段，不应混作同一数据集。当前飞行实际使用的数据仍在仓库原有的 `data/geometry_subset/`，其来源记录位于 `data/geometry_subset/provenance/source_manifest.json`。
- 本次是源码补充与完整性核验，不代表在新目录中完成了端到端重建或 MATLAB 数值验证。

## 完整性核验

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 逐文件记录原相对路径、字节数和 SHA-256；[SHA256SUMS](SHA256SUMS) 可用于通用校验。存档源码保留原换行，通过本目录 `.gitattributes` 禁止 Git 自动转换。

从仓库根目录运行：

```bash
python upstream_matlab/verify_snapshot.py
```

此命令只核对来源文件，不执行 MATLAB、建库或训练。

## English

This directory preserves the historical MATLAB geometry-generation and aerodynamic-database sources, their Python orchestration helpers, and selected reference inputs. All 203 source/input files are byte-for-byte copies with SHA-256 provenance. It does not modify the flight/training implementation or claim an end-to-end database rebuild. Historical absolute paths, additional archived datasets, and MATLAB toolboxes must be checked before execution. Keep `matlab/original` and `matlab/active` off the MATLAB path at the same time to avoid name shadowing.
