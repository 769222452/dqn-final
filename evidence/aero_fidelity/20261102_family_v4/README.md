# 外形族实验第四版（协议修订 3）

冻结提交：`01589db3371acc0f542b4c9b1877db0b6221566d`（aoa-fix）。仅追加结果，未修改任何源码、任务输入、扫描范围、审计、选择规则或容许值。
未重跑第一至三版；后续阶段实际状态见 RUN_SUMMARY.json，失败日志完整保留。

| 项目 | 结果 |
|---|---:|
| 任务总数 | 29 |
| 生成成功 | 29 |
| 生成失败 | 0 |
| 通过完整 check 的有效外形 | 29 |
| 正式选择结果 | SELECTED |
| 目标体积（m³） | 1.018813633812261 |
| 生成墙钟（秒） | 80.344 |
| check 墙钟（秒） | 215.100 |
| CFD 运行完成 / 按冻结规则收敛 | 29 / 29 |
| CFD 总墙钟（秒） | 3076.834 |
| 未取得收敛证据 | 0 |
| 含非物理解点警告的案例 | 9 |

“生成成功”仅表示输出了外形文件；几何有效性由冻结审计判定。生成失败与审计不通过均完整保留。

- [中文执行报告与完整 check 输出](RUN_SUMMARY.md)
- [29 组逐任务结果](family_status.csv)
- [生成失败 job_id 与原始错误](family_failures.json)
- [完整审计与所有选择候选](family_selected/family_audit.json)
- [check 原始输出](operations/check.log)
- [冻结 collect 判定](results/decision.json)
- [冻结 collect 表格](results/decision.txt)
- [负阻力、网格差异和比较覆盖范围记录](results/OUTPUT_OBSERVATIONS.md)
- [29 个 CFD 案例残差、收敛证据和耗时](cfd_case_status.csv)
- [完整原始结果 ZIP](aero_family_v4_01589db_results.zip)
- [包内 SHA-256 清单](SHA256SUMS)
- [下载逐文件校验](DOWNLOAD_VERIFIED.json)

ZIP 包含全部 29 份 status.json、全部生成的原始 .mat、四份 MATLAB 运行日志、任务清单、冻结审计、实际产生的 CFD 配置和日志、环境及版本记录；不含 CFD 网格或 restart 文件。
此目录提供便于浏览的报告副本；完整目录结构在 ZIP 中，SHA256SUMS 路径以解压目录为根。

ZIP SHA-256：`a047d1595e684a9cea72b0932b525a7edd76493b16b49432f595fa66fa017449`。

原始判定为 premise_holds=false，targets_met=0。存在负压力/总阻力、网格异号和 SU2 非物理状态警告；收敛不等于物理正确。原始结果与判据未修改，详细记录见 OUTPUT_OBSERVATIONS。

已保存 [select.json](family_selected/select.json) 和三个入选 .mat。
