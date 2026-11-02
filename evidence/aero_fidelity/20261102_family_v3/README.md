# 外形族实验第三版（协议修订 2）

冻结提交：`78378fab661eff81b0840417a1f7616acf451454`（aoa-fix）。仅追加结果，未修改任何源码、任务输入、扫描范围、审计、选择规则或容许值。
未重跑第一、二版；未执行 cfd-mesh、cfd-run、collect。

| 项目 | 结果 |
|---|---:|
| 任务总数 | 189 |
| 生成成功 | 106 |
| 生成失败 | 83 |
| 通过完整 check 的有效外形 | 106 |
| 正式选择结果 | NO_ADMISSIBLE_FAMILY |
| 目标体积（m³） | 1.018813633812261 |
| 生成墙钟（秒） | 210.703 |
| check 墙钟（秒） | 773.877 |

“生成成功”仅表示输出了外形文件；几何有效性由冻结审计判定。生成失败与审计不通过均完整保留。

- [中文执行报告与完整 check 输出](RUN_SUMMARY.md)
- [189 组逐任务结果](family_status.csv)
- [生成失败 job_id 与原始错误](family_failures.json)
- [完整审计与所有选择候选](family_selected/family_audit.json)
- [check 原始输出](operations/check.log)
- [完整原始结果 ZIP](aero_family_v3_78378fa_results.zip)
- [包内 SHA-256 清单](SHA256SUMS)
- [下载逐文件校验](DOWNLOAD_VERIFIED.json)

ZIP 包含全部 189 份 status.json、全部生成的原始 .mat、四份 MATLAB 运行日志、任务清单、冻结审计、环境和版本记录。
此目录提供便于浏览的报告副本；完整目录结构在 ZIP 中，SHA256SUMS 路径以解压目录为根。

ZIP SHA-256：`8f43684157ce861c3bb52c238759f1b9a47d73aeabeec9f62756de12bf1bb3fe`。

没有可入选外形族，故未生成 select.json 或三个入选 .mat；未以其他外形替代。
