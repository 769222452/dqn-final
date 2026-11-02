# 外形族实验第二版（协议修订 1）

协议基础：`4baf51c6ad962dc646ced8ab4c105b50db131dc0`；实际执行：`e33111d30934e3ff1f2f069510127f02c443b4a9`。

用户授权的唯一源码修复是 `make_family.m` 兼容 cell / struct 任务读取。
`upstream_matlab`、180 组任务数值、冻结原生审计、选择规则及 ±3% 容差均未改变。
首次已完成的 45 组记录完整保留，续跑仅执行缺失的 135 组。

| 项目 | 结果 |
|---|---:|
| 任务总数 | 180 |
| 生成成功 | 110 |
| 生成失败 | 70 |
| 通过完整 check 的有效外形 | 37 |
| 正式选择结果 | NO_ADMISSIBLE_FAMILY |
| 生成执行合计（秒） | 390.955 |
| check 墙钟（秒） | 858.286 |

“生成成功”仅指输出外形文件。所有失败记录保留。未重跑第一版，未执行 cfd-mesh、cfd-run、collect。

- [中文执行报告，含完整 check 输出](RUN_SUMMARY.md)
- [逐任务结果表（180 行）](family_status.csv)
- [生成失败 job_id 与原始错误](family_failures.json)
- [冻结审计与全部选择候选](family_selected/family_audit.json)
- [check 原始输出](operations/check.log)
- [最小兼容补丁](family/AUTHORIZED_COMPATIBILITY_FIX.patch)
- [完整原始结果 ZIP](aero_family_v2_4baf51c_results.zip)
- [包内 SHA-256 清单](SHA256SUMS)
- [下载逐文件核验记录](DOWNLOAD_VERIFIED.json)

ZIP 包含全部 180 份 status.json、全部生成的原始 .mat、首次和续跑 MATLAB 日志、输入清单、原生审计、运行与版本记录。
此目录提供便于浏览的报告副本；完整原始目录结构在 ZIP 中，SHA256SUMS 的路径以解压目录为根。

ZIP SHA-256：`7da3014ff7ba18eda199b7f03596952f54f6817591f43f93982bf0a3bd72b217`。

没有可入选外形族，故未生成 select.json 或三个入选 .mat；未以其他外形替代。
