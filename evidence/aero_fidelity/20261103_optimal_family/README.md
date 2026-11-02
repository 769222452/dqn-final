# 最优外形检验（README §3h）

冻结源提交：`f248c4f7da03a48335cbe1dfca4e2b6e0e58c7da`，分支 aoa-fix。没有修改扫描、生成器、容许值、CFD 设置或判据。

- [完整中文执行报告](RUN_SUMMARY.md)
- [全部证据 ZIP（含第二轮所有 VALID 外形 MAT）](aero_opt_f248c4f_results.zip)
- [ZIP 内逐文件 SHA-256](SHA256SUMS)
- [原始最优外形判定](results/optimum_decision.json)
- [第一轮审计](family_a_selected/family_audit.json)
- [第二轮审计](family_b_selected/family_audit.json)

ZIP SHA-256：`8b1d8515b7b9e2d08681a43373947b7283178a9fe4b8aef6db0560b4260d1238`。

原始成功及失败结果均保留。MAP_ONLY 是审计映射阶段的正常完成状态。Euler 仅在 EVALUATED 后执行；没有运行 collect。完整原始日志保留于 ZIP 和云端，CFD 网格/restart 不入包。
