# 外形族第三版（协议修订 2）执行记录

冻结提交：`78378fab661eff81b0840417a1f7616acf451454`（aoa-fix）。

本次仅执行第三版 family / check；前两版未重跑，未执行 cfd-mesh、cfd-run、collect。未修改源码、参数、扫描范围、审计或选择规则。

本机 MATLAB R2023a，Optimization 与 Symbolic Math 工具箱可用；按用户授权异机流程执行原始 make_family.m。云端 Python 3.11.10，SU2 8.5.0，Open MPI 5.0.11。NP=7、JOBS=2、SLICES=4；容器 CPU 配额 10 核。

相关测试：13 passed。运行前后 upstream_matlab 全部 203 个源文件的大小、SHA-256 及文件清单一致。PYTHONDONTWRITEBYTECODE=1 仅禁止生成 Python 字节码缓存，避免向冻结源目录写入缓存。

任务：189 组；目标体积 1.018813633812261 m³，容差 ±3%。

生成：{'GENERATED': 106, 'GENERATION_FAILED': 83}；墙钟耗时 210.703 s（含 MATLAB 启动）。189 份状态记录的输入逐项与原始任务清单一致。

check：`NO_ADMISSIBLE_FAMILY`；墙钟耗时 773.877 s；退出码 1。

审计分类：`{"VALID": 106, "GENERATION_FAILED": 83}`。

## 生成失败分类

- 74 组：`V5:GeometryFailure | Stream surface 1: Invalid isentropic streamline [V5:GeometryDomain]`
- 9 组：`V5:GeometryFailure | Stream surface 1: Characteristic iteration cap [V5:GeometryNonconverged]`

全部失败 job_id 与原始错误见 family_failures.json；全部任务见 family_status.csv。

## check 完整输出

```text
{
 "jobs": 189,
 "generated": 106,
 "valid": 106,
 "outcomes": {
  "VALID": 106,
  "GENERATION_FAILED": 83
 },
 "status": "NO_ADMISSIBLE_FAMILY"
}
candidates: [('pair_d+2_r0.5', False, 0.385), ('pair_d+4_r0.5', False, 0.326), ('pair_d+6_r0.5', False, 0.667)]
STOP: family selection failed (see runs/aero_fidelity_v3/family_selected/family_audit.json)
```

逐任务冻结审计与所有候选见 family_selected/family_audit.json。
没有可入选外形族，故没有 select.json 或三个入选 .mat；所有生成的原始 .mat 均保留。

全部 MATLAB 日志、任务输入、生成成功及失败记录均保留。SHA256SUMS 覆盖包内全部证据文件，不含清单自身。
