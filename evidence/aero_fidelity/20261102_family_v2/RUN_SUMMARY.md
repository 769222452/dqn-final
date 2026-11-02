# 外形族第二版（协议修订 1）执行记录

协议基础提交：`4baf51c6ad962dc646ced8ab4c105b50db131dc0`（aoa-fix）；实际执行提交：`e33111d30934e3ff1f2f069510127f02c443b4a9`（aero-family-v2-results-20261102）。

首次运行因任务 JSON 字段不同导致 MATLAB cell/struct 读取不兼容，3 个进程在生成前退出；另一个完成 45 组。用户授权后仅修复 make_family.m 的任务读取方式，原 45 个状态记录逐字节保留，仅补跑缺失的 135 组。完整原始日志及补丁均在 family/ 内。上游生成器、180 组任务输入、审计、选择规则均不变。

仅执行版本 2 的 family 和 check；第一版未重跑，未执行任何 CFD 阶段。

环境：本机 MATLAB R2023a，Optimization、Symbolic Math 工具箱可用；按已授权异机流程调用原始 make_family。云端 Python 3.11.10，SU2 8.5.0，Open MPI 5.0.11。NP=7、JOBS=2、SLICES=4；check 使用冻结脚本的 4 个进程。云端容器 CPU 配额 10 核。

相关测试：10 passed；upstream_matlab 203 个文件 SHA-256 全部一致。

生成：{'GENERATION_FAILED': 70, 'GENERATED': 110}；两次执行墙钟合计 390.955 s（含 MATLAB 启动）；初次 220.533 s，续跑 170.422 s；包含排障间隔的端到端耗时 464.285 s。

冻结 check：`NO_ADMISSIBLE_FAMILY`；耗时 858.286 s；退出码 1。

审计分类：`{"GENERATION_FAILED": 70, "FROZEN_AUDIT_INVALID": 73, "VALID": 37}`。

## 生成失败分类

- 27 组：`V5:GeometryFailure | Stream surface 1: Invalid shock streamline [V5:GeometryDomain]`
- 13 组：`V5:GeometryFailure | Stream surface 1: Invalid isentropic streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 2: Invalid shock streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 3: Invalid shock streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 4: Invalid shock streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 5: Invalid shock streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 6: Invalid shock streamline [V5:GeometryDomain]`
- 6 组：`V5:GeometryFailure | Stream surface 7: Invalid shock streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 8: Invalid shock streamline [V5:GeometryDomain]`
- 3 组：`V5:GeometryFailure | Stream surface 9: Invalid shock streamline [V5:GeometryDomain]`
- 1 组：`V5:GeometryFailure | Stream surface 3: Invalid isentropic streamline [V5:GeometryDomain]`
- 2 组：`V5:GeometryFailure | Stream surface 10: Invalid shock streamline [V5:GeometryDomain]`

全部失败 job_id 及原始错误见 family_failures.json；180 行逐任务结果见 family_status.csv。

## check 完整输出

```text
{
 "jobs": 180,
 "generated": 110,
 "valid": 37,
 "outcomes": {
  "GENERATION_FAILED": 70,
  "FROZEN_AUDIT_INVALID": 73,
  "VALID": 37
 },
 "status": "NO_ADMISSIBLE_FAMILY"
}
candidates: [('g42_d+0', False, nan), ('g42_d+2', False, nan), ('g42_d+4', False, nan), ('g42_d+6', False, nan), ('g42_d-2', False, nan), ('pair_d+0_r0.5', False, nan), ('pair_d+0_r0.65', False, nan), ('pair_d+0_r0.8', False, nan), ('pair_d+2_r0.5', False, 0.595), ('pair_d+2_r0.65', False, nan), ('pair_d+2_r0.8', False, nan), ('pair_d+4_r0.5', False, 0.274), ('pair_d+4_r0.65', False, nan), ('pair_d+4_r0.8', False, nan), ('pair_d+6_r0.5', False, 0.156), ('pair_d+6_r0.65', False, nan), ('pair_d+6_r0.8', False, nan), ('pair_d-2_r0.5', False, nan), ('pair_d-2_r0.65', False, nan), ('pair_d-2_r0.8', False, nan)]
STOP: family selection failed (see runs/aero_fidelity_v2/family_selected/family_audit.json)
```

完整逐任务原生审计与选择候选：family_selected/family_audit.json。
未选出满足冻结规则的外形族，因此没有 select.json 或三个入选 .mat；所有生成的原始 .mat 均保留。

全部失败和日志已保留；未扩大扫描、放宽容许值或修改规则。SHA256SUMS 覆盖包内所有证据文件（不含清单自身）。

## 归档完整性复核

审计导入冻结 mesh_checks.py 时，Python 自动生成一个 __pycache__/mesh_checks.cpython-311.pyc，导致严格文件清单检查报告额外文件并中止首次打包。逐个重新核验原始清单中的 203 个文件：大小和 SHA-256 全部一致，没有源码内容修改或缺失。保留该缓存及原始报错，仅恢复打包，没有重跑生成或 check。详见 operations/upstream_post_integrity.json 与 check_and_package.log。
