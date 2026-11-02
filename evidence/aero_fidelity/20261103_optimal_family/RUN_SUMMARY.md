# 最优外形检验执行记录

冻结提交：`f248c4f7da03a48335cbe1dfca4e2b6e0e58c7da`（aoa-fix）。

按 README §3h 执行；原生成器、参数、扫描范围、±3% 体积容许和判据保持不变。第一轮 FAMILY_VERSION=5，第二轮 FAMILY_VERSION=6 使用第一轮完整审计。未运行 collect 阶段。

本机 MATLAB R2023a；云端 Python 3.11.10、SU2 8.5.0、Open MPI 5.0.11。SLICES=4，NP=7，JOBS=2；容器 CPU 配额 10 核。相关测试 21 passed。

## 生成和冻结审计

| 轮次 | 任务 | 生成成功 | 生成失败 | 生成墙钟/s | 审计有效 | 审计状态 |
|---|---:|---:|---:|---:|---:|---|
| A | 420 | 302 | 118 | 532.610 | 302 | MAP_ONLY |
| B | 78 | 78 | 0 | 160.438 | 78 | MAP_ONLY |

MAP_ONLY 表示完成审计并形成映射，不表示实验失败。生成失败及审计无效记录全部保留。

## 阶段耗时

| 阶段 | 状态 | 墙钟/s |
|---|---|---:|
| check-a | COMPLETED | 2230.050 |
| check-b | COMPLETED | 635.125 |
| evaluate | COMPLETED | 136.437 |
| cfd-mesh | COMPLETED | 656.031 |
| cfd-run | COMPLETED | 5153.546 |
| confirm | COMPLETED | 6.308 |

## 第 A 轮生成失败分类

```json
{
  "V5:GeometryFailure | Stream surface 1: Invalid isentropic streamline [V5:GeometryDomain]": 63,
  "V5:GeometryFailure | Stream surface 1: Invalid shock streamline [V5:GeometryDomain]": 25,
  "V5:GeometryFailure | Stream surface 8: Invalid shock streamline [V5:GeometryDomain]": 10,
  "V5:GeometryFailure | Stream surface 3: Invalid shock streamline [V5:GeometryDomain]": 10,
  "V5:GeometryFailure | Stream surface 9: Invalid shock streamline [V5:GeometryDomain]": 10
}
```

全部失败 job_id 和原始信息：family_a/generation_failures.json；逐任务记录：family_a/*/status.json。

## 第 B 轮生成失败分类

```json
{}
```

全部失败 job_id 和原始信息：family_b/generation_failures.json；逐任务记录：family_b/*/status.json。

## 冻结 evaluate_optimum 原始判定

```json
{
  "candidates": 26,
  "status": "EVALUATED",
  "eligible": 26,
  "winners": {
    "4.0": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "6.0": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "8.0": "OPTB_MD10_d+3_r0.35_ax0.99440"
  },
  "winner_design_mach": {
    "4.0": 9.0,
    "6.0": 9.0,
    "8.0": 10.0
  },
  "P1": true,
  "switching_gain": {
    "0.03": 0.0,
    "0.04": 0.00044358758409557275,
    "0.05": 0.0027865784104486035
  },
  "best_single": {
    "0.03": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "0.04": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "0.05": "OPTB_MD10_d+3_r0.35_ax0.99440"
  },
  "P2": false,
  "premise_holds": false,
  "euler_confirmation_set": [
    "OPTB_MD10_d+3_r0.35_ax0.99440",
    "OPTB_MD9_d+2_r0.5_ax0.97550"
  ],
  "euler_confirmation_ids": [
    "OPT_MD10_d+3_r0p35_ax0p99440",
    "OPT_MD9_d+2_r0p5_ax0p97550"
  ]
}
```

## Euler 运行及收敛

工况 24，运行完成 24，按冻结条件收敛 21。

末密度残差为 SU2 rms[Rho] 的 log10 原值；小于 −5，或 CL/CD 两项 Cauchy 都通过，才计为收敛。求解完成与物理判据分开报告。

未收敛工况及末残差：
```json
[
  {
    "case_id": "OPT_MD9_d+2_r0p5_ax0p97550/medium/M6_a4",
    "last_residual": -3.903991667
  },
  {
    "case_id": "OPT_MD9_d+2_r0p5_ax0p97550/medium/M6_a6",
    "last_residual": -4.253701912
  },
  {
    "case_id": "OPT_MD9_d+2_r0p5_ax0p97550/medium/M6_a8",
    "last_residual": -3.93218686
  }
]
```

每个工况的非物理解点和重构警告、Cauchy 证据、末 40 行日志见 operations/cfd_convergence.json；完整日志在 cfd/。逐工况时间从日志创建到 done 估算，阶段总墙钟单独记录。

## Euler 原始复核结果

```json
{
  "euler": {
    "4.0": "OPTB_MD10_d+3_r0.35_ax0.99440",
    "6.0": "OPTB_MD10_d+3_r0.35_ax0.99440",
    "8.0": "OPTB_MD10_d+3_r0.35_ax0.99440"
  },
  "wedge": {
    "4.0": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "6.0": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "8.0": "OPTB_MD10_d+3_r0.35_ax0.99440"
  },
  "wedge_winners_declared": {
    "4.0": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "6.0": "OPTB_MD9_d+2_r0.5_ax0.97550",
    "8.0": "OPTB_MD10_d+3_r0.35_ax0.99440"
  },
  "confirmed": false,
  "cases": 24
}
```

euler_confirmation.json 为冻结脚本原始输出。该脚本使用已覆盖 CL 目标的平均值，不自动要求每个成员覆盖全部三个目标，也不筛除未收敛 history。原始输出与可采信证据必须区分。

实际覆盖核对：所有成员、Mach 和模型均覆盖三个指定 CL：True；所有 Euler 输入收敛：False。

MD9、Mach 6 的攻角 4°、6°、8° 三个工况未收敛。该组等升力拟合和排名不能计作已通过的 Euler 验证。原始复核 confirmed=false 保持原样；不将运行完成说成完整物理验证。详见 operations/confirmation_coverage.json。

本次按冻结协议 GRID_CHECK=0，仅运行 medium 网格，不能据此声称网格收敛。网格构建的原始回退尝试及实际单元数见 operations/cfd-mesh.log 和 mesh_verification.json。

归档包含两轮全部 status、MATLAB 日志和审计，第二轮所有 VALID 成员 MAT、最优外形原始输出及确认外形、全部 CFD 工况配置/history/log、运维时间与哈希。CFD mesh/restart 不入包，云端原始文件不删除。
