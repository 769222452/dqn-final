# 外形族第四版（协议修订 3）执行记录

冻结提交：`01589db3371acc0f542b4c9b1877db0b6221566d`（aoa-fix）。

原始生成器、29 组任务、审计、选择规则和容差均未修改；第一至三版未重跑。MATLAB 在本机运行，其余步骤在云端原提交执行。

环境：MATLAB R2023a，Optimization / Symbolic Math 工具箱授权可用；云端 Python 3.11.10、SU2 8.5.0、Open MPI 5.0.11。

NP=7、JOBS=2、SLICES=4；云端实际 CPU 配额 10 核。相关测试 16 passed；运行前后 203 个 upstream_matlab 文件 SHA-256 全部通过。

生成：29/29 成功，0 失败，墙钟 80.344 s。

check：SELECTED；29 个有效外形；墙钟 215.100 s。

## 生成错误分类

无生成失败。

## check 完整输出

```text
{
 "jobs": 29,
 "generated": 29,
 "valid": 29,
 "outcomes": {
  "VALID": 29
 },
 "status": "SELECTED",
 "selected": {
  "design_key": "per_mach",
  "target_volume_m3": 1.018813633812261,
  "max_volume_deviation": 0.008925663727964439,
  "admissible": true,
  "chosen": {
   "4.0": "FAM4_MD4_pair_d+4_r0.5_ax1.0400",
   "6.0": "FAM4_MD6_pair_d+2_r0.5_ax1.0250",
   "8.0": "FAM4_MD8_pair_d+2_r0.5_ax0.9850"
  },
  "deviation_per_mach": {
   "4.0": -0.006023182169068386,
   "6.0": 0.008925663727964439,
   "8.0": -0.00024170476995577506
  },
  "shock_offset_deg_per_mach": {
   "4.0": 4.0,
   "6.0": 2.0,
   "8.0": 2.0
  }
 }
}
candidates: [('per_mach', True, 0.009)]
```

| 入选构型 | 原任务 | 体积 m³ | 相对目标偏差 |
|---|---|---:|---:|
| FAM_MD4 | FAM4_MD4_pair_d+4_r0.5_ax1.0400 | 1.012677133699 | -0.602318% |
| FAM_MD6 | FAM4_MD6_pair_d+2_r0.5_ax1.0250 | 1.027907221709 | +0.892566% |
| FAM_MD8 | FAM4_MD8_pair_d+2_r0.5_ax0.9850 | 1.018567381697 | -0.024170% |

## 后续阶段

- cfd-mesh: COMPLETED, 退出码 0, 墙钟 655.923 s；完整日志见 operations/cfd-mesh.log。
- cfd-run: COMPLETED, 退出码 0, 墙钟 3076.834 s；完整日志见 operations/cfd-run.log。
- collect: COMPLETED, 退出码 0, 墙钟 7.110 s；完整日志见 operations/collect.log。

CFD case.json 数量 29；有 done 标志 29；按冻结残差或 CL/CD Cauchy 规则收敛 29。逐案例日志、警告与末残差见 cfd_case_status.json / csv。

未取得收敛证据的案例及末残差：[]。rms[Rho] 为 log10 值，密度残差小于 1e-5 对应该值小于 -5。

SU2 非物理解点警告涉及 9 个案例，非物理迎风重构状态警告涉及 28 个案例。原始文字与数量逐案例保留；Cauchy 收敛不确定这些警告对气动力的影响。本报告不改变冻结判据。

逐案例耗时由 su2.log 文件创建时间到 done 标志时间估算（起点精度为秒，包含 MPI 启动和输出）；阶段总墙钟由独立计时器记录。

压缩包保留全部生成记录和 MATLAB 日志、冻结审计及入选外形、实际产生的 CFD 配置与日志；不含 CFD 网格或 restart 文件。所有原始文件仍保留在云端。

## 冻结 collect 原始判定

```json
{
  "per_condition": {
    "0.03": {
      "met": false,
      "at_flight_mach": {
        "4.0": {
          "best": "FAM_MD8",
          "expected": "FAM_MD4",
          "best_is_expected": false,
          "lead_over_worst": 2.9680014064313545,
          "LD": {
            "FAM_MD4": 3.550771756037389,
            "FAM_MD6": 5.210136831224515,
            "FAM_MD8": 14.089467321873089
          }
        },
        "8.0": {
          "best": "FAM_MD6",
          "expected": "FAM_MD8",
          "best_is_expected": false,
          "lead_over_worst": -1.7315726245471192,
          "LD": {
            "FAM_MD4": 12.55117249268011,
            "FAM_MD6": 58.405374218894124,
            "FAM_MD8": -79.83537417771754
          }
        }
      }
    },
    "0.04": {
      "met": false,
      "at_flight_mach": {
        "4.0": {
          "best": "FAM_MD8",
          "expected": "FAM_MD4",
          "best_is_expected": false,
          "lead_over_worst": 2.154760295623448,
          "LD": {
            "FAM_MD4": 4.330429446313153,
            "FAM_MD6": 6.157525712075975,
            "FAM_MD8": 13.661466880227367
          }
        },
        "8.0": {
          "best": "FAM_MD8",
          "expected": "FAM_MD8",
          "best_is_expected": true,
          "lead_over_worst": 2.902765447476041,
          "LD": {
            "FAM_MD4": 10.737458613442033,
            "FAM_MD6": 23.1960985289355,
            "FAM_MD8": 41.90578247024557
          }
        }
      }
    },
    "0.05": {
      "met": false,
      "at_flight_mach": {
        "4.0": {
          "best": "FAM_MD8",
          "expected": "FAM_MD4",
          "best_is_expected": false,
          "lead_over_worst": 1.5847797546997318,
          "LD": {
            "FAM_MD4": 4.8880442941353195,
            "FAM_MD6": 6.713010664017896,
            "FAM_MD8": 12.634517931556514
          }
        },
        "8.0": null
      }
    }
  },
  "premise_holds": false,
  "targets_met": 0
}
```

decision.json 保持原脚本输出。collect 本身不筛选未收敛 history；收敛证据另列以上。网格比较的实际尺寸、棱边重试比例及单元数见各 mesh_info.json。

负压力阻力/负总阻力条目、网格异号结果和原切换收益统计的覆盖范围见 results/OUTPUT_OBSERVATIONS.md 及 JSON。数值收敛不等于物理正确或网格无关，不能忽略这些输出问题解释冻结判定。
