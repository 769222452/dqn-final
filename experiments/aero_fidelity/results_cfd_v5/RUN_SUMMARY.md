# CFD 第五版执行记录（探索性实验）

冻结提交：`dc39a7f9e7c3d78f8bc66c660866907759362c24`（aoa-fix）。

复用第四版三种入选外形，五个文件 SHA-256 逐一一致。本次未运行 MATLAB、family 或 check；未修改冻结源码、求解器设置或判据。

环境：Python 3.11.10、SU2 8.5.0、Open MPI 5.0.11、gmsh 4.15.2；实际 CPU 配额 10 核。按文档使用 NP=7、JOBS=2、BASE_MARKER=1。

相关测试 19 passed；运行前后 upstream_matlab 203 个文件哈希一致。

## 各阶段墙钟

| 阶段 | 状态 | 秒 |
|---|---|---:|
| cfd-mesh | COMPLETED | 662.513 |
| cfd-run | COMPLETED | 2750.911 |
| collect | COMPLETED | 6.779 |

## 网格核验

| 网格 | 第四版四面体 | 第五版四面体 | base_surfaces | 标记正确 | 内部网格内容相同 |
|---|---:|---:|---:|---|---|
| FAM_MD4/medium/mesh.su2 | 347599 | 347599 | 2 | True | True |
| FAM_MD6/coarse/mesh.su2 | 93085 | 93085 | 1 | True | True |
| FAM_MD6/fine/mesh.su2 | 1012692 | 1012692 | 1 | True | True |
| FAM_MD6/medium/mesh.su2 | 284213 | 284213 | 1 | True | True |
| FAM_MD8/medium/mesh.su2 | 257932 | 257932 | 1 | True | True |

MD4 中网格 base_surfaces=2，其余为 1。两个几何面共同组成同一个连通的 x=5 m 尾部封口，并归入唯一 base 物理标记；用户已明确接受这一冻结脚本的原始输出并授权继续。没有将该字段改写为 1，也没有修改网格或参数。详见 operations/base_review.json 和 accepted_base_deviation.json。

## CFD 收敛证据

生成工况 29/29，运行完成 29/29，按原残差或 CL/CD Cauchy 条件收敛 29/29。

未收敛工况及末密度残差：[]。rms[Rho] 保留 SU2 的 log10 值，小于 -5 才对应密度残差小于 1e-5。

非物理解点警告涉及 9 个工况；非物理迎风重构警告涉及 26 个工况。完整文字保留在 su2.log 和 cfd_case_status.json。

逐工况耗时由日志创建到 done 标志估算，起点精度为秒；阶段总墙钟单独计时。

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
          "lead_over_worst": 0.6340949889379475,
          "LD": {
            "FAM_MD4": 4.650020517203619,
            "FAM_MD6": 5.849114311761248,
            "FAM_MD8": 7.598575225621077
          }
        },
        "8.0": {
          "best": "FAM_MD8",
          "expected": "FAM_MD8",
          "best_is_expected": true,
          "lead_over_worst": 0.4951213672764647,
          "LD": {
            "FAM_MD4": 4.599223255413209,
            "FAM_MD6": 5.532842198631211,
            "FAM_MD8": 6.87639696204311
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
          "lead_over_worst": 0.5354760007794448,
          "LD": {
            "FAM_MD4": 5.487759297299135,
            "FAM_MD6": 6.777822257702824,
            "FAM_MD8": 8.426322699057092
          }
        },
        "8.0": {
          "best": "FAM_MD8",
          "expected": "FAM_MD8",
          "best_is_expected": true,
          "lead_over_worst": 0.3968389963116543,
          "LD": {
            "FAM_MD4": 4.996354187101682,
            "FAM_MD6": 5.903454887333679,
            "FAM_MD8": 6.979102367928645
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
          "lead_over_worst": 0.44332828866713236,
          "LD": {
            "FAM_MD4": 6.006269415015046,
            "FAM_MD6": 7.270034607492944,
            "FAM_MD8": 8.669018556047405
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

decision.json/txt 为冻结脚本原始输出，未筛除工况或改写判定；脚本本身读取所有存在的 history，是否取得收敛证据另见逐案例记录。

理想切换收益 Euler：{'best_fixed': 'FAM_MD6', 'gain_pct': 0.0, 'winners': {'M4_CL0.03': 'FAM_MD6', 'M4_CL0.04': 'FAM_MD6', 'M4_CL0.05': 'FAM_MD6', 'M6_CL0.03': 'FAM_MD6', 'M6_CL0.04': 'FAM_MD6', 'M6_CL0.05': 'FAM_MD6', 'M8_CL0.03': 'FAM_MD6', 'M8_CL0.04': 'FAM_MD6', 'M8_CL0.05': 'FAM_MD6'}}；切楔模型：{'best_fixed': 'FAM_MD6', 'gain_pct': 0.0, 'winners': {'M4_CL0.03': 'FAM_MD6', 'M4_CL0.04': 'FAM_MD6', 'M4_CL0.05': 'FAM_MD6', 'M6_CL0.03': 'FAM_MD6', 'M6_CL0.04': 'FAM_MD6', 'M6_CL0.05': 'FAM_MD6', 'M8_CL0.03': 'FAM_MD6', 'M8_CL0.04': 'FAM_MD6', 'M8_CL0.05': 'FAM_MD6'}}。

## 预声明诊断 D1–D3

| 构型 | Mach | 底面项均值 | α间标准差 | 全为负 | 均值偏差 | 最大逐点偏差 |
|---|---:|---:|---:|---|---:|---:|
| FAM_MD4 | 4.0 | 0.00183298389 | 0.000152410896 | False | -0.000440906757 | 0.00102991994 |
| FAM_MD4 | 6.0 | -0.00252234924 | 0.000251380886 | True | -0.000461337599 | 0.000970168878 |
| FAM_MD4 | 8.0 | -0.00421971006 | 0.000314157411 | True | -0.000476616636 | 0.000974763102 |
| FAM_MD6 | 4.0 | 0.00050736402 | 0.000129349459 | False | 0.000323083215 | 0.000421629339 |
| FAM_MD6 | 6.0 | -0.00354821281 | 0.000249337062 | True | -7.02057945e-05 | 0.000210161455 |
| FAM_MD6 | 8.0 | -0.00502689863 | 0.000372307381 | True | -0.000200115917 | 0.000398065366 |
| FAM_MD8 | 4.0 | -0.0018261643 | 4.19121879e-05 | True | 0.000638371573 | 0.000998837361 |
| FAM_MD8 | 6.0 | -0.00416072119 | 0.00010649476 | True | 0.00016806623 | 0.000222833467 |
| FAM_MD8 | 8.0 | -0.00475023661 | 0.000138626802 | True | -6.82568321e-06 | 3.69503667e-05 |

D1 阈值：底面项为负、α 间标准差 <0.0003，与第四版 Euler−切楔偏移之差 <0.0005；均值与逐点差异均提供，不隐去不满足项。

D2（全部 29 个压力 CD >0）：True；非正项：[]。

D3（中/细网格同号且 CL、CD 变化绝对值均 <15%）：{'rel_change_CL': 0.025585088784250948, 'rel_change_CD': 0.017929571620658995, 'same_sign': True, 'pass_': True, 'denominator': 'abs(fine), following frozen collect_su2.grid_convergence'}。

第五版为预声明的探索性后续实验。第四版原始判定仍保留；数值收敛与物理正确、网格无关性分别报告。

## 原切换收益统计的覆盖范围

- euler / FAM_MD8 缺少 [{'mach': 8.0, 'CL_target': 0.05}]，因此被原脚本从完整条件切换收益计算中排除。
- wedge / FAM_MD8 缺少 [{'mach': 8.0, 'CL_target': 0.05}]，因此被原脚本从完整条件切换收益计算中排除。

原始 0% 切换收益按冻结脚本的完整条件筛选口径报告；不能把它解释为三个成员在全部目标 CL 上均已得到比较。decision.json/txt 保持原样。

压缩包包含所有工况的 case.json、case.cfg、history.csv、su2.log、网格统计、诊断、冻结文档及原始 decision；不含 CFD 网格或 restart。所有云端原始文件保留。
