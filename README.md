# 定时定点到达任务：变构乘波体的反事实优势强化学习

> **v2（本分支 `target-rl-v2-run`）：**变构扩大可靠任务域的完整实验，运行说明见 [`README_V2.md`](README_V2.md)，设计见 [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md)。以下是第二轮的说明。


本分支是一个独立的最小工程，只含这项研究需要的代码和数据。研究设计、判据和修订记录见
[`PREREGISTRATION.md`](PREREGISTRATION.md)。

**任务：** 1200 s 时刻同时到达指定航程和指定末速。大气和气动有扰动，每次飞行的目标不同。
**方法：** 反事实优势策略迭代。一个策略打所有目标，每 12 s 选"升力系数 19 档 × 变构程度 4 档"；执行层每 3 s 把升力系数目标修正为"目标 − kg × 航迹角"（kg = 1，第二轮起）；起点是按目标查表的常值动作。

**当前是第二轮（修订 5）。** 第一轮结果（H1、G1 未通过）保留在 `runs/<库名>/` 和分支 `target-rl-results-20261118`；第二轮结果写到 `runs/r2/<库名>/`，用新的测试场景 2300–2319。
**对比：** 预测-校正制导（公平版、强版、不变构版、全信息版）、单一常值动作、按目标查表、第一轮策略。
**构型库：** 主实验用 **F8D2000B**（F8D2000 的 2000 个构型 + 91 个减速板构型，共 2091 个，前缘全部固定为 G42 前缘，不变构对照 F8D_01351）；F8D2000（不加减速板）作敏感性分析；L298 第二轮不再运行。

## 目录

| 路径 | 内容 |
|---|---|
| `configs/task.yaml` | 全部参数：飞行器、约束、构型库、动作、扰动、种子、回报、训练、基线 |
| `data/F8D2000/` | F8D2000 构型库的 P2 气动表和原说明 |
| `data/BrakeLadder/` | 91 个减速板构型的 P2 气动表和说明；与 F8D2000 合起来就是主实验的 F8D2000B |
| `data/L298/` | 298 构型快照的 P2 气动表（第一轮敏感性分析用） |
| `src/target_rl/` | 代码（见下表） |
| `tests/` | 测试；`fixtures/reference_model.json` 是从原工程模型记录的对照值 |
| `scripts/pilot_r2.sh` | 第二轮试跑（1 个种子、20 轮、开发场景），输出 `runs/r2/F8D2000B/PILOT.md` |
| `scripts/run_paper.sh` | 论文全量实验 |
| `scripts/run_all.sh` | 单个库的全流程，分阶段、可续跑 |
| `runs/F8D2000B/region.json` 等 | 各库的目标区域和 T1–T6（已算好并随分支提交，预注册以它们为准，两轮共用） |

| 模块 | 作用 |
|---|---|
| `dynamics.py` | 纵向三自由度质点模型和大气，与原工程逐项一致（测试对照，相对误差 < 1e-12） |
| `aero.py` | 等升力阻力表：每个马赫数和升力系数下，库内构型的最小、最大阻力，以及固定构型的阻力 |
| `scenario.py` | 扰动场景（初始偏差、升阻偏差、沿航程的密度场），与原快速模型的生成器逐位一致 |
| `flight.py` | 飞行环境：0.5 s RK4，每个子步检查全部约束；共用保护内环；在线升阻估计；回报 |
| `region.py` | 名义可达范围、目标区域、T1–T6 规则、训练和测试目标 |
| `baselines.py` | 预测-校正（公平版、强版、不变构）、常值策略调参、全知参照 |
| `ensemble.py`, `policy.py`, `train.py` | 优势网络集合、策略链、反事实优势策略迭代 |
| `evaluate.py`, `report.py`, `paper.py` | 测试集评估、判定、论文表图 |
| `pilot.py` | 第二轮试跑小结 |

## 论文实验（第二轮）

实验设计见 [`EXPERIMENTS.md`](EXPERIMENTS.md)：改动、研究问题、各方法、表 1–6 和图 1–7、计算量估计。

先试跑（约 1 小时），看 `runs/r2/F8D2000B/PILOT.md` 再决定是否全量：

```bash
WORKERS=5 bash scripts/pilot_r2.sh
```

全量（可续跑，试跑的训练会被接着用）：

```bash
WORKERS=5 nohup bash scripts/run_paper.sh > run_paper_r2.log 2>&1 &
```

它依次跑 F8D2000B（完整方案）和 F8D2000（精简方案），最后把表格和图写到 `runs/r2/paper/`。6 核约 1～1.5 天。

## 在云服务器上运行（给 Codex）

需要 Python ≥ 3.10。只依赖 numpy、scipy、PyYAML、pytest，不需要 CasADi 或 MATLAB。

```bash
git clone -b target-rl https://github.com/769222452/dqn-final.git target-rl
```

```bash
cd target-rl && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

主实验（F8D2000B）：

```bash
WORKERS=$(nproc) nohup bash scripts/run_all.sh > run_F8D2000B.log 2>&1 &
```

主实验跑完后，再跑敏感性分析（F8D2000）：

```bash
LIBRARY=F8D2000 WORKERS=$(nproc) nohup bash scripts/run_all.sh > run_F8D2000.log 2>&1 &
```

各阶段（`STAGES` 环境变量可选其中几个），结果都在 `runs/r2/<库名>/` 下（`region.json` 在 `runs/<库名>/`）：

| 阶段 | 命令 | 产出 | 耗时 |
|---|---|---|---|
| `test` | `pytest -q` | `logs/test.log` | < 1 分钟 |
| `region` | `python -m target_rl.region` | `region.json`（已随分支提交，存在则跳过） | 约 2 分钟 |
| `pi0` | `python -m target_rl.train --pi0 --variants "..."` | `start_*.json`（各训练目标的常值动作表） | 约 20～30 分钟 |
| `train` | `python -m target_rl.train --variant V --seed S` | `train/V_sS/state.json` | 最多 60 轮，共 20 个 |
| `eval` | `python -m target_rl.evaluate --methods all` | `eval/L2/*.json` | 见 EXPERIMENTS.md |
| `report` | `python -m target_rl.report` | `report.json`、`REPORT.md` | 秒级 |

直接用 `python -m` 运行时，先设置 `PYTHONPATH=src`，库用环境变量 `TARGET_RL_LIBRARY`（默认 `F8D2000B`）。
耗时以 EXPERIMENTS.md 第 6 节为准（6 核、5 个并行进程估计）。

- **续跑：** 中断后再执行同一条命令即可，已完成的训练迭代和评估用例会跳过。
- **只跑一部分：** 例如 `STAGES="train" VARIANTS="full" SEEDS="17 23" bash scripts/run_all.sh`。
- **并行多个训练：** 可以开几个进程，各自用 `VARIANTS`、`SEEDS` 分工，每个进程的 `WORKERS` 相应调小。
- **不要改** `runs/*/region.json`：T1–T6 已在预注册中固定。
- **不要删** `runs/F8D2000B/train/`：第一轮策略是第二轮的对比方法之一。

### 跑完后提交结果

`runs/cache/` 是可再生的缓存，已在 `.gitignore` 中。其余 `runs/` 下的文件都要提交，建议推到新分支：

```bash
git checkout -b target-rl-results-$(date +%Y%m%d) && git add runs && git commit -m "target-rl results" && git push -u origin HEAD
```

## 判据（摘要，全文见预注册修订 5）

- **主判据 H1′：** 在只有变构才能到的目标 T4～T6 上，完整方法每个训练种子相对公平版预测-校正的命中率差，95% 置信区间下界 > 0，且超限率不更高。置信区间按测试场景做聚类自助法。
- **门槛 G1′：** 5 个种子平均命中率高 20 个百分点以上，且 H1′ 在至少 4 个种子成立。通过后才做第二阶段（认证链）。
- **非劣效（次要）：** 相对强版预测-校正，T4～T6 命中率差 ≥ −10 个百分点，且单次飞行快 50 倍以上。
- **只按主实验（F8D2000B）判定；** F8D2000 同样报告，作敏感性分析。
- **命中：** 无超限，且航程误差 ≤ 5 km、末速误差 ≤ 25 m/s。

## 跨平台说明

Windows 与 Linux 的数学库在最后一位浮点上可能不同（相对差约 1e-16），测试按 1e-12 的相对误差比较。由此产生的
轨迹差异极小，但不同平台上的结果不保证逐位相同；论文结果以实际运行的那台机器为准，并在报告中注明平台。

## 已知局限

- 变构用"同升力下最小到最大阻力之间插值"作代理，不是逐个真实构型，也没有计入 8 s 变构过渡。
- 只计压力阻力；传感器理想、无噪声；只有纵向平面；没有攻角速率限制。
- F8D2000B 的减速板构型弯度较大（最大约 0.16 m），机构可实现性未验证；L298 中阻力最大的构型可能形状较极端；可达区域的低速边界受气动表马赫下限 3.5 限制。
- F8D2000 上"只有变构才能到"的区域只有 29～54 m/s 宽，见 `runs/F8D2000/region.json` 的 `notes`。
