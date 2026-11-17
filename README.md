# 定时定点到达任务：变构乘波体的反事实优势强化学习

> **library-design 分支：** 在 `target-rl` 的基础上增加了构型库设计研究（`src/libdesign/`，报告见
> [`docs/LIBRARY_DESIGN.md`](docs/LIBRARY_DESIGN.md)）和推荐的前缘固定构型库 **F8D2000B**（F8D2000 + 减速板阶梯，
> `LIBRARY=F8D2000B bash scripts/run_all.sh`）。`target-rl` 分支的预注册和默认设置没有改动。

本分支是一个独立的最小工程，只含这项研究需要的代码和数据。研究设计、判据和修订记录见
[`PREREGISTRATION.md`](PREREGISTRATION.md)。

**任务：** 1200 s 时刻同时到达指定航程和指定末速。大气和气动有扰动，每次飞行的目标不同。
**方法：** 反事实优势策略迭代。一个策略打所有目标，每 12 s 选"升力系数 19 档 × 变构程度 4 档"。
**对比：** 预测-校正制导（公平版、强版、不变构版）、常值起点策略、全知参照（只作上界）。
**构型库：** 主实验用项目原有的 298 构型快照（L298，固定构型 G42）；F8D2000（固定构型 F8D_01351）作敏感性分析。

## 目录

| 路径 | 内容 |
|---|---|
| `configs/task.yaml` | 全部参数：飞行器、约束、构型库、动作、扰动、种子、回报、训练、基线 |
| `data/L298/` | 298 构型快照的 P2 气动表（主实验） |
| `data/F8D2000/` | F8D2000 构型库的 P2 气动表（敏感性分析）和原说明 |
| `src/target_rl/` | 代码（见下表） |
| `tests/` | 测试；`fixtures/reference_model.json` 是从原工程模型记录的对照值 |
| `scripts/run_all.sh` | 全流程，分阶段、可续跑 |
| `runs/L298/region.json`、`runs/F8D2000/region.json` | 各库的目标区域和 T1–T6（已算好并随分支提交，预注册以它们为准） |

| 模块 | 作用 |
|---|---|
| `dynamics.py` | 纵向三自由度质点模型和大气，与原工程逐项一致（测试对照，相对误差 < 1e-12） |
| `aero.py` | 等升力阻力表：每个马赫数和升力系数下，库内构型的最小、最大阻力，以及固定构型的阻力 |
| `scenario.py` | 扰动场景（初始偏差、升阻偏差、沿航程的密度场），与原快速模型的生成器逐位一致 |
| `flight.py` | 飞行环境：0.5 s RK4，每个子步检查全部约束；共用保护内环；在线升阻估计；回报 |
| `region.py` | 名义可达范围、目标区域、T1–T6 规则、训练和测试目标 |
| `baselines.py` | 预测-校正（公平版、强版、不变构）、常值策略调参、全知参照 |
| `ensemble.py`, `policy.py`, `train.py` | 优势网络集合、策略链、反事实优势策略迭代 |
| `evaluate.py`, `report.py` | 测试集评估，H1/G1 判定和报告 |

## 在云服务器上运行（给 Codex）

需要 Python ≥ 3.10。只依赖 numpy、scipy、PyYAML、pytest，不需要 CasADi 或 MATLAB。

```bash
git clone -b target-rl https://github.com/769222452/dqn-final.git target-rl
```

```bash
cd target-rl && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

主实验（298 库）：

```bash
WORKERS=$(nproc) nohup bash scripts/run_all.sh > run_L298.log 2>&1 &
```

主实验跑完后，再跑敏感性分析（F8D2000）：

```bash
LIBRARY=F8D2000 WORKERS=$(nproc) nohup bash scripts/run_all.sh > run_F8D2000.log 2>&1 &
```

各阶段（`STAGES` 环境变量可选其中几个），结果都在 `runs/<库名>/` 下：

| 阶段 | 命令 | 产出 | 14 核估计耗时（每个库） |
|---|---|---|---|
| `test` | `pytest -q` | `logs/test.log` | < 1 分钟 |
| `region` | `python -m target_rl.region` | `region.json`（已随分支提交，存在则跳过） | 约 2 分钟 |
| `pi0` | `python -m target_rl.train --pi0` | `pi0_morph.json`、`pi0_fixed.json` | 约 1 分钟 |
| `train` | `python -m target_rl.train --variant V --seed S` | `train/V_sS/state.json` | 每个约 5 分钟（30 轮），共 20 个，合计 1～2 小时 |
| `eval` | `python -m target_rl.evaluate --methods all` | `eval/L2/*.json` | 约 30 分钟 |
| `report` | `python -m target_rl.report` | `report.json`、`REPORT.md` | 秒级 |

直接用 `python -m` 运行时，先设置 `PYTHONPATH=src`，库用环境变量 `TARGET_RL_LIBRARY`（默认 `L298`）。
耗时是按本地 4 核试跑推算的（每轮训练约 18 s，单次飞行约 0.15 s，预测-校正单次 8～25 s），以实际为准。

- **续跑：** 中断后再执行同一条命令即可，已完成的训练迭代和评估用例会跳过。
- **只跑一部分：** 例如 `STAGES="train" VARIANTS="full" SEEDS="17 23" bash scripts/run_all.sh`。
- **并行多个训练：** 可以开几个进程，各自用 `VARIANTS`、`SEEDS` 分工，每个进程的 `WORKERS` 相应调小。
- **不要改** `runs/*/region.json`：T1–T6 已在预注册中固定。

### 跑完后提交结果

`runs/cache/` 是可再生的缓存，已在 `.gitignore` 中。其余 `runs/` 下的文件都要提交，建议推到新分支：

```bash
git checkout -b target-rl-results-$(date +%Y%m%d) && git add runs && git commit -m "target-rl results" && git push -u origin HEAD
```

## 判据（摘要，全文见预注册）

- **主判据 H1：** 在只有变构才能到的目标 T4～T6 上，完整方法每个训练种子相对公平版预测-校正的命中率差，95% 置信区间下界 > 0，且超限率不更高。置信区间按测试场景做聚类自助法。
- **门槛 G1：** 5 个种子平均命中率高 20 个百分点以上，且 H1 在至少 4 个种子成立。
- **只按主实验（L298）判定；** F8D2000 同样报告，作敏感性分析。
- **命中：** 无超限，且航程误差 ≤ 5 km、末速误差 ≤ 25 m/s。

## 已知局限

- 变构用"同升力下最小到最大阻力之间插值"作代理，不是逐个真实构型，也没有计入 8 s 变构过渡。
- 只计压力阻力；传感器理想、无噪声；只有纵向平面；没有攻角速率限制。
- L298 中阻力最大的构型可能形状较极端；可达区域的低速边界受气动表马赫下限 3.5 限制。
- F8D2000 上"只有变构才能到"的区域只有 29～54 m/s 宽，见 `runs/F8D2000/region.json` 的 `notes`。
