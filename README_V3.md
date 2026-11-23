# v3：模型失配下的纯强化学习（本地运行说明）

分支 `target-rl-v3-run`，在 v2（`target-rl-v2-run`）的基础上修改。

**这一轮只看强化学习自己：**不跑预测-校正等其他方法，报告里只有学到的策略和它的起点规则（用来看学习带来了多少）。

## 和 v2 相比改了什么

| 项 | v2 | v3 |
|---|---|---|
| 飞行结束 | 固定 1200 s，判航程和末速 | **速度降到要求的末速 V\* 时结束**，只判航程（误差 ≤ 5 km，且无超限）；最长 2400 s，到时还没降到 V\* 算未命中 |
| 模型失配 | 无（只有密度场和升阻偏差扰动） | **真实对象 ≠ 名义模型**：刹车构型的阻力效果只有表里的 60%，阻力 +20%，阻力随马赫数的斜率 +10%/Ma（Ma 6 为零点），升力 −5%。策略内部的预测、初始升力表、在线偏差估计都只知道名义模型 |
| 变构档位 | 4 档（0, 1/3, 2/3, 1） | 6 档（0, 0.2, 0.4, 0.6, 0.8, 1），仍就近选 2091 个真实构型、8 s 过渡、切换次数不限 |
| 起点规则 | 预测航程误差调升力；末段末速偏高时刹车 | 预测航程误差调升力；预测多飞超过 15 km **且升力系数目标已 ≥ 0.09** 时满刹车（在训练场景 1100–1159 上从 4 种写法中选出，失配下命中 75%） |
| 目标区域 | 固定 1200 s 的可达区域 | 新任务（到 V\* 结束、真实对象）的可达区域，已算好并提交：`runs/v3/F8D2000B/region.json`。网格除了动压设定值，还加了按升力系数分三段的飞法（短航程要靠大升力系数） |
| 训练的变体 | 变构 + 不变构 | 只训练变构（`rel`）；需要不变构时加 `VARIANTS="rel relfixed"` |
| 每轮训练的决策状态 | 25 个（到 1200 s） | 50 个（到 2400 s，飞行结束后的状态自动跳过） |

失配的数值是人为设定的，没有物理依据。容差 5 km 有文献依据，8 s 过渡可以接受，切换次数不限，这三项已确认。

## 环境

Python ≥ 3.10，只需要 numpy、scipy、PyYAML、pytest、matplotlib。Windows 用 Git Bash。

```bash
git clone -b target-rl-v3-run https://github.com/769222452/dqn-final.git target-rl-v3 && cd target-rl-v3
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt matplotlib
```

## 先做一次流程检查（约 15 分钟）

```bash
QUICK=1 WORKERS=4 bash scripts/run_v3.sh
```

结果在 `runs/v3quick/`，数字没有意义，只检查流程能不能跑通。

## 正式运行

```bash
WORKERS=5 nohup bash scripts/run_v3.sh > run_v3.log 2>&1 &
```

Windows Git Bash 如果提示换行符问题：`bash -o igncr scripts/run_v3.sh`。中断后再执行同一条命令即可续跑。

| 阶段 | 内容 | 估计耗时 |
|---|---|---|
| `test` | 全部 39 个测试 | 约 1—2 分钟 |
| `region` | 目标区域（已提交，自动跳过；删掉文件重算约 25 分钟） | 0 |
| `train` | 种子 17，变构策略，最多 120 轮（连续 20 轮没有采纳会提前停） | 云端 4 核实测每轮约 85 秒，120 轮约 3 小时；5—6 个进程会更快 |
| `eval` | 验证集 192 个目标 + 任务域地图（161 个格点 × 12 个场景），起点规则和学到的策略各一遍 | 约 15—30 分钟 |
| `report` | 报告和图 | 秒级 |

起点规则在 300 个训练场景上（失配下）：命中 217 个，超限 14 个。

想先看中间结果：`ITER_CAP=30 bash scripts/run_v3.sh`（每次最多训练 30 轮，可续跑），然后 `STAGES="eval report" bash scripts/run_v3.sh`。

## 结果在哪里

`runs/v3/F8D2000B/`：
- `V3_REPORT_validation.md`：成功率、航程误差、超限、构型切换次数，学习相对起点规则的提升，任务域地图统计；
- `V3_domain_map.png`：成功率地图；
- `train/rel_s17/state.json`：训练状态；`study_v2/<集合>/*.json`：逐用例结果；`logs/`：日志。

## 测试集（只运行一次）

验证集结果记录好之后再运行：

```bash
RUN_TEST=1 STAGES="testset" WORKERS=5 bash scripts/run_v3.sh
```

## 提交结果

```bash
git checkout -b target-rl-v3-results-$(date +%Y%m%d) && git add runs/v3 && git commit -m "v3 results" && git push -u origin HEAD
```

## 代码改动一览（相对 `target-rl-v2-run`）

| 文件 | 改动 |
|---|---|
| `configs/v3.yaml`（新） | v3 全部设定 |
| `src/target_rl/flight.py` | 真实对象的失配（`plant`）；到末速结束（`mission.end: speed`）；命中判定 |
| `src/target_rl/guided.py` | 预测用名义模型；按目标末速查初始升力（`initial_cl_speed`）；预测特征第二项改为预测剩余时间 |
| `src/target_rl/policy.py` | 时间特征按 `mission.duration_s` 换算；起点规则的刹车条件改为预测多飞 |
| `src/target_rl/train.py` | 决策状态覆盖到 `mission.duration_s` |
| `src/target_rl/region.py`、`config.py` | 新任务的可达区域（每条轨迹经过各末速时的航程）；v3 用自己的区域文件 |
| `src/target_rl/study_v2.py` | 评估的方法列表从配置读（v3 只有强化学习和起点规则） |
| `scripts/run_v3.sh`（新） | 全流程脚本 |
| `tests/test_v3.py`（新） | 4 个新测试；原有 35 个测试不变 |

旧配置（task.yaml、study_a/r、v2）不受影响：新功能只有在 v3 配置里打开才生效。

## v3b：单独的采纳场景 + 用在线升力估计的保护回路

v3（种子 17）的结果：训练场景命中从 72% 升到 86%，验证集没有提高（70.3% → 69.3%），是过拟合。
采纳规则只看同一批 300 个训练场景，策略学成了专门对付这些场景的动作（每次飞行的构型切换从 2 次涨到 18 次）。
另外验证集上 14 个超限全是前 170—544 s 的动压超限：保护回路按名义模型算平衡升力系数，真实升力偏小时比例律会稳定在动压上限之上。

v3b（`configs/v3b.yaml`）的两处修改：

1. **单独的采纳场景：**候选策略要在训练场景**和**单独的采纳场景（1400—1599，200 个，不产生标签）上平均回报都提高、两边超限都不增加，才采纳；几个候选里按采纳场景上的回报选。
2. **保护回路：**平衡升力系数除以在线升力估计，动压目标从 0.95 倍上限改为 0.90 倍。起点规则在 300 个训练场景上：超限 14 → 1，命中 217 → 223（只在训练场景上比较过）。

区域沿用 v3（同一个任务）。运行：

```bash
TARGET_RL_CONFIG=configs/v3b.yaml WORKERS=5 nohup bash scripts/run_v3.sh > run_v3b.log 2>&1 &
```

结果在 `runs/v3b/F8D2000B/`（`V3B_REPORT_validation.md`、`V3B_domain_map.png`）。每轮训练比 v3 多飞 3 × 200 次，约慢一半。
