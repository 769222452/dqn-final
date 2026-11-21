# v2 完整实验：变构扩大可靠任务域（本地运行说明）

分支 `target-rl-v2-run`。设计见 [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md)，探索阶段的证据见
[`EXPERIMENT_OVERVIEW_ZH.md`](EXPERIMENT_OVERVIEW_ZH.md)。

**要回答的问题：**
1. 变构把可靠完成的"距离—末速"任务范围扩大了多少（成功率地图、各末速下的最远可靠距离）？
2. 学习得到的策略（变构 / 不变构）相对公平的固定外形基线、相对自己的起点规则，增加了什么？

## 环境

Python ≥ 3.10，只需要 numpy、scipy、PyYAML、pytest、matplotlib（出图）。Windows 用 Git Bash。

```bash
git clone -b target-rl-v2-run https://github.com/769222452/dqn-final.git target-rl-v2 && cd target-rl-v2
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt matplotlib
```

## 先做一次 15 分钟的流程检查

```bash
QUICK=1 WORKERS=4 bash scripts/run_v2.sh
```

小规模跑完测试、训练、评估、报告，结果在 `runs/v2quick/`（数字没有意义，只检查流程；不会污染正式结果）。

## 正式运行

```bash
WORKERS=5 nohup bash scripts/run_v2.sh > run_v2.log 2>&1 &
```

Windows Git Bash 如果提示换行符问题：`bash -o igncr scripts/run_v2.sh`。可续跑：中断后再执行同一条命令即可。

| 阶段 | 内容 | 估计耗时（6 核、5 个进程） |
|---|---|---|
| `test` | 35 个测试 | 1 分钟 |
| `train` | 每个种子训练 `rel`（变构）和 `relfixed`（不变构），最多 120 轮 | 每个变体约 5—9 小时（研究 R 的实际：64 轮 5.3 小时，116 轮 8.5 小时），**每个种子 11—18 小时** |
| `eval` | 验证场景评估（192 个目标 × 各方法）+ 任务域地图（72 点 × 12 场景 × 各方法） | 约 1 小时 |
| `report` | 生成报告和图 | 秒级 |

**建议分两步：**先 `SEEDS=17 WORKERS=5 bash scripts/run_v2.sh`（一个种子，约 12—20 小时）看效果，
再 `SEEDS="17 23" ...` 补第二个种子（已训练的会跳过）。想先看中间结果可以加 `ITER_CAP=30`（每次调用最多训练 30 轮，可续跑），
然后 `STAGES="eval report" bash scripts/run_v2.sh`。

只想看**变构的价值**（不需要学习）：`STAGES="eval report" SEEDS="" bash scripts/run_v2.sh`，约 1 小时，
报告里就有固定外形与变构预测-校正的成功率、任务域地图和远端边界。

## 结果在哪里

`runs/v2/F8D2000B/`：
- `V2_REPORT_validation.md`：主结果表、条件成功率、配对比较与判定、任务域面积和远端边界；
- `V2_domain_map.png`：成功率地图；
- `train/<变体>_s<种子>/state.json`：训练状态；`study_v2/<集合>/*.json`：每个方法的逐用例结果；`logs/`：日志。

## 测试集（只运行一次）

**只有在验证集结果出来、判定记录好之后**才运行：

```bash
RUN_TEST=1 STAGES="testset" WORKERS=5 bash scripts/run_v2.sh
```

不设 `RUN_TEST=1` 时脚本会拒绝运行，防止误用。

## 提交结果

```bash
git checkout -b target-rl-v2-results-$(date +%Y%m%d) && git add runs/v2 && git commit -m "v2 results" && git push -u origin HEAD
```

`runs/cache/` 是可再生的缓存，已在 `.gitignore` 中。

## 代码改动一览（相对 `target-rl-v2`）

| 文件 | 改动 |
|---|---|
| `configs/v2.yaml`（新） | v2 全部设定：目标分布、数据划分、执行层、训练、任务域网格 |
| `src/target_rl/aero.py` | 构型库的"可选阻力档位"表和 `Drag.snap`（就近选真实构型） |
| `src/target_rl/flight.py` | 高度下限保护；就近选构型 + 8 s 过渡（`_morph_exec`）；切换次数；默认关闭，不影响旧配置 |
| `src/target_rl/baselines.py` | `pc_strong_w0`（固定外形基线）；预测内部用连续、瞬时的变构 |
| `src/target_rl/region.py` | v2 目标分布、分层、数据集（train / validation / test / domain） |
| `src/target_rl/study_v2.py`（新） | 方法、评估、任务域地图、配对比较与判定、报告 |
| `scripts/run_v2.sh`（新） | 全流程脚本 |
| `tests/test_v2.py`（新） | 6 个新测试（执行层、目标分布、数据划分、基线）；原有 29 个测试不变 |
