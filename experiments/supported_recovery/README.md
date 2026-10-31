# 按状态与动作证据约束的独立策略改进实验

入口：`scripts/train_supported_recovery.py`。协议：`supported-recovery-policy-improvement/1`。
原训练入口、控制器、奖励、物理约束及已有结果不修改。导入策略只作为冻结参考；其旧阶段不追溯增加门控，不能声称整个旧策略已经消除外推。

新增阶段不再用标签时间的 ±48 秒决定是否生效。支持证据必须匹配物理状态、迎角、CL 历史、构型与变构进度、恢复计划及退出计数、认证条件、控制器身份、参考策略、续飞规则、基准动作和候选动作。只有累计奖励和累计变构次数不进入支持键，因为它们不影响后续控制与奖励增量。

首版只启用精确支持，尚未校准的近邻半径会被拒绝。它可能在首次轨迹偏离后大量回退，所以必须报告实际命中数、阻止的提议和首次改动时刻；不能把支持不足称为收敛或泛化改善。

## 标签与执行

每次查询从同一个完整快照分别续飞基准请求与候选请求。首个 12 秒请求不同，其后采用同一冻结续飞规则，分别飞到绝对任务时间 1200 秒。两臂回报都重新计算，不借用名义轨迹的剩余回报。任一臂未完成则没有训练标签，UNKNOWN 不填零、不伪造负收益。

默认续飞方式为原来的 open-loop 名义请求时间表；可显式选 closed-loop。这个选择计入实验与标签身份。open-loop 优势不是闭环参考策略 Q 值。标签含义是经过原认证器的 CL 请求，执行可能被安全回退替换，日志分别保存 requested 和 actual executed，不能将请求当作已实际执行的物理 CL。

只拟合当前参考版本标签，不混入旧控制器或旧参考版本的优势。相同网络输入/动作若出现不同基准或相互冲突的目标，直接报告别名冲突，不静默平均。原四输入、隐藏层、拟合更新数及均值减标准差选择规则保留；完整控制记忆用于支持门控。均值减标准差仍只是预测启发式。

主动补标使用冻结候选实际访问状态中被门控拒绝的请求。补标期间不改变正在评估的网络；重新拟合后重新开始完整候选飞行。最晚一次同策略复飞若失败，不能用早先成功结果绕过它。

## 命令

所有输出目录必须不存在。以下为一轮小预算机制验证，最多 4 个成对查询（其中 1 个主动查询），即最多 8 条剩余时域续飞；另有 1 条完整参考飞行和最多 3 条完整候选飞行。它不是原默认预算的收敛实验。

在仓库根目录使用已安装 `requirements.txt` 的 Python 3.11 环境运行。以下单行命令兼容 PowerShell 和 Bash：

```text
python scripts/train_supported_recovery.py --out runs/supported_recovery_seed17_one_iteration --model experiments/continuous_recovery/assets/model.npz --initial-policy experiments/supported_recovery/assets/initial_policy.json --seed 17 --pairs 4 --active-pairs 1 --candidates 3 --workers 1
```

脚本每次仅做一轮，不自动追加，不覆盖现有运行目录。默认 `pairs=25`、`active_pairs=5`、`candidates=14`，主动查询包含在总查询数中；失败尝试也扣预算。运行前预留下一批完整证据的磁盘空间及 1.5 GB 余量，不删除文件腾空间。

有限真实分支检查使用 `scripts/check_supported_recovery.py --out <新目录> --model <模型> --initial-policy <初始策略>`。该检查只跑至 60 秒，生成的成对结果永远不是完整训练标签。

```text
python scripts/check_supported_recovery.py --out runs/supported_recovery_check --model experiments/continuous_recovery/assets/model.npz --initial-policy experiments/supported_recovery/assets/initial_policy.json
```

随包模型是本次实际使用的冻结控制器，初始策略来自原 seed101，包含5个已学阶段。seed17用于本次新阶段的拟合；上述命令不表示从随机策略起步。
原始字节的来源和SHA256见 `assets_manifest.json`；模型及证据中的旧本机路径仅记录来源，不需要访问这些路径。

## 已完成的小预算试验

2026-10-31的一轮试验共4组成对查询、8条有效标签（含4条基准零优势）及3次候选完整飞行，耗时4691.9583秒。
三条候选均完成1200秒，航程2501.053414513093 km、回报1.1421055893804557、10次变构，与参考一致。
新增阶段均改动作0次；原有收益门槛也没有被任何提议满足。因此未采纳候选，不能把日志中“缺少支持的提议数”当作实际阻止改动作的次数。
这是一轮机制验证，未重新训练原5个阶段，也不证明性能提升或收敛。

`evidence/validation_result.json` 保存结果核对、479项已通过的单元测试记录及82个运行源码文件的哈希一致性检查；
`evidence/branch_validation.json` 保存两类真实起点的有限分支检查。完整认证轨迹仍在原本地运行目录，不纳入Git。

## 证据与边界

- `state.json`：当前阶段、输入/源码身份、冻结参考、全部当前标签、新策略、拟合诊断和已扣预算。运行中也保存未完成候选请求文件索引。
- `*/candidate_requests.json`：启动飞行前保存完整网络权重及支持表；不能只保存策略哈希。
- `baseline/`、`nominal_pairs/`、`active_pairs/`、`*_candidates/`：完整状态、恢复记忆、请求/实际动作、认证、双网格轨迹、支持原因及回报。
- `summary.json`：参考与最终航程、收益差、实际预算、支持阻止数和真实墙钟时间。最终策略只接受完整 1200 秒且收益严格提高的候选。

没有有效标签时以 `NO_COMPLETE_PAIRED_LABELS` 停止，学习轮数为 0；既有参考仍保存。一轮结束只表示预算到达，不证明收敛、局部半径有效或独立初态泛化改善。
