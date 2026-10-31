# 新连续恢复控制器上的策略重新训练

新入口是 `scripts/train_recovery.py`，协议为 `recovery-policy-improvement/1`。原来的 `scripts/train.py` 继续使用原执行器，行为不变。

这是分支优势网络的近似策略迭代：先生成分支续飞标签，再拟合优势网络，最后完整飞行验证候选。不是按单条轨迹进行TD更新的在线DQN。

## 保持与隔离

- 保持19档绝对CL、12秒上层决策、3秒执行、298构型、8秒变构，以及当前已验证连续恢复控制器的动力学、奖励、预算和认证约束。
- 所有名义轨迹、分支和候选均使用新控制器。当前3秒、未来见证及恢复计划都重新认证；只有PASS可以提交。当前步通过后，冻结双网格轨迹和输入，执行提交只验证绑定并使用该轨迹，不再次积分。
- 分支复制飞行状态、变构状态、CL历史、恢复计划、恢复轮次、退出确认计数，以及策略的保持动作和决策时刻。编译函数可以共享，认证PASS不共享。
- 独立重放的分支起点必须与名义轨迹的完整状态、恢复记忆、策略调度和累计奖励精确一致。前缀不一致保留证据并中止该轮，不能据此生成优势标签。
- 旧检查点只能通过 `--initial-policy` 导入策略链。旧收益、标签、认证结果、训练轮数、停止记录不导入。新标签全部使用新控制器重算。
- 未完成分支不冒充完整回报标签；UNKNOWN保留未认证含义。候选只有完成1200秒并取得更高回报才可采纳。
- 保持原优势网络、拟合参数、跨版本标签衰减、候选生成和停止规则。网络拟合按原实现每次重建优化器，不恢复旧DQN的Adam或Replay。

## 从已验证的学习策略开始

以下命令最多进行1轮完整策略改进。每轮包含多条分支和候选飞行，不能把单次1200秒飞行的耗时当作一轮训练耗时。输出目录必须是仓库内的新目录。默认单worker；示例不会自动运行。

在仓库根目录使用已安装 `requirements.txt` 的 Python 3.11 环境运行。单行命令兼容 PowerShell 和 Bash：

```text
python scripts/train_recovery.py --out runs/recovery_retrain_seed17 --model experiments/continuous_recovery/assets/model.npz --initial-policy experiments/supported_recovery/assets/initial_policy.json --seed 17 --iterations 1 --workers 1
```

seed17控制新训练的抽样与网络拟合；初始策略明确来自旧seed101。它只是起点，不属于“从随机权重独立训练”的实验。

如需从常值CL=0.025起步，使用另一个新输出目录，省略 `--initial-policy` 即可。配置和本地代理模型通过 `--config`、`--model` 指定；不会重新拟合Pre/Post。

## 初始化检查与恢复

`--iterations 0` 只创建独立协议、空标签池和输入快照，不飞行、不拟合、不消耗训练轮数。

同一目录重复运行并保持配置/模型/初始策略/算法参数一致即可续训，`--iterations N` 表示本次最多再进行N轮。恢复可以省略 `--initial-policy`，使用该运行保存的初始策略副本。模型、元数据、源码、物理输入、数值软件环境或算法设置变化会拒绝续训。修改代码后应建立新协议运行，不能继续沿用原标签。

默认保留中断尝试、分支、被拒绝候选和认证证据。`--workers` 仅控制任务并发；认证含墙钟预算，机器负载仍可能改变某些认证结果，不能宣称跨负载逐位复现。

## 输出

- `state.json`：当前策略链、当前名义成绩、全部决策前缀、标签池、版本、拟合检查、历史与停止状态。
- `inputs/`、`source_snapshot/`、`provenance.json`：配置、模型及元数据、初始策略和源码快照及SHA256。
- `iter_*/labels.json`：分支有效性、实际执行动作、完整回报和新优势标签。
- `iter_*/candidate_results.json`：候选完成情况、航程、回报、动作分布、变构次数、耗时及诊断。
- 每条飞行的 `decision_*/`：原始状态、认证池、控制器记忆、证书、双网格执行轨迹与实际动作。
- 中断的轮次保留在 `interrupted_attempts/`。失败候选不能替换已通过的当前策略。

## 有限时段真实验证

`scripts/check_recovery_training.py` 是独立检查入口，先运行至48秒的实际恢复状态，再检查两条相同动作续飞及一条改变CL的分支。它不生成训练标签、不拟合网络，也不宣称完成1200秒。

```text
python scripts/check_recovery_training.py --out runs/recovery_training_branch_check --model experiments/continuous_recovery/assets/model.npz --initial-policy experiments/supported_recovery/assets/initial_policy.json
```

检查要求两条相同动作续飞的完整状态、恢复记忆、奖励、控制节点和策略调度精确一致；改变CL时旧恢复计划作废，而且不污染其他分支。

## 当前步认证轨迹复用

执行证据协议为 `continuous-recovery-execution/1`，用于训练、固定CL评估和冻结策略评估。证据在认证通过时生成，包含完整初态、攻角节点、时间网格、构型、CL、控制器见证、模型/源码身份、配置和两套积分轨迹的不可变副本。提交时这些输入必须完全匹配；证据只能使用一次，不能写入检查点后跨进程恢复，也不能跨状态复用。

首次0.1秒与0.5秒独立审计、未来见证、恢复计划审计、约束与UNKNOWN语义不变。取消的仅是提交时对已认证当前3秒的重复积分。`experiment.commit_audited` 只保留为回归对照，不再由生产入口调用。

每步记录 `execution_protocol`、`certified_trajectory_reused`、`actual_execution_reintegrated`、`execution_integration_calls`，并分别记录 `execution_seal_wall_s` 和 `execution_commit_wall_s`。保存证据的开销也计入训练步耗时。

`scripts/check_recovery_execution.py --out <新目录> --all` 用已保存的400段真实飞行输入，对比原重积分提交和新证据提交，核验状态、奖励、双网格轨迹及审计逐位一致，单独统计实际积分调用与耗时。它不重新求解OCP，不训练策略，不把提交环节加速等同于整轮训练加速。

## 随包资产与历史准备工具

`assets/model.npz` 与 `assets/model.json` 是本次实际使用的冻结模型，无需先重新生成教师标签。模型来源及SHA256见
`../supported_recovery/assets_manifest.json`。初始策略是只含旧5阶段参数的轻量快照，不导入旧标签或收益。

重新制作模型的历史工具仍需要原始实验输入：`prepare_continuous_recovery.py` 的旧 `runs/` 默认目录、
`evaluate_continuous_surrogate.py` 的教师上下文，以及 `check_recovery_execution.py` 的原飞行记录均未随包附带。
使用这些工具时须通过其 `--help` 中的输入参数指定实际数据（如 `--teacher-contexts`、`--reuse-labels` 或 `--flight`）；
它们不是运行上述冻结模型训练命令的前置步骤。

`train_recovery.py` 保留原时间窗口策略阶段；带状态—动作证据检查的独立候选试验使用
`scripts/train_supported_recovery.py`，见 `../supported_recovery/README.md`。
