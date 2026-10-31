# 云端最终结果对应的原始项目源码

本分支直接保存 `archive_v14_final.tgz` 中 `v14_final/v16_b16/repo/` 的完整项目源码和输入数据，对应云端七个种子（17、23、41、59、83、97、101）的最终实验及 held-out-3 评估。

## 来源与核验

- 原云端路径：`/hy-tmp/v14_final/v16_b16/repo`。
- 本地归档路径：`D:/cloud_results/v14_final/v16_b16/repo`。
- 原项目 388 个文件逐字节保留；本次上传未修改代码、配置、代理模型和气动数据，也未重新训练或评估。
- 原项目文件清单 SHA-256：`5d36ff669901a717eda424f055193d1090d3d311a63e41b9ed3e29d4d79fe526`，与当时预登记记录一致。
- 本次另附本说明及 `cloud_provenance/v16_b16/` 来源材料；它们不属于原始 388 个文件。

原始说明见 [README_zh.md](README_zh.md)。登记、运行命令和历史结论分别见 [PREREGISTRATION.md](cloud_provenance/v16_b16/PREREGISTRATION.md)、[run_final.sh](cloud_provenance/v16_b16/run_final.sh)、[RESULTS.md](cloud_provenance/v16_b16/RESULTS.md)。`run_final.sh` 是原实验目录的执行记录，需要原归档中的策略及目录布局，不能在本分支根目录直接执行。

## 对应结果

完整结果、训练后策略、评估数据及日志保存在另一个分支的原始压缩包中：

- [结果分支及下载说明](https://github.com/769222452/dqn-final/tree/cloud-results-v14-final-20261031/cloud_results/v14_final)
- [下载 archive_v14_final.tgz](https://media.githubusercontent.com/media/769222452/dqn-final/cloud-results-v14-final-20261031/cloud_results/v14_final/archive_v14_final.tgz)

解压后的 `v14_final/v16_b16/runs/seed*/state.json` 是对应训练策略；`v14_final/v16_b16/eval/` 保存冻结评估。压缩包中上一级 `v14_final/repo/` 是较早的对照版本；本分支对应的是 `v14_final/v16_b16/repo/`。

## 获取这份源码

```bash
git clone --single-branch --branch cloud-v16-original-20261031 https://github.com/769222452/dqn-final.git dqn-cloud-v16
```

本分支是产生已归档云端结果的历史源码快照。后续本地修改保存在其他分支。
