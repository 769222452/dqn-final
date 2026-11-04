# 原云端 V14 / V16 最终实验结果备份

本目录保存 2026-10-26 从云服务器下载的 `archive_v14_final.tgz` 原件，上传分支为
`cloud-results-v14-final-20261031`。压缩包原始字节不变，大小为 **426,801,390 字节（426.8 MB）**，
通过 Git LFS 保存。它包含 39,404 个文件，解压根目录为 `v14_final/`。

这是原云端实验快照，不包含 2026-10-30 至 31 日的本地连续恢复及状态—动作支持试验。
压缩包内保留当时使用的代码与数据；外层仓库的新实现不能代替原实验源码作为复现口径。

## 在线查看主要结论

- [Held-out-2 原始结果](summaries/heldout2/RESULTS.md)及[预注册](summaries/heldout2/PREREGISTRATION.md)。
- [V16 / held-out-3 最终结果](summaries/v16_b16/RESULTS.md)及[预注册](summaries/v16_b16/PREREGISTRATION.md)。
- [最终判定](summaries/v16_b16/verdict.json)、[逐策略对照汇总](summaries/v16_b16/comparison.json)、
  [测试初态集合](summaries/v16_b16/heldout3_set.json)。

这些是原结果文件的逐字节副本，包含原文中的 PASS 和 FAIL，没有重新计算、删改失败结果或改写结论。
来源路径和SHA256列在 [manifest.json](manifest.json)。

## 下载

先安装 Git 和 [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。
在一个不存在 `dqn-cloud-results` 的目录执行：

```bash
git lfs install
git clone --branch cloud-results-v14-final-20261031 --single-branch https://github.com/769222452/dqn-final.git dqn-cloud-results
cd dqn-cloud-results
git lfs pull --include="cloud_results/v14_final/archive_v14_final.tgz"
```

在已有仓库中则执行：

```bash
git fetch origin cloud-results-v14-final-20261031
git switch cloud-results-v14-final-20261031
git lfs pull --include="cloud_results/v14_final/archive_v14_final.tgz"
```

不要把只有几行 `version`、`oid`、`size` 的 LFS 指针文本当成压缩包；下载后的文件必须是426,801,390字节。
网页的源代码 ZIP 是否包含 LFS 大文件取决于仓库设置，以上 Git LFS 命令可显式取得原文件。

## 校验和解压

SHA256：

```text
1a8d1a38e0153a9349cf77ce97248a6f828c9c176f18c1ffdb2fb50380b99130
```

Linux / Bash：

```bash
cd cloud_results/v14_final
sha256sum -c SHA256SUMS
mkdir extracted
tar -xzf archive_v14_final.tgz -C extracted
```

Windows PowerShell：

```powershell
Set-Location cloud_results/v14_final
Get-FileHash archive_v14_final.tgz -Algorithm SHA256
New-Item -ItemType Directory extracted -ErrorAction Stop
tar -xzf archive_v14_final.tgz -C extracted
```

使用新的解压目录；解压后的文件总大小约1.92 GB。

## 压缩包中的关键位置

| 路径（相对解压根目录） | 内容 |
|---|---|
| `v14_final/RESULTS.md`、`eval/`、`runs/` | Held-out-2 实验及原始运行记录 |
| `v14_final/v16_b16/RESULTS.md`、`verdict.json` | Held-out-3 最终结论 |
| `v14_final/v16_b16/runs/seed*/state.json` | seed17、23、41、59、83、97、101的策略参数与训练历史 |
| `v14_final/v16_b16/eval/` | 泛化评估、各初态及探针的结果 |
| `v14_final/v16_b16/ablation/` | Post控制器消融实验 |
| `v14_final/v16_b16/logs/` | 实验日志 |
| 各运行的 `proofs/` | 逐步认证证据 |
| `v14_final/repo/`、`v14_final/v16_b16/repo/` | 当时的源码与输入快照 |
| `v14_final/v15_single/` | 已撤回、未运行的实验目录，不能视为完成结果 |

本分支只发布现有结果，没有重新训练或追加评估。
