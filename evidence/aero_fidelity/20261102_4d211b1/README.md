# 外形族生成与云端检查记录（2026-11-02）

实验源代码：`aoa-fix` 分支，冻结提交 `4d211b1b9cc282bdf7716c6e03ff5c5156c9d4a7`。
本结果分支以该提交为基础，仅追加实验记录；未修改生成器、选择规则或判据。

## 执行结果

| 设计 Mach | 生成成功 | 生成失败 |
|---|---:|---:|
| 4 | 40 | 0 |
| 6 | 33 | 7 |
| 8 | 32 | 8 |
| 合计 | 105 | 15 |

- 指定单元测试：8/8 通过，2.33 s。
- 原始 120 组扫描全部产生状态记录，4 路 MATLAB，生成阶段墙钟时间 225.266 s。
- “生成成功”表示输出了 `.mat`，不等于通过几何审计。
- 云端 `check` 在底面三角剖分时报 `ear clipping failed (non-simple base polygon)`，异常退出。
- `family_audit.json`、`select.json` 均未生成。**本次没有正式选择判定，不能记作 `NO_ADMISSIBLE_FAMILY`。**
- 未运行 `cfd-mesh`、`cfd-run` 或 `collect`；保留全部失败记录。

## 环境与执行位置

云端无 MATLAB，因此使用项目 `experiments/aero_fidelity/CODEX.md` 第 3 条允许的异机生成方式：
本机 MATLAB R2023a（9.14.0.2206163），Optimization 与 Symbolic Math 工具箱许可证验证通过；
调用原始 `make_family.m` / `upstream_matlab/source/matlab/active/build_geometry_core.m`。
原始生成文件经 SHA-256 校验后上传云端检查。

云端 Python 3.11.10、SU2 8.5.0（conda-forge）、Open MPI 5.0.11、gmsh 4.15.2。
未设置 OpenMP 线程限制时 `nproc=80`；容器 CPU 配额相当于 10 核。
原脚本设置 `OMP_NUM_THREADS=1` 后输出的 `cores: 1` 不代表主机核心数。
参数保持 `NP=7 JOBS=2 SLICES=4`，本次未启动 MPI/CFD。

## 文件入口

- [完整中文报告](RUN_SUMMARY.md)
- [机器可读汇总](RUN_SUMMARY.json)
- [全部120组结果表](family_status.csv)
- [15个生成失败的详细记录](family_failures.json)
- [check完整原始输出](operations/check.log)
- [环境与版本](operations/environment.txt)
- [MATLAB日志、120份状态与105个原始外形](family/)
- [完整原始结果ZIP](aero_family_4d211b1_check_failed.zip)
- [原始包内259个文件的SHA-256清单](SHA256SUMS)
- [下载校验记录](DOWNLOAD_VERIFIED.json)
- [本GitHub结果目录的SHA-256清单](UPLOAD_SHA256SUMS)

原始 ZIP SHA-256：
`1a7c0295ab3c94e69d97d539aecfc9ffa49f93e5c602558a374ba8bb94694519`

原始 ZIP 和原始文件逐字节保留。本目录通过 `.gitattributes` 禁用换行转换以保持哈希一致。
`README.md`、`.gitattributes`、`UPLOAD_SHA256SUMS` 为上传时新增的索引和校验文件。
