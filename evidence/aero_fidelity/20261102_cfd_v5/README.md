# CFD 第五版：底面标记分离，表皮力积分（探索性实验）

冻结源提交：`dc39a7f9e7c3d78f8bc66c660866907759362c24`（aoa-fix）。复用第四版入选外形，未重跑 MATLAB/family/check；未修改冻结参数或判据。

| 项目 | 实际结果 |
|---|---:|
| CFD 工况数 | 29 |
| 运行完成 | 29 |
| 按冻结规则取得收敛证据 | 29 |
| 未收敛 | 0 |
| 并行设置 | NP=7, JOBS=2 |
| 标记设置 | BASE_MARKER=1 |

- [完整中文报告：网格、耗时、残差和诊断](RUN_SUMMARY.md)
- [原始 decision.json](results/decision.json) / [原始 decision.txt](results/decision.txt)
- [逐工况收敛与耗时 CSV](cfd_case_status.csv)
- [README §3f 的 D1–D3 实际值](results/v5_diagnostics.json)
- [逐工况第四版/第五版力系数比较](results/v4_v5_force_comparison.csv)
- [网格标记、单元数和内部内容校验](operations/mesh_verification.json)
- [第四版外形复制哈希核验](operations/family_copy_verification.json)
- [完整原始结果 ZIP](aero_family_v5_dc39a7f_results.zip)
- [ZIP 内逐文件 SHA-256](SHA256SUMS)

ZIP 包含每个工况的 case.json、case.cfg、history.csv、su2.log，以及所有实际产生的报告、网格统计、入选外形、配置和运维记录；不含网格与 restart。全部原始文件在云端保留。

ZIP SHA-256：`03a698fdcec49cf082191d8ae56a2c9872cc05f673df9e18943dde87993bb2c1`。

本版结论为探索性后续实验；第四版主检验仍保持原始记录。求解器收敛、D1–D3 诊断与气动前提判定分别报告，失败工况不剔除，原始 decision 未改写。
