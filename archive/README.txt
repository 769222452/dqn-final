2026-10-30 清理归档（UTF-8）

retired_assets_20261030.zip 保存被移出的 5 份 CSV、6 个旧模块、2 项旧接口测试，
以及本次修改文件的清理前版本。逐成员 SHA256 已在删除原位置文件前校验。
retired_assets_20261030.json 给出 ZIP 哈希、每个成员哈希及处置类别。
两个归档文件和本说明均纳入 data/SHA256SUMS，正常飞行仍校验其完整性。

旧 data/aero 气动数据与当前 NPZ 不同，属于来源数据，不是现用数据的副本。
p2_common_grid.csv 的数值与 NPZ 一致，但它另有几何来源哈希；此信息完整保留。
298 个 native_geometry.mat 原位保留，外形和变构图可以继续读取。
当前 runtime.json、controller.npz、p2_grid.npz、lookahead 资格证据均未改动。

需要旧文件时，先校验归档，再解压至一个不存在的新目录：
  python -m zipfile -e archive/retired_assets_20261030.zip <新的恢复目录>
不要直接覆盖当前仓库。ZIP 保存的是退休资产和被改文件原版，不是独立运行包。
清理改变源码与配置指纹；历史训练不能直接以新指纹续跑或复用旧缓存。
完整清理前本地快照及数值对照位于：
  D:/v6-mission-morph-clean/_runtime/dqn_final_cleanup_20261030_190117/
