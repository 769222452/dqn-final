# F8D2000：固定 G42 三维前缘的上下表面变形构型库

本目录提供 **2000 个有效且坐标不重复的构型、完整原 P2 气动表及静态等升力选型结果**。上下表面各有四个变形参数，固定 G42 的三维前缘，不约束容积。原始库标识为 `fixed_leading_edge_G42_shape_2000_20261114_v1`。

该库作为独立数据集发布。仓库现有飞行程序仍使用原来的 298 构型数据库；本次上传没有替换训练输入或改变气动公式。

## 文件与数据范围

| 路径 | 内容 |
|---|---|
| `geometries/<ID>/native_geometry.mat` | 全部 2000 个构型的完整双精度上下表面网格 |
| `geometries/<ID>/shape_parameters.json` | 上下表面共八个变形系数，单位 m |
| `geometries/<ID>/certificate.json` | 各构型的几何检查结果及原坐标哈希 |
| `aerodynamics/<ID>.csv` | 每构型 360 行气动表，共 720,000 行计算数据 |
| `aerodynamics/p2_grid.npz` | Python 汇总表，原始文件逐字节保留 |
| `aerodynamics/p2_grid.mat` | MATLAB 汇总表，与 NPZ 的数值及构型顺序完全一致 |
| `action_index.csv` / `selected_ids.json` | 连续动作索引与构型 ID 的对应关系 |
| `geometry_inventory.csv` / `geometry_summary.json` | 全库几何信息与统计 |
| `config.json` / `sampling_attempts.csv` | 参数范围、采样方式及接受/拒绝记录 |
| `references/` | G42、F2K_01233、F2K_00021 三个父构型的原始网格，不计入 2000 个候选 |
| `evaluation/` | 仅使用本库 2000 个候选的选型张量、120 工况选型表及固定构型排名 |
| `validation/` | 原求根版本与快速版本的核对记录、已有直接等升力复核记录 |
| `model/` | 生成气动表所用 P2 求值器及原求根版本，原文件逐字节保留 |
| `examples/` | Python、MATLAB 读取示例及完整性检查脚本 |
| `viewer.html` | 全部 2000 构型的离线三维浏览器，下载后用浏览器打开 |
| `figures/` | 三维外形、表面变化、相对最佳固定构型及 G42 的减阻图，PNG/PDF/SVG |
| `metadata.json` / `provenance.json` / `SHA256SUMS` | 单位、参考量、原文件哈希及本发布包校验值 |

构型编号沿用采样顺序，存在间断，**不能把 ID 后缀减一当作数组索引**。Python/CSV 的 `action_index` 从 0 开始；MATLAB 行索引从 1 开始。

## 几何数据

每个 MAT 文件的 `wav_x, wav_y, wav_z` 为下表面，`wav_up_x, wav_up_y, wav_up_z` 为上表面；六个数组均为 **50×50、float64、单位 m**。网格表示一个对称半翼，另一半由 y 镜像得到。原生坐标 z 向下；P2 求值器内部将坐标乘以 `[1,-1,-1]` 转为其气动机体坐标。

`shape_coefficients_m` 按以下顺序存放八个系数：

`upper_front, upper_rear, upper_tail, upper_span, lower_front, lower_rear, lower_tail, lower_span`

相应绝对值上限为 `[0.03,0.03,0.04,0.025,0.025,0.025,0.035,0.02]` m。正系数使原生 z 坐标向下移动。

参数化使用四个边界保持基函数。令 t 为流向列索引归一化到 [0,1]，c=max(0,5−x_LE)/5，e 为前缘半展向坐标归一化到 [0,1]，则

`B = c * [t²(1−t)³/0.03456, t³(1−t)²/0.03456, t²(3−2t), 16t²(1−t)²(2e²−1)]`。

前缘列和塌缩翼尖行的位移设为零；上下表面各自用其四个系数修改父构型的 z。父构型由 `parent_geometry_id` 指定，`inputs10` 仅表示父构型来源，单独使用它不能重建变形后的外形。读取 `native_geometry.mat` 可直接获得完整原始坐标。

本参数族保留父构型的 x/y 网格、平面外形、5 m 长度和镜面对称。2000 个构型分别来自三个父构型 667、667、666 个；保留早期 178 个有效构型，新增 1822 个。容积只记录、不参与筛选。全库前缘相对 G42 的最大三维误差为 5.661×10⁻¹⁶ m。

几何有效性基于固定 XY 网格的完整投影三角片交叠检查，并检查两种对角线剖分。它说明离散曲面符合已记录的几何条件，尚未验证机构可达性、驱动能力或任意构型对的动态过渡。

## 完整气动表

NPZ 与 MAT 的 `CL, CD_pressure, Cm_pressure, model_valid, usable_p2` 形状均为 **[2000,18,20]**，轴顺序为 **[构型, Mach, 迎角]**。字段 `geometry_ids, mach_grid, alpha_deg_grid` 明确给出对应轴。全部 720,000 个网格点均有效且可用于 P2。

- Mach：`3.5,4,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25`。
- 迎角（度）：`-1,0,1,2,3,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12`。
- `CL` 为升力系数，`CD_pressure` 为表面压强阻力系数，`Cm_pressure` 为求值器定义的压强俯仰力矩系数；系数均无量纲。
- 参考面积 `6.931920786884801 m²`，参考长度 `5 m`；力矩参考点在气动机体坐标中为 `[2.5,0,-0.9481139754931033] m`。
- CSV 还包含脱体与真空区域面积比例；每一行都有 Mach 和迎角，不应凭行号猜测工况。

这里使用用户选定的**原 P2 表面压强模型**，不含底阻、摩擦或变构能耗。快速求值器只改变求根迭代实现，保留原方程和分支。已有原版本核对记录位于 `validation/fast_p2.json` 与 `validation/new_p2_reference.json`。

## 等升力选型结果

`evaluation/equal_lift_new_2000.npz` 的 `CD_pressure` 和 `alpha_deg` 轴顺序为 **[2000 个构型,12 个目标 CL,10 个 Mach]**，与原始气动表轴顺序不同。`winner_index` 为从 0 开始的构型索引。

选型范围为 `M=5.75:0.25:8`、`CL=0.015:0.005:0.070`，共 120 个等权工况。采用有效迎角区间内的 PCHIP 插值求等升力迎角，不外推；存在多个可行根时选择正且有限的最小压强阻力。最佳固定构型须覆盖全部工况。

| 指标 | 结果 |
|---|---:|
| 最佳固定构型 | F8D_01351 |
| 分工况获胜构型数 | 6 |
| 分工况选型平均压强阻力系数 | 0.003511290961308272 |
| 最佳固定构型平均压强阻力系数 | 0.003525778781675314 |
| 相对最佳固定构型的平均减阻率 | 0.410911% |
| 相对外部参考 G42 的平均减阻率 | 8.963618% |

平均减阻率定义为 `100 × (1 − 平均选型 CD / 平均基准 CD)`。这些数值是静态等升力压强减阻结果，不能直接当作航程收益。G42 仅作为参考，不计入候选数。

已有直接等升力复核共 892 点、22 个构型，本次选型组合相对其最佳固定构型的直接复核减阻率为 0.410676%。该复核同时涉及历史旧库比较对象，故 `validation/direct_equal_lift/` 中包含部分外部 ID；它不是 2000 个构型的穷尽直接寻优。两格存在已检查候选与插值赢家次序翻转，最大阻力差 0.013523%。原记录完整保留。

## 读取和检查

Python 需要 NumPy 与 SciPy。在仓库根目录运行：

```bash
python datasets/F8D2000/examples/load_library.py F8D_01351
python datasets/F8D2000/examples/verify_dataset.py
```

MATLAB 在仓库根目录运行：

```matlab
addpath('datasets/F8D2000/examples');
data = load_library('F8D_01351');
surf(data.geometry.wav_x, data.geometry.wav_y, data.geometry.wav_z);
% data.CL, data.CD_pressure, data.Cm_pressure 均为 [18,20]
% 可选：用原模型重新计算这一构型的单个工况（迎角传入弧度）
addpath('datasets/F8D2000/model');
g = data.geometry;
lo = cat(3,g.wav_x,g.wav_y,g.wav_z);
up = cat(3,g.wav_up_x,g.wav_up_y,g.wav_up_z);
result = p2_fast(lo,up,6.5,deg2rad(4));
```

为避免重复保存同一批大数组，发布包以逐构型 MAT 保存完整几何数据，没有再附原来的 175 MB 汇总几何 NPZ。所有 2000 份 MAT 的六个坐标数组已逐项与原汇总数组核对，数值完全一致，没有降采样或降低精度。

![相对最佳固定构型的静态等升力减阻](figures/03_相对最佳固定构型减阻.png)
