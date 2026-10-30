强化学习候选几何数据库：298 个构型快照（2026-10-15）

一、目录用途
本目录包含项目使用的 298 个候选构型及对应的气动数据、几何参数和校验记录。
candidate_inventory.csv 记录几何参数、体积/厚度比与几何有效性信息；完整性与导出一致性见 manifest.json 和 validation.json。
该文件夹可整体复制到另一位置使用；mat_path 均相对于本文件夹根目录。

二、主要数据
candidate_inventory.csv：298 个构型的几何参数、硬门槛结果、几何哈希、本地 MAT 路径。
action_index.csv：固定的 action_index（0～297）与 geometry_id 对应关系，按 ID 字典序排列。
geometries/<geometry_id>/native_geometry.mat：298 份原始几何文件，逐文件校验身份与 SHA256。
归档成员 data/geometry_subset/aerodynamics/p2_common_grid.csv：107280 条共同网格气动记录，保留无效点与失败原因。
aerodynamics/p2_grid.npz：可直接加载的数值张量，轴为 [构型, Mach, 迎角]，尺寸 [298,18,20]。
归档成员 data/geometry_subset/aerodynamics/equal_lift.csv：32184 条等升力结果，含不可行状态与 298 个构型之间的排名。
benchmarks/：108 点理想选型、最佳固定构型比较与收益定义。
metadata/：网格、参考面积/长度、模型设置、有效性与使用约定。
provenance/：源数据路径/哈希和原始气动、等升力算法代码留档。
figures/：各 Mach–目标 CL 工况下压力阻力最小的构型分区图（PNG/SVG）。
manifest.json、validation.json：文件校验清单和导出核对结果。

三、加载方式
在本文件夹中使用有 NumPy 的 Python 环境：
  python -B load_database.py --verify

Python 程序示例（将本文件夹加入模块搜索路径后）：
  from load_database import load_database
  db = load_database(verify=True)
  ids = db['geometry_ids']
  CL = db['CL']                   # [298,18,20]
  CD = db['CD_pressure']          # [298,18,20]
  valid = db['usable_p2']         # 模型有效、有限 CL/CD、CD>0
  mach = db['mach_grid']
  alpha_deg = db['alpha_deg_grid']

四、有效性与强化学习接入约定
保留了所有共同网格节点，无效气动点没有伪造为零或填补。
model_valid 是原模型标记；usable_p2 是可用气动系数掩码。
screening_eligible 标记 CL>=0.026150265744858465 的数据点；该标记仅供数据分析使用。
不要将此 CL 门槛当成所有任务状态的约束；有效的低升力/负升力数据仍被保留。
Mach 覆盖 3.5～8.25，迎角覆盖 -1～12 度，网格非等距；以 metadata/domain.json 为准。
缺失/无效节点不能跨越插值，不能越界外推。NPZ 提供网格数据，不自带连续查询插值器。
几何参数的定义/单位沿用原始 Teacher，原始输入在 inputs10 与几何 MAT 中保留。
动作编号只标识候选几何，未验证任意两构型之间的连续变形可达性。
本数据库尚未提供变形速率、切换耗能、结构/执行机构、配平或动态转移模型。
该数据快照供飞行气动计算与离线数据库分析使用；飞行代码读取 p2_grid.npz。

五、基准指标与适用范围
1.084360786%：在本数据快照包含的 298 个构型内，逐工况自由选型相对最佳固定构型的理想化减阻指标。
该指标是 108 个 Mach–目标 CL 工况的等权平均压力阻力系数；每个工况允许调整迎角。
固定构型须覆盖全部 108 个工况；本数据快照中有 234 个构型满足这一覆盖条件。
理想化指标不计构型切换成本，也不模拟真实飞行轨迹，不代表航程提升或经验证的实际变形收益。
气动数据来自低阶 P2 压力阻力模型，不包含摩擦阻力和底阻，也不是 CFD 验证结果。

六、2026-10-30 包整理
两个 CSV 导出已无损迁入仓库 archive/retired_assets_20261030.zip。
当前 NPZ、298 份 MAT、几何编号、元数据及验证结果字节均未改变。
manifest.json 校验本目录现存文件；原完整清单与 CSV 同在归档内。
validation.json 保留原导出时的核验结果，不代表 CSV 仍位于本目录。
