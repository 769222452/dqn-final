# 全时域一致CL跟踪的小规模实验

这是旁支独立实验，协议为 `continuous-alpha-uniform-pointwise-tracking/2`。只增加本目录文件，不修改原连续迎角实验、默认训练入口、代理模型或物理配置。

## 设计

1. 删除当前3秒独享的优化预算；每个RK阶段都使用同一逐点CL误差带。带宽0.0001来自上一版CL尺度0.1乘以绝对平方误差预算1e-6的平方根，是本实验精度参数，不是物理极限。
2. 每个规划积分小步有独立非负CL松弛量，作用于该小步全部RK阶段。逐点误差约束是 `|CL-CL_cmd| <= band+slack`，并在全时域按实际时间积分惩罚归一化松弛量。有限L1权重100是预先声明的工程选择；任何松弛都记录，不能宣称松弛必然是安全所必需。
3. 保留全时域CL平方误差和原经济目标权重。保留攻角连续性、原变化率限值与原变化率代价，增加变化率变化的平方代价，包含上一已执行区间的真实斜率和间隔。新增代价复用原控制变化代价权重，乘数1。它不构成C1连续性的硬保证，也不是辨识后的俯仰/执行器模型。
4. 每个Pre/Final只求解一个NLP，保留原IPOPT预算。重用原0.1/0.5秒全规划和当前区间物理审计；未通过不得执行。只对CL误差引入松弛，不放宽动压、热、载荷、状态、攻角、变化率或气动表范围。
5. 保留真实Pre/Final各60秒、每3秒执行、298构型、Generator和8秒变构。末段仍由剩余时间截断。重规划不重置已执行攻角斜率。

## 有界验证

从上一实验保存的0、48、375、489秒状态，比较同一Final-OCP上下文；再从489秒状态开始，真实滚动执行12秒双OCP。预先固定配置，不根据结果临时调参。数据范围不是完整任务，不作为1200秒成功或航程收益证据。

```powershell
python -B -m pytest -p no:cacheprovider experiments/uniform_tracking_v2/test_controller.py
python -B experiments/uniform_tracking_v2/check.py --source runs/continuous_tracking_20261030/full_current_priority --out experiments/uniform_tracking_v2/runs/trial_01
```

冷启动编译和建图时间单独反映在首个冻结案例墙钟时间里，NLP实际求解时间保存在各阶段日志。短闭环耗时与旧保存记录只是同机初步对照，不能替代重复计时试验。
