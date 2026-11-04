# 外形族实验前半段结果

- 分支：aoa-fix；提交：`4d211b1b9cc282bdf7716c6e03ff5c5156c9d4a7`。
- 指定测试：8 项全部通过（2.33 s）。
- 云端：Python 3.11.10、SU2 8.5.0（conda-forge）、Open MPI 5.0.11、gmsh 4.15.2。
- 云端无 MATLAB；按 CODEX.md 第 3 条允许的方式，在本机 MATLAB R2023a（9.14.0.2206163）生成，Optimization / Symbolic Math 工具箱许可证均通过验证，再将原始数据传回云端运行 check。
- 云端 nproc（未设置 OpenMP 线程限制时）=80；容器 CPU quota/period=1000000/100000，相当于 10 核，内存上限 16642998272 字节。原脚本设置 OMP_NUM_THREADS=1 后显示 nproc=1，不能视为主机核心数。
- 参数保持 NP=7、JOBS=2、SLICES=4；本次仅启动 4 路本机 MATLAB，未运行 MPI/CFD。
- 120 组全部有记录：生成成功 105 组，生成失败 15 组；外形生成阶段墙钟耗时 225.266 s（3.754 min）。四个 MATLAB 进程均正常退出。成功输出的非有限坐标数均为 0。
- 完整失败条目在 `family_failures.json`，全部任务在 `family_status.csv`。

| 原始错误信息 | 组数 |
|---|---:|
| `Stream surface 1: Invalid isentropic streamline [V5:GeometryDomain]` | 5 |
| `Stream surface 3: Invalid shock streamline [V5:GeometryDomain]` | 5 |
| `Stream surface 1: Invalid shock streamline [V5:GeometryDomain]` | 5 |

## check 结果

原始 check 在闭合底面三角剖分时异常退出，尚未写出任何正式选择判定。`family_audit.json` 和 `select.json` 未生成，`family_selected/` 不存在。
这不是 `NO_ADMISSIBLE_FAMILY` 判定；不能据此宣称不存在满足体积规则的外形族。
没有修改生成器、扫描值、选择规则或 ±3% 容许，没有重跑、跳过失败输入或运行 cfd-mesh / cfd-run / collect。

完整输出：

```text
Traceback (most recent call last):
  File "/hy-tmp/dqn-aero/experiments/aero_fidelity/family/check_family.py", line 105, in <module>
    sys.exit(main())
             ^^^^^^
  File "/hy-tmp/dqn-aero/experiments/aero_fidelity/family/check_family.py", line 58, in main
    r.update(audit(d/'native_geometry.mat'))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/hy-tmp/dqn-aero/experiments/aero_fidelity/family/check_family.py", line 37, in audit
    V, T = cfd.closed_surface(mat)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/hy-tmp/dqn-aero/scripts/cfd/make_su2_cases.py", line 136, in closed_surface
    for t in _base_triangles(loop[:, 1:]):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/hy-tmp/dqn-aero/scripts/cfd/make_su2_cases.py", line 101, in _base_triangles
    return _ear_clip(poly)
           ^^^^^^^^^^^^^^^
  File "/hy-tmp/dqn-aero/scripts/cfd/make_su2_cases.py", line 74, in _ear_clip
    raise RuntimeError('ear clipping failed (non-simple base polygon)')
RuntimeError: ear clipping failed (non-simple base polygon)
STOP: family selection failed (see runs/aero_fidelity/family_selected/family_audit.json)
```
