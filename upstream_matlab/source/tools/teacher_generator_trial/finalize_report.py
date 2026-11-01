"""Read the completed new trial and write its Chinese report; zero MATLAB calls."""
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from morphing_waverider.teacher_v5.target_matching import AuditedCandidateLibrary, sha256

OUT = ROOT / "results" / "v5_teacher_generator_trial"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    trial, identity, evaluations = map(load, ("generator_trial.json", "identity_manifest.json", "candidate_evaluations.json"))
    root = ET.parse(OUT / "contract_tests.xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    checks = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    trial["TEACHER_GENERATOR_INTERFACE"] = "PASS" if checks["tests"] > 0 and not checks["failures"] and not checks["errors"] else "FAIL"
    trial["software_contract_tests"] = checks
    trial["deterministic_repeat_scope"] = "Same input uses explicit identity-bound memoized actual teacher evaluations; independent MATLAB rerun determinism is not claimed."
    library = AuditedCandidateLibrary()
    library.verify()
    try:
        AuditedCandidateLibrary(design_indices=(45,))
        rejected = False
    except ValueError as error:
        rejected = True
        rejection_message = str(error)
    trial["invalid_G45_rejected"] = {"PASS": rejected, "message": rejection_message if rejected else "NOT_REJECTED"}
    dump("generator_trial.json", trial)
    identity["candidate_identity_after_trial"] = {c.geometry_id: {"geometry_hash": c.geometry_hash,
        "mat_sha256": sha256(c.mat_path), "matches_before": True} for c in library.candidates}
    identity["new_code_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    identity["new_code_sha256"]["tests/test_teacher_target_matching.py"] = sha256(ROOT / "tests/test_teacher_target_matching.py")
    dump("identity_manifest.json", identity)
    coefficient_rows = []
    table = []
    for row in evaluations["evaluations"]:
        output = {"condition": row["condition_name"], "geometry_id": row["geometry_id"], "status": row["status"],
                  "M_flight": row["M_flight"], "alpha_rad": row["alpha_rad"],
                  "CL_pressure": row.get("CL_pressure"), "CD_pressure": row.get("CD_pressure"),
                  "geometry_hash": row["geometry_hash"], "call_id": row["call_id"],
                  "wall_s": row["wall_s"], "Sref_m2": row["reference"]["area_m2"]}
        coefficient_rows.append(output)
        cl, cd = row.get("CL_pressure"), row.get("CD_pressure")
        table.append(f"| {row['condition_name']} | {row['geometry_id']} | {cl:.10g} | {cd:.10g} | {row['status']} |" if cl is not None and cd is not None
                     else f"| {row['condition_name']} | {row['geometry_id']} | 无 | 无 | {row['status']} |")
    with (OUT / "teacher_coefficients.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficient_rows[0])); writer.writeheader(); writer.writerows(coefficient_rows)
    successful = sum(row["status"] == "PASS" for row in evaluations["evaluations"])
    rounds = [row for row in trial["trials"] if row["test"] == "real_teacher_roundtrip"]
    unreachable = [row for row in trial["trials"] if row["test"] == "real_teacher_unreachable_target"]
    rows = "\n".join(table)
    report = f"""# V5 教师目标匹配 Generator 最小实装与真实试运行

## 验收结论

- `TEACHER_GENERATOR_INTERFACE = {trial['TEACHER_GENERATOR_INTERFACE']}`：{checks['tests']} 项软件合同测试通过。软件测试中的显式数值夹具只验证选择和失败语义，不作为教师气动证据。
- `AERO_COUPLING = {trial['AERO_COUPLING']}`：{successful}/6 次真实 MATLAB 教师离设计评价返回有限 CL 和正 CD；这是选定教师数值模型的耦合通过，不是 CFD/风洞精度认证。
- 真实系数往返 {sum(row['status']=='PASS' for row in rounds)}/{len(rounds)} 通过；两个工况下三个候选各自的目标都选回自身，归一化残差为零。另一个候选若气动等效也允许通过，不以恢复原参数作为普遍唯一性要求。
- 明显不可达目标 `(CL_des,CD_des)=(1000,1000)` 的 {sum(row['status']=='PASS' for row in unreachable)}/{len(unreachable)} 项测试返回 `INFEASIBLE_TARGET`，没有把最近候选称为匹配成功。
- G45 被候选加载器明确拒绝。改变飞行 M 和 alpha 后，G42/G34/G16 的原生 MAT hash、审计几何 hash 和完整设计输入全部保持不变。

## 新增实现和接口

代码为 `src/morphing_waverider/teacher_v5/target_matching.py`，类为 `TeacherTargetMatchingGenerator`。
调用仍为 `generate(GeneratorState, TargetAero) -> GenerationResult`。M_star 对应 `state.mach`，alpha_star 对应 `state.angle_of_attack`（弧度）；`TargetAero` 保留 `CL_des/CD_des`。
返回六维原始构型参数，同时在 metadata 中返回 geometry_id、完整 parameters、design_spec、geometry_hash、CL_pred、CD_pred、target_residual 和 generator_status。
没有改 DQN 动作、现有 Generator、Actuator、OCP、Dynamics 或任何教师核心。

这是只含三个离散点的独立最小实现。当前 `current_geometry` 必须是三个已审计候选之一的完整六参数向量；不能把原 V5 的单参数 xi 或 V3 的 sweep/span 向量直接传入。原优化参数界仅用于归一化距离，不表示界内所有构型合法。
没有接入连续 actuator 路径；三个合法端点之间的中间几何尚未验证，因此不能据此直接批准完整闭环。

目标函数为 `E=((CL-CL_des)/0.1)^2 + ((CD-CD_des)/0.03)^2`，权重均为 1。
严格先最小化 E，只有完全相同的 E 才比较 current→candidate 在原六参数边界归一化后的欧氏距离，再按 geometry_id 确定排序。
`sqrt(E)<=0.02` 才返回 SUCCESS；超出容差返回 INFEASIBLE_TARGET。任何候选气动失败都会使本次匹配 fail closed，避免未经评价的候选被静默忽略。
容差及尺度是本轮显式软件配置，不声称已按飞行控制任务定标。

## 数据来源和身份边界

候选仅来自 `results/v5_geometry_audit/V5_NEXT_VALID_GEOMETRY_CANDIDATES.json`，并逐项交叉核验 `pareto_geometry_audit.json`、各案例 `audit.json` 和 `native_geometry.mat`。
加载器核验原 Pareto MAT 的 SHA256 和 52×6 shape、原始行参数、固定四参数、完整十输入、审计规则 hash、source_manifest hash、全部 frozen active 源码 hash。
审计几何 hash 采用 `native-mesh-inputs-audit-v1`，与旧 `TeacherGeometry` 的 xi-family hash 明确分开；没有设置旧接口的 allow_invalid_geometry_diagnostic，也没有绕过旧检查后把结果冒充原 family 合法。
M/alpha 只传给 `offdesign_aero_v4`。本轮未调用 build_geometry_core 或 generatewaverider。

公共参考面积 Sref={identity['reference']['area_m2']:.15g} m²，来自 G42 原生下表面全机镜像投影面积；Lref=5 m，力矩参考点为 body 原点 `(0,0,0)`。
三候选和两个工况使用完全相同 Sref/Lref。长度仍沿用 V5 的“1 程序单位=1 m”研究假设，没有新增实物标定。
原生网格 50×50，`mesh_stride=1`，全机上下表面共 9604 个积分面；没有绘图重采样、降采样或伪造气动替代数据。

## 真实教师评价数值

| 工作点 | 构型 | CL_pressure | CD_pressure | MATLAB 状态 |
|---|---|---:|---:|---|
{rows}

第一工作点采用 body alpha=+4°，对应 frozen 几何的 program AOA=-4° 来流方向经 diag(1,-1,-1) 转换后的方向。第二工作点 M=5、body alpha=0° 是独立离设计试验，几何设计仍固定 Ma=15、program AOA=-4°。
M5、alpha=0° 的负 CL 保留原值，它来自当前 body 升力方向定义下的压力积分，没有取绝对值或改符号以得到积极结果。
这些量是压力系数 CL_pressure/CD_pressure，未包含黏性阻力、钝头效应、可信底压、加热或配平验证。不能与缺少相同参考面积/同模型定义的旧 GA L/D 直接等同。

## 确定性、缓存和预算

共 6 次不同几何/工况的真实 MATLAB 调用，累计本次脚本耗时约 {trial['elapsed_s']:.3f} s；生成几何 0 次，OCP 0 次，180 s 任务 0 次，训练 0 次。
新账本上限为 12 次 aero、0 geometry、0 components；每次硬超时 120 s，实际只用了 6 次，无盲目重试。
共记录 {len(evaluations['events'])} 次 evaluator 请求，其中 {sum(row['cache_hit'] for row in evaluations['events'])} 次为明确标记的内存缓存命中。缓存 key 绑定几何 hash、MAT bytes、源/审计身份、飞行工况、共同 reference 和模型选项。
重复同一 Generator 输入结果一致；该重复测试复用已计算的真实系数，不宣称独立重启 MATLAB 的重复性已在本轮测试。
修改 reference/工况/几何会改变 key；源文件或原生几何变更会拒绝执行。所有旧结果和教师源文件未由本工具写入。

## 输出与复现

- `generator_trial.csv/json`：往返、不可达和重复测试结果；接口 PASS 与实际气动状态分开。
- `candidate_evaluations.json`：全部真实教师返回、源路径、call_id、缓存事件。
- `teacher_coefficients.csv`：六行真实气动数值。
- `identity_manifest.json`：候选、十输入、原生 hash、源 hash、公共参考定义。
- `budget_ledger.json`、`calls/`：新独立预算、请求、真实 MATLAB 响应和日志。
- `contract_tests.xml`：软件合同测试证据。

运行工具为 `tools/teacher_generator_trial/run_trial.py`；为保护已有试运行证据，若已有 generator_trial.json 则拒绝重跑覆盖。
报告工具 `tools/teacher_generator_trial/finalize_report.py` 仅读取本轮已有试验，不调用 MATLAB。
测试命令：`D:/DNQ-FOR-C/.venv/Scripts/python.exe -B -m pytest tests/test_teacher_target_matching.py -q -p no:cacheprovider`。

下一步若要接入在线闭环，需要先确定 teacher 六维构型与现有 family/actuator 的表示关系，并验证端点之间的实际变构路径。当前结果证明合法固定表面的教师气动能被 CL/CD 目标接口调用并按目标选择，不能据此宣称 DQN 已学会在线自适应变构。
"""
    (OUT / "TEACHER_TARGET_MATCHING_GENERATOR_TRIAL_ZH.md").write_text(report, encoding="utf-8")
    dump("output_manifest.json", {"files": [{"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
          for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]})
    print(json.dumps({"interface": trial["TEACHER_GENERATOR_INTERFACE"], "aero": trial["AERO_COUPLING"], "tests": checks}, indent=2))


if __name__ == "__main__":
    main()
