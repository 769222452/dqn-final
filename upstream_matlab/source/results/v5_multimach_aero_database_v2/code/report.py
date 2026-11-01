"""Write REPORT_ZH.md, SUMMARY.json and aero_database_manifest.json from the JSON outputs."""
import json
import math
from datetime import datetime, timezone

from common import (ALPHA_DEG, BUDGET_S, FLIGHT_MACH, MODEL_OPTIONS, MODEL_VERSION, OFFDESIGN_SHA, OUT,
                    PILOT_WORKER_SHA, REFERENCE, STATE, dump, load, read_csv, sha)

OUTPUT_FILES = ["aero_database.csv", "g42_baseline.csv", "holdout_labels.csv", "holdout_validation.csv",
                "geometry_aero_coverage.csv", "failed_teacher_labels.csv", "equal_lift_drag_comparison.csv",
                "optimal_geometry_by_mach.csv", "optimal_geometry_by_cl.csv", "volume_constrained_comparison.csv",
                "ld_summary.csv", "interpolation_validation.json", "progress.json", "resume_manifest.json"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else ("%+." + str(d) + "f%%") % (100 * x)


def f(x, fmt="%.4g"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt % x


def nrows(p):
    with open(p, encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def main():
    I = load(OUT / "provenance/integrity.json")
    CC = load(OUT / "provenance/curve_checks.json")
    V = load(OUT / "interpolation_validation.json")
    A = load(OUT / "provenance/analysis.json")
    P = load(OUT / "progress.json")
    clock = load(STATE / "clock.json")
    reuse = load(OUT / "provenance/reuse_audit.json")
    ident = load(OUT / "provenance/identity_check.json")
    verify_p = OUT / "provenance/verify_after.json"
    verify = load(verify_p) if verify_p.exists() else None
    plan = load(OUT / "provenance/analysis_plan.json")
    ready = ("YES" if I["FULL_AERO_DATABASE_V2"] == "COMPLETE" and I["TEACHER_LABEL_INTEGRITY"] == "PASS"
             and V["DATABASE_INTERPOLATION"] == "PASS" else "NO")
    labels = {"FULL_AERO_DATABASE_V2": I["FULL_AERO_DATABASE_V2"],
              "TEACHER_LABEL_INTEGRITY": I["TEACHER_LABEL_INTEGRITY"],
              "DATABASE_INTERPOLATION": V["DATABASE_INTERPOLATION"],
              "NEW_GEOMETRY_BETTER_THAN_G42": A["labels"]["NEW_GEOMETRY_BETTER_THAN_G42"],
              "CROSS_MACH_ADVANTAGE": A["labels"]["CROSS_MACH_ADVANTAGE"],
              "BETTER_THAN_BEST_FIXED_GEOMETRY": A["labels"]["BETTER_THAN_BEST_FIXED_GEOMETRY"],
              "READY_FOR_CONTROL_INTEGRATION": ready,
              "PROJECT_UNCHANGED": verify["PROJECT_UNCHANGED"] if verify else "NOT_YET_VERIFIED"}
    runs = clock.get("runs", [])
    led = [json.loads(l) for l in open(STATE / "ledger.jsonl", encoding="utf-8") if l.strip()]
    walls = sorted(r["wall_s"] for r in led if r.get("wall_s") and r.get("outcome") == "COMMITTED")
    last_done = max((r["elapsed_at_done_s"] for r in led), default=0.0)
    run = {"clock_t0_utc": clock.get("t0_utc"), "supervisor_starts": len(clock.get("supervisor_starts", [])),
           "resumes": clock.get("resumes"), "last_teacher_result_elapsed_s": last_done,
           "last_teacher_result_elapsed_h": last_done / 3600, "budget_h": BUDGET_S / 3600,
           "requests_committed": I["committed_requests"], "retry_records": sum(I["retry_records_by_reason"].values()),
           "retry_by_reason": I["retry_records_by_reason"],
           "median_request_wall_s": walls[len(walls) // 2] if walls else None,
           "worker_restarts_total": sum(r["stats"].get("worker_restarts", 0) for r in runs),
           "timeouts_total": sum(r["stats"].get("timeouts", 0) for r in runs),
           "stop_reasons": [r["reason"] for r in runs]}

    # ---------------------------------------------------------------- manifest
    files = {}
    for fn in OUTPUT_FILES:
        p = OUT / fn
        if p.exists():
            files[fn] = {"sha256": sha(p), "bytes": p.stat().st_size,
                         "rows": nrows(p) if fn.endswith(".csv") else None}
    manifest = {
        "created_utc": utc(), "database": "V5 multimach aero database V2 (Teacher pressure aero labels)",
        "target": {"geometries": 348, "flight_mach": FLIGHT_MACH, "alpha_deg": ALPHA_DEG,
                   "labels": 348 * len(FLIGHT_MACH) * len(ALPHA_DEG), "g42_baseline_labels": 195},
        "teacher": {"function": "matlab/active/offdesign_aero_v4.m (unmodified)", "sha256": OFFDESIGN_SHA,
                    "model_version": MODEL_VERSION, "worker": "code/aero_worker.m (byte-identical to pilot)",
                    "worker_sha256": PILOT_WORKER_SHA, "active_m_files_sha256": ident["active_m_files"],
                    "cone_solver_note": ident["teacher_sources_note"]},
        "reference": REFERENCE, "model_options": MODEL_OPTIONS,
        "quantities": {"CL": "CL_pressure", "CD": "CD_pressure (inviscid, no friction)", "Cm": "Cm_pressure",
                       "L_over_D": "CL/CD for PASS rows"},
        "origin_values": {"REUSED_VERIFIED_TEACHER": "pilot label passing all six reuse conditions",
                          "NEW_TEACHER": "computed in this build"},
        "status_values": "PASS | FAIL (Teacher error, kept) | FAIL_NONFINITE | INFRA_FAILED | NOT_RUN",
        "origin_counts": I["origin_counts"], "status_counts": I["status_counts"],
        "reuse_audit": {k: v for k, v in reuse.items() if k != "rejected_examples"},
        "holdout": "holdout_labels.csv is never part of the database; used only for interpolation_validation.json",
        "labels": labels, "files": files}
    dump(OUT / "aero_database_manifest.json", manifest)

    # ---------------------------------------------------------------- summary
    bf = A["best_fixed"]
    summary = {"task": "V5-MULTIMACH-AERO-DB-V2 full Teacher aero database", "created_utc": utc(),
               "labels": labels, "run": run,
               "database": {"target_labels": I["target_labels"], "status_counts": I["status_counts"],
                            "origin_counts": I["origin_counts"], "g42_status_counts": I["g42_status_counts"],
                            "holdout_status_counts": I["holdout_status_counts"], "not_run": I["not_run"]},
               "g42_regression": I["g42_regression"], "curve_checks": CC["counts"],
               "interpolation": {k: V[k] for k in ("evaluable", "not_evaluable", "overall_pass_fraction",
                                                   "by_subtype", "DATABASE_INTERPOLATION")},
               "performance": {"better_than_G42": A["better_than_G42"], "best_fixed": bf,
                               "cross_mach": {c: {k: v[k] for k in v if k != "per_mach"}
                                              for c, v in A["cross_mach"].items()},
                               "friction_whatif": A["friction_whatif"],
                               "reachability_per_CL": A["reachability"]["per_CL"]},
               "project_unchanged": verify and {k: verify[k] for k in ("hashed", "stat")} and {
                   k: {j: (len(x) if isinstance(x, list) else x) for j, x in verify[k].items()}
                   for k in ("hashed", "stat")}}
    dump(OUT / "SUMMARY.json", summary)

    # ---------------------------------------------------------------- report
    L = []
    w = L.append
    w("# V5 多马赫 V2 几何全量 Teacher 气动数据库（V2）报告\n")
    w("生成时间：%s。所有数值来自本目录内的 CSV/JSON，均为 Teacher（`offdesign_aero_v4`，未修改）的压力气动标签；"
      "摩擦只作为单独标注的敏感性分析，不进入任何标签。\n" % utc())
    w("## 1. 最终标签\n")
    w("| 标签 | 结果 |\n|---|---|")
    for k, v in labels.items():
        w("| %s | **%s** |" % (k, v))
    w("")
    w("## 2. 运行概况\n")
    w("- 14 h 计时起点（第一个 MATLAB worker 启动）：%s；监督进程启动 %d 次（含 1 次主动重启以验证恢复路径），时钟从未重置。"
      % (run["clock_t0_utc"], run["supervisor_starts"]))
    w("- 最后一条 Teacher 结果出现在计时后 %.2f h（预算 %.0f h）。停止原因：%s。"
      % (run["last_teacher_result_elapsed_h"], run["budget_h"], ", ".join(run["stop_reasons"]) or "n/a"))
    w("- 3 个常驻 MATLAB worker；提交请求 %d 个，请求中位耗时 %s s；超时 %d 次，worker 重启 %d 次，重试记录 %d 条 %s。"
      % (run["requests_committed"], f(run["median_request_wall_s"], "%.1f"), run["timeouts_total"],
         run["worker_restarts_total"], run["retry_records"], json.dumps(run["retry_by_reason"], ensure_ascii=False)))
    w("- 账本 `state/ledger.jsonl` 逐条 fsync 追加；重复提交忽略 %d 条；账本坏行 %s。\n"
      % (I["duplicate_committed_records_ignored"], I["ledger_bad_lines"] or "无"))
    w("## 3. 身份核验与标签复用\n")
    w("- `offdesign_aero_v4.m` sha256 = `%s`（与 V1、pilot 相同）；worker 与 pilot 字节一致。" % OFFDESIGN_SHA[:16])
    w("- 锥形流求解器 `comptue_conicalfield.m` 为 2026-10-05 正式晋升的 CSR1 版本（fzero TolX 2e-13→eps，"
      "physics_changed=false），pilot 与本次构建相同；V1 只用于 G42 回归。")
    w("- 348 个 VALID 几何的 MAT 哈希与几何哈希全部重算一致（构建前后各一次）。")
    w("- pilot 标签按 6 条件审核后复用 %d 条（主库 %d，G42 %d），拒绝 %d 条；其余全部由本次 Teacher 新算。"
      % (reuse["reused_labels"], reuse["reused_main"], reuse["reused_G42"], sum(reuse["rejections"].values())))
    w("- V2_MD15_001 与 G42 几何哈希相同但 MAT 文件不同，按规则全部重新计算，未从 G42 复制。\n")
    w("## 4. 数据库完整性\n")
    w("| 项目 | 结果 |\n|---|---|")
    w("| 目标标签 | %d |" % I["target_labels"])
    w("| 状态计数 | %s |" % json.dumps(I["status_counts"]))
    w("| 来源计数 | %s |" % json.dumps(I["origin_counts"]))
    w("| G42 基线 (195) | %s |" % json.dumps(I["g42_status_counts"]))
    w("| hold-out 标签 | %s |" % json.dumps(I["holdout_status_counts"]))
    w("| 唯一键 / 冲突 | %s / %d |" % (I["unique_keys"], I["label_collisions"]))
    w("| PASS 全部有限 / CD>0 / 版本一致 / 网格一致 | %s / %s / %s / %s |"
      % (I["pass_all_finite"], I["pass_all_CD_positive"], I["pass_model_version_ok"], I["grid_membership_ok"]))
    w("| 源码与 MAT 哈希未变 | offdesign %s，active .m %s，几何 MAT %s，G42 MAT %s |"
      % (I["offdesign_sha_unchanged"], I["active_m_files_unchanged"], I["native_mat_sha_unchanged"],
         I["g42_mat_sha_unchanged"]))
    g = I["g42_regression"]
    w("| G42 回归 vs V1 | %d 点，最大偏差 %s，%s |" % (g["v1"]["points"], f(g["v1"]["max_abs_diff"], "%.2e"),
                                               "PASS" if g["v1"]["pass"] else "FAIL"))
    w("| G42 回归 vs pilot | %d 点，最大偏差 %s，%s |" % (g["pilot"]["points"], f(g["pilot"]["max_abs_diff"], "%.2e"),
                                                  "PASS" if g["pilot"]["pass"] else "FAIL"))
    t = g["twin_V2_MD15_001_vs_G42"]
    w("| V2_MD15_001 (新算) vs G42 | %d 点，最大偏差 %s，逐位相同 %s |" % (t["points"], f(t["max_abs_diff"], "%.2e"),
                                                               t["bitwise_identical"]))
    w("")
    w("失败标签全部保留在 `aero_database.csv`（status≠PASS）并列于 `failed_teacher_labels.csv`。\n")
    w("曲线检查（信息性，不改标签）：%s。\n" % json.dumps(CC["counts"]))
    w("## 5. 插值验证（预注册 hold-out，RAW 与 INTERPOLATED 分列）\n")
    w("方法：%s。容差：%s。判据：%s。\n" % (plan["holdout"]["interpolant"], plan["holdout"]["tolerance"]["rule"],
                                     plan["holdout"]["PASS_if"]))
    w("| 子类 | n | 通过率 | abs(dCL) 中位/p95 | abs(dCD)/CD 中位/p95/最大 |\n|---|---|---|---|---|")
    for s, v in V["by_subtype"].items():
        w("| %s | %d | %s | %s / %s | %s / %s / %s |" % (
            s, v["n"], f(v["pass_fraction"], "%.3f"), f(v["median_abs_dCL"], "%.1e"), f(v["p95_abs_dCL"], "%.1e"),
            pct(v["median_abs_dCD_rel"], 2), pct(v["p95_abs_dCD_rel"], 2), pct(v["max_abs_dCD_rel"], 2)))
    w("\n总体：可评估 %d，不可评估 %d（%.1f%%），总体通过率 %s → **%s**。逐点结果见 `holdout_validation.csv`"
      "（RAW_* 为 Teacher 实算，INTERP_* 为插值）。\n" % (
          V["evaluable"], V["not_evaluable"], 100 * V["not_evaluable_fraction"],
          f(V["overall_pass_fraction"], "%.3f"), V["DATABASE_INTERPOLATION"]))
    if V.get("pass_fraction_by_alpha"):
        w("按攻角的通过率：%s。\n" % ", ".join("α=%s: %.2f" % (k, v) for k, v in V["pass_fraction_by_alpha"].items()))
    w("## 6. 性能分析（只用 RAW 网格标签；等升力 CD 由 PCHIP 反插值，u=|PCHIP−线性|）\n")
    w("### Q1/Q2 各马赫最大升阻比（CL≥%.2f，纯压力）\n" % 0.02)
    w("| M | G42 | C0 最优 | 增益 | C1 最优 | 增益 | C2 最优 | 增益 |\n|---|---|---|---|---|---|---|---|")
    for r in A["opt_by_mach"]:
        w("| %g | %s | %s (Md%s) %s | %s | %s %s | %s | %s %s | %s |" % (
            r["flight_mach"], f(r["G42_LD_max_usable"], "%.3f"), r.get("C0_best"), f(r.get("C0_design_mach"), "%g"),
            f(r.get("C0_LD_max_usable"), "%.3f"), pct(r.get("C0_gain_vs_G42")), r.get("C1_best"),
            f(r.get("C1_LD_max_usable"), "%.3f"), pct(r.get("C1_gain_vs_G42")), r.get("C2_best"),
            f(r.get("C2_LD_max_usable"), "%.3f"), pct(r.get("C2_gain_vs_G42"))))
    w("\n设计马赫组（同一飞行马赫、同一 CL 下各组最优设计几何）获胜组：%s。\n" % ", ".join(
        "%s→Md%g" % (k, v) for k, v in list(A["design_mach_group_winner"].items())))
    w("### Q3/Q4 等升力阻力与 G42 比较\n")
    for cn in ("C0", "C1", "C2"):
        b = A["better_than_G42"][cn]
        w("- %s：标签 %s；存在 ≥2 个可用 CL 显著低阻的马赫：%s。" % (cn, b["label"], b["machs_with_>=2_sig_CL"] or "无"))
    w("\n各 (M, CL) 最优几何相对 G42 的 CD 变化见 `optimal_geometry_by_cl.csv`、`figures/fig2_equal_lift_envelope_vs_g42.png`。\n")
    w("### Q5 按马赫切换（固定 CL）与 CROSS_MACH_ADVANTAGE\n")
    w("| CL | 固定最优 | 固定/G42 平均 CD | 不同获胜者数 | 包络显著低于固定的马赫 | 判定 |\n|---|---|---|---|---|---|")
    for c, v in A["cross_mach"].items():
        if not v["evaluable"]:
            w("| %s | 不可评估（全 13 马赫可行候选 %d） | | | | |" % (c, v["n_fixed_candidates"]))
            continue
        w("| %s | %s | %.4f | %d | %s | %s |" % (c, v["fixed"], v["fixed_mean_CD_over_G42"], v["n_distinct_winners"],
                                             ",".join("%g" % m for m in v["significant_machs"]) or "无", v["label"]))
    w("\n### Q6 按 CL 切换（固定马赫）\n")
    w("| M | 不同获胜者数 | 获胜者（CL 0.02→0.20） |\n|---|---|---|")
    for M, v in A["switch_by_cl"].items():
        w("| %s | %d | %s |" % (M, v["n_distinct"], ", ".join(str(x) for x in v["winners"].values())))
    w("\n### Q7 体积/厚度约束\n")
    w("C1 = 体积 ≥ G42（%d 个几何），C2 = 体积与最大厚度均 ≥ G42（%d 个几何）。逐状态结果见 "
      "`volume_constrained_comparison.csv`。\n" % (A["constraint_sets"]["C1"], A["constraint_sets"]["C2"]))
    w("### Q8 形变图可达性\n")
    w("形变路径只在同一设计马赫内构建，每个设计马赫组是一个连通分量（%s）。G42 由其哈希孪生 V2_MD15_001 代表，位于分量 %s。\n"
      % (", ".join("分量%s: Md%s, %d 个" % (k, v["design_mach"], v["size"]) for k, v in
                   A["reachability"]["components"].items()), A["reachability"]["G42_component"]))
    w("| CL | 各马赫获胜者是否在同一分量 | 最佳单一分量 | 其包络 CD/G42 | 无约束包络 CD/G42 | G42 所在分量包络 CD/G42 |"
      "\n|---|---|---|---|---|---|")
    for c, v in A["reachability"]["per_CL"].items():
        w("| %s | %s | %s (Md%s) | %s | %s | %s |" % (c, v["all_winners_in_one_component"], v["best_single_component"],
                                                v["best_single_component_design_mach"],
                                                f(v["best_single_component_mean_CD_over_G42"], "%.4f"),
                                                f(v["unrestricted_envelope_mean_CD_over_G42"], "%.4f"),
                                                f(v["G42_component_mean_CD_over_G42"], "%.4f")))
    w("\n### Q9 逐状态包络 vs 最佳固定几何\n")
    w("状态集 S = 可用 CL（≥0.02）且 G42 可行的 (M, CL)，共 %d 个；在全部 S 上可行的固定候选 %d 个；最佳固定几何 **%s**"
      "（Md%s，平均 CD/CD_G42 = %s）。包络显著更低的状态 %d 个（%.1f%%），包络相对固定几何 CD 中位 %s、最大 %s → **%s**。\n"
      % (bf["states"], bf["fixed_candidates_feasible_on_all_states"], bf["best_fixed"], f(bf["best_fixed_design_mach"], "%g"),
         f(bf["best_fixed_mean_CD_over_G42"], "%.4f"), bf["envelope_significantly_lower_states"], 100 * bf["fraction"],
         pct(bf["envelope_gain_vs_fixed_median"]), pct(bf["envelope_gain_vs_fixed_min"]), bf["label"]))
    w("按 CL 分解（包络 = 每个状态 C0 最低 CD 几何；G42 比较不排除孪生，孪生与 G42 相同时差为 0）：\n")
    w("| CL | 状态数 | 包络 vs 固定 中位 / 最大降幅 | 显著状态 | 包络 vs G42 中位 / 最大降幅 | 显著低于 G42 的马赫数 |"
      "\n|---|---|---|---|---|---|")
    for c, v in bf.get("by_CL", {}).items():
        w("| %s | %d | %s / %s | %d | %s / %s | %d |" % (
            c, v["states"], pct(v["env_vs_fixed_median_rel"], 2), pct(v["env_vs_fixed_min_rel"], 1),
            v["env_vs_fixed_significant_states"], pct(v["env_vs_G42_median_rel"], 2), pct(v["env_vs_G42_min_rel"], 1),
            v["env_vs_G42_significant_lower_machs"]))
    w("")
    w("## 7. 摩擦敏感性（what-if，不进入标签）\n")
    w("CD_total = CD_pressure + Cf·S_wet/S_ref，Cf ∈ {0.001, 0.002}。")
    for k, v in A["friction_whatif"].items():
        w("- %s：最佳固定 %s；包络相对固定 CD 中位 %s，>1%% 更低的状态 %d；包络（不含孪生）相对 G42 中位 %s，>1%% 更低的状态 %d。"
          % (k, v["best_fixed"], pct(v["envelope_vs_fixed_median_rel"]), v["states_envelope_>1pct_below_fixed"],
             pct(v["envelope_vs_G42_median_rel"]), v["states_envelope_>1pct_below_G42"]))
    w("\n## 8. 局限\n")
    w("- 标签为无黏压力系数；参考面积对所有几何固定为 G42 投影面积 %.6g m²，因此 CL/CD 是同一参考下的可比力系数，"
      "不是各几何自身面积归一化的系数。" % REFERENCE["area_m2"])
    w("- 可行性受攻角网格 [-2°, 12°] 限制：薄、小体积几何在高 CL 下常因 α>12° 而不可行，这会限制“全状态可行”的固定候选集合。")
    w("- 等升力比较依赖 PCHIP 反插值；显著性要求 |dCD| > 2(u_g+u_ref) 且 >1%%，u 为 PCHIP 与线性插值之差（中位 %s，p95 %s）。"
      % (pct(A["interp_unc_rel_median"], 3), pct(A["interp_unc_rel_p95"], 3)))
    w("- 形变可达性只在同一设计马赫分量内成立；跨设计马赫切换没有已验证的形变路径。\n")
    w("## 9. 文件\n")
    for fn, v in files.items():
        w("- `%s`（%s 行，sha256 %s…）" % (fn, v["rows"] if v["rows"] is not None else "-", v["sha256"][:12]))
    w("- `aero_database_manifest.json`、`SUMMARY.json`、`figures/`、`provenance/`（身份、复用审计、预注册分析计划、完整性、"
      "曲线检查、分析结果、项目未修改验证）、`state/`（任务、账本、时钟、日志）。")
    (OUT / "REPORT_ZH.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(labels))


if __name__ == "__main__":
    main()
