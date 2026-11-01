"""P1a: mission-driven design conditions from the stage-1 feasible fixed trajectories.

Weight of a sample = q * Sref * V * dt = drag work per unit CD on that trajectory
(J). Each trajectory contributes equally. Conditions are (Mach, CL) bins; the
fixed-trajectory weights only rank geometry directions, they are not new ranges.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import BANDS, OUT, REFERENCE, STAGE1, dump, load, read_csv, write_csv

MACH_BIN = 0.25
CL_BIN = 0.005
SCORE_CL_MIN = 0.015   # below: initial pull-down transient, kept in the table, not scored


def band_of(m):
    for k, (lo, hi) in BANDS.items():
        if lo <= m < hi or (k == "M7-8" and abs(m - hi) < 1e-9):
            return k
    return "M4.5-5" if m < 5 else "M>8"


def build():
    comp = load(STAGE1 / "comparison.json")
    feas = [c["geometry"] for c in comp["candidates"] if c.get("status") == "BEST_FOUND_FIXED_TRAJECTORY"]
    sref = REFERENCE["area_m2"]
    cells = defaultdict(lambda: defaultdict(list))
    totals = {}
    for g in feas:
        rows = read_csv(STAGE1 / g / "trajectory_0p5s.csv")
        wsum = 0.0
        for r in rows:
            m, cl = float(r["Mach"]), float(r["CL_actual"])
            q, v = float(r["q_pa"]), float(r["V_mps"])
            w = q * sref * v * 0.5
            wsum += w
            key = (np.floor(m / MACH_BIN) * MACH_BIN, np.floor(cl / CL_BIN) * CL_BIN)
            c = cells[key]
            for k, x in (("M", m), ("CL", cl), ("q", q), ("V", v), ("h", float(r["h_m"])),
                         ("alpha", np.degrees(float(r["alpha_rad"]))), ("w", w),
                         ("drag_work", float(r["D_n"]) * v * 0.5), ("t", float(r["time_s"]))):
                c[k].append(x)
            c["geometry"].append(g)
        totals[g] = wsum
    n_traj = len(feas)
    rows = []
    for (mb, cb), c in sorted(cells.items()):
        w = np.array(c["w"]) / np.array([totals[g] for g in c["geometry"]]) / n_traj
        src = {g: c["geometry"].count(g) for g in feas if g in c["geometry"]}
        m_mean = float(np.average(c["M"], weights=w))
        cl_mean = float(np.average(c["CL"], weights=w))
        rows.append({
            "condition_id": f"C_M{mb:.2f}_CL{cb:.3f}", "band": band_of(m_mean),
            "Mach_bin_lo": round(mb, 4), "Mach_bin_hi": round(mb + MACH_BIN, 4),
            "CL_bin_lo": round(cb, 4), "CL_bin_hi": round(cb + CL_BIN, 4),
            "Mach": m_mean, "CL_required": cl_mean,
            "q_pa": float(np.average(c["q"], weights=w)), "V_mps": float(np.average(c["V"], weights=w)),
            "h_m": float(np.average(c["h"], weights=w)), "alpha_deg_on_source": float(np.average(c["alpha"], weights=w)),
            "duration_s_per_trajectory": len(c["M"]) * 0.5 / n_traj,
            "drag_work_weight": float(w.sum()),
            "drag_work_j_on_source_per_trajectory": float(np.sum(c["drag_work"]) / n_traj),
            "t_first_s": float(min(c["t"])), "t_last_s": float(max(c["t"])),
            "scored": cl_mean >= SCORE_CL_MIN,
            "source_trajectories": json.dumps(src)})
    s = sum(r["drag_work_weight"] for r in rows)
    for r in rows:
        r["drag_work_weight"] /= s
    write_csv(OUT / "mission_design_conditions.csv", rows)
    bands = {}
    for b in list(BANDS) + ["M4.5-5", "M>8"]:
        sel = [r for r in rows if r["band"] == b]
        if not sel:
            continue
        w = np.array([r["drag_work_weight"] for r in sel])
        cl = np.array([r["CL_required"] for r in sel])
        bands[b] = {"conditions": len(sel), "duration_s_per_trajectory": sum(r["duration_s_per_trajectory"] for r in sel),
                    "drag_work_share": float(w.sum()),
                    "CL_weighted_mean": float(np.average(cl, weights=w)),
                    "CL_p10_p90": [float(np.quantile(np.repeat(cl, np.maximum(1, (w * 1e4).astype(int))), q)) for q in (.1, .9)],
                    "q_pa_weighted_mean": float(np.average([r["q_pa"] for r in sel], weights=w)),
                    "V_mps_weighted_mean": float(np.average([r["V_mps"] for r in sel], weights=w)),
                    "h_m_range": [min(r["h_m"] for r in sel), max(r["h_m"] for r in sel)],
                    "time_window_s": [min(r["t_first_s"] for r in sel), max(r["t_last_s"] for r in sel)],
                    "scored_weight_share": float(sum(r["drag_work_weight"] for r in sel if r["scored"]))}
    summary = {"source_trajectories": feas, "stage1_comparison_sha_config": comp.get("comparison_scope"),
               "weight_definition": "q*Sref*V*dt per 0.5 s sample (drag work per unit CD), normalised per trajectory, trajectories equally weighted",
               "bins": {"Mach": MACH_BIN, "CL": CL_BIN}, "score_CL_min": SCORE_CL_MIN,
               "n_conditions": len(rows), "n_scored": sum(r["scored"] for r in rows),
               "unscored_weight": float(sum(r["drag_work_weight"] for r in rows if not r["scored"])),
               "bands": bands}
    dump(OUT / "mission_design_conditions_summary.json", summary)
    return rows, summary


def report(rows, summary):
    L = ["# 任务驱动设计工况（V6 第二阶段 P1）", "",
         f"来源：第一阶段 4 条可行固定构型轨迹 {', '.join(summary['source_trajectories'])}"
         "（`results/fixed_geometry_baselines/<几何>/trajectory_0p5s.csv`，0.5 s 采样）。",
         "", "## 定义", "",
         "- 工况 = (Mach, CL) 分箱：Mach 每 0.25、CL 每 0.005。",
         "- 权重 = 单位 CD 的阻力功 q·Sref·V·dt（D = q·Sref·CD，P_drag = D·V）；每条轨迹先归一化，4 条轨迹等权。",
         f"- CL < {SCORE_CL_MIN} 的工况（起始下压过渡段）保留在表中，但不参与几何评分（权重合计 {summary['unscored_weight']:.4f}）。",
         "- 这些工况来自已有固定轨迹，只用于确定几何优化的优先方向；新构型飞行时不要求遵守这些 CL 或轨迹。",
         "", "## 分段汇总", "",
         "| 段 | 工况数 | 每条轨迹时长 (s) | 阻力功占比 | 加权 CL | CL p10–p90 | 加权 q (kPa) | 加权 V (m/s) | 高度 (km) | 时间窗 (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for b, s in summary["bands"].items():
        L.append(f"| {b} | {s['conditions']} | {s['duration_s_per_trajectory']:.1f} | {s['drag_work_share']:.3f} | "
                 f"{s['CL_weighted_mean']:.4f} | {s['CL_p10_p90'][0]:.4f}–{s['CL_p10_p90'][1]:.4f} | "
                 f"{s['q_pa_weighted_mean'] / 1e3:.1f} | {s['V_mps_weighted_mean']:.0f} | "
                 f"{s['h_m_range'][0] / 1e3:.1f}–{s['h_m_range'][1] / 1e3:.1f} | "
                 f"{s['time_window_s'][0]:.0f}–{s['time_window_s'][1]:.0f} |")
    top = sorted(rows, key=lambda r: -r["drag_work_weight"])[:12]
    L += ["", "## 权重最高的 12 个工况", "",
          "| 工况 | 段 | Mach | CL | q (kPa) | V (m/s) | 时长 (s/轨迹) | 权重 |", "|---|---|---|---|---|---|---|---|"]
    for r in top:
        L.append(f"| {r['condition_id']} | {r['band']} | {r['Mach']:.3f} | {r['CL_required']:.4f} | {r['q_pa'] / 1e3:.1f} | "
                 f"{r['V_mps']:.0f} | {r['duration_s_per_trajectory']:.1f} | {r['drag_work_weight']:.4f} |")
    L += ["", "## 用于几何评分的指标", "",
          "对几何 g：J_g = Σ_c w_c·CD_g(M_c, CL_c) / Σ_c w_c·CD_G42(M_c, CL_c)，CD 为同一 Mach、同一 CL 下的 Teacher 压力阻力"
          "（PCHIP 反插值，迎角限于 Teacher 表范围）。分段评分只对该段工况求和。J < 1 表示在固定轨迹工况上阻力功低于 G42。",
          "这是固定轨迹上的阻力功近似，不等于新构型的最终航程；航程只由第 P4 节的完整 1200 s 仿真给出。",
          "", "数据：`mission_design_conditions.csv`、`mission_design_conditions_summary.json`。"]
    (OUT / "mission_design_conditions_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    r, s = build()
    report(r, s)
    print(json.dumps(s, indent=1, ensure_ascii=False))
