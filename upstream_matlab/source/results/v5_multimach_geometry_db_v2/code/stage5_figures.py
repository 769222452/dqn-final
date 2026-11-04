"""Stage 5: figures. Every panel is drawn from real stored data.

Sources: the MATLAB native geometry .mat files, the frozen-rule audit records,
the stage 0 feasibility probe tables, and the morph-path validation tables.
Nothing is sketched, idealised or interpolated for display beyond the plotted
mesh itself. Labels are in English so that the figures render identically
without a Chinese font installed; the reports describe them in Chinese.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mmg
from mmg import OUT, PARAMS
from geometry_metrics import load_surfaces

FIG = OUT / "figures"
CLS_COLOR = {"VALID": "#1b7837", "INVALID": "#b2182b", "AMBIGUOUS": "#f1a340",
             "GENERATION_FAILED": "#999999"}
MACH_COLOR = {4.0: "#313695", 5.0: "#4575b4", 6.0: "#74add1", 7.0: "#f46d43",
              8.0: "#d73027", 15.0: "#542788"}
manifest = {}


def save(fig, name, title, sources):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    manifest[name] = {"title": title, "data_sources": sources, "sha256": mmg.sha(path)}
    print(f"[fig] {name}", flush=True)


def inv():
    return mmg.read_csv(OUT / "geometry_inventory.csv")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- figure 1
def fig_design_mach_feasibility():
    rows = inv()
    machs = sorted({float(r["Ma_design"]) for r in rows})
    counts = {m: Counter(r["classification"] for r in rows if float(r["Ma_design"]) == m)
              for m in machs}
    keys = ["VALID", "INVALID", "GENERATION_FAILED"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    bottom = np.zeros(len(machs))
    for k in keys:
        v = np.array([counts[m].get(k, 0) for m in machs], float)
        ax[0].bar([f"{m:g}" for m in machs], v, bottom=bottom, color=CLS_COLOR[k], label=k)
        bottom += v
    ax[0].set_xlabel("Ma_design")
    ax[0].set_ylabel("candidates")
    ax[0].set_title("Candidate outcome by design Mach (real MATLAB calls)")
    ax[0].legend(fontsize=8)
    tot = [sum(counts[m].values()) for m in machs]
    val = [counts[m].get("VALID", 0) for m in machs]
    gen = [counts[m].get("VALID", 0) + counts[m].get("INVALID", 0) for m in machs]
    ax[1].plot([f"{m:g}" for m in machs], 100 * np.array(gen) / tot, "o-", label="generated %")
    ax[1].plot([f"{m:g}" for m in machs], 100 * np.array(val) / tot, "s-", label="VALID %")
    for i, m in enumerate(machs):
        ax[1].annotate(f"{val[i]}/{tot[i]}", (i, 100 * val[i] / tot[i]),
                       textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax[1].set_xlabel("Ma_design")
    ax[1].set_ylabel("percent of candidates")
    ax[1].set_title("Generation and validity rate by design Mach")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    save(fig, "fig01_design_mach_feasibility.png",
         "Candidate outcome and success rate versus design Mach",
         ["geometry_inventory.csv"])


# ---------------------------------------------------------------- figure 2
def fig_failure_mechanism():
    rows = []
    for f in ("feasibility_scan.csv", "feasibility_scan_b.csv", "feasibility_scan_c.csv"):
        p = OUT / "stage0_feasibility" / f
        if p.exists():
            rows += mmg.read_csv(p)
    import re
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    pts = defaultdict(list)
    for r in rows:
        m = re.search(r"Stream surface (\d+)", r.get("error_message") or "")
        if m:
            pts[float(r["Ma_design"])].append(int(m[1]))
    machs = sorted(pts)
    ax[0].boxplot([pts[m] for m in machs], tick_labels=[f"{m:g}" for m in machs])
    ax[0].set_xlabel("Ma_design")
    ax[0].set_ylabel("first failing stream surface index (of 50)")
    ax[0].set_title("MOC characteristic net folds earlier at low design Mach")
    ax[0].grid(alpha=.3)
    ids = Counter()
    for r in rows:
        msg = (r.get("error_message") or "").strip()
        if r["status"] != "GENERATED" and msg:
            key = msg.split(":", 1)[-1].strip()
            key = re.sub(r"\s*\[.*\]$", "", key)
            ids[key] += 1
    names = [k for k, _ in ids.most_common()]
    ax[1].barh(names, [ids[k] for k in names], color="#b2182b")
    ax[1].set_xlabel("probe count")
    ax[1].set_title("Generator domain errors (stage 0 probes)")
    ax[1].grid(alpha=.3, axis="x")
    save(fig, "fig02_failure_mechanism.png",
         "Where and why the generator fails as the design Mach drops",
         ["stage0_feasibility/feasibility_scan*.csv"])


# ---------------------------------------------------------------- figure 3
def fig_design_space_map():
    rows = inv()
    machs = sorted({float(r["Ma_design"]) for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), machs):
        sub = [r for r in rows if float(r["Ma_design"]) == m]
        for cls in ("GENERATION_FAILED", "INVALID", "VALID"):
            s = [r for r in sub if r["classification"] == cls]
            if not s:
                continue
            x = [min(float(r[p]) for p in ("a1", "a2", "b1", "b2")) for r in s]
            y = [float(r["d1"]) for r in s]
            ax.scatter(x, y, s=16, c=CLS_COLOR[cls], label=cls, alpha=.8,
                       edgecolors="none")
        ax.axhline(0.109, color="k", ls="--", lw=.8)
        ax.set_title(f"Ma_design = {m:g}")
        ax.grid(alpha=.3)
    for ax in axes[-1]:
        ax.set_xlabel("min(a1,a2,b1,b2)  [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("d1  (leading-edge Bezier)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Measured feasible region: generation needs angle, validity needs small d1 "
                 "(dashed line: largest d1 observed among VALID geometries)")
    save(fig, "fig03_design_space_map.png",
         "Design-space map of generation and validity by design Mach",
         ["geometry_inventory.csv"])


# ---------------------------------------------------------------- figure 4
def pick_representatives():
    rows = [r for r in inv() if r["classification"] == "VALID" and r["native_geometry_path"]]
    met = {r["geometry_id"]: r for r in mmg.read_csv(OUT / "geometry_metrics.csv")}
    out = {}
    for m in sorted({float(r["Ma_design"]) for r in rows}):
        sub = [r for r in rows if float(r["Ma_design"]) == m]
        sub.sort(key=lambda r: -(fnum(met.get(r["geometry_id"], {}).get("volumetric_efficiency")) or 0))
        out[m] = sub[0]
    return out


def fig_valid_geometries_3d():
    reps = pick_representatives()
    fig = plt.figure(figsize=(16, 9))
    for i, (m, r) in enumerate(reps.items(), start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        lo, up = load_surfaces(OUT / r["native_geometry_path"])
        for g, c in ((lo, "#2166ac"), (up, "#d6604d")):
            ax.plot_surface(g[:, :, 0], g[:, :, 1], g[:, :, 2], color=c, alpha=.85,
                            linewidth=0, antialiased=False, rstride=2, cstride=2)
        ax.set_title(f"{r['geometry_id']}  Ma_design={m:g}\n"
                     f"a=({float(r['a1']):.1f},{float(r['a2']):.1f},"
                     f"{float(r['b1']):.1f},{float(r['b2']):.1f})  d1={float(r['d1']):.3f}",
                     fontsize=8)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=24, azim=-132)
    fig.suptitle("Native half-skin surfaces of the highest volumetric-efficiency VALID geometry "
                 "at each design Mach (program coordinates, lower blue / upper red)")
    save(fig, "fig04_valid_geometries_3d.png",
         "Representative VALID native geometries per design Mach",
         ["native_geometries/*.mat", "geometry_metrics.csv"])


# ---------------------------------------------------------------- figure 5
def fig_cross_sections():
    reps = pick_representatives()
    fracs = (0.25, 0.50, 0.75, 1.00)
    fig, axes = plt.subplots(len(fracs), len(reps), figsize=(16, 10), sharex=True)
    for col, (m, r) in enumerate(reps.items()):
        lo, up = load_surfaces(OUT / r["native_geometry_path"])
        x0 = float(min(lo[:, :, 0].min(), up[:, :, 0].min()))
        x1 = float(max(lo[:, :, 0].max(), up[:, :, 0].max()))
        for row, f in enumerate(fracs):
            ax = axes[row, col]
            xt = x0 + f * (x1 - x0)
            for g, c, lab in ((lo, "#2166ac", "lower"), (up, "#d6604d", "upper")):
                ys, zs = [], []
                for i in range(g.shape[0]):
                    xs = g[i, :, 0]
                    if xs.min() <= xt <= xs.max():
                        ys.append(np.interp(xt, xs, g[i, :, 1]))
                        zs.append(np.interp(xt, xs, g[i, :, 2]))
                o = np.argsort(ys)
                ax.plot(np.asarray(ys)[o], np.asarray(zs)[o], color=c, lw=1.2, label=lab)
            ax.invert_yaxis()
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"x/L={f:.2f}\nprogram z", fontsize=7)
            if row == 0:
                ax.set_title(f"Ma_design={m:g}\n{r['geometry_id']}", fontsize=8)
            if row == len(fracs) - 1:
                ax.set_xlabel("program y", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Native cross sections of the representative VALID geometries "
                 "(z axis inverted so the body appears upright)")
    save(fig, "fig05_cross_sections.png", "Cross sections at four stations per design Mach",
         ["native_geometries/*.mat"])


# ---------------------------------------------------------------- figure 6
def fig_metrics_vs_mach():
    met = [r for r in mmg.read_csv(OUT / "geometry_metrics.csv") if r["classification"] == "VALID"]
    machs = sorted({float(r["Ma_design"]) for r in met})
    keys = [("volumetric_efficiency", "V^(2/3) / S_planform"),
            ("volume_full", "volume (program units^3)"),
            ("planform_area_full", "planform area"),
            ("aspect_ratio_span_over_length", "span / length")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (k, lab) in zip(axes.ravel(), keys):
        data = [[fnum(r[k]) for r in met if float(r["Ma_design"]) == m and fnum(r[k]) is not None]
                for m in machs]
        ax.boxplot(data, tick_labels=[f"{m:g}" for m in machs], showfliers=False)
        for i, m in enumerate(machs, start=1):
            v = data[i - 1]
            ax.scatter(np.full(len(v), i) + np.random.default_rng(7).normal(0, .04, len(v)), v,
                       s=12, color=MACH_COLOR.get(m, "#444"), alpha=.8, zorder=3)
        ax.set_xlabel("Ma_design")
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=.3)
    fig.suptitle("Geometric metrics of the VALID geometries, computed from the stored native meshes")
    save(fig, "fig06_metrics_vs_mach.png", "Geometric metric distributions per design Mach",
         ["geometry_metrics.csv"])


# ---------------------------------------------------------------- figure 7
def fig_g42_regression():
    reg = mmg.load(OUT / "g42_regression.json")
    old = mmg.OLD_AUDIT / "cases" / "design_42" / "native_geometry.mat"
    new = OUT / "cases" / "REG_G42_MD15" / "native_geometry.mat"
    lo_a, up_a = load_surfaces(old)
    lo_b, up_b = load_surfaces(new)
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_surface(lo_a[:, :, 0], lo_a[:, :, 1], lo_a[:, :, 2], color="#2166ac", alpha=.9,
                    linewidth=0, rstride=2, cstride=2)
    ax.plot_surface(up_a[:, :, 0], up_a[:, :, 1], up_a[:, :, 2], color="#d6604d", alpha=.9,
                    linewidth=0, rstride=2, cstride=2)
    ax.set_title("frozen reference design_42 (G42)", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=24, azim=-132)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(lo_b[:, :, 0], lo_b[:, :, 1], lo_b[:, :, 2], color="#2166ac", alpha=.9,
                    linewidth=0, rstride=2, cstride=2)
    ax.plot_surface(up_b[:, :, 0], up_b[:, :, 1], up_b[:, :, 2], color="#d6604d", alpha=.9,
                    linewidth=0, rstride=2, cstride=2)
    ax.set_title("regenerated by this task's call layer", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=24, azim=-132)
    ax = fig.add_subplot(1, 3, 3)
    d = [abs(a - b).max() for a, b in ((lo_a, lo_b), (up_a, up_b))]
    ax.bar(["lower surface", "upper surface"], d, color="#1b7837")
    ax.set_ylabel("max |coordinate difference|")
    ax.set_ylim(-1e-16, max(max(d), 1e-16) * 1.5 + 1e-16)
    ax.set_title(f"regression {reg['GEOMETRY_REGRESSION']}\n"
                 f"geometry_hash identical: {reg['geometry_hash_identical']}", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    fig.suptitle("G42 @ Ma_design=15 regression through the independent matlab/active call layer")
    save(fig, "fig07_g42_regression.png", "G42 regression reference versus regenerated",
         ["results/v5_geometry_audit/cases/design_42/native_geometry.mat",
          "cases/REG_G42_MD15/native_geometry.mat", "g42_regression.json"])


# ---------------------------------------------------------------- figure 8
def fig_morph_graph():
    nodes = mmg.read_csv(OUT / "morph_path_nodes.csv")
    edges = mmg.read_csv(OUT / "morph_path_edges.csv")
    pos = {r["geometry_id"]: (float(r["a1"]), float(r["d1"])) for r in nodes}
    machs = sorted({float(r["Ma_design"]) for r in nodes})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m in zip(axes.ravel(), machs):
        for e in edges:
            if float(e["Ma_design"]) != m:
                continue
            a, b = pos[e["from_geometry_id"]], pos[e["to_geometry_id"]]
            ok = e["status"] == "CONNECTED"
            ax.plot([a[0], b[0]], [a[1], b[1]], "-" if ok else ":",
                    color="#1b7837" if ok else "#b2182b", lw=1.4 if ok else 1.0, zorder=1)
        sub = [r for r in nodes if float(r["Ma_design"]) == m]
        iso = [r for r in sub if int(r["degree_connected"]) == 0]
        ax.scatter([pos[r["geometry_id"]][0] for r in sub], [pos[r["geometry_id"]][1] for r in sub],
                   s=40, c="#2166ac", zorder=2, label="VALID node")
        if iso:
            ax.scatter([pos[r["geometry_id"]][0] for r in iso],
                       [pos[r["geometry_id"]][1] for r in iso],
                       s=90, facecolors="none", edgecolors="#b2182b", zorder=3,
                       label="isolated node")
        ax.set_title(f"Ma_design={m:g}: {sum(1 for e in edges if float(e['Ma_design'])==m and e['status']=='CONNECTED')}"
                     f"/{sum(1 for e in edges if float(e['Ma_design'])==m)} edges connected", fontsize=9)
        ax.set_xlabel("a1 [deg]", fontsize=8)
        ax.set_ylabel("d1", fontsize=8)
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Morph-path graph per design Mach. Solid green: every intermediate geometry "
                 "really generated and VALID. Dotted red: blocked edge.")
    save(fig, "fig08_morph_graph.png", "Morph-path connectivity per design Mach",
         ["morph_path_nodes.csv", "morph_path_edges.csv"])


# ---------------------------------------------------------------- figure 9
def fig_morph_validation():
    rows = mmg.read_csv(OUT / "morph_path_validation.csv")
    edges = {r["edge_id"]: r for r in mmg.read_csv(OUT / "morph_path_edges.csv")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    by_rho = defaultdict(Counter)
    for r in rows:
        by_rho[float(r["rho"])][r["classification"]] += 1
    rhos = sorted(by_rho)
    bottom = np.zeros(len(rhos))
    for cls in ("VALID", "INVALID", "GENERATION_FAILED"):
        v = np.array([by_rho[r].get(cls, 0) for r in rhos], float)
        ax[0].bar([f"{r:.2f}" for r in rhos], v, bottom=bottom, color=CLS_COLOR[cls], label=cls)
        bottom += v
    ax[0].set_xlabel("rho along the edge")
    ax[0].set_ylabel("intermediate geometries")
    ax[0].set_title("Intermediate geometry outcome by path fraction")
    ax[0].legend(fontsize=8)
    for eid, e in edges.items():
        sub = sorted([r for r in rows if r["edge_id"] == eid], key=lambda r: float(r["rho"]))
        t = [fnum(r["min_section_thickness"]) for r in sub]
        if any(v is None for v in t):
            continue
        ax[1].plot([float(r["rho"]) for r in sub], t,
                   color="#1b7837" if e["status"] == "CONNECTED" else "#b2182b",
                   alpha=.5, lw=.9)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("rho along the edge")
    ax[1].set_ylabel("min section thickness")
    ax[1].set_title("Audited minimum section thickness along every tested edge")
    ax[1].grid(alpha=.3)
    save(fig, "fig09_morph_validation.png", "Morph-path intermediate validation",
         ["morph_path_validation.csv", "morph_path_edges.csv"])


# ---------------------------------------------------------------- figure 10
def fig_angle_growth_map():
    rows = [r for r in inv() if r["classification"] in ("VALID", "INVALID")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for r in rows:
        a1, a2 = float(r["a1"]), float(r["a2"])
        b1, b2 = float(r["b1"]), float(r["b2"])
        m = float(r["Ma_design"])
        v = r["classification"] == "VALID"
        ax[0].scatter(a1, a2 / a1, s=22 if v else 12,
                      color=MACH_COLOR.get(m, "#444"), marker="o" if v else "x",
                      alpha=.9 if v else .35, linewidths=.8)
        ax[1].scatter(b1, b2 / b1, s=22 if v else 12,
                      color=MACH_COLOR.get(m, "#444"), marker="o" if v else "x",
                      alpha=.9 if v else .35, linewidths=.8)
    for a, lab in ((ax[0], ("a1 [deg]", "a2 / a1")), (ax[1], ("b1 [deg]", "b2 / b1"))):
        a.axvline(8.0, color="k", ls="--", lw=.8)
        a.axhline(1.0, color="k", ls=":", lw=.8)
        a.set_xlabel(lab[0])
        a.set_ylabel(lab[1])
        a.grid(alpha=.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=MACH_COLOR[m], label=f"Ma_design {m:g}")
               for m in sorted(MACH_COLOR)]
    handles += [plt.Line2D([], [], marker="o", ls="", color="#444", label="VALID"),
                plt.Line2D([], [], marker="x", ls="", color="#444", label="INVALID")]
    ax[0].legend(handles=handles, fontsize=7, ncol=2)
    fig.suptitle("Root shock angle versus streamwise growth ratio. Dashed line: frozen optimizer "
                 "upper bound of 8 deg; everything right of it is the declared EXTENDED space.")
    save(fig, "fig10_angle_growth_map.png", "Shock angle and growth ratio of generated geometries",
         ["geometry_inventory.csv"])


def main():
    fig_design_mach_feasibility()
    fig_failure_mechanism()
    fig_design_space_map()
    fig_valid_geometries_3d()
    fig_cross_sections()
    fig_metrics_vs_mach()
    fig_g42_regression()
    fig_morph_graph()
    fig_morph_validation()
    fig_angle_growth_map()
    mmg.dump(OUT / "figures" / "figure_manifest.json",
             {"utc": mmg.now(), "count": len(manifest),
              "note": "All panels are drawn from stored MATLAB geometry and audit records.",
              "figures": manifest})
    print(json.dumps({"figures": len(manifest)}))


if __name__ == "__main__":
    main()
