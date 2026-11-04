"""Stage 8: figures (wedge-model predictions only; none of them is a CFD-validated statement)."""
import collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
from common import ROOT, OUT, read_csv

F = OUT / "figures"; F.mkdir(exist_ok=True)
NOTE = "TANGENT_WEDGE_SHOCK_EXPANSION prediction (pressure-only); NOT CFD-validated"
top = read_csv(OUT / "screening/top5_by_condition.csv"); cross = read_csv(OUT / "offdesign/cross_mach_performance.csv")
near = read_csv(OUT / "offdesign/nearby_performance.csv"); eq = read_csv(OUT / "offdesign/equal_lift_comparison.csv")
allsc = read_csv(OUT / "screening/all_candidate_scores.csv"); sel = read_csv(OUT / "screening/selected_unique_geometries.csv")
Ma = {r["geometry_id"]: float(r["Ma_design"]) for r in sel}
Ms = sorted({float(r["M_flight"]) for r in top}); As = sorted({float(r["alpha_flight_deg"]) for r in top})
cm = plt.get_cmap("viridis"); nrm = plt.Normalize(8, 15)


def fin(fig, name):
    fig.text(0.5, 0.005, NOTE, ha="center", fontsize=8, color="firebrick"); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(F / name, dpi=130); plt.close(fig)


# 1 Top1 map
t1 = {(float(r["M_flight"]), float(r["alpha_flight_deg"])): r for r in top if r["rank"] == "1"}
Z = np.array([[float(t1[(m, a)]["Ma_design"]) for m in Ms] for a in As]); fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto", extent=(Ms[0] - .25, Ms[-1] + .25, As[0] - .5, As[-1] + .5), vmin=8, vmax=15)
for i, a in enumerate(As):
    for j, m in enumerate(Ms): ax.text(m, a, t1[(m, a)]["geometry_id"].split("_")[1][:7] + f"\n{float(t1[(m,a)]['L_over_D']):.1f}", ha="center", va="center", fontsize=6, color="w")
ax.set_xlabel("flight Mach"); ax.set_ylabel("alpha [deg]"); ax.set_title("Top-1 geometry per condition (colour = design Mach; text = family / L/D)"); fig.colorbar(im, label="design Mach"); fin(fig, "01_top1_distribution.png")
# 2 Top5 frequency
cnt = collections.Counter(r["geometry_id"] for r in top).most_common(15); fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([k.replace("S2_", "") for k, _ in cnt][::-1], [v for _, v in cnt][::-1], color=[cm(nrm(Ma[k])) for k, _ in cnt][::-1]); ax.set_xlabel("appearances in Top5 (of 49 conditions)"); ax.set_title("Top5 frequency (colour = design Mach)"); fin(fig, "02_top5_frequency.png")
# 3 L/D vs flight Mach, alpha = 6
fig, ax = plt.subplots(figsize=(8, 5.5)); Mg = sorted({float(r["M_flight"]) for r in cross})
for g in Ma:
    y = [next(float(r["L_over_D"]) for r in cross if r["geometry_id"] == g and float(r["M_flight"]) == m and float(r["alpha_flight"]) == 6.0) for m in Mg]
    ax.plot(Mg, y, color=cm(nrm(Ma[g])), lw=1.2, alpha=.8, ls="--" if g == "G42" else "-")
ax.set_xlabel("flight Mach"); ax.set_ylabel("L/D (pressure only)"); ax.set_title("selected geometries, alpha=6 deg (colour = design Mach; dashed = G42)"); sm = plt.cm.ScalarMappable(norm=nrm, cmap=cm); fig.colorbar(sm, ax=ax, label="design Mach"); fin(fig, "03_LD_vs_flight_mach.png")
# 4 same Mach comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, m in zip(axs, (5.5, 7.5)):
    rows = sorted([r for r in cross if float(r["M_flight"]) == m and float(r["alpha_flight"]) == 7.0], key=lambda r: -float(r["L_over_D"]))
    ax.bar(range(len(rows)), [float(r["L_over_D"]) for r in rows], color=[cm(nrm(Ma[r["geometry_id"]])) for r in rows]); ax.set_title(f"M={m}, alpha=7 deg: L/D of all selected geometries"); ax.set_xlabel("geometry (sorted)"); ax.set_ylabel("L/D")
fin(fig, "04_same_mach_comparison.png")
# 5/6 nearby
freq = [k for k, _ in collections.Counter(r["geometry_id"] for r in top).most_common(3)] + ["G42"]
for fn, key, lab, off in (("05_nearby_mach.png", "M_query", "flight Mach", ("M-", "CENTER", "M+")), ("06_nearby_alpha.png", "alpha_query", "alpha [deg]", ("ALPHA-", "CENTER", "ALPHA+"))):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for g in freq:
        c0 = [r for r in near if r["geometry_id"] == g and r["condition_id"] == "M6.50_A07.00"] or [r for r in near if r["geometry_id"] == g][:5]
        c0 = [r for r in c0 if r["offset"] in off]; c0.sort(key=lambda r: float(r[key]))
        for ax, v in zip(axs, ("CL", "CD_pressure")): ax.plot([float(r[key]) for r in c0], [float(r[v]) for r in c0], "o-", label=g.replace("S2_", "") + f" ({c0[0]['condition_id']})")
    axs[0].set_ylabel("CL"); axs[1].set_ylabel("CD_pressure"); [a.set_xlabel(lab) for a in axs]; axs[0].legend(fontsize=6); fig.suptitle("nearby evaluations (real model calls) around the centre condition"); fin(fig, fn)
# 7 design Mach vs flight Mach
fig, ax = plt.subplots(figsize=(7.5, 5)); cc = collections.Counter((float(r["M_flight"]), round(float(r["Ma_design"]))) for r in top)
ax.scatter([k[0] for k in cc], [k[1] for k in cc], s=[40 * v for v in cc.values()], alpha=.6); ax.set_xlabel("flight Mach"); ax.set_ylabel("design Mach (rounded)"); ax.set_title("design Mach of Top5 members vs flight Mach (marker area = count)"); fin(fig, "07_design_vs_flight_mach.png")
# 8 equal lift CD
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, tg in zip(axs, sorted({float(r["CL_target"]) for r in eq})):
    for g in Ma:
        rr = sorted([r for r in eq if r["geometry_id"] == g and float(r["CL_target"]) == tg and r["status"] == "OK"], key=lambda r: float(r["M_flight"]))
        if rr: ax.plot([float(r["M_flight"]) for r in rr], [float(r["CD_at_target"]) for r in rr], color=cm(nrm(Ma[g])), lw=1)
    ax.set_title(f"CL = {tg:.4f}"); ax.set_xlabel("flight Mach")
axs[0].set_ylabel("CD_pressure at equal CL"); fin(fig, "08_equal_lift_cd.png")
# 9 volume vs L/D
fig, ax = plt.subplots(figsize=(8, 5.5)); sc = [r for r in allsc if r["condition_id"] == "M6.50_A07.00" and r["aero_status"] == "OK"]; ids = set(Ma)
ax.scatter([float(r["volume_ratio_G42"]) for r in sc], [float(r["L_over_D"]) for r in sc], s=8, c="lightgrey", label="all 453 pool geometries")
s2 = [r for r in sc if r["geometry_id"] in ids]; ax.scatter([float(r["volume_ratio_G42"]) for r in s2], [float(r["L_over_D"]) for r in s2], s=30, c=[cm(nrm(float(r["Ma_design"]))) for r in s2], label="selected")
ax.axvline(.85, color="r", ls=":", label="exploratory volume screen"); ax.set_xlabel("volume / G42 volume"); ax.set_ylabel("L/D"); ax.set_title("M=6.5, alpha=7 deg: L/D vs volume ratio"); ax.legend(fontsize=7); ax.set_xlim(0, 3); fin(fig, "09_volume_vs_LD.png")
# 10 3D native
show = [freq[0], freq[1], freq[2], "G42"]; fig = plt.figure(figsize=(14, 7))
for k, g in enumerate(show, 1):
    S = sio.loadmat(str(ROOT / next(r["mat_path"] for r in sel if r["geometry_id"] == g))); ax = fig.add_subplot(2, 2, k, projection="3d")
    for nm, col in (("wav_", "tab:blue"), ("wav_up_", "tab:orange")):
        X, Y, Z = (S[nm + c] for c in "xyz")
        for sgn in (1, -1): ax.plot_surface(X, sgn * -Y, -Z, color=col, alpha=.5, linewidth=0, rstride=2, cstride=2)
    ax.set_title(f"{g} (Ma_design {Ma[g]:.2f})", fontsize=8); ax.view_init(20, -60); ax.set_box_aspect((5, 2, 1))
fin(fig, "10_native_3d.png")
print(sorted(p.name for p in F.glob("*.png")))
