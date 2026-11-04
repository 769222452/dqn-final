"""Stage-2 figures (matplotlib, Agg)."""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import OUT, load  # noqa: E402

FIG = OUT / "figures"


def tradeoff():
    from pilot import candidate_pool
    pool, _ = candidate_pool()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, scen, zoom in zip(axes.ravel(), ("EXPLORATORY", "CONSERVATIVE") * 2, (False, False, True, True)):
        if zoom:
            ax.set_xlim(0.975, 1.06)
            ax.set_ylim(0.84, 1.08)
        for ev, c, mk in (("V2_TEACHER", "0.6", "o"), ("NEW_TEACHER", "tab:red", "^")):
            p = [x for x in pool if x["evidence"] == ev and x["scenario"][scen]]
            ax.scatter([x["J"]["M7-8"] for x in p], [x["J"]["M5-7"] for x in p], s=14, c=c, marker=mk,
                       label=f"{ev} ({len(p)})", alpha=.8)
        ax.axvline(1, c="k", lw=.6, ls=":")
        ax.axhline(1, c="k", lw=.6, ls=":")
        ax.set_xlabel("J M7-8 (iso-CL drag work / G42)")
        ax.set_ylabel("J M5-7 (iso-CL drag work / G42)")
        ax.set_title(f"{scen}: high-Mach vs low/mid-Mach" + (" (zoom)" if zoom else ""))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "band_tradeoff.png", dpi=130)
    plt.close(fig)


def paths():
    for p in sorted((OUT / "morph").glob("*.json")):
        rec = load(p)
        n = rec["nodes"]
        s = [x["s"] for x in n]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(s, [x.get("volume_ratio_G42") or np.nan for x in n], "o-", label="volume/G42")
        ax[0].plot(s, [x.get("thickness_ratio_G42") or np.nan for x in n], "s-", label="thickness/G42")
        ax[0].axhline(1, c="k", lw=.6, ls=":")
        ax[0].set_xlabel("path progress s")
        ax[0].legend()
        ax[0].set_title(f"{rec['path']}  [{rec['status']}]", fontsize=9)
        ax[1].plot(s, [x["cum_max_displacement_m"] for x in n], "o-", label="sum max displacement")
        ax[1].plot(s, [x["cum_rms_displacement_m"] for x in n], "s-", label="sum RMS displacement")
        ax[1].set_xlabel("s")
        ax[1].set_ylabel("m")
        ax[1].legend()
        for c in rec["blend_checks"]:
            if "cd_rel_residual_max" in c:
                ax[2].plot(c["mid"], c["cd_rel_residual_max"], "o", c=f"C{c['depth']}")
                ax[2].plot(c["mid"], c["cl_rel_residual_max"], "x", c=f"C{c['depth']}")
        ax[2].axhline(rec["tolerances"]["CD_rel"], c="r", lw=.8)
        ax[2].set_xlabel("midpoint s (o: CD, x: CL; colour=depth)")
        ax[2].set_ylabel("Teacher vs blend rel. residual")
        fig.tight_layout()
        fig.savefig(FIG / f"path_{rec['path']}.png", dpi=120)
        plt.close(fig)
        try:
            from geometry import surfaces
            fig, ax = plt.subplots(figsize=(7, 5))
            cm = plt.get_cmap("viridis")
            for x in n:
                if x.get("native_geometry_mat"):
                    lo, _ = surfaces(x["native_geometry_mat"])
                    ax.plot(lo[0, :, 0], lo[0, :, 1], c=cm(x["s"] / max(s)), lw=.8)
                    ax.plot(lo[-1, :, 0], lo[-1, :, 1], c=cm(x["s"] / max(s)), lw=.8)
                    ax.plot(lo[:, 0, 0], lo[:, 0, 1], c=cm(x["s"] / max(s)), lw=.8)
                    ax.plot(lo[:, -1, 0], lo[:, -1, 1], c=cm(x["s"] / max(s)), lw=.8)
            ax.set_aspect("equal")
            ax.set_title(f"lower-surface grid boundary (x-y), colour = s: {rec['path']}", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIG / f"path_{rec['path']}_planform.png", dpi=120)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print("planform figure skipped:", e)


def pilot():
    from deliverables import _passes
    res = _passes()
    ok = {k: v for k, v in res.items() if v.get("status") == "BEST_FOUND"}
    if not ok:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    names = sorted(ok, key=lambda k: ok[k]["range_m"])
    ax[0].barh(range(len(names)), [ok[k]["range_m"] / 1e3 for k in names], color="tab:blue")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=7)
    ax[0].set_xlim(min(ok[k]["range_m"] for k in names) / 1e3 - 20, max(ok[k]["range_m"] for k in names) / 1e3 + 10)
    ax[0].set_xlabel("best-found range at 1200 s (km)")
    for k in names:
        f = OUT / ("pilot" if ok[k]["best_pass"] == "A" else "pilot_B") / k / "trajectory_0p5s.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        t = [float(r["time_s"]) for r in rows]
        ax[1].plot(t, [float(r["Mach"]) for r in rows], lw=.8, label=k)
        ax[2].plot(t, [float(r["path_s"]) for r in rows], lw=.8, label=k)
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("Mach")
    ax[2].set_xlabel("t (s)")
    ax[2].set_ylabel("path progress s")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "pilot_1200s.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    tradeoff()
    paths()
    pilot()
