"""Is T invariant to c_coll?  The claim the rho=1 argument rests on.

    .venv/bin/python pace-analysis/fig_T_ccoll.py

Reads results/ccoll_J/data.csv (no new simulation) and draws T against c_coll,
one line per c_idle, one panel per scenario. The answer is no in general: T is
flat only at small c_idle and short windows, and rises by up to 92% with c_coll
at large c_idle under basic access at W = 1680.

The second figure is the comparison that decides whether the rho = 1 argument
survives: the swing in ln T against the swing in alpha (ln rho)^2 over the same
c_coll range. Where the T curve is above, c_coll is acting on efficiency, not
just on the share, and the optimum is not at rho = 1.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "ccoll_J", "data.csv")
OUT = os.path.join(ROOT, "results", "figure")
ALPHA = 0.5


def load():
    rows = [{k: (v if k == "access" else float(v)) for k, v in r.items()}
            for r in csv.DictReader(open(SRC))]
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = load()
    CI = sorted({r["c_idle"] for r in R})
    CC = sorted({r["c_coll"] for r in R})
    NV = sorted({int(r["n_vis"]) for r in R})
    panels = [(ac, w) for ac in ("rts", "basic") for w in (420, 1680)]
    cmap = plt.get_cmap("plasma")

    def sel(ac, w, nv, ci):
        d = {r["c_coll"]: r for r in R if r["access"] == ac
             and int(r["w_eff"]) == w and int(r["n_vis"]) == nv
             and r["c_idle"] == ci}
        return [d[c] for c in CC]

    # ── figure 1: T vs c_coll ────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 10.5), sharex=True)
    for i, (ac, w) in enumerate(panels):
        for j, nv in enumerate(NV):
            ax = axes[i][j]
            for k, ci in enumerate(CI):
                rows = sel(ac, w, nv, ci)
                ax.plot(CC, [r["T"] for r in rows], "-o", ms=2.5, lw=1.1,
                        color=cmap(0.05 + 0.85 * k / (len(CI) - 1)),
                        label=rf"$c_i$={ci}" if (i, j) == (0, 0) else None)
            sw = max(max(r["T"] for r in sel(ac, w, nv, ci))
                     / min(r["T"] for r in sel(ac, w, nv, ci)) - 1
                     for ci in CI)
            ax.set_title(rf"{ac}, $W$={w}, $N_v$={nv}"
                         "\n" rf"worst swing {100*sw:.0f}%", fontsize=8)
            ax.grid(color="0.92", lw=0.4)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(r"total useful airtime $T$")
            if i == 3:
                ax.set_xlabel(r"$c_\mathrm{coll}$")
    axes[0][0].legend(fontsize=6.5, ncol=2)
    fig.suptitle(r"$T$ is NOT invariant to $c_\mathrm{coll}$: it is flat only at "
                 r"small $c_\mathrm{idle}$ and short windows", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig11-1.{ext}"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)

    # ── figure 2: which term drives J along c_coll ───────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3), sharey=True)
    for i, (ac, w) in enumerate(panels):
        ax = axes[i]
        for nv, mk in zip(NV, ("o", "s", "^", "D")):
            rt = []
            for ci in CI:
                rows = sel(ac, w, nv, ci)
                lt = [math.log(r["T"]) for r in rows]
                lp = [ALPHA * math.log(max(r["rho"], 1e-9)) ** 2 for r in rows]
                rt.append((max(lp) - min(lp)) / max(max(lt) - min(lt), 1e-9))
            ax.semilogy(CI, rt, mk + "-", ms=4, lw=1.2, label=rf"$N_v$={nv}")
        ax.axhline(1.0, color="k", lw=1.0, ls="--")
        ax.set_title(f"{ac}, $W$={w}", fontsize=9)
        ax.set_xlabel(r"$c_\mathrm{idle}$")
        ax.grid(color="0.92", lw=0.4, which="both")
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel(r"swing ratio  $\Delta\alpha(\ln\rho)^2\,/\,\Delta\ln T$")
            ax.legend(fontsize=7)
    fig.suptitle(r"above the dashed line the share term drives $J$ along "
                 rf"$c_\mathrm{{coll}}$; below it, efficiency does  ($\alpha$={ALPHA})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig11-2.{ext}"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig11-1.* and fig11-2.*")


if __name__ == "__main__":
    main()
