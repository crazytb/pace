"""What actually keeps the visitors' X = ln tau together?

    .venv/bin/python pace-analysis/homogeneity.py

The analysis leans on a homogeneous population (section 3.2) and credits
solo-copy for it. But every visitor also STARTS at the same tau_0 = 1/W_eff,
and most epochs act identically on every viable visitor, so the shared start
may be doing the work on its own.

Two measurements over the FULL scenario set, at the shipped (1.2, 1.2):

  1. Epoch decomposition. An epoch can only pull two viable visitors apart if
     they receive different updates. Idle (everyone x c_idle), native solo
     (nobody updates) and a native-only collision (every viable visitor is a
     listener, so everyone / c_coll) are all pure translations. Only a
     collision containing at least one visitor splits the population.

  2. pace_nocopy. Same algorithm with the solo-copy assignment removed, so
     the only thing holding the population together is the shared start and
     the shared updates. If the spread stays small, solo-copy is not the
     synchroniser; if it grows, it is.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO

C_COLL = C_IDLE = 1.20
VISITS = 30
SCEN = [(nv, 10, w, ac) for ac in ("rts", "basic")
        for w in (105, 210, 420, 840, 1680) for nv in (5, 10, 20, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "homogeneity")


def measure(core):
    scn = CO.Scn(*core)
    row = {"access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
           "n_nat": scn.n_nat}
    for m in ("pace", "pace_nocopy"):
        a = CO.aggregate(
            CO.batch(scn, C_IDLE, C_COLL, CO.EVAL_SEEDS, VISITS, mode=m),
            scn, 0.0)
        tag = "" if m == "pace" else "_nc"
        row[f"x_sd{tag}"] = round(a["x_sd"], 5)
        row[f"tau_cv{tag}"] = round(a["tau_cv"], 5)
        row[f"T{tag}"] = round(a["T"], 5)
        row[f"rho{tag}"] = round(a["rho"], 5)
        row[f"solo_vis{tag}"] = round(a["solo_vis_frac"], 5)
        row[f"coll_vis{tag}"] = round(a["coll_vis_per_ep"], 5)
        row[f"coll_nat{tag}"] = round(a["coll_nat_per_ep"], 5)
        row[f"solo_nat{tag}"] = round(a["solo_nat_frac"], 5)
        row[f"idle{tag}"] = round(a["idle_ep_frac"], 5)
        row[f"epochs{tag}"] = round(a["epochs_per_visit"], 3)
    # the only epochs that can spread the population, and the only ones that
    # collapse it
    row["split_frac"] = row["coll_vis"]
    row["merge_frac"] = row["solo_vis"]
    return row


def summarise(rows):
    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"  {name:<34s} min {xs.min():8.4f}  median "
                f"{np.median(xs):8.4f}  max {xs.max():8.4f}")

    o = ["", f"{len(rows)} scenarios at (c_coll, c_idle) = ({C_COLL}, "
             f"{C_IDLE}), {len(CO.EVAL_SEEDS)} seeds x {VISITS} visits", ""]
    o.append("  --- epoch decomposition (fractions of all epochs) ---")
    for k, lab in (("idle", "idle          (translate)"),
                   ("solo_nat", "native solo   (no update)"),
                   ("coll_nat", "native-only collision (transl.)"),
                   ("solo_vis", "visitor solo  (MERGE)"),
                   ("coll_vis", "visitor collision (SPLIT)")):
        o.append(stat(lab, [r[k] for r in rows]))
    o.append("")
    o.append(stat("neutral epochs (no spread change)",
                  [r["idle"] + r["solo_nat"] + r["coll_nat"] for r in rows]))
    o.append(stat("split / merge  ratio",
                  [r["coll_vis"] / max(r["solo_vis"], 1e-9) for r in rows]))
    o.append("")
    o.append("  --- spread of X = ln tau across viable visitors ---")
    o.append(stat("sd(X)  with solo-copy", [r["x_sd"] for r in rows]))
    o.append(stat("sd(X)  WITHOUT solo-copy", [r["x_sd_nc"] for r in rows]))
    o.append(stat("ratio  nocopy / pace",
                  [r["x_sd_nc"] / max(r["x_sd"], 1e-9) for r in rows]))
    o.append("")
    o.append(stat("T      with solo-copy", [r["T"] for r in rows]))
    o.append(stat("T      WITHOUT solo-copy", [r["T_nc"] for r in rows]))
    o.append(stat("T ratio nocopy / pace",
                  [r["T_nc"] / max(r["T"], 1e-9) for r in rows]))
    o.append("")
    hdr = ("  scenario              idle  sNat cNat  sVis  cVis |   sd(X)"
           "  sdX_nc  ratio |     T    T_nc")
    o.append(hdr)
    for r in sorted(rows, key=lambda r: (r["access"], r["w_eff"], r["n_vis"])):
        n = f"{r['access']}_W{r['w_eff']}_v{r['n_vis']}"
        o.append(f"  {n:<20s}{r['idle']:6.3f}{r['solo_nat']:6.3f}"
                 f"{r['coll_nat']:6.3f}{r['solo_vis']:6.3f}{r['coll_vis']:6.3f}"
                 f" |{r['x_sd']:8.4f}{r['x_sd_nc']:8.4f}"
                 f"{r['x_sd_nc']/max(r['x_sd'],1e-9):7.1f} |"
                 f"{r['T']:7.3f}{r['T_nc']:8.3f}")
    return "\n".join(o)


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NV = sorted({r["n_vis"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    cmap = plt.get_cmap("viridis")
    for ac, ls, mk in (("rts", "-", "o"), ("basic", "--", "s")):
        for i, nv in enumerate(NV):
            sel = sorted((r for r in rows
                          if r["access"] == ac and r["n_vis"] == nv),
                         key=lambda r: r["w_eff"])
            col = cmap(0.08 + 0.84 * i / (len(NV) - 1))
            w = [r["w_eff"] for r in sel]
            axes[0].semilogx(w, [r["coll_vis"] for r in sel], ls + mk, ms=4,
                             lw=1.2, color=col, label=f"{ac}, $N_v$={nv}")
            axes[1].loglog(w, [r["x_sd"] for r in sel], ls + mk, ms=4,
                           lw=1.2, color=col)
            axes[1].loglog(w, [r["x_sd_nc"] for r in sel], ls + mk, ms=4,
                           lw=1.2, color=col, alpha=0.35)
            axes[2].semilogx(w, [r["T_nc"] / r["T"] for r in sel], ls + mk,
                             ms=4, lw=1.2, color=col)
    for ax, t in zip(axes, ("fraction of epochs that SPLIT the population\n"
                            "(collision containing a visitor)",
                            r"spread sd($X$), $X=\ln\tau$" "\n"
                            "solid: PACE   faded: solo-copy removed",
                            "airtime ratio, solo-copy removed / PACE")):
        ax.set_title(t, fontsize=8.5)
        ax.set_xlabel(r"$W_\mathrm{eff}$ (slots)")
        ax.grid(color="0.9", lw=0.4, which="both")
        ax.set_axisbelow(True)
    axes[2].axhline(1.0, color="k", lw=1.0)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("what holds the visitor population together at "
                 r"$(c_\mathrm{coll},c_\mathrm{idle})=(1.2,1.2)$, "
                 "all 40 scenarios", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    d = os.path.join(ROOT, "results", "figure")
    for ext in ("png", "pdf", "eps"):
        fig.savefig(os.path.join(d, f"fig13-1.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    return os.path.join(d, "fig13-1.*")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(measure, SCEN), 1):
            rows.append(r)
            print(f"[{i}/{len(SCEN)}] {r['access']}_W{r['w_eff']}"
                  f"_v{r['n_vis']}", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = summarise(rows)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("\nwrote", os.path.join(OUT, "data.csv"), "and", figure(rows))


if __name__ == "__main__":
    main()
