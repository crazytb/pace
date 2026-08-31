"""Does tau_0 = 1/W_eff matter, at the shipped (c_coll, c_idle) = (1.2, 1.2)?

    .venv/bin/python pace-analysis/init_scan.py

Two arms over the FULL scenario set (access x W_eff x N_vis, N_nat = 10):

  one_probe   tau_0 = 1/W_eff        the shipped rule
  uniform     tau_0 ~ U(0,1) i.i.d.  no knowledge of the window at all

Paired on the PPDU stream, so a difference is attributable to the start point.
Section 4.5's standing rule applies: every claim is a min/median/max over all
40 scenarios, and no row is quoted as if it stood for the set.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO

C_COLL = C_IDLE = 1.20
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VISITS = 30
SCEN = [(nv, 10, w, ac) for ac in ("rts", "basic")
        for w in (105, 210, 420, 840, 1680) for nv in (5, 10, 20, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "init_scan")


def measure(core):
    scn = CO.Scn(*core)
    arms = {t: CO.batch(scn, C_IDLE, C_COLL, CO.EVAL_SEEDS, VISITS, tau0=t)
            for t in ("one_probe", "uniform")}
    row = {"access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
           "n_nat": scn.n_nat}
    for t, rows in arms.items():
        m = CO.aggregate(rows, scn, 0.0)
        row[f"T_{t}"] = round(m["T"], 5)
        row[f"rho_{t}"] = round(m["rho"], 5)
        row[f"coll_{t}"] = round(m["coll_frac"], 5)
        row[f"idle_{t}"] = round(m["idle_frac"], 5)
        row[f"cv_{t}"] = round(m["tau_cv"], 5)
    for a in ALPHAS:
        ju = CO.objective(row["T_uniform"], row["rho_uniform"], a)
        jo = CO.objective(row["T_one_probe"], row["rho_one_probe"], a)
        row[f"dJ_a{a}"] = round(ju - jo, 5)
    ci = CO.bootstrap(arms["uniform"], arms["one_probe"], scn, 0.5)
    row["dJ_a0.5_lo"], row["dJ_a0.5_hi"] = (round(v, 5) for v in ci["dJ_ci"])
    return row


def summarise(rows):
    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"  {name:<28s} min {xs.min():7.4f}  median "
                f"{np.median(xs):7.4f}  max {xs.max():7.4f}")

    out = ["", f"{len(rows)} scenarios, (c_coll, c_idle) = "
               f"({C_COLL}, {C_IDLE}), {len(CO.EVAL_SEEDS)} seeds x "
               f"{VISITS} visits", ""]
    out.append(stat("T uniform / T one_probe",
                    [r["T_uniform"] / r["T_one_probe"] for r in rows]))
    out.append(stat("rho uniform", [r["rho_uniform"] for r in rows]))
    out.append(stat("rho one_probe", [r["rho_one_probe"] for r in rows]))
    for a in ALPHAS:
        out.append(stat(f"exp(dJ), alpha={a}",
                        [math.exp(r[f"dJ_a{a}"]) for r in rows]))
    sig = sum(r["dJ_a0.5_hi"] < 0 for r in rows)
    out.append("")
    out.append(f"  uniform worse than one_probe at alpha=0.5, 95% paired "
               f"bootstrap CI excluding 0: {sig}/{len(rows)}")
    out.append(f"  uniform better, CI excluding 0: "
               f"{sum(r['dJ_a0.5_lo'] > 0 for r in rows)}/{len(rows)}")
    out.append("")
    hdr = ("  scenario                T_1/W    T_U   ratio    rho_1/W  rho_U"
           "   expdJ(.5)")
    out.append(hdr)
    for r in sorted(rows, key=lambda r: (r["access"], r["w_eff"], r["n_vis"])):
        name = f"{r['access']}_W{r['w_eff']}_v{r['n_vis']}"
        out.append(f"  {name:<20s} {r['T_one_probe']:6.3f} "
                   f"{r['T_uniform']:6.3f} {r['T_uniform']/r['T_one_probe']:7.3f}"
                   f"   {r['rho_one_probe']:6.3f} {r['rho_uniform']:6.3f}"
                   f"   {math.exp(r['dJ_a0.5']):7.3f}")
    return "\n".join(out)


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    WS = sorted({r["w_eff"] for r in rows})
    NV = sorted({r["n_vis"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    cmap = plt.get_cmap("viridis")
    for ac, ls, mk in (("rts", "-", "o"), ("basic", "--", "s")):
        for i, nv in enumerate(NV):
            sel = sorted((r for r in rows
                          if r["access"] == ac and r["n_vis"] == nv),
                         key=lambda r: r["w_eff"])
            col = cmap(0.08 + 0.84 * i / (len(NV) - 1))
            lab = rf"{ac}, $N_v$={nv}"
            axes[0].semilogx([r["w_eff"] for r in sel],
                             [r["T_one_probe"] for r in sel], ls + mk,
                             ms=4, lw=1.2, color=col, label=lab)
            axes[1].semilogx([r["w_eff"] for r in sel],
                             [r["T_uniform"] for r in sel], ls + mk,
                             ms=4, lw=1.2, color=col)
            axes[2].loglog([r["w_eff"] for r in sel],
                           [r["T_uniform"] / r["T_one_probe"] for r in sel],
                           ls + mk, ms=4, lw=1.2, color=col)
    for ax, t in zip(axes, (r"$\tau_0=1/W_\mathrm{eff}$ (shipped)",
                            r"$\tau_0\sim U(0,1)$",
                            r"efficiency ratio $T_U/T_{1/W}$")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(r"$W_\mathrm{eff}$ (slots)")
        ax.grid(color="0.9", lw=0.4, which="both")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"total useful airtime $T$")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[2].axhline(1.0, color="k", lw=1.0)
    axes[2].set_ylim(4e-3, 1.6)
    axes[2].set_ylabel(r"uniform / one-probe")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle(r"one-probe initialisation at $(c_\mathrm{coll},"
                 r"c_\mathrm{idle})=(1.2,1.2)$, all 40 scenarios",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    d = os.path.join(ROOT, "results", "figure")
    for ext in ("png", "pdf", "eps"):
        fig.savefig(os.path.join(d, f"fig12-1.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    return os.path.join(d, "fig12-1.*")


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
