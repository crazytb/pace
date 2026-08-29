"""J against c_coll, with c_idle held fixed as an explicit axis.

    .venv/bin/python pace-analysis/fig_ccoll_J.py

One figure per (access, W_eff, N_vis, c_idle); five lines per figure, one per
alpha. c_idle is a GRID VALUE, not a function of c_coll, so a slope on the x
axis is attributable to c_coll alone -- which is the whole point after section
4.5.21, where sweeping the two together made c_coll look decisive when the
dependence was mostly the ramp equation dragging c_idle along.

c_idle = 1.0 is included as a control: it disables the idle update entirely, so
tau can only ever fall from its 1/W_eff start.

T and rho do not depend on alpha, so each (scenario, c_idle, c_coll) is
simulated once and all five alpha curves are read off the same measurement.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import coeff_oracle as CO

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CIDLE = (1.0, 1.2, 1.4, 1.6)
CCOLL = (1.02, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.55, 1.70, 1.85, 2.00)
N_NAT = 10
SCEN = [(nv, N_NAT, w, ac) for ac in ("basic", "rts")
        for w in (420, 1680) for nv in (5, 10, 20, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ccoll_J")


def measure(job):
    core, ci = job
    scn = CO.Scn(*core)
    rows = []
    for cc in CCOLL:
        m = CO.aggregate(CO.batch(scn, ci, cc, CO.EVAL_SEEDS, 30), scn, 0.0)
        rows.append({"access": scn.access, "w_eff": scn.w_eff,
                     "n_vis": scn.n_vis, "n_nat": scn.n_nat,
                     "c_idle": ci, "c_coll": cc,
                     "T": round(m["T"], 5), "rho": round(m["rho"], 5),
                     "tau_cv": round(m["tau_cv"], 5)})
    return rows


def objective(T, rho, a):
    return math.log(max(T, 1e-9)) - a * math.log(max(rho, 1e-9)) ** 2


def plot(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    keys = sorted({(r["access"], r["w_eff"], r["n_vis"], r["c_idle"])
                   for r in rows})
    for ac, w, nv, ci in keys:
        sel = sorted((r for r in rows
                      if (r["access"], r["w_eff"], r["n_vis"], r["c_idle"])
                      == (ac, w, nv, ci)), key=lambda r: r["c_coll"])
        ccs = [r["c_coll"] for r in sel]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for i, a in enumerate(ALPHAS):
            ax.plot(ccs, [objective(r["T"], r["rho"], a) for r in sel],
                    "-o", ms=3.5, lw=1.4, color=cmap(0.08 + 0.84 * i / 4),
                    label=rf"$\alpha$={a}")
        ax.set_xlabel(r"$c_\mathrm{coll}$")
        ax.set_ylabel(r"$J=\ln T-\alpha(\ln\rho)^2$")
        ax.set_title(rf"{ac},  $W$={w},  $N_v$={nv},  $N_n$={N_NAT}"
                     "\n" rf"$c_\mathrm{{idle}}$={ci} (held fixed)", fontsize=9)
        ax.grid(color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        stem = os.path.join(outdir, f"ccollJ_{ac}_W{w}_v{nv}_ci{ci}")
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    return len(keys)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(core, ci) for core in SCEN for ci in CIDLE]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(measure, jobs), 1):
            rows += part
            r = part[0]
            print(f"[{i}/{len(jobs)}] {r['access']}_W{r['w_eff']}"
                  f"_v{r['n_vis']}_ci{r['c_idle']}", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    n = plot(rows, OUT)
    print(f"\nwrote {len(rows)} rows and {n} figures to {OUT}")


if __name__ == "__main__":
    main()
