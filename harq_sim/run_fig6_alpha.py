# -*- coding: utf-8 -*-
"""Paper Fig eval-6 (fig6-1/fig6-2): the objective against N_vis, per alpha.

    .venv/bin/python harq_sim/run_fig6_alpha.py
    .venv/bin/python harq_sim/run_fig6_alpha.py --fast

The other evaluation figures report total useful airtime, which is only half of
what the scheme is for: PACE is meant to raise channel airtime while keeping
the visitors' share close to proportional. This figure puts both on one axis,

    J = ln T - alpha (ln rho)^2

where T is useful airtime over the window and rho is the visitors' airtime
share divided by their population share, so rho = 1 is proportional and the
penalty is symmetric in over- and under-service. alpha is the operator's
weighting, swept rather than fixed because there is no single right value.

T and rho do not depend on alpha, so one simulation per (scheme, N_vis) serves
every curve.

Same setting as Fig eval-1: N_nat = 10, W_eff = 420, both access modes,
tau_0 = 1/W_eff re-initialised each visit.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_step9_fig17 as _f17
import run_step9_fig25 as _f25

METHODS = ["dcf_excl", "pace", "oracle"]
LABEL = {"dcf_excl": "Standard NPCA", "pace": "PACE",
         "oracle": "FS (fair share)"}
STYLE = {"dcf_excl": dict(color="#525252", ls="-.", marker="x", ms=6, lw=1.9),
         "pace": dict(color="#ff7f0e", ls="-", marker="^", ms=6, lw=2.2),
         "oracle": dict(color="#2ca02c", ls="--", marker="D", ms=5, lw=1.8)}

NV_LIST = [5, 10, 20, 50]
M_FIX, W_FIX = 10, 420
ALPHAS = [0.0, 0.5, 1.0]
ACCESS = [("basic", "nocd", 0),
          ("rts", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]

FULL_SEEDS, FULL_REPS, FULL_VISITS = [42, 123, 456, 789, 1234], 12, 40
FAST_SEEDS, FAST_REPS, FAST_VISITS = [42], 4, 20

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")
OUT_DIR = os.path.join(ROOT, "results", "fig6")


def run(method, n_vis, coll_cost, succ_oh, seeds, reps, visits):
    old = (_f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF)
    _f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF = n_vis, M_FIX, W_FIX
    m_idx = METHODS.index(method)
    sv = sn = 0.0
    try:
        for s in seeds:
            for r in range(reps):
                rng_p = np.random.default_rng(s * 10001 + r * 71 + 7)
                rng = np.random.default_rng(s * 200003 + r * 3163
                                            + m_idx * 29 + n_vis * 211)
                for v in range(visits):
                    ppdus = _f25._sample_ppdus25(rng_p)
                    tau0 = (np.full(n_vis, 1.0 / W_FIX)
                            if method == "pace" else None)
                    air, _c, _i, _o, _k = _f25._run_visit25(
                        ppdus, rng, method, tau0, coll_cost, succ_oh)
                    if v >= visits // 2:
                        sv += float(air[:n_vis].sum())
                        sn += float(air[n_vis:].sum())
    finally:
        _f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF = old
    norm = len(seeds) * reps * (visits - visits // 2) * W_FIX
    tot = sv + sn
    pop = n_vis / (n_vis + M_FIX)
    return {"T": tot / norm,
            "rho": (sv / tot) / pop if tot > 0 else 0.0}


def objective(T, rho, alpha):
    return (math.log(max(T, 1e-9))
            - alpha * math.log(max(rho, 1e-9)) ** 2)


def main():
    ap = argparse.ArgumentParser(
        description="Paper Fig eval-6 (fig6-1/fig6-2) — objective vs N_vis")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="plot from the cached results/fig6/data.csv")
    ap.add_argument("--out-dir", default=FIG_DIR)
    a = ap.parse_args()
    seeds, reps, visits = ((FAST_SEEDS, FAST_REPS, FAST_VISITS) if a.fast
                           else (FULL_SEEDS, FULL_REPS, FULL_VISITS))
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    if a.replot:
        with open(os.path.join(OUT_DIR, "data.csv")) as fh:
            rows = [{"access": r["access"], "method": r["method"],
                     "n_vis": int(r["n_vis"]), "T": float(r["T"]),
                     "rho": float(r["rho"])} for r in csv.DictReader(fh)]
        return plot(rows, a.out_dir)

    rows = []
    for acc, cc, oh in ACCESS:
        for m in METHODS:
            for nv in NV_LIST:
                r = run(m, nv, cc, oh, seeds, reps, visits)
                rows.append({"access": acc, "method": m, "n_vis": nv,
                             "T": round(r["T"], 5), "rho": round(r["rho"], 5)})
                print(f"  {acc:>5} {m:<9} N_vis={nv:<3} "
                      f"T={r['T']:.3f} rho={r['rho']:.3f}", flush=True)
    with open(os.path.join(OUT_DIR, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot(rows, a.out_dir)


def plot(rows, out_dir):
    for i, (acc, _cc, _oh) in enumerate(ACCESS, start=1):
        fig, axes = plt.subplots(1, len(ALPHAS),
                                 figsize=(3.33 * len(ALPHAS), 3.2), sharex=True)
        for ax, al in zip(axes, ALPHAS):
            for m in METHODS:
                sel = [x for x in rows if x["access"] == acc and x["method"] == m]
                sel.sort(key=lambda x: x["n_vis"])
                ax.plot([x["n_vis"] for x in sel],
                        [objective(x["T"], x["rho"], al) for x in sel],
                        label=LABEL[m], **STYLE[m])
            ax.set_xscale("log")
            ax.set_xticks(NV_LIST)
            ax.set_xticklabels([str(n) for n in NV_LIST])
            ax.set_xlabel(r"visitor STAs $N_\mathrm{vis}$")
            ax.set_title(rf"$\alpha={al}$", fontsize=10)
            ax.grid(color="0.9", lw=0.4)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(r"$J=\ln T-\alpha(\ln\rho)^2$")
        axes[0].legend(fontsize=7.5, loc="best")
        fig.tight_layout()
        stem = os.path.join(out_dir, f"fig6-{i}")
        for ext in ("eps", "png", "pdf"):
            fig.savefig(f"{stem}.{ext}", format=ext, dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure -> {stem}.pdf")
    print(f"\nFig 6 complete (c = {_f17.PND_C_COLL}) -> {out_dir}/fig6-*")


if __name__ == "__main__":
    main()
