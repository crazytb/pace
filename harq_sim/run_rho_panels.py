# -*- coding: utf-8 -*-
"""Proportionality companions to the window sweep and the ablation.

    .venv/bin/python harq_sim/run_rho_panels.py

fig1 already has a rho companion in fig2, but fig3 (window sweep) and fig5
(ablation) report total airtime only, so a scheme that buys airtime out of the
natives' share looks the same as one that does not. These are the missing
panels:

    fig3-3 / fig3-4   rho against the visiting duration   (from fig27's data)
    fig5-3 / fig5-4   rho against N_vis, per variant      (from fig29's data)

rho = (visitor airtime / total airtime) / (N_vis / (N_vis + N_nat)), so rho = 1
is proportional to population and the reference line is drawn there.

No new simulation: both source CSVs already carry succ_v and succ_n per seed,
which is everything rho needs.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_step9_fig17 as _f17

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")

N_NAT = 10

STYLE = {
    "dcf_excl": dict(color="#525252", ls="-.", marker="x", ms=6, lw=1.9),
    "pace": dict(color="#ff7f0e", ls="-", marker="^", ms=7, lw=2.2),
    "oracle": dict(color="#2ca02c", ls="--", marker="D", ms=6, lw=1.8),
    "pace_wrule": dict(color="#d62728", ls=":", marker="v", ms=6, lw=2.0),
    "pace_noexcl": dict(color="#1f77b4", ls="--", marker="o", ms=6, lw=1.9),
    "pace_rand": dict(color="#9467bd", ls=":", marker="s", ms=6, lw=1.9),
}
LABEL = {
    "dcf_excl": "Standard NPCA (CSMA/CA)", "pace": "PACE",
    "oracle": "Fair share (FS)", "pace_noexcl": "No self-exclusion",
    "pace_wrule": r"PACE, $c=\exp(C/\sqrt{W_\mathrm{eff}})$",
    "pace_rand": r"Naive $\tau_0\sim\mathcal{U}(0,1)$",
}


def load(path):
    return [{k: (v if k in ("access", "method") else float(v))
             for k, v in r.items()} for r in csv.DictReader(open(path))]


def rho_of(rows, n_vis):
    """Pool the seeds first, then form rho once -- it is a ratio of sums."""
    sv = sum(r["succ_v"] for r in rows)
    sn = sum(r["succ_n"] for r in rows)
    tot = sv + sn
    if tot <= 0:
        return 0.0
    return (sv / tot) / (n_vis / (n_vis + N_NAT))


def panel(ax, xs, series, xlabel, xticklabels=None):
    for m, ys in series:
        ax.plot(xs, ys, label=LABEL[m], **STYLE[m])
    ax.axhline(1.0, color="0.5", ls="--", lw=1.0, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels or [str(x) for x in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Visitor airtime proportionality")
    ax.grid(color="0.9", lw=0.4)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext in ("eps", "png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{ext}"), format=ext,
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{stem}.pdf")


def window_sweep():
    rows = load(os.path.join(ROOT, "results", "step9", "fig27", "data.csv"))
    ws = sorted({int(r["W_eff"]) for r in rows})
    n_vis = int(rows[0]["N"])
    for i, acc in enumerate(("basic", "rts"), start=3):
        series = []
        for m in ("dcf_excl", "pace", "pace_wrule", "oracle"):
            ys = [rho_of([r for r in rows if r["access"] == acc
                          and int(r["W_eff"]) == w and r["method"] == m], n_vis)
                  for w in ws]
            series.append((m, ys))
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        panel(ax, ws, series, "Visiting duration (ms)",
              [f"{w * 9 / 1000:.1f}".rstrip("0").rstrip(".") for w in ws])
        ax.set_title(f"{'basic access' if acc == 'basic' else 'RTS/CTS'}"
                     rf",  $N_\mathrm{{vis}}={n_vis}$,  "
                     rf"$c={_f17.PND_C_COLL}$", fontsize=10)
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        save(fig, f"fig3-{i}")


def ablation():
    rows = load(os.path.join(ROOT, "results", "step9", "fig29", "data.csv"))
    nvs = sorted({int(r["N_visitor"]) for r in rows})
    order = ["pace", "pace_noexcl", "pace_rand", "dcf_excl"]
    for i, acc in enumerate(("basic", "rts"), start=3):
        series = []
        for m in order:
            ys = [rho_of([r for r in rows if r["access"] == acc
                          and int(r["N_visitor"]) == nv and r["method"] == m], nv)
                  for nv in nvs]
            series.append((m, ys))
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        panel(ax, nvs, series, r"Number of visitor STAs $N_\mathrm{vis}$")
        ax.set_title(f"{'basic access' if acc == 'basic' else 'RTS/CTS'}"
                     rf",  ablation,  $c={_f17.PND_C_COLL}$", fontsize=10)
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        save(fig, f"fig5-{i}")


if __name__ == "__main__":
    print(f"rho panels at c = {_f17.PND_C_COLL}")
    window_sweep()
    ablation()
