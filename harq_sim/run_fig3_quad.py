# -*- coding: utf-8 -*-
"""Paper Fig eval-3 as one full-width row: airtime and rho, both access modes.

    .venv/bin/python harq_sim/run_fig3_quad.py

Replaces the stacked fig3-1/3-2 (airtime) and fig3-3/3-4 (rho) pairs with four
quarter-width panels for a single figure* row. Panels are drawn small (2.8 x
2.3 in) so the 10pt fonts survive the ~0.6 scale of a 0.245\\textwidth render.
Reads fig27's cached CSV; no simulation. The legend appears only in the first
panel and applies to all four.
"""
from __future__ import annotations

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

import run_step9_fig26 as _f26

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")
CSV = os.path.join(ROOT, "results", "step9", "fig27", "data.csv")

N_NAT = 10
METHODS = ["dcf_excl", "pace", "pace_wrule", "oracle"]
STYLE = dict(_f26._STYLE_26)
STYLE["pace_wrule"] = dict(color="#d62728", ls=":", lw=2.0, marker="v", ms=5)
LABEL = dict(_f26._LABEL_26)
LABEL["pace"] = "PACE-static"
LABEL["pace_wrule"] = "PACE-dynamic"

# smaller markers/lines for the quarter-width render
for st in STYLE.values():
    st["ms"] = max(3.5, st.get("ms", 5) * 0.75)
    st["lw"] = st.get("lw", 1.8) * 0.85


def load():
    with open(CSV) as fh:
        return [{k: (v if k in ("access", "method") else float(v))
                 for k, v in r.items()} for r in csv.DictReader(fh)]


def series(rows, access, metric):
    ws = sorted({int(r["W_eff"]) for r in rows})
    out = {}
    for m in METHODS:
        ys = []
        for w in ws:
            sel = [r for r in rows if r["access"] == access
                   and int(r["W_eff"]) == w and r["method"] == m]
            if metric == "useful":
                ys.append(float(np.mean([r["useful"] for r in sel])))
            else:                       # rho as a ratio of sums over seeds
                sv = sum(r["succ_v"] for r in sel)
                tot = sv + sum(r["succ_n"] for r in sel)
                n_vis = int(sel[0]["N"])
                ys.append((sv / tot) / (n_vis / (n_vis + N_NAT)))
        out[m] = ys
    return ws, out


def panel(rows, access, metric, out_stem, legend=False):
    fig, ax = plt.subplots(figsize=(2.8, 2.3))
    ws, data = series(rows, access, metric)
    xs = ws
    for m in METHODS:
        ax.plot(xs, data[m], label=LABEL[m], **STYLE[m])
    if metric == "rho":
        ax.axhline(1.0, color="0.5", ls="--", lw=0.9, zorder=0)
        ax.set_ylabel(r"Proportionality $\rho$")
    else:
        ax.set_ylabel("Total airtime / $W_\\mathrm{eff}$")
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w * 9 / 1000:.1f}".rstrip("0").rstrip(".")
                        for w in ws])
    ax.minorticks_off()
    ax.set_xlabel("Visiting duration (ms)")
    if legend:
        ax.legend(fontsize=6.5, frameon=True, loc="best",
                  handlelength=1.4, borderpad=0.25, labelspacing=0.25)
    ax.grid(color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("eps", "png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{out_stem}.{ext}"), format=ext,
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{out_stem}.pdf")


def main():
    rows = load()
    panel(rows, "basic", "useful", "fig3q-1", legend=True)
    panel(rows, "rts", "useful", "fig3q-2")
    panel(rows, "basic", "rho", "fig3q-3")
    panel(rows, "rts", "rho", "fig3q-4")
    print("Fig 3 quad complete")


if __name__ == "__main__":
    main()
