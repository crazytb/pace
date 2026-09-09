# -*- coding: utf-8 -*-
"""Paper Fig eval-1 as a side-by-side pair: total airtime vs N_vis.

    .venv/bin/python harq_sim/run_fig1_pair.py

Emits fig1q-1 (basic) and fig1q-2 (rts) at full plot size (5.0 x 2.5 in) for
a stacked two-row, column-width render. Reads fig26_visitor's cached CSV; no
simulation. The legend appears only in the first panel.
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

import run_step9_fig26 as _f26

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")
CSV = os.path.join(ROOT, "results", "step9", "fig26_visitor", "data.csv")

METHODS = ["dcf_excl", "pace", "oracle"]
STYLE = {m: dict(_f26._STYLE_26[m]) for m in METHODS}
LABEL = dict(_f26._LABEL_26)
LABEL["pace"] = "PACE-static"


def main():
    with open(CSV) as fh:
        rows = [{k: (v if k in ("access", "method") else float(v))
                 for k, v in r.items()} for r in csv.DictReader(fh)]
    nvs = sorted({int(r["N_visitor"]) for r in rows})
    for i, acc in enumerate(("basic", "rts"), start=1):
        fig, ax = plt.subplots(figsize=(5.0, 2.2))
        for m in METHODS:
            ys = [float(np.mean([r["useful"] for r in rows
                                 if r["access"] == acc and r["method"] == m
                                 and int(r["N_visitor"]) == n]))
                  for n in nvs]
            ax.plot(nvs, ys, label=LABEL[m], **STYLE[m])
        ax.set_xticks(nvs)
        ax.set_xlabel("Number of visitor STAs $N_\\mathrm{vis}$")
        ax.set_ylabel("Total airtime / $W_\\mathrm{eff}$")
        ax.set_ylim(0.0, 0.8)
        if acc == "basic":
            ax.legend(fontsize=7.5, frameon=True, loc="best")
        ax.grid(color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        stem = os.path.join(FIG_DIR, f"fig1q-{i}")
        for ext in ("eps", "png", "pdf"):
            fig.savefig(f"{stem}.{ext}", format=ext, dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {stem}.pdf")
    print("Fig 1 pair complete")


if __name__ == "__main__":
    main()
