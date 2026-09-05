# -*- coding: utf-8 -*-
"""Paper Fig eval-4b (fig4-3/fig4-4): tracking at a short and a long window.

    .venv/bin/python harq_sim/run_fig4_windows.py
    .venv/bin/python harq_sim/run_fig4_windows.py --fast

fig4-1 and fig4-2 show the tracking at the reference window only, where a fixed
coefficient and a window-scaled one differ by 1.50 against 1.64 and the two
traces nearly coincide. The point of the scaling is what happens away from that
window, so this draws the same trace at the ends of the sweep:

    W_eff = 100 slots (0.9 ms)   the fixed value ramps too slowly to arrive
    W_eff = 1680 slots (15 ms)   the fixed value keeps climbing past the target

PACE-dynamic uses c = exp(C / sqrt(W_eff)), which is 2.76 at the short window
and 1.28 at the long one, against the fixed 1.5 in both. C is calibrated at
alpha = 0.5 (section 4.5.40).

Reuses fig28's instrumented visit, with its module-level W patched per panel.
"""
from __future__ import annotations

import argparse
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
import run_step9_fig28 as _f28

WINDOWS = [100, 1680]
MODES = ["dcf_excl", "pace", "pace_dyn", "oracle"]
ACCESS = [("basic", "nocd", 0),
          ("rts", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")


def main():
    ap = argparse.ArgumentParser(
        description="Paper Fig eval-4b (fig4-3/fig4-4) — tracking vs window")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out-dir", default=FIG_DIR)
    a = ap.parse_args()
    visits = _f28.FAST_VISITS if a.fast else _f28.FULL_VISITS
    os.makedirs(a.out_dir, exist_ok=True)
    _f28._LABEL = dict(_f28._LABEL)
    _f28._LABEL["pace"] = "PACE-static (measured)"
    _f28._LABEL["pace_dyn"] = "PACE-dynamic (measured)"

    saved_w, saved_bins = _f28.W, _f28.N_BINS
    try:
        for i, (acc, cc, oh) in enumerate(ACCESS, start=3):
            fig, axes = plt.subplots(1, len(WINDOWS),
                                     figsize=(4.4 * len(WINDOWS), 3.1))
            for ax, w in zip(axes, WINDOWS):
                _f28.W = w
                # a 0.9 ms visit has only a handful of epochs, so the default
                # 42 bins leave most of them empty and the trace breaks up
                _f28.N_BINS = max(6, min(saved_bins, int(w / 12)))
                for mode in MODES:
                    xs, ys = _f28.binned(mode, cc, oh, visits)
                    st = {k: v for k, v in _f28._STYLE[mode].items()
                          if k not in ("marker", "ms")}
                    ax.plot(xs, ys, label=_f28._LABEL[mode], **st)
                ax.set_yscale("log")
                ax.set_xlabel("Elapsed time in the visit (ms)", fontsize=9)
                ax.set_title(rf"$W_\mathrm{{eff}}={w}$ slots "
                             rf"({w * 9 / 1000:.2f} ms):  fixed $c=1.5$ vs "
                             rf"dynamic $c={math.exp(_f28.C_WRULE / math.sqrt(w)):.2f}$",
                             fontsize=9)
                ax.grid(True, ls=":", lw=0.6, alpha=0.7)
                ax.tick_params(labelsize=8)
                print(f"  {acc} W={w} done", flush=True)
            axes[0].set_ylabel("Per-slot transmission rate", fontsize=9)
            axes[0].legend(fontsize=7, loc="best")
            fig.tight_layout()
            stem = os.path.join(a.out_dir, f"fig4-{i}")
            for ext in ("eps", "png", "pdf"):
                fig.savefig(f"{stem}.{ext}", format=ext, dpi=300,
                            bbox_inches="tight")
            plt.close(fig)
            print(f"  Figure -> {stem}.pdf")
    finally:
        _f28.W, _f28.N_BINS = saved_w, saved_bins
    print(f"\nFig 4b complete -> {a.out_dir}/fig4-3, fig4-4")


if __name__ == "__main__":
    main()
