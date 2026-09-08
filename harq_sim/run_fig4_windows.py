# -*- coding: utf-8 -*-
"""Paper Fig eval-4b (fig4-3/fig4-4): tracking at a short and a long window.

    .venv/bin/python harq_sim/run_fig4_windows.py
    .venv/bin/python harq_sim/run_fig4_windows.py --fast

fig4-1 and fig4-2 show the tracking at the reference window only, where a fixed
coefficient and a window-scaled one differ by 1.50 against 1.64 and the two
traces nearly coincide. The point of the scaling is what happens away from that
window, so this draws the same trace at the ends of the sweep:

    W_eff = 200 slots (1.8 ms)   the fixed value ramps too slowly to arrive
    W_eff = 1680 slots (15 ms)   the fixed value keeps climbing past the target

PACE-dynamic uses c = exp(C / sqrt(W_eff)), which is 2.05 at the short window
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

WINDOWS = [200, 1680]
MODES = ["dcf_excl", "pace", "pace_dyn", "oracle"]
ACCESS = [("basic", "nocd", 0),
          ("rts", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")


def main():
    ap = argparse.ArgumentParser(
        description="Paper Fig eval-4b (fig4q-1..4) — tracking vs window")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out-dir", default=FIG_DIR)
    a = ap.parse_args()
    visits = _f28.FAST_VISITS if a.fast else _f28.FULL_VISITS
    os.makedirs(a.out_dir, exist_ok=True)
    _f28._LABEL = dict(_f28._LABEL)
    _f28._LABEL["dcf_excl"] = "Standard NPCA"
    _f28._LABEL["pace"] = "PACE-static"
    _f28._LABEL["pace_dyn"] = "PACE-dynamic"

    saved_w, saved_bins = _f28.W, _f28.N_BINS
    idx = 0
    try:
        for acc, cc, oh in ACCESS:
            for w in WINDOWS:
                idx += 1
                _f28.W = w
                # a short visit has only a handful of epochs, so the default
                # 42 bins leave most of them empty and the trace breaks up
                _f28.N_BINS = max(6, min(saved_bins, int(w / 12)))
                fig, ax = plt.subplots(figsize=(2.8, 2.3))
                curves = {m: _f28.binned(m, cc, oh, visits) for m in MODES}
                common = np.logical_and.reduce(
                    [ok for m, (_x, _y, ok) in curves.items()
                     if m != "oracle"])
                for mode in MODES:
                    xs, ys, _ok = curves[mode]
                    if mode != "oracle":
                        ys = np.where(common, ys, np.nan)
                    # bridge structurally empty bins so the line stays whole
                    m = ~np.isnan(ys)
                    st = {k: v for k, v in _f28._STYLE[mode].items()
                          if k not in ("marker", "ms")}
                    st["lw"] = st.get("lw", 1.8) * 0.85
                    ax.plot(xs[m], ys[m], label=_f28._LABEL[mode], **st)
                ax.set_yscale("log")
                ax.set_xlabel("Elapsed time in the visit (ms)")
                ax.set_ylabel("Per-slot transmission rate")
                ax.grid(color="0.9", lw=0.4)
                ax.set_axisbelow(True)
                ax.minorticks_off()
                if acc == "basic" and w == WINDOWS[1]:
                    # the long-window panel is empty below the ramp on the
                    # right, so the shared legend lives there
                    ax.legend(fontsize=6, loc="lower right", frameon=True,
                              handlelength=1.4, borderpad=0.25,
                              labelspacing=0.25)
                fig.tight_layout()
                stem = os.path.join(a.out_dir, f"fig4q-{idx}")
                for ext in ("eps", "png", "pdf"):
                    fig.savefig(f"{stem}.{ext}", format=ext, dpi=300,
                                bbox_inches="tight")
                plt.close(fig)
                print(f"  {acc} W={w} -> {stem}.pdf", flush=True)
    finally:
        _f28.W, _f28.N_BINS = saved_w, saved_bins
    print(f"\nFig 4 quad complete -> {a.out_dir}/fig4q-1..4")


if __name__ == "__main__":
    main()
