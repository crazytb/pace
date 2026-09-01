# -*- coding: utf-8 -*-
"""Paper Fig eval-7 (fig7-1/fig7-2): PACE-static against PACE-dynamic.

    .venv/bin/python harq_sim/run_fig7_dynamic.py
    .venv/bin/python harq_sim/run_fig7_dynamic.py --fast

Every other figure holds W_eff fixed per scenario, but a real visit does not:
W_eff is NPCA_PPDU_REM_DUR, the remaining duration of the OBSS PPDU that
triggered the switch, and it differs from one transition to the next. The
standard hands that number to the STA in the trigger frame, so a coefficient
computed from it costs nothing extra.

  PACE-static    c_coll = c_idle = 1.5, one value for every visit
  PACE-dynamic   c_coll = c_idle = exp(C / sqrt(W_eff)), recomputed per visit

Section 4.5.40 derives the square root: a log-domain MIMD is a stochastic
approximation whose stationary spread is eps sqrt(E), staying near the target
bounds eps by delta/sqrt(E), and the epoch budget E is close to proportional to
the window. C is calibrated once, at alpha = 0.5.

The axis is the SPREAD of the window distribution, because that is what the two
should differ on. W_eff is drawn log-uniformly around a fixed geometric mean,
with the spread s meaning W in [W_mid/s, W_mid*s]; s = 1 is the degenerate case
every other figure runs, where a static value tuned to that window is already
right, and the gap should open as s grows.
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

C_WRULE = 10.16               # calibrated, alpha = 0.5 (section 4.5.40)
C_STATIC = 1.5                # the shipped fixed value
W_MID = 420                   # geometric centre of the window distribution
SPREADS = [1.0, 1.5, 2.0, 3.0, 4.0]
N_VIS, N_NAT = 20, 10
METHODS = ["dcf_excl", "pace_static", "pace_dynamic", "oracle"]
LABEL = {"dcf_excl": "Standard NPCA (CSMA/CA)",
         "pace_static": r"PACE-static ($c=1.5$)",
         "pace_dynamic": r"PACE-dynamic ($c=\exp(C/\sqrt{W_\mathrm{eff}})$)",
         "oracle": "Fair share (FS)"}
STYLE = {"dcf_excl": dict(color="#525252", ls="-.", marker="x", ms=6, lw=1.9),
         "pace_static": dict(color="#ff7f0e", ls="-", marker="^", ms=7, lw=2.2),
         "pace_dynamic": dict(color="#d62728", ls=":", marker="v", ms=6, lw=2.0),
         "oracle": dict(color="#2ca02c", ls="--", marker="D", ms=6, lw=1.8)}
ACCESS = [("basic", "nocd", 0),
          ("rts", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]

FULL_SEEDS, FULL_REPS, FULL_VISITS = [42, 123, 456, 789, 1234], 10, 60
FAST_SEEDS, FAST_REPS, FAST_VISITS = [42], 4, 30

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figure")
OUT_DIR = os.path.join(ROOT, "results", "fig7")


def draw_w(rng, spread):
    """Log-uniform in [W_MID/spread, W_MID*spread]; degenerate at spread = 1."""
    if spread <= 1.0:
        return W_MID
    lo, hi = math.log(W_MID / spread), math.log(W_MID * spread)
    return int(round(math.exp(rng.uniform(lo, hi))))


def run(method, spread, coll_cost, succ_oh, seeds, reps, visits):
    old = (_f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF,
           _f17.PND_C_COLL, _f17.PND_C_IDLE)
    _f25.N_VISITOR, _f25.N_NATIVE = N_VIS, N_NAT
    m_idx = METHODS.index(method)
    sv = sn = 0.0
    span = 0.0
    try:
        for s in seeds:
            for r in range(reps):
                rng_p = np.random.default_rng(s * 10001 + r * 71 + 7)
                rng = np.random.default_rng(s * 200003 + r * 3163 + m_idx * 29)
                # the window sequence is exogenous, so every method sees the
                # same one: its own stream, keyed without the method index
                rng_w = np.random.default_rng(s * 7717 + r * 131)
                for v in range(visits):
                    w = draw_w(rng_w, spread)
                    _f25.W_REF = w
                    if method == "pace_dynamic":
                        _f17.PND_C_COLL = _f17.PND_C_IDLE = \
                            math.exp(C_WRULE / math.sqrt(w))
                    elif method == "pace_static":
                        _f17.PND_C_COLL = _f17.PND_C_IDLE = C_STATIC
                    ppdus = _f25._sample_ppdus25(rng_p)
                    tau0 = (np.full(N_VIS, 1.0 / w)
                            if method.startswith("pace") else None)
                    mode = ("pace" if method.startswith("pace") else method)
                    air, _c, _i, _o, _k = _f25._run_visit25(
                        ppdus, rng, mode, tau0, coll_cost, succ_oh)
                    if v >= visits // 2:
                        sv += float(air[:N_VIS].sum())
                        sn += float(air[N_VIS:].sum())
                        span += w
    finally:
        (_f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF,
         _f17.PND_C_COLL, _f17.PND_C_IDLE) = old
    tot = sv + sn
    pop = N_VIS / (N_VIS + N_NAT)
    return {"T": tot / span, "rho": (sv / tot) / pop if tot > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser(
        description="Paper Fig eval-7 (fig7-1/fig7-2) — static vs dynamic c")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out-dir", default=FIG_DIR)
    a = ap.parse_args()
    seeds, reps, visits = ((FAST_SEEDS, FAST_REPS, FAST_VISITS) if a.fast
                           else (FULL_SEEDS, FULL_REPS, FULL_VISITS))
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for acc, cc, oh in ACCESS:
        for m in METHODS:
            for sp in SPREADS:
                r = run(m, sp, cc, oh, seeds, reps, visits)
                rows.append({"access": acc, "method": m, "spread": sp,
                             "T": round(r["T"], 5), "rho": round(r["rho"], 5)})
                print(f"  {acc:>5} {m:<13} spread={sp:<4} "
                      f"T={r['T']:.3f} rho={r['rho']:.3f}", flush=True)
    with open(os.path.join(OUT_DIR, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for i, (acc, _cc, _oh) in enumerate(ACCESS, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        for ax, key, ylab in ((axes[0], "T", r"Total airtime / $W_\mathrm{eff}$"),
                              (axes[1], "rho", "Visitor airtime proportionality")):
            for m in METHODS:
                sel = sorted((x for x in rows if x["access"] == acc
                              and x["method"] == m), key=lambda x: x["spread"])
                ax.plot([x["spread"] for x in sel], [x[key] for x in sel],
                        label=LABEL[m], **STYLE[m])
            if key == "rho":
                ax.axhline(1.0, color="0.5", ls="--", lw=1.0, zorder=0)
            ax.set_xlabel(r"window spread $s$   ($W_\mathrm{eff}\in"
                          r"[W_0/s,\,W_0 s]$)")
            ax.set_ylabel(ylab)
            ax.grid(color="0.9", lw=0.4)
            ax.set_axisbelow(True)
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{'basic access' if acc == 'basic' else 'RTS/CTS'}: "
                     r"a per-transition window, $W_0=420$ slots "
                     rf"({W_MID * 9 / 1000:.2f} ms), $N_\mathrm{{vis}}={N_VIS}$",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        stem = os.path.join(a.out_dir, f"fig7-{i}")
        for ext in ("eps", "png", "pdf"):
            fig.savefig(f"{stem}.{ext}", format=ext, dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure -> {stem}.pdf")
    print(f"\nFig 7 complete -> {a.out_dir}/fig7-*")


if __name__ == "__main__":
    main()
