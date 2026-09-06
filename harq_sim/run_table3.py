# -*- coding: utf-8 -*-
"""Table III: per-attempt outcome of visitor transmissions.

    .venv/bin/python harq_sim/run_table3.py

The table in the manuscript had no generator in the repository, so the numbers
could not be reproduced when the coefficient changed. This is that generator,
written to the caption's own definition: the setting of the tracking figure
(fig28: N_vis = 20, N_nat = 10, W_eff = 420) in steady state, classifying every
visitor transmission attempt.

An attempt is one visitor's transmission in one contention epoch. It succeeds
if it was the only transmission on the channel that epoch, and collides
otherwise. "Att./frame" is attempts divided by successful frames, i.e. how many
times a visitor has to transmit to land one.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import run_step9_fig17 as _f17
import run_step9_fig24 as _f24
import run_step9_fig25 as _f25

N_V, M, W = 20, 10, 420
SEEDS = [42, 123, 456, 789, 1234]
REPS, VISITS = 8, 60
SCHEMES = [("Standard", "dcf_excl"), ("PACE-static", "pace"),
           ("PACE-dynamic", "pace_dyn"), ("FS", "oracle")]
C_WRULE = 10.16          # section 4.5.40, calibrated at alpha = 0.5
ACCESS = [("basic", "nocd", 0), ("RTS/CTS", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "table3")


def tally(mode: str, coll_cost, succ_oh: int) -> dict:
    """Count visitor attempts, successes and collisions over the steady half."""
    old = (_f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF)
    _f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF = N_V, M, W
    saved_c = (_f17.PND_C_COLL, _f17.PND_C_IDLE)
    if mode == "pace_dyn":
        _f17.PND_C_COLL = _f17.PND_C_IDLE = float(np.exp(C_WRULE / np.sqrt(W)))
        mode = "pace"
        is_pace = True
    else:
        is_pace = mode == "pace"
    att = succ = 0
    try:
        for s in SEEDS:
            for r in range(REPS):
                rng_p = np.random.default_rng(s * 10001 + r * 71 + 7)
                rng = np.random.default_rng(s * 200003 + r * 3163)
                for v in range(VISITS):
                    ppdus = _f25._sample_ppdus25(rng_p)
                    tau0 = (np.full(N_V, 1.0 / W) if is_pace else None)
                    st: dict = {}
                    _f25._run_visit25(ppdus, rng, mode, tau0, coll_cost,
                                      succ_oh, stats=st)
                    if v < VISITS // 2:
                        continue
                    # every visitor transmitter in every epoch is one attempt;
                    # the successful ones are the visitor solos
                    a = st.get("coll_txv", 0) + st.get("solo_vis", 0)
                    att += a
                    succ += st.get("solo_vis", 0)
                    st.clear()
    finally:
        _f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF = old
        _f17.PND_C_COLL, _f17.PND_C_IDLE = saved_c
    return {"att": att, "succ": succ, "coll": att - succ}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for acc, cc, oh in ACCESS:
        for label, mode in SCHEMES:
            t = tally(mode, cc, oh)
            a, s = t["att"], t["succ"]
            rows.append({"access": acc, "scheme": label,
                         "succ_pct": round(100.0 * s / max(a, 1), 1),
                         "coll_pct": round(100.0 * (a - s) / max(a, 1), 1),
                         "att_per_frame": round(a / max(s, 1), 1),
                         "attempts": a, "successes": s})
            print(f"  {acc:>8} {label:<9} "
                  f"succ {rows[-1]['succ_pct']:5.1f}%  "
                  f"coll {rows[-1]['coll_pct']:5.1f}%  "
                  f"att/frame {rows[-1]['att_per_frame']:4.1f}", flush=True)

    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tex = [r"\begin{tabular}{llccc}", r"\hline",
           r"\textbf{Access} & \textbf{Scheme} & \textbf{Succ.} & "
           r"\textbf{Coll.} & \textbf{Att./frame} \\", r"\hline"]
    for i, r in enumerate(rows):
        first = (i % len(SCHEMES) == 0)
        acc = r["access"] if first else ""
        if first and i:
            tex.append(r"\hline")
        tex.append(f"{acc:<8}& {r['scheme']:<9}& ${r['succ_pct']:.0f}\\%$ & "
                   f"${r['coll_pct']:.0f}\\%$ & ${r['att_per_frame']:.1f}$ \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    txt = "\n".join(tex)
    open(os.path.join(OUT, "table3.tex"), "w").write(txt + "\n")
    print()
    print(txt)
    print()
    print(f"c_coll = c_idle = {_f17.PND_C_COLL}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
