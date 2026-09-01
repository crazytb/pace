# -*- coding: utf-8 -*-
"""Is the best coefficient a closed-form function of W_eff alone?

    .venv/bin/python harq_sim/c_of_w.py

fig3-1 shows PACE crossing below fair share under basic access at long windows,
and fig3-3 shows why: proportionality climbs from 0.37 to 0.93 across the same
sweep. A single coefficient is wrong in opposite directions at the two ends --
too timid in a short window, too eager in a long one -- which is the signature
of a coefficient that should depend on the window.

This measures c*(W_eff) directly in the manuscript's own setting and tests
whether

    ln c* = C / sqrt(W_eff)     equivalently   c* = exp(C / sqrt(W_eff))

fits, which is the form a data collapse supported earlier (exponent 0.44 [0.38,
0.52] under RTS/CTS and 0.54 [0.46, 0.62] under basic). Other exponents are fit
alongside so the 1/2 is tested rather than assumed.

The objective is the paper's own: total useful airtime against the fair-share
reference, with rho reported next to it, since a coefficient that buys airtime
out of the natives' share is not an improvement.
"""
from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import run_step9_fig17 as _f17
import run_step9_fig25 as _f25

CS = [1.10, 1.20, 1.30, 1.40, 1.50, 1.65, 1.80, 2.00, 2.25, 2.50]
WS = [100, 200, 420, 840, 1680]
NVS = [10, 20, 50]
ACCESS = [("basic", "nocd", 0), ("rts", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]
SEEDS = [42, 123, 456, 789, 1234]
REPS, VISITS = 10, 40

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "c_of_w")


def run(method, n_vis, w_eff, coll_cost, succ_oh, c):
    old = (_f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF,
           _f17.PND_C_COLL, _f17.PND_C_IDLE)
    _f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF = n_vis, 10, w_eff
    _f17.PND_C_COLL = _f17.PND_C_IDLE = c
    sv = sn = 0.0
    try:
        for s in SEEDS:
            for r in range(REPS):
                rng_p = np.random.default_rng(s * 10001 + r * 71 + 7)
                rng = np.random.default_rng(s * 200003 + r * 3163
                                            + n_vis * 211 + w_eff * 7)
                for v in range(VISITS):
                    ppdus = _f25._sample_ppdus25(rng_p)
                    tau0 = (np.full(n_vis, 1.0 / w_eff)
                            if method == "pace" else None)
                    air, _c, _i, _o, _k = _f25._run_visit25(
                        ppdus, rng, method, tau0, coll_cost, succ_oh)
                    if v >= VISITS // 2:
                        sv += float(air[:n_vis].sum())
                        sn += float(air[n_vis:].sum())
    finally:
        (_f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF,
         _f17.PND_C_COLL, _f17.PND_C_IDLE) = old
    norm = len(SEEDS) * REPS * (VISITS - VISITS // 2) * w_eff
    tot = sv + sn
    pop = n_vis / (n_vis + 10)
    return {"T": tot / norm, "rho": (sv / tot) / pop if tot > 0 else 0.0}


def peak(cs, ts):
    """Log-quadratic vertex through the best point and its neighbours."""
    i = int(np.argmax(ts))
    if i in (0, len(cs) - 1):
        return float(cs[i]), True
    x = np.log([cs[i - 1], cs[i], cs[i + 1]])
    y = np.array([ts[i - 1], ts[i], ts[i + 1]])
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0:
        return float(cs[i]), True
    return float(math.exp(-b / (2 * a))), False


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for acc, cc, oh in ACCESS:
        for w in WS:
            for nv in NVS:
                orc = run("oracle", nv, w, cc, oh, 1.5)
                for c in CS:
                    m = run("pace", nv, w, cc, oh, c)
                    rows.append({"access": acc, "w_eff": w, "n_vis": nv, "c": c,
                                 "T": round(m["T"], 5),
                                 "rho": round(m["rho"], 5),
                                 "T_fs": round(orc["T"], 5),
                                 "rho_fs": round(orc["rho"], 5)})
                print(f"  [{acc} W{w} v{nv}] done", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    o = ["", "c* that maximises total useful airtime, per (access, W_eff, N_vis)",
         "  measured: T = useful airtime / W_eff; c* by log-quadratic vertex",
         "  'edge' means the peak sat at a grid end and the cell is excluded", ""]
    o.append(f"  {'access':>6}{'W_eff':>7}{'N_vis':>7}{'c*':>8}"
             f"{'T at c*':>9}{'T at 1.5':>10}{'rho at c*':>11}{'rho at 1.5':>12}")
    pts = []
    for acc, _cc, _oh in ACCESS:
        for w in WS:
            for nv in NVS:
                sel = sorted((r for r in rows if r["access"] == acc
                              and r["w_eff"] == w and r["n_vis"] == nv),
                             key=lambda r: r["c"])
                cs = [r["c"] for r in sel]
                ts = [r["T"] for r in sel]
                cstar, edge = peak(cs, ts)
                at15 = [r for r in sel if r["c"] == 1.50][0]
                tag = "edge" if edge else f"{cstar:.3f}"
                o.append(f"  {acc:>6}{w:>7}{nv:>7}{tag:>8}"
                         f"{max(ts):9.3f}{at15['T']:10.3f}"
                         f"{sel[int(np.argmax(ts))]['rho']:11.3f}"
                         f"{at15['rho']:12.3f}")
                if not edge:
                    pts.append((acc, w, nv, cstar))
    o.append("")
    if len(pts) >= 4:
        o.append("  fitting ln c* = C * W_eff^(-theta)")
        lw = np.array([math.log(p[1]) for p in pts])
        lc = np.array([math.log(math.log(p[3])) for p in pts])
        th, b = np.polyfit(lw, lc, 1)
        o.append(f"    free exponent: theta = {-th:.3f}  "
                 f"(C = {math.exp(b):.2f}),  residual sd(log) = "
                 f"{np.std(lc - np.polyval([th, b], lw)):.3f}")
        for theta, name in ((0.5, "sqrt"), (1.0, "linear"), (0.0, "constant")):
            C = float(np.mean([math.log(p[3]) * p[1] ** theta for p in pts]))
            pred = np.array([C * p[1] ** (-theta) for p in pts])
            act = np.array([math.log(p[3]) for p in pts])
            o.append(f"    theta = {theta} ({name}): C = {C:6.2f},  "
                     f"ln c* pred/act median {np.median(pred/act):.3f}, "
                     f"sd(log) {np.std(np.log(pred/act)):.3f}")
        o.append("")
        o.append(f"  {'access':>6}{'W_eff':>7}{'N_vis':>7}{'c*':>8}"
                 f"{'c = exp(C/sqrt(W))':>20}")
        C = float(np.mean([math.log(p[3]) * math.sqrt(p[1]) for p in pts]))
        for acc, w, nv, cstar in pts:
            o.append(f"  {acc:>6}{w:>7}{nv:>7}{cstar:8.3f}"
                     f"{math.exp(C / math.sqrt(w)):20.3f}")
    txt = "\n".join(o)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
