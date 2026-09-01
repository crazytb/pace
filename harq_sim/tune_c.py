# -*- coding: utf-8 -*-
"""Retune the MIMD coefficient on the manuscript's own evaluation grid.

    .venv/bin/python harq_sim/tune_c.py

c_coll = c_idle = 1.2 is inherited from PND, whose analysis is infinite-horizon.
The paper's own grid is finite-window and spans W_eff over 16.8x, so the value
is worth checking rather than assuming.

Run exactly as the paper states its algorithm (Table II, Section V): tau_0 =
1/W_eff, re-initialised every visit, c_coll = c_idle. Grid is the union of what
fig26 and fig27 sweep -- N_vis in {5,10,20,50} at W_eff = 420, and W_eff in
{100,200,420,840,1680} at N_vis = 20 -- with N_nat = 10 and both access modes,
which is what a single fixed coefficient has to serve.

Reported per candidate: total useful airtime T against the oracle's, and the
proportionality rho, since the paper's evaluation frame is (airtime, fairness)
and a coefficient that buys airtime by suppressing the natives is not a win.
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

CS = [1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.60, 1.75, 1.90, 2.10]
SEEDS = [42, 123, 456, 789, 1234]
REPS, VISITS = 12, 40
ACCESS = [("basic", "nocd", 0), ("rts", _f25.COLL_RTS_24M, _f25.OH_SUCC_24M)]

# the union of fig26's and fig27's axes; N_nat is 10 throughout, per Table II
GRID = ([(nv, 420) for nv in (5, 10, 20, 50)]
        + [(20, w) for w in (100, 200, 840, 1680)])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "tune_c")


def run(method, n_vis, w_eff, coll_cost, succ_oh, c):
    """One (method, scenario, coefficient) point, second half of each sequence."""
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


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for acc, cc, oh in ACCESS:
        for n_vis, w_eff in GRID:
            orc = run("oracle", n_vis, w_eff, cc, oh, 1.2)
            for c in CS:
                m = run("pace", n_vis, w_eff, cc, oh, c)
                rows.append({"access": acc, "n_vis": n_vis, "w_eff": w_eff,
                             "c": c, "T": round(m["T"], 5),
                             "rho": round(m["rho"], 5),
                             "T_oracle": round(orc["T"], 5),
                             "rho_oracle": round(orc["rho"], 5),
                             "T_ratio": round(m["T"] / orc["T"], 5)})
            print(f"[{acc} v{n_vis} W{w_eff}] done", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    o = ["", "Retuning c = c_coll = c_idle on the manuscript's own grid",
         f"  tau_0 = 1/W_eff every visit; N_nat = 10; {len(GRID)} scenarios "
         f"x 2 access = {2*len(GRID)} cells", ""]
    o.append("  measured: T = useful airtime / W_eff (0..1);  "
             "T/T_oracle, 1.00 = oracle;  rho = visitor share / population share")
    o.append("")
    o.append(f"  {'c':>6}{'T/T_orc med':>13}{'T/T_orc min':>13}"
             f"{'rho med':>10}{'T med':>9}")
    for c in CS:
        s = [r for r in rows if r["c"] == c]
        tr = np.array([r["T_ratio"] for r in s])
        o.append(f"  {c:6.2f}{np.median(tr):13.3f}{tr.min():13.3f}"
                 f"{np.median([r['rho'] for r in s]):10.3f}"
                 f"{np.median([r['T'] for r in s]):9.3f}")
    o.append("")
    for acc in ("basic", "rts"):
        o.append(f"  --- {acc}: T/T_oracle median by c ---")
        o.append("    " + "".join(f"{c:>7.2f}" for c in CS))
        o.append("    " + "".join(
            f"{np.median([r['T_ratio'] for r in rows if r['c'] == c and r['access'] == acc]):7.3f}"
            for c in CS))
    o.append("")
    o.append("  --- T/T_oracle by W_eff (both access, median) ---")
    o.append("    " + f"{'W_eff':>7}" + "".join(f"{c:>7.2f}" for c in CS))
    for w in (100, 200, 420, 840, 1680):
        s = [r for r in rows if r["w_eff"] == w]
        if not s:
            continue
        o.append(f"    {w:>7}" + "".join(
            f"{np.median([r['T_ratio'] for r in s if r['c'] == c]):7.3f}"
            for c in CS))
    o.append("")
    o.append("  --- T/T_oracle by N_vis at W_eff = 420 (both access, median) ---")
    o.append("    " + f"{'N_vis':>7}" + "".join(f"{c:>7.2f}" for c in CS))
    for nv in (5, 10, 20, 50):
        s = [r for r in rows if r["w_eff"] == 420 and r["n_vis"] == nv]
        o.append(f"    {nv:>7}" + "".join(
            f"{np.median([r['T_ratio'] for r in s if r['c'] == c]):7.3f}"
            for c in CS))
    txt = "\n".join(o)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
