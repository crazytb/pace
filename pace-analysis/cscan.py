# -*- coding: utf-8 -*-
"""Why does the scale constant C differ between basic and RTS/CTS?

    .venv/bin/python pace-analysis/cscan.py

Section 4.5.27 adopted c_idle = c_coll = exp(C/sqrt(W_eff)), so C is fitted on
the DIAGONAL eps_idle = eps_coll -- which neither cidle_scan (c_coll pinned at
1.2) nor dbsweep2 (separate axes) actually samples. Part (a) measures it there.

The two access modes differ in two ways at once: a collision costs 12 slots
under RTS/CTS and about 85 under basic (max L_i over the colliders), and a
success carries a 10-slot handshake under RTS/CTS and none under basic. Part
(b) sweeps the two costs independently, holding everything else fixed, so a
difference in C can be attributed rather than guessed at.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import params as P

ALPHAS = (0.25, 0.5, 1.0)
SEEDS = CO.EVAL_SEEDS[:30]
VISITS = 20
EPS = np.geomspace(0.05, 1.20, 14)
N_NAT = 10

DIAG = [(nv, N_NAT, w, ac) for ac in ("rts", "basic")
        for w in (105, 210, 420, 840, 1680) for nv in (5, 10, 20, 50)]

# part (b): rts timing, but the collision cost and the handshake swept apart
COSTS = (1, 4, 12, 30, 60, 100, 150)
OHS = (0, 5, 10, 20)
COST_SCEN = [(nv, N_NAT, w) for w in (420, 1680) for nv in (10, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "cscan")


def _diag(job):
    core = job
    scn = CO.Scn(*core)
    rows = []
    for e in EPS:
        c = math.exp(e)
        m = CO.aggregate(CO.batch(scn, c, c, SEEDS, VISITS), scn, 0.0)
        rows.append({"access": scn.access, "w_eff": scn.w_eff,
                     "n_vis": scn.n_vis, "coll_cost": -1, "succ_oh": -1,
                     "eps": round(float(e), 5),
                     "T": round(m["T"], 5), "rho": round(m["rho"], 5)})
    return rows


def _cost(job):
    nv, nn, w, cc, oh = job
    scn = CO.Scn(nv, nn, w, "rts")
    rows = []
    for e in EPS:
        c = math.exp(e)
        m = CO.aggregate(CO.batch(scn, c, c, SEEDS, VISITS,
                                  access=(cc, oh)), scn, 0.0)
        rows.append({"access": "custom", "w_eff": w, "n_vis": nv,
                     "coll_cost": cc, "succ_oh": oh, "eps": round(float(e), 5),
                     "T": round(m["T"], 5), "rho": round(m["rho"], 5)})
    return rows


def peak(rows, alpha):
    """Log-quadratic fit through the best point and its neighbours, so the
    answer is not just which grid point happened to win."""
    js = [CO.objective(r["T"], r["rho"], alpha) for r in rows]
    i = int(np.argmax(js))
    if i in (0, len(rows) - 1):
        return float(rows[i]["eps"]), True          # pinned at an edge
    x = np.log([rows[j]["eps"] for j in (i - 1, i, i + 1)])
    y = np.array([js[j] for j in (i - 1, i, i + 1)])
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0:
        return float(rows[i]["eps"]), True
    return float(math.exp(-b / (2 * a))), False


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(_diag, DIAG), 1):
            rows += part
            print(f"[diag {i}/{len(DIAG)}]", flush=True)
        jobs = [(nv, nn, w, cc, 10) for nv, nn, w in COST_SCEN for cc in COSTS]
        jobs += [(nv, nn, w, 12, oh) for nv, nn, w in COST_SCEN for oh in OHS
                 if oh != 10]
        for i, part in enumerate(ex.map(_cost, jobs), 1):
            rows += part
            print(f"[cost {i}/{len(jobs)}]", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    o = ["", "(a) C = eps* sqrt(W_eff) on the diagonal c_idle = c_coll, "
             f"N_nat = {N_NAT}", ""]
    for a in ALPHAS:
        per = {}
        for ac in ("rts", "basic"):
            cs = []
            for w in (105, 210, 420, 840, 1680):
                for nv in (5, 10, 20, 50):
                    sel = sorted((r for r in rows if r["access"] == ac
                                  and r["w_eff"] == w and r["n_vis"] == nv),
                                 key=lambda r: r["eps"])
                    e, edge = peak(sel, a)
                    if not edge:
                        cs.append(e * math.sqrt(w))
            per[ac] = np.array(cs)
        r, b = per["rts"], per["basic"]
        o.append(f"  alpha = {a}   (clean peaks: rts {len(r)}/20, "
                 f"basic {len(b)}/20)")
        o.append(f"    rts    C  min {r.min():6.2f}  median {np.median(r):6.2f}"
                 f"  max {r.max():6.2f}")
        o.append(f"    basic  C  min {b.min():6.2f}  median {np.median(b):6.2f}"
                 f"  max {b.max():6.2f}")
        o.append(f"    ratio basic/rts of medians: "
                 f"{np.median(b)/np.median(r):.3f}")
        o.append("")

    o.append("(b) attribution: rts timing, one cost varied at a time")
    o.append("")
    for a in ALPHAS:
        o.append(f"  alpha = {a}")
        o.append("    collision cost sweep (handshake held at 10 slots)")
        o.append("      " + "".join(f"{c:>9}" for c in COSTS) + "   <- slots")
        for nv, nn, w in COST_SCEN:
            line = ""
            for cc in COSTS:
                sel = sorted((r for r in rows if r["coll_cost"] == cc
                              and r["succ_oh"] == 10 and r["w_eff"] == w
                              and r["n_vis"] == nv), key=lambda r: r["eps"])
                e, edge = peak(sel, a)
                line += f"{e*math.sqrt(w):9.2f}" if not edge else f"{'edge':>9}"
            o.append(f"      W{w} v{nv}: {line}")
        o.append("    handshake sweep (collision held at 12 slots)")
        o.append("      " + "".join(f"{c:>9}" for c in sorted(OHS))
                 + "   <- slots")
        for nv, nn, w in COST_SCEN:
            line = ""
            for oh in sorted(OHS):
                sel = sorted((r for r in rows if r["succ_oh"] == oh
                              and r["coll_cost"] == 12 and r["w_eff"] == w
                              and r["n_vis"] == nv), key=lambda r: r["eps"])
                e, edge = peak(sel, a)
                line += f"{e*math.sqrt(w):9.2f}" if not edge else f"{'edge':>9}"
            o.append(f"      W{w} v{nv}: {line}")
        o.append("")
    txt = "\n".join(o)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
