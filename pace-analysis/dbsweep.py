# -*- coding: utf-8 -*-
"""Does the drift-balance pair land where the sweep's optimum is?

    .venv/bin/python pace-analysis/dbsweep.py

N_nat = 0 throughout, which is where the rule's own assumptions hold: no
natives, every visitor starts at 1/W_eff, rho == 1 so J = ln T and alpha drops
out. Section 4.5.25 measured that solo-copy is then close to a self-loop.

The c_coll axis runs far past anything swept before. eps_coll >= ~3 already
means "one collision puts tau on the 1e-4 floor", so the grid's top end covers
every larger prediction as one policy rather than pretending to resolve them.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import driftbalance as DB

C_M = 9.27                                   # section 4.5.23's fitted scale
SEEDS = CO.EVAL_SEEDS[:30]
VISITS = 20
EPS_I = np.geomspace(0.08, 1.00, 8)
EPS_C = np.geomspace(0.02, 8.00, 12)
SCEN = [(nv, 0, w, ac) for ac in ("rts", "basic")
        for w in (105, 210, 420, 840, 1680) for nv in (5, 10, 20, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "driftbalance")


def measure(core):
    scn = CO.Scn(*core)
    rows = []
    for ei in EPS_I:
        for ec in EPS_C:
            m = CO.aggregate(CO.batch(scn, math.exp(ei), math.exp(ec),
                                      SEEDS, VISITS), scn, 0.0)
            rows.append({"access": scn.access, "w_eff": scn.w_eff,
                         "n_vis": scn.n_vis, "eps_idle": round(float(ei), 5),
                         "eps_coll": round(float(ec), 5),
                         "c_idle": round(math.exp(ei), 5),
                         "c_coll": round(math.exp(ec), 5),
                         "T": round(m["T"], 5), "tau_cv": round(m["tau_cv"], 5),
                         "x_sd": round(m["x_sd"], 5)})
    return rows


def analyse(rows):
    out = []
    keys = sorted({(r["access"], r["w_eff"], r["n_vis"]) for r in rows})
    for ac, w, nv in keys:
        sel = [r for r in rows if (r["access"], r["w_eff"], r["n_vis"])
               == (ac, w, nv)]
        best = max(sel, key=lambda r: r["T"])
        # the rule
        ei_hat = C_M / math.sqrt(w)
        r_st = DB.r_star(nv, ac)
        ec_hat = r_st * ei_hat
        # nearest grid point to the rule, for a like-for-like read of the cost
        near = min(sel, key=lambda r: (math.log(r["eps_idle"] / ei_hat) ** 2
                                       + math.log(r["eps_coll"]
                                                  / min(ec_hat,
                                                        EPS_C[-1])) ** 2))
        # the shipped pair
        ship = min(sel, key=lambda r: (math.log(r["c_idle"] / 1.2) ** 2
                                       + math.log(r["c_coll"] / 1.2) ** 2))
        # how flat is the eps_coll axis at the best eps_idle?
        col = sorted((r for r in sel if r["eps_idle"] == best["eps_idle"]),
                     key=lambda r: r["eps_coll"])
        out.append({
            "access": ac, "w_eff": w, "n_vis": nv,
            "r_star": round(r_st, 3),
            "eps_i_hat": round(ei_hat, 4), "eps_c_hat": round(ec_hat, 4),
            "eps_i_star": round(best["eps_idle"], 4),
            "eps_c_star": round(best["eps_coll"], 4),
            "r_swept": round(best["eps_coll"] / best["eps_idle"], 3),
            "T_star": round(best["T"], 5),
            "T_hat": round(near["T"], 5),
            "T_ship": round(ship["T"], 5),
            "G_hat": round(near["T"] / best["T"], 4),
            "G_ship": round(ship["T"] / best["T"], 4),
            "T_at_top_eps_c": round(col[-1]["T"], 5),
            "T_at_low_eps_c": round(col[0]["T"], 5),
        })
    return out


def summarise(a):
    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"  {name:<36s} min {xs.min():9.4f}  median "
                f"{np.median(xs):9.4f}  max {xs.max():9.4f}")

    o = ["", f"{len(a)} scenarios, N_nat = 0, rho == 1 so J = ln T", ""]
    o.append(stat("r* predicted  (eps_c / eps_i)", [r["r_star"] for r in a]))
    o.append(stat("r  swept      (eps_c / eps_i)", [r["r_swept"] for r in a]))
    o.append(stat("ratio  predicted / swept",
                  [r["r_star"] / max(r["r_swept"], 1e-9) for r in a]))
    o.append("")
    o.append(stat("eps_idle  predicted / swept",
                  [r["eps_i_hat"] / r["eps_i_star"] for r in a]))
    o.append(stat("eps_coll  predicted / swept",
                  [r["eps_c_hat"] / r["eps_c_star"] for r in a]))
    o.append("")
    o.append(stat("G = T(rule) / T(best)", [r["G_hat"] for r in a]))
    o.append(stat("G = T(shipped 1.2,1.2) / T(best)", [r["G_ship"] for r in a]))
    o.append("")
    o.append(stat("T at the TOP of the eps_coll axis",
                  [r["T_at_top_eps_c"] / r["T_star"] for r in a]))
    o.append(stat("T at the BOTTOM of the eps_coll axis",
                  [r["T_at_low_eps_c"] / r["T_star"] for r in a]))
    o.append("")
    hdr = ("  scenario             r*      r_swept |  e_i^   e_i*   e_c^   e_c*"
           " |  T*     G_rule  G_ship")
    o.append(hdr)
    for r in sorted(a, key=lambda r: (r["access"], r["w_eff"], r["n_vis"])):
        n = f"{r['access']}_W{r['w_eff']}_v{r['n_vis']}"
        o.append(f"  {n:<18s}{r['r_star']:7.1f}{r['r_swept']:12.2f} |"
                 f"{r['eps_i_hat']:6.2f}{r['eps_i_star']:7.2f}"
                 f"{r['eps_c_hat']:7.2f}{r['eps_c_star']:7.2f} |"
                 f"{r['T_star']:7.3f}{r['G_hat']:8.3f}{r['G_ship']:8.3f}")
    return "\n".join(o)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(measure, SCEN), 1):
            rows += part
            r = part[0]
            print(f"[{i}/{len(SCEN)}] {r['access']}_W{r['w_eff']}"
                  f"_v{r['n_vis']}", flush=True)
    with open(os.path.join(OUT, "grid.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = analyse(rows)
    with open(os.path.join(OUT, "compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(a[0]))
        w.writeheader()
        w.writerows(a)
    txt = summarise(a)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
