# -*- coding: utf-8 -*-
"""The objective-targeted rule against the simulation oracle, WITH natives.

    .venv/bin/python pace-analysis/dbsweep2.py

Section 4.5.26 tested the rule at N_nat = 0, where rho == 1 forces the target
down to argmax T -- a point so far below 1/N that collisions are rare and r*
runs to 14-132. With natives and alpha > 0 the fairness term pulls the target
back up, r_J lands in 0.6-4.2, and the basic-access degeneracy disappears.
This is the version worth checking against simulation.

    tau_J = argmax [ln T(tau) - alpha (ln rho(tau))^2]      analytic
    r_J   = A0(tau_J) / Pc_lis(tau_J)                       derived
    s     = C_m / sqrt(W_eff)                               calibrated
    (c_idle*, c_coll*) = (exp s, exp[r_J s])

T and rho do not depend on alpha, so one simulation per grid point serves every
alpha; only the oracle and the rule move.

Three readings, because they answer different questions:
  G_rule   the whole rule against the oracle -- includes the calibrated scale
  G_ratio  r_J judged with eps_idle pinned at the oracle's own value, which is
           the only clean test of the DERIVED half
  G_ship   the shipped (1.2, 1.2), for reference
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import driftbalance as DB

C_M = 9.27
N_NAT = 10
ALPHAS = (0.25, 0.5, 1.0)
SEEDS = CO.EVAL_SEEDS[:30]
VISITS = 20
EPS_I = np.geomspace(0.08, 1.00, 8)
EPS_C = np.geomspace(0.02, 5.00, 12)
SCEN = [(nv, N_NAT, w, ac) for ac in ("rts", "basic")
        for w in (105, 210, 420, 840, 1680) for nv in (5, 10, 20, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "driftbalance_J")


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
                         "T": round(m["T"], 5), "rho": round(m["rho"], 5)})
    return rows


def analyse(rows):
    out = []
    keys = sorted({(r["access"], r["w_eff"], r["n_vis"]) for r in rows})
    for ac, w, nv in keys:
        sel = [r for r in rows if (r["access"], r["w_eff"], r["n_vis"])
               == (ac, w, nv)]
        ship = min(sel, key=lambda r: (math.log(r["eps_idle"] / math.log(1.2)) ** 2
                                       + math.log(r["eps_coll"]
                                                  / math.log(1.2)) ** 2))
        for a in ALPHAS:
            J = {id(r): CO.objective(r["T"], r["rho"], a) for r in sel}
            best = max(sel, key=lambda r: J[id(r)])
            rj = DB.r_J(nv, N_NAT, ac, a)
            s = C_M / math.sqrt(w)
            # the whole rule, snapped to the grid
            rule = min(sel, key=lambda r: (math.log(r["eps_idle"] / s) ** 2
                                           + math.log(r["eps_coll"]
                                                      / min(rj * s,
                                                            EPS_C[-1])) ** 2))
            # the derived half only: eps_idle pinned at the oracle's own choice
            row = [r for r in sel if r["eps_idle"] == best["eps_idle"]]
            ec_t = min(rj * best["eps_idle"], EPS_C[-1])
            ratio = min(row, key=lambda r: abs(math.log(r["eps_coll"] / ec_t)))
            out.append({
                "access": ac, "w_eff": w, "n_vis": nv, "alpha": a,
                "tau_J": round(DB.tau_J(nv, N_NAT, ac, a), 6),
                "r_J": round(rj, 4),
                "r_oracle": round(best["eps_coll"] / best["eps_idle"], 4),
                "eps_i_rule": round(s, 4),
                "eps_i_oracle": round(best["eps_idle"], 4),
                "J_oracle": round(J[id(best)], 5),
                "G_rule": round(math.exp(J[id(rule)] - J[id(best)]), 4),
                "G_ratio": round(math.exp(J[id(ratio)] - J[id(best)]), 4),
                "G_ship": round(math.exp(J[id(ship)] - J[id(best)]), 4),
                "T_rule": round(rule["T"], 5), "rho_rule": round(rule["rho"], 5),
                "T_ship": round(ship["T"], 5), "rho_ship": round(ship["rho"], 5),
                "row_span": round(max(J[id(r)] for r in row)
                                  - min(J[id(r)] for r in row), 5),
            })
    return out


def _F(rho):
    return math.exp(-math.log(max(rho, 1e-9)) ** 2)


def summarise(a):
    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"  {name:<38s} min {xs.min():8.4f}  median "
                f"{np.median(xs):8.4f}  max {xs.max():8.4f}")

    o = ["", f"{len(a)} (scenario, alpha) rows, N_nat = {N_NAT}", ""]
    for a_ in ALPHAS:
        s = [r for r in a if r["alpha"] == a_]
        o.append(f"  --- alpha = {a_} ---")
        o.append(stat("G_rule   (whole rule / oracle)",
                      [r["G_rule"] for r in s]))
        o.append(stat("G_ratio  (derived r_J only)", [r["G_ratio"] for r in s]))
        o.append(stat("G_ship   (1.2, 1.2)", [r["G_ship"] for r in s]))
        o.append(stat("r_J predicted", [r["r_J"] for r in s]))
        o.append(stat("r  oracle", [r["r_oracle"] for r in s]))
        o.append(stat("J span along the pinned eps_coll row",
                      [r["row_span"] for r in s]))
        o.append("")
    dom = sum(1 for r in a if r["T_rule"] >= r["T_ship"]
              and _F(r["rho_rule"]) >= _F(r["rho_ship"]))
    dmd = sum(1 for r in a if r["T_rule"] <= r["T_ship"]
              and _F(r["rho_rule"]) <= _F(r["rho_ship"]))
    o.append(f"  rule Pareto-dominates (1.2,1.2): {dom}/{len(a)}")
    o.append(f"  rule Pareto-dominated by (1.2,1.2): {dmd}/{len(a)}")
    o.append("")
    o.append("  scenario            a    tau_J     r_J  r_orac | G_rule G_ratio"
             "  G_ship  span")
    for r in sorted(a, key=lambda r: (r["access"], r["w_eff"], r["n_vis"],
                                      r["alpha"])):
        n = f"{r['access']}_W{r['w_eff']}_v{r['n_vis']}"
        o.append(f"  {n:<18s}{r['alpha']:5.2f}{r['tau_J']:9.5f}"
                 f"{r['r_J']:8.2f}{r['r_oracle']:8.2f} |{r['G_rule']:7.3f}"
                 f"{r['G_ratio']:8.3f}{r['G_ship']:8.3f}{r['row_span']:6.3f}")
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
