# -*- coding: utf-8 -*-
"""Is there a LOCAL, N-free setpoint an outer loop could steer c toward?

    .venv/bin/python pace-analysis/setpoint.py

Section 4.5.27 left the scale as the only coefficient worth getting right
(r = 1 is within 2% of the oracle ratio), so the question for a per-visit
adaptation is whether a station can tell from its own visit whether its single
c was too big or too small.

Everything a visitor can count for free during a visit: idle epochs, epochs
where it was a silent listener to a collision, epochs where someone soloed.
The drift equation says n_I / n_C estimates A0(tau) / Pc_lis(tau), so that
ratio reports which tau the station actually operated at -- without knowing
N_vis, N_nat, or W_eff.

For an outer loop to work, the ratio observed at the optimal c must be roughly
the SAME across scenarios. If it is, that value is the setpoint and

    c(next) = (1-lam) c(now) + lam * f(observed - setpoint)

converges to it. If it is not, there is nothing to steer toward.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO

ALPHAS = (0.25, 0.5, 1.0)
SEEDS = CO.EVAL_SEEDS[:30]
VISITS = 20
EPS = np.geomspace(0.05, 1.20, 14)
SCEN = [(nv, 10, w, ac) for ac in ("rts", "basic")
        for w in (105, 210, 420, 840, 1680) for nv in (5, 10, 20, 50)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "setpoint")


def measure(core):
    scn = CO.Scn(*core)
    rows = []
    for e in EPS:
        c = math.exp(e)
        m = CO.aggregate(CO.batch(scn, c, c, SEEDS, VISITS), scn, 0.0)
        rows.append({
            "access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
            "eps": round(float(e), 5),
            "T": round(m["T"], 5), "rho": round(m["rho"], 5),
            # everything below is countable by a visitor from its own visit
            "idle": round(m["idle_ep_frac"], 5),
            "coll_vis": round(m["coll_vis_per_ep"], 5),
            "coll_nat": round(m["coll_nat_per_ep"], 5),
            "solo_vis": round(m["solo_vis_frac"], 5),
            "solo_nat": round(m["solo_nat_frac"], 5),
        })
    return rows


def peak(rows, alpha):
    js = [CO.objective(r["T"], r["rho"], alpha) for r in rows]
    i = int(np.argmax(js))
    if i in (0, len(rows) - 1):
        return None
    x = np.log([rows[j]["eps"] for j in (i - 1, i, i + 1)])
    y = np.array([js[j] for j in (i - 1, i, i + 1)])
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0:
        return None
    return float(math.exp(-b / (2 * a))), i


def interp(rows, i, e_star, key):
    """Read a countable quantity at the fitted peak, log-linear between the two
    bracketing grid points."""
    lo = max(0, min(i, len(rows) - 2))
    x0, x1 = math.log(rows[lo]["eps"]), math.log(rows[lo + 1]["eps"])
    y0, y1 = rows[lo][key], rows[lo + 1][key]
    t = (math.log(e_star) - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def summarise(rows):
    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"    {name:<30s} min {xs.min():8.4f}  median "
                f"{np.median(xs):8.4f}  max {xs.max():8.4f}  "
                f"spread {xs.max()/max(xs.min(),1e-9):6.2f}x")

    keys = sorted({(r["access"], r["w_eff"], r["n_vis"]) for r in rows})
    o = ["", "candidate setpoints, read at the objective-optimal eps on the "
             "diagonal c_idle = c_coll", ""]
    for a in ALPHAS:
        rec = []
        for ac, w, nv in keys:
            sel = sorted((r for r in rows if (r["access"], r["w_eff"],
                                              r["n_vis"]) == (ac, w, nv)),
                         key=lambda r: r["eps"])
            p = peak(sel, a)
            if p is None:
                continue
            e, i = p
            g = lambda k: interp(sel, i, e, k)
            idle, cv, cn, sv, sn = (g("idle"), g("coll_vis"), g("coll_nat"),
                                    g("solo_vis"), g("solo_nat"))
            rec.append({"access": ac, "w_eff": w, "n_vis": nv, "alpha": a,
                        "eps_star": e, "idle": idle, "coll_vis": cv,
                        "coll_nat": cn, "solo_vis": sv, "solo_nat": sn,
                        "nI_over_nCvis": idle / max(cv, 1e-9),
                        "nI_over_nCall": idle / max(cv + cn, 1e-9),
                        "nI_over_nSvis": idle / max(sv, 1e-9),
                        "sv_over_cv": sv / max(cv, 1e-9)})
        o.append(f"  alpha = {a}   ({len(rec)}/{len(keys)} clean peaks)")
        for k, lab in (("idle", "idle fraction"),
                       ("coll_vis", "visitor-collision fraction"),
                       ("solo_vis", "visitor-solo fraction"),
                       ("nI_over_nCvis", "n_idle / n_coll(visitor)"),
                       ("nI_over_nCall", "n_idle / n_coll(all)"),
                       ("sv_over_cv", "n_solo(vis) / n_coll(vis)")):
            o.append(stat(lab, [r[k] for r in rec]))
        o.append("")
        if a == ALPHAS[0]:
            hdr = ("    scenario             eps*   idle  cVis  cNat  sVis |"
                   "  I/Cv   I/Call  Sv/Cv")
            o.append(hdr)
            for r in sorted(rec, key=lambda r: (r["access"], r["w_eff"],
                                                r["n_vis"])):
                n = f"{r['access']}_W{r['w_eff']}_v{r['n_vis']}"
                o.append(f"    {n:<18s}{r['eps_star']:7.3f}{r['idle']:7.3f}"
                         f"{r['coll_vis']:6.3f}{r['coll_nat']:6.3f}"
                         f"{r['solo_vis']:6.3f} |{r['nI_over_nCvis']:7.2f}"
                         f"{r['nI_over_nCall']:8.2f}{r['sv_over_cv']:7.2f}")
            o.append("")
    return "\n".join(o), rec


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(measure, SCEN), 1):
            rows += part
            r = part[0]
            print(f"[{i}/{len(SCEN)}] {r['access']}_W{r['w_eff']}"
                  f"_v{r['n_vis']}", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt, _ = summarise(rows)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
