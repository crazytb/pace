# -*- coding: utf-8 -*-
"""Can the setpoint be built from what one station can actually count?

    .venv/bin/python pace-analysis/localq.py

Section 4.5.28's q uses n_coll_vis -- collisions containing at least one
visitor. A silent station cannot observe that: nothing decodes during a
collision, so it cannot tell whose frames collided. Only 28% of collisions
involve a visitor at the shipped point, so the distinction is not cosmetic.

Four candidates, in decreasing order of what they assume:

  q_glob  idle / (idle + solo_vis + coll_vis)      section 4.5.28; NOT local
  q_pop   idle / (idle + solo_vis + coll_txv/Nv)   population mean; needs N_vis
  q_self  idle / (idle + solo_vis + coll_self)     local, mixed scale
  q_own   idle / (idle + solo_self + coll_self)    local, own transmissions only
  idle    idle fraction                            local, known to fail on N_nat

coll_self counts the collisions the tagged station was in, which is exactly
what a standard DCF station already tracks to drive its backoff. solo_self is
its own successful transmissions.

Judged the way section 4.5.28 judged the others: steer every scenario to one
global setpoint, and report what that costs against the per-scenario oracle.
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
SCEN = [(nv, nn, w, ac) for ac in ("rts", "basic") for w in (210, 420, 1680)
        for nv in (5, 20, 50) for nn in (0, 5, 10, 20)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "localq")

CANDS = ("q_glob", "q_pop", "q_self", "q_own", "idle")


def q_of(r: dict, kind: str) -> float:
    i, sv, ss = r["idle"], r["solo_vis"], r["solo_self"]
    cv, ct, cs = r["coll_vis"], r["coll_txv"], r["coll_self"]
    if kind == "idle":
        return i
    den = {"q_glob": i + sv + cv, "q_pop": i + sv + ct,
           "q_self": i + sv + cs, "q_own": i + ss + cs}[kind]
    return i / den if den > 0 else 1.0


def measure(core):
    scn = CO.Scn(*core)
    rows = []
    for e in EPS:
        c = math.exp(e)
        m = CO.aggregate(CO.batch(scn, c, c, SEEDS, VISITS), scn, 0.0)
        rows.append({
            "access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
            "n_nat": scn.n_nat, "eps": round(float(e), 5),
            "T": round(m["T"], 5), "rho": round(m["rho"], 5),
            "idle": round(m["idle_ep_frac"], 6),
            "solo_vis": round(m["solo_vis_frac"], 6),
            "solo_self": round(m["solo_self_per_ep"], 6),
            "coll_vis": round(m["coll_vis_per_ep"], 6),
            "coll_txv": round(m["coll_txv_per_ep"], 6),
            "coll_self": round(m["coll_self_per_ep"], 6)})
    return rows


def analyse(rows):
    keys = sorted({(r["access"], r["w_eff"], r["n_vis"], r["n_nat"])
                   for r in rows})

    def rows_of(k):
        return sorted((r for r in rows
                       if (r["access"], r["w_eff"], r["n_vis"], r["n_nat"])
                       == k), key=lambda r: r["eps"])

    def opt(k, a):
        rw = rows_of(k)
        js = [CO.objective(r["T"], r["rho"], a) for r in rw]
        i = int(np.argmax(js))
        return None if i in (0, len(rw) - 1) else (rw, js, i)

    def solve(rw, kind, target):
        """eps whose observable equals the target (the observable falls in eps)."""
        v = [q_of(r, kind) for r in rw]
        for j in range(len(rw) - 1):
            if (v[j] - target) * (v[j + 1] - target) <= 0:
                t = (target - v[j]) / ((v[j + 1] - v[j]) or 1e-12)
                return math.exp(math.log(rw[j]["eps"])
                                + t * (math.log(rw[j + 1]["eps"])
                                       - math.log(rw[j]["eps"])))
        return rw[0]["eps"] if v[0] < target else rw[-1]["eps"]

    def J_at(rw, e, a):
        lo = min(range(len(rw) - 1),
                 key=lambda j: abs(math.log(rw[j]["eps"] / e)))
        lo = min(lo, len(rw) - 2)
        x0, x1 = math.log(rw[lo]["eps"]), math.log(rw[lo + 1]["eps"])
        t = max(0.0, min(1.0, (math.log(e) - x0) / (x1 - x0)))
        T = rw[lo]["T"] + t * (rw[lo + 1]["T"] - rw[lo]["T"])
        rho = rw[lo]["rho"] + t * (rw[lo + 1]["rho"] - rw[lo]["rho"])
        return CO.objective(T, rho, a)

    out = ["", f"{len(keys)} scenarios (N_nat 0/5/10/20), steering all of them "
               "to ONE global setpoint", ""]
    monot = {c: [0, 0] for c in CANDS}
    for k in keys:
        rw = rows_of(k)
        for c in CANDS:
            v = [q_of(r, c) for r in rw]
            monot[c][0] += 1
            monot[c][1] += int(all(v[j] >= v[j + 1] - 1e-9
                                   for j in range(len(v) - 1)))
    out.append("  monotone-decreasing in eps (needed for the update's sign):")
    for c in CANDS:
        out.append(f"    {c:<8s} {monot[c][1]:3d}/{monot[c][0]}")
    out.append("")
    out.append(f"  {'alpha':>6}{'signal':>9}{'setpoint':>10}{'G min':>9}"
               f"{'median':>9}{'max':>8}{'<0.90':>10}{'spread of':>11}")
    out.append(f"  {'':>6}{'':>9}{'':>10}{'':>9}{'':>9}{'':>8}{'':>10}"
               f"{'own opt':>11}")
    for a in ALPHAS:
        for c in CANDS:
            tg = []
            for k in keys:
                o = opt(k, a)
                if o is None:
                    continue
                rw, js, i = o
                tg.append(q_of(rw[i], c))
            P = float(np.median(tg))
            G = []
            for k in keys:
                o = opt(k, a)
                if o is None:
                    continue
                rw, js, i = o
                G.append(math.exp(J_at(rw, solve(rw, c, P), a) - js[i]))
            g, t = np.array(G), np.array(tg)
            out.append(f"  {a:>6}{c:>9}{P:10.3f}{g.min():9.3f}"
                       f"{np.median(g):9.3f}{g.max():8.3f}"
                       f"{int((g < 0.90).sum()):6d}/{len(g)}"
                       f"{t.max()/max(t.min(), 1e-9):11.2f}x")
        out.append("")
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(measure, SCEN), 1):
            rows += part
            if i % 12 == 0:
                print(f"[{i}/{len(SCEN)}]", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = analyse(rows)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
