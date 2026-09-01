# -*- coding: utf-8 -*-
"""Do the ratio and the scale really separate under carry?

    .venv/bin/python pace-analysis/sepgrid.py

Section 4.5.33 measured the ratio axis at one scale (eps_idle = 0.03) and found
the derived r_J beating r = 1. The claim that came with it -- that r sets the
operating point while eps_idle only sets how fast you reach it -- was never
tested, because the scale never moved. This sweeps both.

Two things also change from 4.5.33, both because they are more faithful:

  * NATIVE STATE CARRIES between visits. The OBSS does not stop when the
    visitor's NPCA window closes, so the natives' (CW, backoff) should persist;
    resetting them every visit is the odd choice. The engine took native_init
    already but never returned the end state, so it now leaves it in the stats
    dict (passive, no rng consumed).
  * tau carries too, which is the regime 4.5.31 and 4.5.33 established.

The separation is testable as two independent statements:

  settled tau depends on r, not on eps_idle
  convergence time depends on eps_idle, not much on r

alpha is post-hoc: T and rho do not depend on it, so one run serves all three.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import params as P

EPS_I = tuple(round(float(x), 5) for x in np.geomspace(0.008, 0.30, 6))
RATIOS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
ALPHAS = (0.25, 0.5, 1.0)
SEEDS = CO.EVAL_SEEDS[:8]
VISITS = 100   # small eps_idle needs many visits to settle under carry
HALF = VISITS // 2
SCEN = [(nv, nn, w, ac) for ac in ("rts", "basic")
        for w in (210, 420, 1680) for nv in (5, 20, 50) for nn in (5, 10, 20)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "sepgrid")


def run(scn, eps_i, eps_c):
    """Sequences with BOTH tau and the native contention state carried."""
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = scn.n_vis, scn.n_nat
    ci, cc = math.exp(eps_i), math.exp(eps_c)
    av = an = 0.0
    nat_tx = nat_slots = 0
    paths = []
    try:
        with P.coefficients(cc, ci), P.window(scn.w_eff):
            for s in SEEDS:
                rng_p, rng = CO._rngs(scn, s)
                tau = np.full(scn.n_vis, 1.0 / scn.w_eff)
                nat = None
                path = []
                for v in range(VISITS):
                    path.append(float(np.mean(tau)))
                    st: dict = {}
                    air, _, _, _, carry = f25._run_visit25(
                        f25._sample_ppdus25(rng_p), rng, "pace", tau,
                        *P.ACCESS[scn.access], native_init=nat, stats=st)
                    if v >= HALF:
                        av += float(air[:scn.n_vis].sum())
                        an += float(air[scn.n_vis:].sum())
                        nat_tx += st.get("nat_tx", 0)
                        nat_slots += st.get("nat_slots", 0)
                    tau = carry
                    nat = st.get("native_end")
                paths.append(path)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    p = np.array(paths)                       # seeds x visits
    med = np.median(p, axis=0)
    settled = float(np.mean(med[HALF:]))
    # first visit after which the trajectory stays within 10% of settled
    bad = np.where(np.abs(np.log(np.maximum(med, 1e-12) / settled))
                   >= math.log(1.10))[0]
    conv = int(bad[-1] + 1) if len(bad) else 0
    n = len(SEEDS) * (VISITS - HALF)
    tot = av + an
    pop = scn.n_vis / (scn.n_vis + scn.n_nat)
    return {"tau_settled": round(settled, 6), "conv_visits": conv,
            "T": round(tot / (n * scn.w_eff), 5),
            "rho": round((av / tot) / pop if tot > 0 else 0.0, 5),
            "tau_nat": round(nat_tx / max(nat_slots, 1), 5)}


def measure(core):
    scn = CO.Scn(*core)
    out = []
    for ei in EPS_I:
        for r in RATIOS:
            m = run(scn, ei, r * ei)
            out.append({"access": scn.access, "w_eff": scn.w_eff,
                        "n_vis": scn.n_vis, "n_nat": scn.n_nat,
                        "eps_i": round(ei, 5), "r": r,
                        "eps_c": round(r * ei, 5), **m})
    return out


def summarise(rows):
    keys = sorted({(r["access"], r["w_eff"], r["n_vis"], r["n_nat"])
                   for r in rows})

    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"    {name:<40s} min {xs.min():8.3f}  median "
                f"{np.median(xs):8.3f}  max {xs.max():8.3f}")

    o = ["", f"{len(keys)} scenarios x {len(EPS_I)} eps_idle x {len(RATIOS)} r"
             f"   (tau AND native state carried)", ""]

    o.append("  SEPARATION TEST 1 -- does the operating point depend on r only?")
    sr, se = [], []
    for k in keys:
        sel = [x for x in rows
               if (x["access"], x["w_eff"], x["n_vis"], x["n_nat"]) == k]
        for ei in EPS_I:                    # spread across r at fixed scale
            v = [x["tau_settled"] for x in sel if x["eps_i"] == ei]
            if min(v) > 0:
                sr.append(max(v) / min(v))
        for r in RATIOS:                    # spread across scale at fixed r
            v = [x["tau_settled"] for x in sel if x["r"] == r]
            if min(v) > 0:
                se.append(max(v) / min(v))
    o.append(stat("settled tau spread over r  (scale fixed)", sr))
    o.append(stat("settled tau spread over eps_i (r fixed)", se))
    o.append("")

    o.append("  SEPARATION TEST 2 -- does convergence time depend on eps_idle?")
    o.append(f"    {'eps_idle':>10}{'conv visits (median)':>22}")
    for ei in EPS_I:
        v = [x["conv_visits"] for x in rows if x["eps_i"] == ei]
        o.append(f"    {ei:10.4f}{np.median(v):22.0f}")
    o.append(f"    {'r':>10}{'conv visits (median)':>22}")
    for r in RATIOS:
        v = [x["conv_visits"] for x in rows if x["r"] == r]
        o.append(f"    {r:10.2f}{np.median(v):22.0f}")
    o.append("")

    o.append("  BEST (eps_i, r) per scenario, and what the scale costs")
    for a in ALPHAS:
        rs, es, g1 = [], [], []
        for k in keys:
            sel = [x for x in rows
                   if (x["access"], x["w_eff"], x["n_vis"], x["n_nat"]) == k]
            J = {id(x): CO.objective(x["T"], x["rho"], a) for x in sel}
            b = max(sel, key=lambda x: J[id(x)])
            rs.append(b["r"])
            es.append(b["eps_i"])
            # best achievable if the scale is fixed at the grid's midpoint
            mid = EPS_I[len(EPS_I) // 2]
            bm = max((x for x in sel if x["eps_i"] == mid),
                     key=lambda x: J[id(x)])
            g1.append(math.exp(J[id(bm)] - J[id(b)]))
        o.append(f"    alpha = {a}:  best r median {np.median(rs):.2f} "
                 f"(range {min(rs)}-{max(rs)}), best eps_i median "
                 f"{np.median(es):.4f}")
        o.append(stat(f"      G with the scale pinned at {mid:.4f}", g1))
    o.append("")

    o.append("  tau_nat under native carry (compare 4.5.32's fresh-init values)")
    for nn in (5, 10, 20):
        v = [x["tau_nat"] for x in rows if x["n_nat"] == nn]
        o.append(f"    N_nat = {nn:>2}: median {np.median(v):.4f}  "
                 f"(fresh-init was 0.0613 / 0.0498 / 0.0417)")
    return "\n".join(o)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(measure, SCEN), 1):
            rows += part
            print(f"[{i}/{len(SCEN)}]", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = summarise(rows)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
