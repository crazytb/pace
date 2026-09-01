# -*- coding: utf-8 -*-
"""Under carry, does the drift equilibrium become predictive?

    .venv/bin/python pace-analysis/carryeq.py

Section 4.5.27 derived the coefficient ratio by putting the drift zero on the
objective's optimum, and the sweep rejected it: r = 1 beat it and the sign was
wrong. The reason given was Theorem 2 -- an equilibrium is a place to be held,
and a visit that resets to 1/W_eff every time never gets there, so the ramp
wants a small down-step where the equilibrium wants a large one.

Carrying tau across visits removes that objection. The population no longer
re-ramps from scratch; it settles. If it settles where the drift equation says
it should, then the rejected design order comes back:

    tau_J = argmax J(tau)                        analytic, gate passed in 4.5.27
    r_J   = A0(tau_J) / Pc_lis(tau_J)            analytic
    eps   = whatever converges in the visit budget
    carry tau between visits

This measures the premise: sweep r with the scale held fixed, let the sequence
settle, and compare the settled tau against tau_eq(c_idle, c_coll).

tau_nat is taken from results/taunat/data.csv per scenario, not from the 0.052
constant -- section 4.5.32 measured that it moves 3.7x and the constant is
within 10% in only 69 of 360 configurations.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import driftbalance as DB
import params as P

EPS_I = 0.03                      # inside warm's optimal band (section 4.5.31)
RATIOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
SEEDS = CO.EVAL_SEEDS[:16]
VISITS = 60                       # long enough for the carry to settle
SCEN = [(nv, nn, w, ac) for ac in ("rts", "basic") for w in (420, 1680)
        for nv in (5, 20, 50) for nn in (5, 10, 20)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "carryeq")
TAUNAT = os.path.join(ROOT, "results", "taunat", "data.csv")


def _tau_nat_table():
    rows = [{k: (v if k == "access" else float(v)) for k, v in r.items()}
            for r in csv.DictReader(open(TAUNAT))]
    return rows


_TN = _tau_nat_table()


def tau_nat_for(nv, nn, w, ac):
    """Measured attempt rate for this population, nearest available N_vis."""
    cand = [r for r in _TN if r["n_nat"] == nn and r["access"] == ac
            and r["w_eff"] == w and r["c"] == 1.2]
    if not cand:
        cand = [r for r in _TN if r["n_nat"] == nn and r["c"] == 1.2]
    best = min(cand, key=lambda r: abs(r["n_vis"] - nv))
    return best["tau_nat"]


def settle(scn, eps_i, eps_c):
    """Run warm-carry sequences and report the tau the population settles at."""
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = scn.n_vis, scn.n_nat
    ci, cc = math.exp(eps_i), math.exp(eps_c)
    t_sum = n_cnt = 0
    av = an = 0.0
    half = VISITS // 2
    try:
        with P.coefficients(cc, ci), P.window(scn.w_eff):
            for s in SEEDS:
                rng_p, rng = CO._rngs(scn, s)
                tau = np.full(scn.n_vis, 1.0 / scn.w_eff)
                for v in range(VISITS):
                    if v >= half:
                        t_sum += float(np.mean(tau))
                        n_cnt += 1
                    air, _, _, _, carry = f25._run_visit25(
                        f25._sample_ppdus25(rng_p), rng, "pace", tau,
                        *P.ACCESS[scn.access])
                    if v >= half:
                        av += float(air[:scn.n_vis].sum())
                        an += float(air[scn.n_vis:].sum())
                    tau = carry
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    n = len(SEEDS) * (VISITS - half)
    tot = av + an
    pop = scn.n_vis / (scn.n_vis + scn.n_nat)
    return {"tau_settled": t_sum / max(n_cnt, 1),
            "T": tot / (n * scn.w_eff),
            "rho": (av / tot) / pop if tot > 0 else 0.0}


def measure(core):
    scn = CO.Scn(*core)
    tn = tau_nat_for(scn.n_vis, scn.n_nat, scn.w_eff, scn.access)
    out = []
    for r in RATIOS:
        eps_c = r * EPS_I
        m = settle(scn, EPS_I, eps_c)
        # prediction: drift zero for this coefficient pair, with the measured
        # native rate rather than the 0.052 constant
        old = DB.TAU_NAT
        DB.TAU_NAT = tn
        try:
            pred = _tau_eq(math.exp(EPS_I), math.exp(eps_c), scn.n_vis,
                           scn.n_nat)
        finally:
            DB.TAU_NAT = old
        out.append({"access": scn.access, "w_eff": scn.w_eff,
                    "n_vis": scn.n_vis, "n_nat": scn.n_nat,
                    "r": r, "eps_i": EPS_I, "eps_c": round(eps_c, 5),
                    "tau_nat": tn,
                    "tau_settled": round(m["tau_settled"], 6),
                    "tau_eq_pred": round(pred, 6) if pred else float("nan"),
                    "T": round(m["T"], 5), "rho": round(m["rho"], 5)})
    return out


def _tau_eq(c_idle, c_coll, n, n_nat):
    """drift zero, recomputed here so the patched TAU_NAT is picked up."""
    from scipy.optimize import brentq
    ei, ec = math.log(c_idle), math.log(c_coll)

    def f(x):
        tau = math.exp(x)
        a = (1.0 - DB.TAU_NAT) ** n_nat
        b = (n_nat * DB.TAU_NAT * (1.0 - DB.TAU_NAT) ** (n_nat - 1)
             if n_nat else 0.0)
        A0 = (1.0 - tau) ** n * a
        s0 = (1.0 - tau) ** (n - 1) * a
        s1 = (n - 1) * tau * (1.0 - tau) ** (n - 2) * a + \
            (1.0 - tau) ** (n - 1) * b
        return ei * A0 - ec * (1.0 - tau) * (1.0 - s0 - s1)
    lo, hi = math.log(1e-9), math.log(0.95)
    if f(lo) <= 0 or f(hi) >= 0:
        return None
    return math.exp(brentq(f, lo, hi))


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
    ok = [r for r in rows if r["tau_eq_pred"] == r["tau_eq_pred"]]
    ratio = np.array([r["tau_settled"] / r["tau_eq_pred"] for r in ok])
    o = ["", f"{len(rows)} runs; drift equilibrium predicted for {len(ok)}", ""]
    o.append(f"  settled / predicted:  min {ratio.min():.3f}  median "
             f"{np.median(ratio):.3f}  max {ratio.max():.3f}")
    o.append(f"  within 25%: {int((abs(ratio-1) < 0.25).sum())}/{len(ratio)}"
             f"   within 50%: {int((abs(ratio-1) < 0.5).sum())}/{len(ratio)}")
    o.append("")
    o.append("  does the settled tau move with r the way the theory says?")
    o.append(f"    {'r':>6}{'pred median':>13}{'settled median':>16}"
             f"{'ratio':>8}")
    for r in RATIOS:
        s = [x for x in ok if x["r"] == r]
        if not s:
            continue
        p = np.median([x["tau_eq_pred"] for x in s])
        m = np.median([x["tau_settled"] for x in s])
        o.append(f"    {r:>6}{p:13.5f}{m:16.5f}{m/p:8.2f}")
    txt = "\n".join(o)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
