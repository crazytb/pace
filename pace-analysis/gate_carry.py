# -*- coding: utf-8 -*-
"""Does the analytic model still hold once tau and the natives both carry?

    .venv/bin/python pace-analysis/gate_carry.py

Section 4.5.27(b) put the model through a gate -- feed it the engine's measured
tau, compare the T and rho it predicts against the engine's -- and it passed on
T (0.943-1.074) and missed on rho by a median 0.874. That was with the natives
reset every visit and N_nat fixed at 10.

Section 4.5.34 then used the model with the natives carried and N_nat from 5 to
20, and its J diverged badly from the engine's (-1.110 against -0.424 in one
case). The derived ratio's failure sits downstream of that, so the gate has to
be re-run here before anything else is worth fixing.

The tau fed to the model is the EPOCH-AVERAGED tau over viable visitors
(stats["tau_sum"]/stats["tau_cnt"]), not the visit-start value sepgrid.py
recorded: the model's probabilities are per contention epoch, so that is the
quantity they consume.

Both population sizes are reported -- N_vis, and the time-averaged E|V| the
self-exclusion correction gives -- since which one belongs in the model is
exactly what section 4.5.34 could not settle.
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
import viability as V

SEEDS = CO.EVAL_SEEDS[:10]
VISITS = 80
HALF = VISITS // 2
POINTS = ((0.017, 1.0), (0.017, 3.0), (0.034, 2.0), (0.034, 5.0),
          (0.070, 3.0), (0.145, 5.0))
SCEN = [(nv, nn, w, ac) for ac in ("rts", "basic")
        for w in (210, 420, 1680) for nv in (5, 20, 50) for nn in (5, 10, 20)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "gate_carry")


def n_eff(w_eff, n_vis):
    """Time-average of E|V| across the window."""
    ws = np.arange(1, w_eff + 1)
    return float(n_vis * np.mean([V.f_len(w - P.L_HS) for w in ws]))


def run(scn, eps_i, eps_c):
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = scn.n_vis, scn.n_nat
    av = an = 0.0
    tsum = tcnt = 0.0
    ntx = nsl = 0
    try:
        with P.coefficients(math.exp(eps_c), math.exp(eps_i)), \
                P.window(scn.w_eff):
            for s in SEEDS:
                rng_p, rng = CO._rngs(scn, s)
                tau = np.full(scn.n_vis, 1.0 / scn.w_eff)
                nat = None
                for v in range(VISITS):
                    st: dict = {}
                    air, _, _, _, carry = f25._run_visit25(
                        f25._sample_ppdus25(rng_p), rng, "pace", tau,
                        *P.ACCESS[scn.access], native_init=nat, stats=st)
                    if v >= HALF:
                        av += float(air[:scn.n_vis].sum())
                        an += float(air[scn.n_vis:].sum())
                        tsum += st.get("tau_sum", 0.0)
                        tcnt += st.get("tau_cnt", 0)
                        ntx += st.get("nat_tx", 0)
                        nsl += st.get("nat_slots", 0)
                    tau = carry
                    nat = st.get("native_end")
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    n = len(SEEDS) * (VISITS - HALF)
    tot = av + an
    pop = scn.n_vis / (scn.n_vis + scn.n_nat)
    return {"tau_bar": tsum / max(tcnt, 1),
            "tau_nat": ntx / max(nsl, 1),
            "T": tot / (n * scn.w_eff),
            "rho": (av / tot) / pop if tot > 0 else 0.0}


def measure(core):
    scn = CO.Scn(*core)
    out = []
    for ei, r in POINTS:
        m = run(scn, ei, r * ei)
        row = {"access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
               "n_nat": scn.n_nat, "eps_i": ei, "r": r,
               "tau_bar": round(m["tau_bar"], 6),
               "tau_nat": round(m["tau_nat"], 5),
               "T": round(m["T"], 5), "rho": round(m["rho"], 5)}
        old = DB.TAU_NAT
        try:
            DB.TAU_NAT = m["tau_nat"]
            for tag, n in (("nv", scn.n_vis),
                           ("ev", max(n_eff(scn.w_eff, scn.n_vis), 1.05))):
                _, _, T = DB.airtimes(m["tau_bar"], n, scn.n_nat, scn.access)
                rho = DB.rho_model(m["tau_bar"], n, scn.n_nat, scn.access)
                row[f"T_{tag}"] = round(T, 5)
                row[f"rho_{tag}"] = round(rho, 5)
        finally:
            DB.TAU_NAT = old
        out.append(row)
    return out


def summarise(rows):
    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"    {name:<34s} min {xs.min():7.3f}  median "
                f"{np.median(xs):7.3f}  max {xs.max():7.3f}  "
                f"within20% {int((abs(xs-1) < 0.2).sum()):>3}/{len(xs)}")

    o = ["", f"GATE re-run under carry: {len(rows)} runs "
             f"({len(SCEN)} scenarios x {len(POINTS)} coefficient points)",
         "  model fed the engine's epoch-averaged tau and measured tau_nat", ""]
    for tag, lab in (("nv", "n = N_vis"), ("ev", "n = time-averaged E|V|")):
        o.append(f"  --- {lab} ---")
        o.append(stat("T model/engine", [r[f"T_{tag}"] / r["T"] for r in rows]))
        o.append(stat("rho model/engine",
                      [r[f"rho_{tag}"] / max(r["rho"], 1e-9) for r in rows]))
        o.append("")
    o.append("  4.5.27(b) reference, fresh natives, N_nat=10:")
    o.append("    T   0.943 / 1.015 / 1.074      rho 0.842 / 0.874 / 1.124")
    o.append("")
    o.append("  by N_nat (n = N_vis):")
    for nn in (5, 10, 20):
        s = [r for r in rows if r["n_nat"] == nn]
        o.append(f"    N_nat={nn:>2}  T {np.median([r['T_nv']/r['T'] for r in s]):.3f}"
                 f"   rho {np.median([r['rho_nv']/max(r['rho'],1e-9) for r in s]):.3f}")
    o.append("  by access (n = N_vis):")
    for ac in ("rts", "basic"):
        s = [r for r in rows if r["access"] == ac]
        o.append(f"    {ac:>6}  T {np.median([r['T_nv']/r['T'] for r in s]):.3f}"
                 f"   rho {np.median([r['rho_nv']/max(r['rho'],1e-9) for r in s]):.3f}")
    o.append("  by W_eff (n = N_vis):")
    for w in (210, 420, 1680):
        s = [r for r in rows if r["w_eff"] == w]
        o.append(f"    W={w:>5}  T {np.median([r['T_nv']/r['T'] for r in s]):.3f}"
                 f"   rho {np.median([r['rho_nv']/max(r['rho'],1e-9) for r in s]):.3f}")
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
