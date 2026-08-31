# -*- coding: utf-8 -*-
"""Should tau carry across visits instead of resetting to 1/W_eff?

    .venv/bin/python pace-analysis/carrytau.py

Two things prompted this. The reset is odd on its face -- a station that just
finished an NPCA visit throws away everything it learned. And the manuscript's
own figures do NOT reset: run_step9_fig25.run_config seeds tau at 1/N_visitor
and then feeds each visit's end-of-visit tau into the next one, while every
analysis in section 4.5 resets to 1/W_eff. The two describe different
algorithms, which is worth settling.

Three arms, each swept over its own eps grid so none is judged at another's
optimum:

  cold_w   tau_0 = 1/W_eff every visit          the analysis convention
  cold_nv  tau_0 = 1/N_vis  every visit         fig24's cold baseline
  warm     tau_0 = previous visit's end tau     what run_config does
           (first visit of a sequence starts at 1/W_eff)

fig24 measured warm carry against cold_nv and found it did not win, blaming
tau inflation: the end-of-visit tau reflects the shrunken viable set rather
than the full population, and native wins give solo-copy no chance to correct.
That comparison started 20x higher than 1/W_eff does, so it does not settle
this one. tau0_mean is reported to test the inflation story directly.

This calls _run_visit25 directly rather than going through coeff_oracle.batch:
carrying state between visits is not something the shared helper should learn
for one experiment.
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
EPS = np.geomspace(0.008, 1.20, 20)   # low end widened: warm wants eps ~0.25x of cold
SEEDS = CO.EVAL_SEEDS[:16]
VISITS = 30                      # second half is the steady state
ARMS = ("cold_w", "cold_nv", "warm")
SCEN = [(nv, nn, w, ac) for ac in ("rts", "basic") for w in (210, 420, 1680)
        for nv in (5, 20, 50) for nn in (0, 5, 10, 20)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "carrytau")


def one(scn: CO.Scn, arm: str, eps: float) -> dict:
    """Sequences of VISITS visits; only the second half is scored, so a warm
    arm is judged after its carry has settled."""
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = scn.n_vis, scn.n_nat
    c = math.exp(eps)
    av = an = 0.0
    t0_sum = 0.0
    t0_cnt = 0
    half = VISITS // 2
    try:
        with P.coefficients(c, c), P.window(scn.w_eff):
            for s in SEEDS:
                rng_p, rng = CO._rngs(scn, s)
                tau = None
                for v in range(VISITS):
                    if tau is None:
                        tau = np.full(scn.n_vis,
                                      1.0 / scn.n_vis if arm == "cold_nv"
                                      else 1.0 / scn.w_eff)
                    if v >= half:
                        t0_sum += float(np.mean(tau))
                        t0_cnt += 1
                    air, _, _, _, carry = f25._run_visit25(
                        f25._sample_ppdus25(rng_p), rng, "pace", tau,
                        *P.ACCESS[scn.access])
                    if v >= half:
                        av += float(air[:scn.n_vis].sum())
                        an += float(air[scn.n_vis:].sum())
                    tau = carry if arm == "warm" else None
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    n = len(SEEDS) * (VISITS - half)
    tot = av + an
    pop = scn.n_vis / (scn.n_vis + scn.n_nat) if scn.n_nat else 1.0
    return {"access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
            "n_nat": scn.n_nat, "arm": arm, "eps": round(float(eps), 5),
            "T": round(tot / (n * scn.w_eff), 5),
            "rho": round((av / tot) / pop if tot > 0 else 0.0, 5),
            "tau0": round(t0_sum / max(t0_cnt, 1), 6)}


def measure(core):
    scn = CO.Scn(*core)
    return [one(scn, a, e) for a in ARMS for e in EPS]


def summarise(rows):
    keys = sorted({(r["access"], r["w_eff"], r["n_vis"], r["n_nat"])
                   for r in rows})

    def best(k, arm, a):
        sel = sorted((r for r in rows
                      if (r["access"], r["w_eff"], r["n_vis"], r["n_nat"]) == k
                      and r["arm"] == arm), key=lambda r: r["eps"])
        js = [CO.objective(r["T"], r["rho"], a) for r in sel]
        i = int(np.argmax(js))
        return js[i], sel[i], (i in (0, len(sel) - 1))

    def stat(name, xs):
        xs = np.asarray(xs, float)
        return (f"    {name:<34s} min {xs.min():8.4f}  median "
                f"{np.median(xs):8.4f}  max {xs.max():8.4f}")

    o = ["", f"{len(keys)} scenarios (N_nat 0/5/10/20), each arm judged at its "
             "OWN best eps", ""]
    for a in ALPHAS:
        o.append(f"  alpha = {a}")
        for arm in ("cold_nv", "warm"):
            g, t, r, e = [], [], [], []
            for k in keys:
                j0, r0, b0 = best(k, "cold_w", a)
                j1, r1, b1 = best(k, arm, a)
                if b0 or b1:
                    continue
                g.append(math.exp(j1 - j0))
                t.append(r1["T"] / r0["T"])
                r.append(r1["rho"] / max(r0["rho"], 1e-9))
                e.append(r1["eps"] / r0["eps"])
            o.append(f"    --- {arm} vs cold_w (1/W_eff), n = {len(g)} ---")
            o.append(stat("exp(J_arm - J_cold_w)", g))
            o.append(stat("T ratio", t))
            o.append(stat("rho ratio", r))
            o.append(stat("eps* ratio", e))
            o.append(f"      arm wins: {sum(x > 1 for x in g)}/{len(g)}")
        # inflation diagnostic at each arm's own optimum
        w = [best(k, "warm", a)[1]["tau0"] / best(k, "cold_w", a)[1]["tau0"]
             for k in keys
             if not best(k, "warm", a)[2] and not best(k, "cold_w", a)[2]]
        o.append(stat("warm tau_0 / cold_w tau_0", w))
        o.append("")
    return "\n".join(o)


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
    txt = summarise(rows)
    print(txt)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
