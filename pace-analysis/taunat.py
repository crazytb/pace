# -*- coding: utf-8 -*-
"""Is tau_nat really a constant 0.052?

    .venv/bin/python pace-analysis/taunat.py

params.TAU_NAT was measured at N_nat = 10, the engine default, and drift.py has
treated it as exogenous ever since -- params.py flags the self-consistent
coupling as future work. Two things it cannot be right about:

  1. More natives collide more, double their CW more, and back off. tau_nat
     must fall with N_nat.
  2. Visitors collide with natives too, so a more aggressive visitor population
     drives the natives' CW up. tau_nat must depend on the visitor side.

Measured straight off the engine's passive counters: tau_nat = nat_tx /
nat_slots, attempts per native per contention epoch, which is exactly the
quantity drift.py's p0n/p1n consume.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import params as P

SEEDS = CO.EVAL_SEEDS[:20]
VISITS = 25
N_NATS = (2, 5, 10, 20, 40)
N_VISS = (0, 5, 20, 50)
WS = (210, 420, 1680)
CS = (1.05, 1.20, 1.60)
SCEN = [(nv, nn, w, ac, c) for ac in ("rts", "basic") for w in WS
        for nn in N_NATS for nv in N_VISS for c in CS]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "taunat")


def measure(job):
    nv, nn, w, ac, c = job
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    # n_vis = 0 is the natives-on-their-own control; the engine needs at least
    # one visitor slot, so use one and silence it with c = 1 and tau at floor.
    solo_nat = nv == 0
    f25.N_VISITOR, f25.N_NATIVE = (1 if solo_nat else nv), nn
    st: dict = {}
    try:
        with P.coefficients(c, c), P.window(w):
            scn = CO.Scn(max(nv, 1), nn, w, ac)
            for s in SEEDS:
                rng_p, rng = CO._rngs(scn, s)
                for _ in range(VISITS):
                    tau0 = (np.full(1, 1e-4) if solo_nat
                            else np.full(nv, 1.0 / w))
                    f25._run_visit25(f25._sample_ppdus25(rng_p), rng, "pace",
                                     tau0, *P.ACCESS[ac], stats=st)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    slots = st.get("nat_slots", 0)
    return {"access": ac, "w_eff": w, "n_nat": nn, "n_vis": nv, "c": c,
            "tau_nat": round(st.get("nat_tx", 0) / max(slots, 1), 5),
            "epochs": st.get("epochs", 0),
            "idle_frac": round(st.get("idle", 0) / max(st.get("epochs", 1), 1),
                               4)}


def summarise(rows):
    def med(**kw):
        sel = [r["tau_nat"] for r in rows
               if all(r[k] == v for k, v in kw.items())]
        return float(np.median(sel)) if sel else float("nan")

    o = ["", f"tau_nat measured off the engine, {len(rows)} configurations",
         f"params.TAU_NAT = {P.TAU_NAT} (measured at N_nat = 10 only)", ""]
    all_t = np.array([r["tau_nat"] for r in rows])
    o.append(f"  over everything: min {all_t.min():.4f}  median "
             f"{np.median(all_t):.4f}  max {all_t.max():.4f}  "
             f"spread {all_t.max()/all_t.min():.1f}x")
    o.append("")
    o.append("  (1) dependence on N_nat  [median over the rest]")
    o.append("      " + "".join(f"{n:>10}" for n in N_NATS) + "   <- N_nat")
    o.append("      " + "".join(f"{med(n_nat=n):10.4f}" for n in N_NATS))
    o.append(f"      ratio N_nat=2 / N_nat=40: "
             f"{med(n_nat=2)/med(n_nat=40):.2f}x")
    o.append("")
    o.append("  (2) dependence on the VISITOR population")
    o.append("      " + "".join(f"{n:>10}" for n in N_VISS) + "   <- N_vis")
    o.append("      " + "".join(f"{med(n_vis=n):10.4f}" for n in N_VISS))
    o.append("      by visitor aggressiveness c:")
    o.append("      " + "".join(f"{c:>10}" for c in CS) + "   <- c")
    o.append("      " + "".join(f"{med(c=c):10.4f}" for c in CS))
    o.append("")
    o.append("  (3) the two together, at W_eff = 420, rts")
    hdr = "N_nat down, N_vis across"
    o.append("      " + hdr.rjust(12) + "".join(f"{n:>9}" for n in N_VISS))
    for nn in N_NATS:
        row = "".join(f"{med(n_nat=nn, n_vis=nv, w_eff=420, access='rts'):9.4f}"
                      for nv in N_VISS)
        o.append(f"      {nn:>12}" + row)
    o.append("")
    o.append("  (4) how far is the constant 0.052 off, per configuration?")
    err = np.array([r["tau_nat"] / P.TAU_NAT for r in rows])
    o.append(f"      measured / 0.052:  min {err.min():.2f}  median "
             f"{np.median(err):.2f}  max {err.max():.2f}")
    o.append(f"      within 10% of 0.052: "
             f"{int((np.abs(err-1) < 0.10).sum())}/{len(err)}")
    o.append("")
    o.append("  (5) other axes [median]")
    for w in WS:
        o.append(f"      W_eff = {w:>5}: {med(w_eff=w):.4f}")
    for ac in ("rts", "basic"):
        o.append(f"      access = {ac:>5}: {med(access=ac):.4f}")
    return "\n".join(o)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(measure, SCEN), 1):
            rows.append(r)
            if i % 60 == 0:
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
