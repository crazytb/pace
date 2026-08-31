# -*- coding: utf-8 -*-
"""Adapt the single coefficient across visits, instead of calibrating C_m.

    .venv/bin/python pace-analysis/outerloop.py

Section 4.5.27 left c_idle = c_coll = exp(C_m / sqrt(W_eff)) as the design
rule, with C_m a fitted constant that section 4.5.28 then showed moves by
1.27-1.46x with the success overhead alone. This replaces it with a loop the
station can run on its own counters.

The observable is the visitor's share of the epochs it can act on:

    q = n_idle / (n_idle + n_solo_vis + n_coll_vis)

The denominator drops native solos and native-only collisions, which is what
makes the setpoint nearly free of N_nat -- the raw idle fraction has its
ceiling set by the natives and its optimum moves 0.78 -> 0.47 as N_nat goes
0 -> 20, while q's stays at 0.78 -> 0.69.

q falls as c rises (a bigger step ramps tau up faster, trading idles for
collisions), so the update is

    ln c <- ln c + lam (q_hat - q*)

clipped to a sane box. Every quantity is counted by the station during its own
visit; nothing here needs N_vis, N_nat, or a calibration constant.

Section 4.5.28 measured the ideal version of this -- solving for the eps whose
converged statistics hit q* -- at a median G_J of 0.975/0.949/0.934 for alpha
0.25/0.5/1.0. This file asks the harder question: does a noisy one-visit
estimate, fed through an EMA, actually get there, and how fast.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO
import params as P

# section 4.5.30: q_self is built only from what a station can count -- the
# collisions it was itself in, which standard DCF already tracks to drive its
# backoff. q_glob's coll_vis needs to know whether ANY visitor was in a
# collision, which nothing decodes during.
Q_STAR = {"q_self": {0.25: 0.820, 0.5: 0.816, 1.0: 0.807},
          "q_glob": {0.25: 0.761, 0.5: 0.743, 1.0: 0.709}}
SIGNAL = "q_self"
LAM = 0.15
C_BOX = (1.02, 4.0)
N_VISITS = 120
SEQS = 24
ALPHAS = (0.25, 0.5, 1.0)
SCEN = [(nv, nn, w, ac) for ac in ("rts", "basic")
        for w in (210, 420, 1680) for nv in (5, 20, 50) for nn in (0, 5, 10, 20)]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "outerloop")


def run_sequence(scn: CO.Scn, seed: int, alpha: float, c0: float,
                 lam: float = LAM, n_visits: int = N_VISITS,
                 signal: str = SIGNAL) -> dict:
    """One station sequence: c carried visit to visit, tau reset each visit.

    tau_0 stays at 1/W_eff -- it is a given condition, not a knob (section 7).
    """
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = scn.n_vis, scn.n_nat
    q_star = Q_STAR[signal][alpha]
    c = float(c0)
    trace, av, an = [], 0.0, 0.0
    try:
        rng_p, rng = CO._rngs(scn, seed)
        with P.window(scn.w_eff):
            for k in range(n_visits):
                st: dict = {}
                with P.coefficients(c, c):
                    air, _, _, _, _ = f25._run_visit25(
                        f25._sample_ppdus25(rng_p), rng, "pace",
                        np.full(scn.n_vis, 1.0 / scn.w_eff),
                        *P.ACCESS[scn.access], stats=st)
                idle = st.get("idle", 0)
                sv = st.get("solo_vis", 0)
                cv = (st.get("coll_self", 0) if signal == "q_self"
                      else st.get("coll_vis", 0))
                den = idle + sv + cv
                q = idle / den if den else q_star
                trace.append((c, q))
                # q decreases as c grows, so chase the setpoint upward in c
                c = float(np.clip(math.exp(math.log(c) + lam * (q - q_star)),
                                  *C_BOX))
                if k >= n_visits // 2:            # second half = steady state
                    av += float(air[:scn.n_vis].sum())
                    an += float(air[scn.n_vis:].sum())
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    n = n_visits - n_visits // 2
    tot = av + an
    norm = n * scn.w_eff
    pop = scn.n_vis / (scn.n_vis + scn.n_nat) if scn.n_nat else 1.0
    rho = (av / tot) / pop if tot > 0 else 0.0
    return {"c_final": trace[-1][0], "trace": trace,
            "T": tot / norm, "rho": rho,
            "J": CO.objective(tot / norm, rho, alpha)}


def job(args):
    core, alpha, c0 = args
    scn = CO.Scn(*core)
    out = []
    for s in CO.EVAL_SEEDS[:SEQS]:
        r = run_sequence(scn, s, alpha, c0)
        out.append((r["c_final"], r["T"], r["rho"], r["J"],
                    [t[0] for t in r["trace"]]))
    cs = np.array([o[0] for o in out])
    # pool the airtime the way aggregate() does, then form J once
    T = float(np.mean([o[1] for o in out]))
    rho = float(np.mean([o[2] for o in out]))
    tr = np.array([o[4] for o in out])          # SEQS x N_VISITS
    return {"access": scn.access, "w_eff": scn.w_eff, "n_vis": scn.n_vis,
            "n_nat": scn.n_nat, "alpha": alpha, "c0": c0,
            "c_med": float(np.median(cs)), "c_lo": float(cs.min()),
            "c_hi": float(cs.max()), "T": T, "rho": rho,
            "J": CO.objective(T, rho, alpha),
            "c_path": [float(x) for x in np.median(tr, axis=0)]}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(core, a, c0) for core in SCEN for a in ALPHAS
            for c0 in (1.05, 1.20, 2.50)]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(job, jobs), 1):
            rows.append(r)
            if i % 40 == 0:
                print(f"[{i}/{len(jobs)}]", flush=True)
    with open(os.path.join(OUT, "data.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "c_path"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "c_path"})
    np.save(os.path.join(OUT, "paths.npy"),
            np.array([r["c_path"] for r in rows]))
    print("wrote", OUT)
    return rows


if __name__ == "__main__":
    main()
