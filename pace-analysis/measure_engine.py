"""Measure from the simulator the two quantities drift.py cannot derive.

    .venv/bin/python pace-analysis/measure_engine.py

1. tau_nat, the native attempt probability PER CONTENTION EPOCH. drift.py takes
   it as an exogenous constant; this checks that the approximation holds. It is
   not obvious that it should: natives decrement their backoff only on idle
   epochs (run_step9_fig25.py:311-313), so the freeze coupling could pull the
   effective rate far below the saturated-DCF Bianchi value of 0.0525.

2. The contention-epoch budget of one visit. This is the binding constraint in
   Theorem 2. W_eff = 420 slots sounds generous, but a success costs 35-110
   slots and a native success 60, so the number of decision points is small.

Both feed PACE_TWC_ANALYSIS.md sections 1 and 4.3.
"""
from __future__ import annotations

import numpy as np

import drift
import params as P

f25 = P.engine()
VISITS, REPS = P.VISITS, P.REPS


def measure(n_vis: int, n_nat: int, coll_cost, succ_oh: int,
            mode: str = "pace") -> dict:
    """Run REPS x VISITS visits and return per-epoch tallies plus airtime."""
    f25.N_VISITOR, f25.N_NATIVE = n_vis, n_nat
    st: dict = {}
    air_v = air_n = tau_end = 0.0
    counted = 0
    try:
        for r in range(REPS):
            rng_p = np.random.default_rng(10001 + r * 71 + 7)
            rng = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
            for v in range(VISITS):
                ppdus = f25._sample_ppdus25(rng_p)
                tau0 = (np.full(n_vis, P.TAU_0)
                        if mode.startswith(("pace", "oracle")) else None)
                air, _c, _i, _o, carry = f25._run_visit25(
                    ppdus, rng, mode, tau0, coll_cost, succ_oh, stats=st)
                if v >= VISITS // 2:          # steady state
                    air_v += air[:n_vis].sum()
                    air_n += air[n_vis:].sum()
                    if carry is not None:
                        tau_end += float(np.mean(carry))
                    counted += 1
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE   # restore module defaults
    norm = counted * P.W_EFF
    e = st["epochs"]
    return {
        "epochs_per_visit": e / (REPS * VISITS),
        "tau_nat": st["nat_tx"] / st["nat_slots"],
        "tau_mean": st.get("tau_sum", 0.0) / max(st.get("tau_cnt", 1), 1),
        "tau_end": tau_end / counted if counted else 0.0,
        "f_idle": st.get("idle", 0) / e,
        "f_solo_vis": st.get("solo_vis", 0) / e,
        "f_solo_nat": st.get("solo_nat", 0) / e,
        "f_coll": st.get("coll", 0) / e,
        "air_v": air_v / norm,
        "air_n": air_n / norm,
    }


def _main() -> None:
    for label, cc, oh in (("RTS/CTS", *P.ACCESS['rts']),
                          ("basic", "nocd", 0)):
        print(f"\n=== {label} (N_native=10) ===")
        print(f"{'N_vis':>6} {'epochs':>8} {'tau_nat':>9} {'idle':>6} "
              f"{'s_vis':>6} {'s_nat':>6} {'coll':>6} {'air_v':>7}")
        for nv in (10, 20, 50):
            m = measure(nv, 10, cc, oh)
            print(f"{nv:>6} {m['epochs_per_visit']:8.1f} {m['tau_nat']:9.5f} "
                  f"{m['f_idle']:6.3f} {m['f_solo_vis']:6.3f} "
                  f"{m['f_solo_nat']:6.3f} {m['f_coll']:6.3f} {m['air_v']:7.3f}")

    print("\n=== Theorem 2: how far short of equilibrium (RTS/CTS) ===")
    print(f"{'N_vis':>6} {'tau_end':>9} {'tau*':>9} {'climbed':>8} "
          f"{'needed':>7} {'reached':>8}")
    for nv in (10, 20, 50):
        m = measure(nv, 10, *P.ACCESS['rts'])
        ts = drift.tau_star(nv, m["tau_nat"])
        t0 = P.TAU_0
        c = np.log(m["tau_end"] / t0) / np.log(drift.C_MIMD)
        need = np.log(ts / t0) / np.log(drift.C_MIMD)
        print(f"{nv:>6} {m['tau_end']:9.5f} {ts:9.5f} {c:8.1f} "
              f"{need:7.1f} {c / need:8.2f}")


def _self_check() -> None:
    m = measure(20, 10, *P.ACCESS['rts'])
    # the constant-tau_nat approximation drift.py relies on
    assert 0.045 < m["tau_nat"] < 0.060, m["tau_nat"]
    # the epoch budget that makes Theorem 2 bind
    assert 10 < m["epochs_per_visit"] < 40, m["epochs_per_visit"]
    # tau must end below the (already depressed) equilibrium
    assert m["tau_end"] < drift.tau_star(20, m["tau_nat"]), m["tau_end"]
    # the passive hook must not perturb the simulation
    f25.N_VISITOR, f25.N_NATIVE = 20, 10
    out = []
    for use in (True, False):
        rng_p = np.random.default_rng(99)
        rng = np.random.default_rng(1234)
        acc = []
        for _ in range(15):
            p = f25._sample_ppdus25(rng_p)
            a, c, i, o, _ = f25._run_visit25(
                p, rng, "pace", np.full(20, P.TAU_0),
                *P.ACCESS['rts'],
                stats=({} if use else None))
            acc.append((float(a.sum()), c, i, o))
        out.append(acc)
    f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    assert out[0] == out[1], "stats hook perturbed the simulation"
    print("\nmeasure_engine.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
