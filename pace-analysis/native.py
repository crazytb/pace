# -*- coding: utf-8 -*-
"""A self-consistent native attempt rate, replacing the tau_nat constant.

    .venv/bin/python pace-analysis/native.py

params.TAU_NAT = 0.052 was measured once, at N_nat = 10 with the natives reset
every visit. Section 4.5.32 measured it moving 3.7x with N_nat, N_vis, the
visitor coefficient and W_eff; section 4.5.34 measured native carry halving it
again; and section 4.5.35 traced the analytic model's failure under carry to
exactly this, with the gate degrading monotonically in N_nat (T 0.940 / 0.864 /
0.688 at N_nat 5 / 10 / 20).

The engine's natives are not standard DCF and Bianchi's equation does not apply
to them, because they decrement ONLY on idle epochs (run_step9_fig25.py:359).
That rule is what makes them tractable instead:

  * a native with residual b > 0 steps down one, and only, on an idle epoch
  * a native at b = 0 transmits, and whether it succeeds or collides it is
    resolved and redraws in that same epoch -- so it occupies b = 0 for exactly
    one epoch

So each native runs a renewal cycle: draw b ~ U[0, W_s), spend b/A epochs
counting down where A is the probability an epoch is idle from its point of
view, then one epoch transmitting. The fraction of time at b = 0 is the attempt
probability, and it closes on itself because A depends on how often the OTHER
natives are at zero.

    A    = (1 - q0)^(N_nat - 1) (1 - tau)^N_vis        idle, as the tagged native sees it
    P(stage = k) proportional to (1 - A)^k             success resets, collision doubles
    E[cycle] = (E[W] - 1) / (2A) + 1
    q0   = 1 / E[cycle]                                 = tau_nat

One scalar fixed point, no state explosion, no calibration constant.
"""
from __future__ import annotations

import math

import numpy as np

import params as P

CW_MIN = P.CW_MIN                      # 16
CW_MAX = P.CW_MAX                      # 1023
STAGES = 1 + int(math.ceil(math.log2(CW_MAX / CW_MIN)))


def _cw(stage: int) -> int:
    return min(CW_MIN * 2 ** stage, CW_MAX)


def _e_window(a: float) -> float:
    """E[CW] over the stationary stage distribution.

    A native climbs a stage on every collision and drops to stage 0 on every
    success, so the stage is geometric with success probability A, truncated at
    the retry limit."""
    p = np.array([(1.0 - a) ** k for k in range(STAGES)])
    p /= p.sum()
    return float(sum(pk * _cw(k) for k, pk in enumerate(p)))


def tau_nat(n_nat: int, n_vis: float = 0.0, tau_vis: float = 0.0,
            tol: float = 1e-12, iters: int = 200) -> float:
    """Attempt probability per native per contention epoch.

    tau_vis is the visitors' common attempt probability; they matter only
    through how often they deny the natives an idle slot to count down in.

    n_vis must be the VIABLE count, not the population. Measuring the visitor
    silence the natives actually see against (1 - mean tau)^n gave a median
    1.36 with the population and 1.08 with the viable count, and the epoch CV
    of tau is 0.007-0.149, so self-exclusion and not cross-sectional spread is
    what the substitution was getting wrong (section 4.5.37)."""
    if n_nat <= 0:
        return 0.0
    vis = (1.0 - tau_vis) ** n_vis
    q = 0.05                                        # anything in (0, 1)
    for _ in range(iters):
        a = (1.0 - q) ** (n_nat - 1) * vis
        a = min(max(a, 1e-9), 1.0)
        cycle = (_e_window(a) - 1.0) / (2.0 * a) + 1.0
        new = 1.0 / cycle
        if abs(new - q) < tol:
            q = new
            break
        q = 0.5 * q + 0.5 * new                     # damped, the map is steep
    return float(min(max(q, 1e-9), 1.0))


def idle_share(n_nat: int, n_vis: float = 0.0, tau_vis: float = 0.0) -> float:
    """P(no native transmits) -- what p0n needs, without assuming a constant."""
    return (1.0 - tau_nat(n_nat, n_vis, tau_vis)) ** n_nat


def _self_check() -> None:
    # the fixed point must exist and be monotone in the load
    prev = 1.0
    for n in (2, 5, 10, 20, 40):
        t = tau_nat(n)
        assert 0.0 < t < 0.5, (n, t)
        assert t < prev, "more natives must back each other off"
        prev = t
    # visitors deny idle slots, so they depress it too
    assert tau_nat(10, 20, 0.05) < tau_nat(10, 20, 0.0)

    print("native.py self-check passed")
    print()
    print("tau_nat, natives alone (compare the engine's CARRY-regime values)")
    print(f"  {'N_nat':>6}{'model':>9}{'engine (carry)':>16}"
          f"{'engine (fresh)':>16}")
    eng_carry = {5: 0.0255, 10: 0.0178, 20: 0.0126}
    eng_fresh = {2: 0.0759, 5: 0.0613, 10: 0.0498, 20: 0.0417, 40: 0.0373}
    for n in (2, 5, 10, 20, 40):
        c = f"{eng_carry[n]:.4f}" if n in eng_carry else "-"
        print(f"  {n:>6}{tau_nat(n):9.4f}{c:>16}{eng_fresh[n]:16.4f}")
    print()
    print("with visitors present (N_vis = 20)")
    print(f"  {'N_nat':>6}{'tau_vis=0':>11}{'0.02':>9}{'0.05':>9}")
    for n in (5, 10, 20):
        print(f"  {n:>6}{tau_nat(n, 20, 0.0):11.4f}"
              f"{tau_nat(n, 20, 0.02):9.4f}{tau_nat(n, 20, 0.05):9.4f}")


if __name__ == "__main__":
    _self_check()
