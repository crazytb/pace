# -*- coding: utf-8 -*-
"""One duration-free fixed point for both populations.

    .venv/bin/python pace-analysis/joint.py

The natives run Bianchi's backoff chain and the visitors run PACE's log-domain
MIMD, and neither attempt probability depends on how long a frame takes: the
backoff chain counts decrement opportunities (Bianchi eq. 7 and 9 contain no
T_s or T_c) and the drift equation counts events, not their durations. So the
two close on each other through nothing but mutual silence:

    A_n = (1 - tau_n)^(N_nat - 1) (1 - tau_v)^n_v      a native sees an idle epoch
    tau_n = 1 / (E[b]/A_n + 1),  E[b] = (E[W] - 1)/2   Bianchi, on the epoch clock
    P(stage = k) proportional to (1 - A_n)^k

    eps_idle A0(tau_v, tau_n) = eps_coll Pc_lis(tau_v, tau_n)   drift zero

Solve the pair jointly and the coefficients map to an operating point with no
PPDU distribution, no collision cost and no handshake anywhere in it.

That matters because section 4.5.35's gate failed on T and rho, which are the
duration-dependent half. This file tests the other half on its own.

n_v is the VIABLE count, not the population: section 4.5.37 measured the
substitution error at a median 1.36 with N_vis and 1.08 with E|V|.

Bianchi's chain is the natives' own model, not an approximation of it -- his
counter also freezes while the channel is busy (p. 538). The only change is the
clock: his virtual slot is one decrement opportunity, the engine's epoch is one
contention decision, and the two differ by the factor A_n.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

import params as P

CW_MIN, CW_MAX = P.CW_MIN, P.CW_MAX
STAGES = 1 + int(math.ceil(math.log2(CW_MAX / CW_MIN)))


def _e_window(coll: float) -> float:
    """E[CW] with the stage geometric in the collision probability."""
    w = np.array([min(CW_MIN * 2 ** k, CW_MAX) for k in range(STAGES)])
    p = np.array([coll ** k for k in range(STAGES)])
    return float((p * w).sum() / p.sum())


def _tau_nat_given(tau_v: float, n_v: float, n_nat: int,
                   iters: int = 300) -> float:
    """Bianchi's chain on the epoch clock, with the visitors' silence folded in."""
    if n_nat <= 0:
        return 0.0
    vis = (1.0 - tau_v) ** n_v
    q = 0.03
    for _ in range(iters):
        a = min(max((1.0 - q) ** (n_nat - 1) * vis, 1e-9), 1.0)
        new = 1.0 / ((_e_window(1.0 - a) - 1.0) / (2.0 * a) + 1.0)
        if abs(new - q) < 1e-13:
            return float(new)
        q = 0.5 * q + 0.5 * new
    return float(q)


def _drift(tau_v: float, tau_n: float, n_v: float, n_nat: int,
           eps_i: float, eps_c: float) -> float:
    """E[d ln tau] per epoch for a tagged viable visitor. No durations."""
    a = (1.0 - tau_n) ** n_nat
    b = n_nat * tau_n * (1.0 - tau_n) ** (n_nat - 1) if n_nat else 0.0
    A0 = (1.0 - tau_v) ** n_v * a
    s0 = (1.0 - tau_v) ** (n_v - 1) * a
    s1 = ((n_v - 1) * tau_v * (1.0 - tau_v) ** (n_v - 2) * a
          + (1.0 - tau_v) ** (n_v - 1) * b)
    return eps_i * A0 - eps_c * (1.0 - tau_v) * (1.0 - s0 - s1)


def solve(n_v: float, n_nat: int, eps_i: float, eps_c: float,
          iters: int = 120) -> tuple[float, float] | None:
    """(tau_vis, tau_nat) at the joint fixed point, or None if the visitor
    drift has no interior zero (it is upward everywhere, so tau runs to the
    clip -- the runaway regime section 4.5.31 saw at large eps)."""
    tau_n = _tau_nat_given(0.02, n_v, n_nat)
    tau_v = 0.02
    for _ in range(iters):
        lo, hi = math.log(1e-9), math.log(0.95)
        f = lambda x: _drift(math.exp(x), tau_n, n_v, n_nat, eps_i, eps_c)
        if f(lo) <= 0 or f(hi) >= 0:
            return None
        new_v = math.exp(brentq(f, lo, hi))
        new_n = _tau_nat_given(new_v, n_v, n_nat)
        if abs(new_v - tau_v) < 1e-12 and abs(new_n - tau_n) < 1e-12:
            tau_v, tau_n = new_v, new_n
            break
        tau_v = 0.5 * tau_v + 0.5 * new_v
        tau_n = 0.5 * tau_n + 0.5 * new_n
    return float(tau_v), float(tau_n)


def _self_check() -> None:
    # the two halves must respond to each other in the right direction
    # The native half on its own must back off as its own population grows.
    assert (_tau_nat_given(0.01, 15.0, 20)
            < _tau_nat_given(0.01, 15.0, 10)
            < _tau_nat_given(0.01, 15.0, 5))
    # In the COUPLED system tau_nat need not be monotone in N_nat: the visitors
    # retreat as well and hand the natives back the idle slots. What must grow
    # is the channel-level pressure.
    v0, n0 = solve(15.0, 10, 0.03, 0.09)
    v1, n1 = solve(15.0, 20, 0.03, 0.09)
    assert v1 < v0, "more natives must push the visitors down"
    assert 1 - (1 - n1) ** 20 > 1 - (1 - n0) ** 10, "pressure must grow"
    # a bigger down-step must lower the visitors' operating point
    v3, _ = solve(15.0, 10, 0.03, 0.15)
    assert v3 < v0, "raising eps_coll must lower tau_vis"
    print("joint.py self-check passed")
    print()
    print("  duration-free: no PPDU length, collision cost or handshake enters")
    print(f"  {'n_v':>6}{'N_nat':>7}{'eps_i':>8}{'r':>5}{'tau_vis':>10}"
          f"{'tau_nat':>10}{'pressure':>10}")
    for n_v in (4.0, 15.0, 40.0):
        for n_nat in (5, 10, 20):
            for r in (2.0, 5.0):
                s = solve(n_v, n_nat, 0.03, r * 0.03)
                if s is None:
                    continue
                tv, tn = s
                pr = 1.0 - (1.0 - tn) ** n_nat
                print(f"  {n_v:6.0f}{n_nat:>7}{0.03:8.3f}{r:5.1f}"
                      f"{tv:10.5f}{tn:10.5f}{pr:10.4f}")


if __name__ == "__main__":
    _self_check()
