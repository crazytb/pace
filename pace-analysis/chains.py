# -*- coding: utf-8 -*-
"""Both populations as Markov chains, solved for stationary DISTRIBUTIONS.

    .venv/bin/python pace-analysis/chains.py

joint.py closed the two populations on each other but used the drift ZERO for
the visitors, which gives only the mean of ln tau. The coupling needs

    E[(1 - tau)^n] = sum_k pi(k) (1 - tau_k)^n,   not   (1 - mean tau)^n

and section 4.5.37 only ruled out the CROSS-SECTIONAL spread (the epoch CV, at
0.007-0.149). The spread of a single station's tau over TIME was never measured;
it is what the drift zero throws away.

Native chain. Bianchi's two-dimensional chain (his Fig. 4) transferred to the
epoch clock: because the counter only moves on an idle epoch, every k > 0 state
gains a self-loop of probability 1 - A. A self-loop does not change which states
are visited, only how long each is occupied, so

    pi_epoch(i,k) proportional to pi_Bianchi(i,k) * (1/A if k > 0 else 1)

which is the renewal expression joint.py already uses. Bianchi's closed form
(his eqs. 2-7) carries over unchanged.

Visitor chain. X = ln tau on a lattice, and per epoch a tagged viable visitor
takes +eps_idle when nobody transmits, -eps_coll when it stayed silent through
a collision, and stays put otherwise. With r = 1 this is birth-death and has a
product form; with r != 1 the steps are unequal, so it is not reversible and the
balance equations are solved as a linear system instead. Either way the state
space is small -- the lattice spans ln(tau_max/tau_0), a few tens of steps.

Everything here is duration-free: no PPDU length, collision cost or handshake.
"""
from __future__ import annotations

import math

import numpy as np

import params as P

CW_MIN, CW_MAX = P.CW_MIN, P.CW_MAX
STAGES = 1 + int(math.ceil(math.log2(CW_MAX / CW_MIN)))
TAU_FLOOR, TAU_CEIL = 1e-4, 1.0


def _e_window(coll: float) -> float:
    w = np.array([min(CW_MIN * 2 ** k, CW_MAX) for k in range(STAGES)])
    p = np.array([coll ** k for k in range(STAGES)])
    return float((p * w).sum() / p.sum())


def tau_nat_given(tau_v_moments, n_nat: int, iters: int = 300) -> float:
    """Bianchi on the epoch clock. tau_v_moments is E[(1-tau_v)^n_v], the
    visitors' silence -- passed as the expectation, not as a plug-in."""
    if n_nat <= 0:
        return 0.0
    q = 0.03
    for _ in range(iters):
        a = min(max((1.0 - q) ** (n_nat - 1) * tau_v_moments, 1e-9), 1.0)
        new = 1.0 / ((_e_window(1.0 - a) - 1.0) / (2.0 * a) + 1.0)
        if abs(new - q) < 1e-13:
            return float(new)
        q = 0.5 * q + 0.5 * new
    return float(q)


def _rates(tau: np.ndarray, n_v: float, n_nat: int, tau_n: float):
    """Up (idle) and down (collision with the tagged station silent) rates."""
    a = (1.0 - tau_n) ** n_nat
    b = n_nat * tau_n * (1.0 - tau_n) ** (n_nat - 1) if n_nat else 0.0
    up = (1.0 - tau) ** n_v * a
    s0 = (1.0 - tau) ** (n_v - 1) * a
    s1 = ((n_v - 1) * tau * (1.0 - tau) ** (n_v - 2) * a
          + (1.0 - tau) ** (n_v - 1) * b)
    dn = (1.0 - tau) * (1.0 - s0 - s1)
    return up, dn


def visitor_dist(n_v: float, n_nat: int, tau_n: float, eps_i: float,
                 eps_c: float, m: int = 600):
    """Stationary distribution of X = ln tau on a uniform lattice.

    The lattice spacing is chosen so both steps land on it: with r rational the
    walk is exactly representable, and otherwise the steps are rounded to the
    nearest node, which is the same rational approximation dp.lattice makes."""
    lo, hi = math.log(TAU_FLOOR), math.log(0.999 * TAU_CEIL)
    x = np.linspace(lo, hi, m)
    h = x[1] - x[0]
    ui = max(int(round(eps_i / h)), 1)
    uc = max(int(round(eps_c / h)), 1)
    tau = np.exp(x)
    up, dn = _rates(tau, n_v, n_nat, tau_n)

    # one-step transition matrix, reflecting at the clip boundaries
    Q = np.zeros((m, m))
    idx = np.arange(m)
    hi_i = np.minimum(idx + ui, m - 1)
    lo_i = np.maximum(idx - uc, 0)
    Q[idx, hi_i] += up
    Q[idx, lo_i] += dn
    Q[idx, idx] += 1.0 - up - dn

    # stationary distribution by power iteration (the chain is small and mixes)
    pi = np.full(m, 1.0 / m)
    for _ in range(20000):
        nxt = pi @ Q
        if np.abs(nxt - pi).sum() < 1e-14:
            pi = nxt
            break
        pi = nxt
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return tau, pi


def solve(n_v: float, n_nat: int, eps_i: float, eps_c: float,
          iters: int = 40, m: int = 600):
    """Joint fixed point on the two stationary DISTRIBUTIONS.

    Returns (mean tau_v, tau_n, E[(1-tau_v)^n_v], sd of ln tau_v)."""
    tau_n = 0.02
    out = None
    for _ in range(iters):
        tau, pi = visitor_dist(n_v, n_nat, tau_n, eps_i, eps_c, m)
        sil = float((pi * (1.0 - tau) ** n_v).sum())
        new_n = tau_nat_given(sil, n_nat)
        mean_v = float((pi * tau).sum())
        lx = np.log(tau)
        sd = float(math.sqrt(max((pi * lx ** 2).sum()
                                 - ((pi * lx).sum()) ** 2, 0.0)))
        out = (mean_v, new_n, sil, sd)
        if abs(new_n - tau_n) < 1e-12:
            break
        tau_n = 0.5 * tau_n + 0.5 * new_n
    return out


def _self_check() -> None:
    # the distribution must reproduce the drift zero's location roughly, and
    # must be genuinely spread -- if it were a point mass the mean-field
    # substitution would already have been exact
    mv, mn, sil, sd = solve(15.0, 10, 0.03, 0.09)
    assert 1e-4 < mv < 0.5 and 0.0 < mn < 0.2
    assert sd > 0.01, "a point mass would mean the drift zero was enough"
    # raising the down-step must lower the operating point
    mv2, _, _, _ = solve(15.0, 10, 0.03, 0.30)
    assert mv2 < mv
    print("chains.py self-check passed")
    print()
    print("  stationary distribution of X = ln tau, and what the plug-in misses")
    print(f"  {'n_v':>5}{'N_nat':>6}{'r':>5}{'mean tau_v':>12}{'sd(ln tau)':>12}"
          f"{'E[(1-t)^n]':>12}{'(1-E t)^n':>12}{'ratio':>7}")
    for n_v in (4.0, 15.0, 40.0):
        for n_nat in (5, 20):
            for r in (1.0, 3.0, 5.0):
                mv, mn, sil, sd = solve(n_v, n_nat, 0.03, r * 0.03)
                plug = (1.0 - mv) ** n_v
                print(f"  {n_v:5.0f}{n_nat:>6}{r:5.1f}{mv:12.5f}{sd:12.4f}"
                      f"{sil:12.5f}{plug:12.5f}{sil/plug:7.2f}")


if __name__ == "__main__":
    _self_check()
