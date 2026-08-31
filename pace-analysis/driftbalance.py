# -*- coding: utf-8 -*-
"""Drift-balance design rule, and what it predicts against the swept optimum.

    .venv/bin/python pace-analysis/driftbalance.py          # self-check
    .venv/bin/python pace-analysis/driftbalance.py --sweep  # the comparison

The rule under test, in the homogeneous no-native setting where solo-copy is a
self-loop (section 4.5.25 measured that it very nearly is):

    1. tau_target = argmax_tau T(tau)      from a saturated airtime model
    2. r*         = A0(tau_target) / Pc(tau_target)        [= eps_coll/eps_idle]
    3. eps_idle   = C_m / sqrt(W_eff)      the empirical scale, section 4.5.23
    4. c_idle_hat = exp(eps_idle),  c_coll_hat = exp(r* eps_idle)

Only step 2 is derived. Step 3 is a fitted constant, so this rule can identify
the RATIO of the two coefficients but never their scale -- which is exactly the
split Proposition 1 predicts.

IMPORTANT -- the engine does not satisfy the "everyone gets the same update"
assumption. On a collision the transmitters HOLD their tau and only the
listeners divide (run_step9_fig25.py:359-362). So the drift seen by a tagged
station is

    D(X) = eps_idle * A0(tau) - eps_coll * Pc_lis(tau)

with Pc_lis conditioned on the tagged station having stayed silent. Using the
unconditional collision probability instead inflates the down-step and moves
r* by roughly a factor (1-tau)^-1 times the split correction. drift.drift()
already uses the listener-conditioned form for the r = 1 case; the functions
here reduce to it exactly (asserted in _self_check).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import params as P

TAU_NAT = P.TAU_NAT
E_L = 0.5 * (P.PPDU_V_LO + P.PPDU_V_HI)          # 62.5 slots
L_NAT = P.PPDU_NATIVE                            # 50 slots


# ─── epoch outcome probabilities, as the engine resolves them ────────────────

def p0n(n_nat: int, tau_nat: float = TAU_NAT) -> float:
    return (1.0 - tau_nat) ** n_nat


def p1n(n_nat: int, tau_nat: float = TAU_NAT) -> float:
    return n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1) if n_nat else 0.0


def A0(tau: float, n: int, n_nat: int = 0) -> float:
    """No one transmits: every viable visitor takes the +eps_idle step."""
    return (1.0 - tau) ** n * p0n(n_nat)


def Psv(tau: float, n: int, n_nat: int = 0) -> float:
    """Exactly one visitor and no native: the solo-copy (self-loop) epoch."""
    return n * tau * (1.0 - tau) ** (n - 1) * p0n(n_nat)


def Psn(tau: float, n: int, n_nat: int = 0) -> float:
    """Exactly one native and no visitor: visitors take no step at all."""
    return (1.0 - tau) ** n * p1n(n_nat)


def Pc_lis(tau: float, n: int, n_nat: int = 0) -> float:
    """Collision WITH the tagged station silent -- the only epoch that costs it
    eps_coll. Transmitters hold, so the unconditional collision probability is
    the wrong quantity here."""
    a, b = p0n(n_nat), p1n(n_nat)
    silent_others_0 = (1.0 - tau) ** (n - 1) * a
    silent_others_1 = (n - 1) * tau * (1.0 - tau) ** (n - 2) * a + \
        (1.0 - tau) ** (n - 1) * b
    return (1.0 - tau) * (1.0 - silent_others_0 - silent_others_1)


def drift(x: float, n: int, eps_i: float, eps_c: float, n_nat: int = 0) -> float:
    """E[dX] per contention epoch for a tagged viable visitor, X = ln tau."""
    tau = math.exp(x)
    return eps_i * A0(tau, n, n_nat) - eps_c * Pc_lis(tau, n, n_nat)


# ─── the airtime model that sets the target ─────────────────────────────────

def _coll_cost(tau: float, n: int, n_nat: int, access: str) -> float:
    """Slots burned by a collision. RTS/CTS is the 12-slot constant; basic
    charges max L_i over the colliders, so it is an order statistic (the same
    E[max | E[M]] approximation dp.basic_collision_cost uses)."""
    if access == "rts":
        return float(P.L_COL)
    pc = 1.0 - A0(tau, n, n_nat) - Psv(tau, n, n_nat) - Psn(tau, n, n_nat)
    mu = n * tau + n_nat * TAU_NAT
    m = max((mu - Psv(tau, n, n_nat) - Psn(tau, n, n_nat)) / max(pc, 1e-12), 2.0)
    vis = P.PPDU_V_LO + (P.PPDU_V_HI - P.PPDU_V_LO) * m / (m + 1.0)
    return max(vis, float(L_NAT)) if n_nat else vis


def T_model(tau: float, n: int, access: str, n_nat: int = 0) -> float:
    """Useful airtime fraction under saturation, per the engine's slot costs."""
    oh = P.L_HS if access == "rts" else 0
    pi, psv, psn = A0(tau, n, n_nat), Psv(tau, n, n_nat), Psn(tau, n, n_nat)
    pc = 1.0 - pi - psv - psn
    span = (pi * 1.0 + psv * (E_L + oh) + psn * (L_NAT + oh)
            + pc * _coll_cost(tau, n, n_nat, access))
    return (psv * E_L + psn * L_NAT) / span if span > 0 else 0.0


def tau_target(n: int, access: str, n_nat: int = 0) -> float | None:
    """argmax_tau T(tau). None when the optimum is visitor silence, which is
    what happens under basic access with natives present (section 4.5.4)."""
    r = minimize_scalar(lambda x: -T_model(math.exp(x), n, access, n_nat),
                        bounds=(math.log(1e-6), math.log(0.6)), method="bounded")
    t = math.exp(r.x)
    return None if t < 1.5e-6 else t


# ─── the design rule ────────────────────────────────────────────────────────

def r_star(n: int, access: str, n_nat: int = 0) -> float | None:
    """eps_coll / eps_idle that puts the drift zero at the airtime optimum."""
    t = tau_target(n, access, n_nat)
    if t is None:
        return None
    return A0(t, n, n_nat) / Pc_lis(t, n, n_nat)


def design_hat(n: int, w_eff: int, access: str, c_m: float,
               n_nat: int = 0) -> tuple[float, float] | None:
    """(c_idle_hat, c_coll_hat). The ratio is derived, the scale c_m is not."""
    r = r_star(n, access, n_nat)
    if r is None:
        return None
    eps_i = c_m / math.sqrt(w_eff)
    return math.exp(eps_i), math.exp(r * eps_i)


def tau_eq(c_idle: float, c_coll: float, n: int, n_nat: int = 0) -> float | None:
    """The other direction: the drift equilibrium implied by a coefficient pair.
    Theorem 2 says a visit ends long before this is reached, so treating it as a
    prediction of the engine's operating point is expected to fail; measuring
    how badly is the point."""
    eps_i, eps_c = math.log(c_idle), math.log(c_coll)
    lo, hi = math.log(1e-9), math.log(0.95)
    f = lambda x: drift(x, n, eps_i, eps_c, n_nat)
    if f(lo) <= 0 or f(hi) >= 0:
        return None
    return math.exp(brentq(f, lo, hi))


# ─── self-check ─────────────────────────────────────────────────────────────

def _self_check() -> None:
    import drift as D

    # the listener-conditioned drift must reproduce drift.py at r = 1
    for n in (5, 10, 20, 50):
        for nn in (0, 10):
            for tau in (0.002, 0.02, 0.1):
                mine = drift(math.log(tau), n, 1.0, 1.0, nn)
                theirs = D.drift(tau, n, TAU_NAT if nn else 0.0)
                assert abs(mine - theirs) < 1e-12, (n, nn, tau, mine, theirs)

    # the four outcome probabilities must partition the epoch
    for n in (5, 20):
        for nn in (0, 10):
            tau = 0.03
            tot = (A0(tau, n, nn) + Psv(tau, n, nn) + Psn(tau, n, nn))
            assert 0.0 < tot < 1.0
            # a silent tagged station sees exactly one of: idle, someone else
            # solo, collision-without-me
            assert Pc_lis(tau, n, nn) < 1.0 - tau + 1e-12

    # the airtime optimum sits well below 1/n once collisions cost real slots
    for access in ("rts", "basic"):
        for n in (5, 20, 50):
            t = tau_target(n, access)
            assert t is not None and t < 1.0 / n, (access, n, t)

    # basic access with natives degenerates to visitor silence (section 4.5.4)
    assert tau_target(20, "basic", 10) is None
    assert r_star(20, "basic", 10) is None

    # equilibrium round trip: a pair built from r* must put the zero back at
    # the target it was built from
    for access in ("rts", "basic"):
        for n in (5, 20, 50):
            t = tau_target(n, access)
            r = r_star(n, access)
            eps_i = 0.05
            back = tau_eq(math.exp(eps_i), math.exp(r * eps_i), n)
            assert abs(math.log(back / t)) < 1e-6, (access, n, t, back)

    print("driftbalance self-check passed")
    print()
    print(f"{'access':<7}{'N':>4}{'n_nat':>6}{'tau_target':>12}{'1/N':>9}"
          f"{'ratio':>8}{'A0':>8}{'Pc_lis':>9}{'r*':>10}")
    for access in ("rts", "basic"):
        for nn in (0, 10):
            for n in (5, 10, 20, 50):
                t = tau_target(n, access, nn)
                if t is None:
                    print(f"{access:<7}{n:>4}{nn:>6}{'silence':>12}")
                    continue
                r = r_star(n, access, nn)
                print(f"{access:<7}{n:>4}{nn:>6}{t:12.5f}{1/n:9.4f}"
                      f"{(1/n)/t:8.1f}{A0(t, n, nn):8.4f}"
                      f"{Pc_lis(t, n, nn):9.4f}{r:10.2f}")
    print()
    print("c_coll_hat = exp(r* * C_m/sqrt(W_eff)), C_m = 9.27 (section 4.5.23)")
    print(f"{'access':<7}{'N':>4}{'W':>6}{'eps_idle':>10}{'c_idle^':>10}"
          f"{'eps_coll':>10}{'c_coll^':>14}")
    for access in ("rts", "basic"):
        for n in (5, 50):
            for w in (105, 420, 1680):
                d = design_hat(n, w, access, 9.27)
                ei = 9.27 / math.sqrt(w)
                print(f"{access:<7}{n:>4}{w:>6}{ei:10.4f}{d[0]:10.4f}"
                      f"{math.log(d[1]):10.4f}{d[1]:14.4g}")


if __name__ == "__main__":
    _self_check()
