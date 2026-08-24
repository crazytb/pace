"""PACE drift analysis — stochastic equilibrium of the MIMD rule.

Reproduces every number quoted in PACE_TWC_ANALYSIS.md sections 4.2 and 4.3.
Run standalone to print the self-verifying tables:

    .venv/bin/python pace-analysis/drift.py

Model. With c_idle = c_coll = c the reachable probabilities form the lattice
tau = tau_0 * c^k, so a STA's state is the integer k and the MIMD rule becomes
a birth-death walk: idle -> k+1, solo success -> k unchanged (the copied value
equals the listener's own under homogeneity), collision -> k-1. The drift
E[dk] per contention epoch is set to zero to locate the equilibrium.

Half-duplex: a colliding STA cannot hear its own collision, so on a collision
of m STAs only the n-m listeners step down.

Native load: a native solo success leaves visitor tau untouched (it advertises
no tau), but a collision among natives still pushes every listening visitor
down, because the rule reads the slot outcome and not its cause. The rule
therefore cannot tell "my group is too aggressive" from "another group is
busy", which depresses the equilibrium below fair share.

ponytail: tau_nat is an exogenous constant here. It is measured per contention
epoch by measure_engine.py (0.049-0.056, matching saturated-DCF Bianchi) which
justifies the approximation; a self-consistent coupling with the natives'
frozen-backoff dynamics is future work.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

import params as P

W_EFF = P.W_EFF
C_MIMD = P.C_MIMD
N_NATIVE = P.N_NATIVE
TAU_NAT_MEASURED = P.TAU_NAT
TAU_FLOOR = P.TAU_FLOOR


def p0n(tau_nat: float, n_nat: int = N_NATIVE) -> float:
    """Probability that no native transmits in an epoch."""
    return (1.0 - tau_nat) ** n_nat


def p1n(tau_nat: float, n_nat: int = N_NATIVE) -> float:
    """Probability that exactly one native transmits (a neutral epoch)."""
    return n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1)


def drift_all_update(tau: float, n: int) -> float:
    """E[dk] assuming every STA observes the outcome. 2*P_idle + P_succ - 1."""
    return 2 * (1 - tau) ** n + n * tau * (1 - tau) ** (n - 1) - 1


def drift(tau: float, n: int, tau_nat: float = 0.0) -> float:
    """E[dk] per epoch with the half-duplex correction and native load.

    Derivation. E[dk] = P_idle - E[(n-V)/n ; V+M >= 2]. Since E[(n-V)/n] = 1-tau,
    subtracting the three V+M <= 1 cases gives the closed form below.
    """
    a, b = p0n(tau_nat), p1n(tau_nat)
    return (2 * (1 - tau) ** n * a
            + (n - 1) * tau * (1 - tau) ** (n - 1) * a
            + (1 - tau) ** n * b
            - (1 - tau))


def equilibrium_exists(tau_nat: float) -> bool:
    """A positive equilibrium needs upward drift at tau -> 0."""
    return 2 * p0n(tau_nat) + p1n(tau_nat) > 1.0


def tau_star(n: int, tau_nat: float = 0.0, all_update: bool = False) -> float | None:
    """Zero-drift transmission probability, or None if the equilibrium is gone."""
    if all_update:
        return brentq(drift_all_update, 1e-12, 0.9, args=(n,))
    if not equilibrium_exists(tau_nat):
        return None
    return brentq(drift, 1e-12, 0.95, args=(n, tau_nat))


def collapse_threshold(n_nat: int = N_NATIVE) -> float:
    """Native attempt rate above which visitor tau decays to the engine floor."""
    return brentq(lambda t: 2 * p0n(t, n_nat) + p1n(t, n_nat) - 1, 1e-9, 0.5)


def k_gap(n: int, tau_nat: float = 0.0, tau_0: float = P.TAU_0) -> float | None:
    """Net upward lattice steps from tau_0 to the equilibrium."""
    ts = tau_star(n, tau_nat)
    return None if ts is None else math.log(ts / tau_0) / math.log(C_MIMD)


def _main() -> None:
    x_star = brentq(lambda x: (2 + x) * math.exp(-x) - 1, 0.5, 5.0)
    print(f"continuum limit  (2+x)e^-x = 1  ->  x* = {x_star:.6f}")
    print(f"  residual {abs((2 + x_star) * math.exp(-x_star) - 1):.2e}\n")

    print("Theorem 1a — visitor-only channel")
    print(f"{'n':>4} {'tau*':>10} {'n.tau* all':>12} {'n.tau* half-dup':>16} {'1/n':>8}")
    for n in (10, 20, 30, 50):
        ta = tau_star(n, all_update=True)
        tb = tau_star(n, 0.0)
        print(f"{n:>4} {tb:10.5f} {n * ta:12.4f} {n * tb:16.4f} {1 / n:8.4f}")

    print(f"\nTheorem 1b — shared channel (N_native={N_NATIVE})")
    print(f"collapse threshold tau_nat = {collapse_threshold():.5f}")
    ns = (10, 20, 30, 50)
    print(f"{'tau_nat':>9} |" + "".join(f"{'n=' + str(n):>10}" for n in ns))
    for tn in (0.0, 0.018, TAU_NAT_MEASURED, 0.111, 0.125):
        row = f"{tn:9.4f} |"
        for n in ns:
            ts = tau_star(n, tn)
            row += f"{'collapse':>10}" if ts is None else f"{n * ts:10.4f}"
        print(row + ("   <- measured" if tn == TAU_NAT_MEASURED else ""))

    print(f"\nTheorem 2 — steps needed from tau_0 = 1/W_eff = {P.TAU_0:.5f}")
    print(f"{'n':>4} {'k_gap (solo)':>14} {'k_gap (shared)':>16}")
    for n in (10, 20, 50):
        print(f"{n:>4} {k_gap(n, 0.0):14.1f} {k_gap(n, TAU_NAT_MEASURED):16.1f}")
    print("  measured epoch budget per visit: 17-24  (measure_engine.py)")


def _self_check() -> None:
    """Assertions guarding every claim the manuscript draws from this module."""
    x_star = brentq(lambda x: (2 + x) * math.exp(-x) - 1, 0.5, 5.0)
    assert abs(x_star - 1.146193) < 1e-5

    # zero drift really is zero, both variants
    for n in (10, 20, 50):
        assert abs(drift_all_update(tau_star(n, all_update=True), n)) < 1e-9
        for tn in (0.0, 0.02, TAU_NAT_MEASURED):
            assert abs(drift(tau_star(n, tn), n, tn)) < 1e-9

    # native-free case must reduce to the half-duplex equation
    for n in (10, 50):
        assert abs(tau_star(n, 0.0) - brentq(
            lambda t: (2 * (1 - t) ** n + (n - 1) * t * (1 - t) ** (n - 1)
                       - (1 - t)), 1e-12, 0.9)) < 1e-12

    # n.tau* is nearly n-invariant (the core of Theorem 1)
    for tn in (0.0, TAU_NAT_MEASURED):
        vals = [n * tau_star(n, tn) for n in (10, 20, 30, 50)]
        assert max(vals) / min(vals) < 1.10, vals

    # solo channel sits above fair share, measured native load below it
    assert all(n * tau_star(n, 0.0) > 1.10 for n in (10, 20, 30, 50))
    assert all(n * tau_star(n, TAU_NAT_MEASURED) < 0.75 for n in (10, 20, 30, 50))

    # collapse threshold brackets correctly
    tc = collapse_threshold()
    assert 0.111 < tc < 0.112
    assert equilibrium_exists(tc - 1e-4) and not equilibrium_exists(tc + 1e-4)
    assert tau_star(20, 0.125) is None

    # Theorem 2: the gap exceeds the measured epoch budget's useful fraction
    assert k_gap(10, TAU_NAT_MEASURED) > 15.0
    assert k_gap(50, TAU_NAT_MEASURED) > 8.0
    print("\ndrift.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
