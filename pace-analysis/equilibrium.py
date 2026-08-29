"""Section 4.5.4 — the equilibrium layer: the objective's operating point and
the MIMD coefficient ratio that puts the drift equilibrium there.

    .venv/bin/python pace-analysis/equilibrium.py

The two-dimensional choice of (c_idle, c_coll) is reduced to one dimension
analytically. Writing eps_idle = ln c_idle and eps_coll = ln c_coll, the MIMD
rule is a random walk in ln tau whose per-epoch drift is

    D(tau) = eps_idle * A0(tau) - eps_coll * (1 - A0(tau) - A1(tau))

as seen by a listening STA, where A0 is the probability it hears an idle epoch
and A1 the probability it hears exactly one other transmitter. Setting D = 0,

    r = eps_idle / eps_coll = (1 - A0 - A1) / A0                        (*)

so the equilibrium tau is a function of the RATIO alone. The scale eps_coll
cancels: it sets how fast the walk moves and how much it jitters, not where it
settles. That is the scale invariance of section 4.5.

The design problem therefore splits cleanly:

  * WHERE to sit is an equilibrium question, answered here. Pick the operating
    point tau_J that maximises J = ln T - alpha*(ln rho)^2 in the saturated
    renewal-reward limit, then read r* off (*).
  * HOW FAST to get there is a finite-horizon question, and (*) says nothing
    about it. That is optimise.py's job.

The renewal-reward quantities here are the infinite-horizon limit: every STA is
viable, frames are drawn from the full U{LO..HI}, and no deadline truncates
anything. The deadline lives in the DP, which is the right division of labour
and is why this layer stays closed form.

ponytail: tau_nat is exogenous, as in drift.py. It is measured per contention
epoch (params.TAU_NAT) and validate.sensitivity() shows the answer barely moves
with it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import dp
import params as P

TAU_LO, TAU_HI = 1e-6, 0.60          # bracket for the operating-point search


# ─── renewal reward at a fixed tau ───────────────────────────────────────────

def epoch(tau: float, n_vis: int, tau_nat: float = None, n_nat: int = None,
          access: str = "rts") -> dict:
    """Per-epoch outcome probabilities, expected cost and expected airtime.

    Saturated limit: every visitor is viable and E[L] is the full mean, so this
    is the equilibrium the finite window is heading towards rather than what a
    single visit achieves.
    """
    tau_nat = P.TAU_NAT if tau_nat is None else tau_nat
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    coll_cost, succ_oh = P.ACCESS[access]

    e_len = (P.PPDU_V_LO + P.PPDU_V_HI) / 2.0
    q0v = (1.0 - tau) ** n_vis
    q1v = n_vis * tau * (1.0 - tau) ** (n_vis - 1)
    q0n = (1.0 - tau_nat) ** n_nat
    q1n = n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1)

    p_idle = q0v * q0n
    p_svis = q1v * q0n
    p_snat = q0v * q1n
    p_coll = max(1.0 - p_idle - p_svis - p_snat, 0.0)

    if coll_cost == "nocd":
        c_coll = float(dp.basic_collision_cost(
            P.W_EFF, np.array([tau]), n_vis, tau_nat, n_nat)[0])
    else:
        c_coll = float(coll_cost)

    cost = (p_idle * 1.0
            + p_svis * (e_len + succ_oh)
            + p_snat * (P.PPDU_NATIVE + succ_oh)
            + p_coll * c_coll)
    return {"p_idle": p_idle, "p_svis": p_svis, "p_snat": p_snat,
            "p_coll": p_coll, "cost": cost,
            "air_vis": p_svis * e_len, "air_nat": p_snat * P.PPDU_NATIVE}


def outcome_eq(tau: float, n_vis: int, n_nat: int = None, **kw) -> dict:
    """Equilibrium (T, rho): the two axes the objective trades off."""
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    e = epoch(tau, n_vis, n_nat=n_nat, **kw)
    tot = e["air_vis"] + e["air_nat"]
    if e["cost"] <= 0.0 or tot <= 0.0:
        return {"T": 0.0, "rho": 0.0, "visitor": 0.0, "native": 0.0}
    share = e["air_vis"] / tot
    return {"T": tot / e["cost"],
            "visitor": e["air_vis"] / e["cost"],
            "native": e["air_nat"] / e["cost"],
            "rho": share / (n_vis / (n_vis + n_nat))}


def objective_eq(tau: float, n_vis: int, alpha: float, **kw) -> float:
    """J = ln T - alpha*(ln rho)^2, evaluated at the equilibrium.

    Same functional form as dp.objective, so the operating point chosen here and
    the finite-horizon value optimised there are commensurable.
    """
    o = outcome_eq(tau, n_vis, **kw)
    if o["T"] <= 0.0 or o["rho"] <= 0.0:
        return -math.inf
    return math.log(o["T"]) - alpha * math.log(o["rho"]) ** 2


def tau_J(n_vis: int, alpha: float, **kw) -> float:
    """The operating point: argmax over tau of the equilibrium objective.

    Unimodal in tau on (0, TAU_HI): the airtime term rises then falls as
    collisions take over, and the fairness term is monotone increasing, so a
    bounded scalar search is enough.
    """
    res = minimize_scalar(lambda t: -objective_eq(t, n_vis, alpha, **kw),
                          bounds=(TAU_LO, TAU_HI), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x)


# ─── the ratio that puts the equilibrium there ───────────────────────────────

def a0_a1(tau: float, n_vis: int, tau_nat: float = None,
          n_nat: int = None) -> tuple[float, float]:
    """A listening STA's probability of hearing an idle epoch, and of hearing
    exactly one other transmitter.

    The listener is one of the n_vis visitors, so it contends against n_vis-1
    other visitors and n_nat natives. A solo success leaves its lattice index
    alone (it copies a tau equal to its own under homogeneity), which is why A1
    appears in (*) as neither an up nor a down step.
    """
    tau_nat = P.TAU_NAT if tau_nat is None else tau_nat
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    p0n = (1.0 - tau_nat) ** n_nat
    p1n = n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1) if n_nat else 0.0
    a0 = (1.0 - tau) ** (n_vis - 1) * p0n
    a1 = ((n_vis - 1) * tau * (1.0 - tau) ** (n_vis - 2) * p0n
          + (1.0 - tau) ** (n_vis - 1) * p1n)
    return a0, a1


def r_star(tau: float, n_vis: int, **kw) -> float:
    """r = eps_idle/eps_coll placing the zero-drift equilibrium at tau.

    Monotone increasing in tau: a more aggressive target needs larger upward
    steps relative to the downward ones to hold against the collisions it
    causes. r = 1 is not a neutral choice, it is the particular target where a
    listener's idle and collision probabilities coincide.
    """
    a0, a1 = a0_a1(tau, n_vis, **kw)
    if a0 <= 0.0:
        return math.inf
    return (1.0 - a0 - a1) / a0


def r_star_continuum(x: float) -> float:
    """Native-free Poisson limit of r_star, with x = n*tau held fixed.

    A0 -> e^-x and A1 -> x e^-x, so r* -> e^x - 1 - x. This is the form quoted
    in PACE_TWC_ANALYSIS.md section 4.5.3; native load raises it sharply, by
    about 7x at N_nat = 10, so the closed form is a lower bound in a shared BSS.
    """
    return math.exp(x) - 1.0 - x


def design(n_vis: int, alpha: float, **kw) -> dict:
    """One scenario's equilibrium layer: the operating point and the ratio.

    r_star reads the channel, not the reward, so it takes only the population
    arguments; access mode reaches the objective but not the drift.
    """
    pop = {k: v for k, v in kw.items() if k in ("tau_nat", "n_nat")}
    t = tau_J(n_vis, alpha, **kw)
    o = outcome_eq(t, n_vis, **kw)
    return {"tau_J": t, "x_J": t * n_vis, "r_star": r_star(t, n_vis, **pop),
            "T_eq": o["T"], "rho_eq": o["rho"],
            "J_eq": objective_eq(t, n_vis, alpha, **kw)}


# ─── reporting ───────────────────────────────────────────────────────────────

def _main() -> None:
    print("=== the ratio is the dial (N_vis=20, N_nat=10, measured tau_nat) ===")
    print(f"{'x=n.tau':>8} {'r*':>8} {'r* (no nat)':>12} {'e^x-1-x':>10}")
    for x in (0.17, 0.378, 0.64, 1.0, 1.5):
        print(f"{x:8.3f} {r_star(x / 20, 20):8.4f} "
              f"{r_star(x / 20, 20, tau_nat=0.0, n_nat=0):12.4f} "
              f"{r_star_continuum(x):10.4f}")

    for access in ("rts", "basic"):
        print(f"\n=== operating point vs alpha ({access}, N_nat=10) ===")
        print(f"{'N_vis':>6} {'alpha':>6} {'tau_J':>8} {'x_J':>7} {'T_eq':>7} "
              f"{'rho_eq':>7} {'r*':>8} {'c_idle at c_coll=1.2':>21}")
        for n_vis in (10, 20, 50):
            for al in (0.0, 0.05, 0.2, 0.5):
                d = design(n_vis, al, access=access)
                c_i = P.C_MIMD ** d["r_star"]
                print(f"{n_vis:>6} {al:6.2f} {d['tau_J']:8.5f} {d['x_J']:7.3f} "
                      f"{d['T_eq']:7.4f} {d['rho_eq']:7.4f} "
                      f"{d['r_star']:8.4f} {c_i:21.4f}")

    print("\n=== native load raises the required ratio (x_J from alpha=0.05) ===")
    print(f"{'N_nat':>6} {'tau_J':>8} {'x_J':>7} {'r*':>8}")
    for nn in (0, 5, 10, 20):
        d = design(20, 0.05, n_nat=nn, tau_nat=P.TAU_NAT if nn else 0.0)
        print(f"{nn:>6} {d['tau_J']:8.5f} {d['x_J']:7.3f} {d['r_star']:8.4f}")


def _self_check() -> None:
    # (*) must reduce to the drift equation drift.py already solves: at the
    # r = 1 equilibrium the formula must return exactly 1.
    import drift as D
    for n in (10, 20, 50):
        for tn in (0.0, P.TAU_NAT):
            nn = P.N_NATIVE if tn else 0
            ts = D.tau_star(n, tn)
            assert abs(r_star(ts, n, tau_nat=tn, n_nat=nn) - 1.0) < 1e-9, (n, tn)

    # native-free r* must converge to the continuum form as n grows
    for x in (0.2, 0.5, 1.0):
        big = r_star(x / 500, 500, tau_nat=0.0, n_nat=0)
        assert abs(big - r_star_continuum(x)) / r_star_continuum(x) < 0.01, x

    # r* is monotone increasing in tau, which is what makes it a usable dial
    for n in (10, 20, 50):
        rs = [r_star(x / n, n) for x in np.linspace(0.05, 2.0, 40)]
        assert all(a < b for a, b in zip(rs, rs[1:])), n
        assert all(math.isfinite(v) and v > 0 for v in rs)

    # native load raises r*, contrary to the plan's stated hypothesis
    prev = -1.0
    for nn in (0, 5, 10, 20):
        v = r_star(0.378 / 20, 20, tau_nat=P.TAU_NAT if nn else 0.0, n_nat=nn)
        assert v > prev, (nn, v, prev)
        prev = v

    # the objective is a dial in the same direction as dp.objective: raising
    # alpha must not lower the chosen operating point or its rho
    for access in ("rts", "basic"):
        pt = pr = -1.0
        for al in (0.0, 0.05, 0.2, 0.5):
            d = design(20, al, access=access)
            assert d["tau_J"] >= pt - 1e-9, (access, al, d["tau_J"], pt)
            assert d["rho_eq"] >= pr - 1e-9, (access, al, d["rho_eq"], pr)
            assert TAU_LO < d["tau_J"] < TAU_HI, d["tau_J"]
            pt, pr = d["tau_J"], d["rho_eq"]

    # the airtime-only optimum must be a genuine interior maximum of T_eq
    for n in (10, 20, 50):
        t0 = tau_J(n, 0.0)
        base = outcome_eq(t0, n)["T"]
        assert all(outcome_eq(t0 * f, n)["T"] <= base + 1e-12
                   for f in (0.5, 0.8, 1.25, 2.0)), n

    # probabilities and costs stay physical over the whole search bracket
    for t in (TAU_LO, 0.01, 0.1, TAU_HI):
        e = epoch(t, 20)
        assert abs(e["p_idle"] + e["p_svis"] + e["p_snat"] + e["p_coll"]
                   - 1.0) < 1e-12
        assert e["cost"] >= 1.0 and 0.0 < outcome_eq(t, 20)["T"] < 1.0
    print("\nequilibrium.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
