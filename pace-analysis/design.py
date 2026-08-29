"""The coefficient design rule: c_coll by minimax, c_idle by ramp arrival.

    .venv/bin/python pace-analysis/design.py

This replaces the calibrated rule of optimise.py (eps_idle = C / sqrt(W_eff)).
That rule fitted a power law and then fitted its constant; C moved 7.1-11.8
across scenarios and a 15% error in it cost up to 8% of utility at alpha <= 0.2
(section 4.5.20), so the constant was load bearing and undefended.

The replacement has no fitted constant.

c_idle -- ramp arrival.  Theorem 2 says the walk does not equilibrate inside a
visit, so the operating point is set by how far the ramp climbs, not by where
the equilibrium sits. Integrate the drift forward from the Phase (a) start and
solve for the up step that lands on the throughput-optimal target of Eq. (9):

    dX/dn = (1 - tau) (eps_idle A0(tau) - eps_coll Pc(tau)),   X = ln tau
    X(0)  = ln(1 / W_eff)
    X(E)  = ln(1 / N),   N = N_vis + N_nat

The epoch budget closes itself: each epoch consumes lambda(tau) slots and the
visit has W_eff of them, so E never has to be supplied. One bounded root find.

Why this beats a power law: the same equation spans both regimes. At short
windows the ln(W)/E term dominates and eps_idle ~ ln(W)/W; at long windows the
eps_coll Pc/A0 term dominates and r -> r*, the equilibrium ratio. Section
4.5.14's measured critical window is that handover, which also explains why the
fitted exponent came out near -1/2 and drifted with N_vis: it was the local
slope of a crossover, not a power law.

c_coll -- NOT determined by performance.  J does not identify it. Measured
with c_idle re-optimised at every c_coll, so the two are actually separated:
the best attainable J moves by under 2% across c_coll in [1.05, 1.60] at the
design load, and by under 7% against the worst load in [5, 30]. Sweeping the
two coefficients together looked far more decisive than that, but the ramp
equation drags c_idle along and the apparent dependence was mostly its error,
not c_coll's. Only two things constrain the choice, and neither is J:

    feasibility   Proposition 2, D(tau_0) > 0, satisfied by any ramp solution
    homogeneity   tau CV small enough that the dropped size-bias term stays
                  far below one up step

The default below sits inside both with margin. It is an input, not a result.

ponytail: the integrator is a fixed-step forward Euler over the window. The
answer moves by under 0.1% between 200 and 2000 steps, so an adaptive solver
would buy nothing.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

import dp
import equilibrium as EQ
import params as P

# A feasible default, NOT an optimum: see the module docstring. Any value in
# roughly [1.05, 1.4] performs within a few percent, so this is reported as a
# choice rather than derived.
C_COLL = 1.20

STEPS = 400


def _rates(tau: float, n_vis: int, n_nat: int, access: str, w_rem: float):
    """Listener-conditioned outcome probabilities and the slots an epoch costs.

    A0/A1/Pc condition on the tagged STA staying silent, which is the only case
    in which it updates at all (half duplex). The unconditional drift carries an
    extra factor (1 - tau); it is positive, so it cannot move a zero crossing,
    but it does set the pace and so appears in the integration.

    The collision cost comes from dp.basic_collision_cost rather than a local
    stand-in, so basic access uses the same order-statistic model as the DP and
    there is one place to fix if it is ever wrong.
    """
    a0, a1 = EQ.a0_a1(tau, n_vis, n_nat=n_nat)
    pc = max(1.0 - a0 - a1, 0.0)
    coll_cost, oh = P.ACCESS[access]
    if coll_cost == "nocd":
        l_col = float(dp.basic_collision_cost(
            max(int(w_rem), P.min_start()), np.array([tau]), n_vis,
            P.TAU_NAT, n_nat)[0])
    else:
        l_col = float(coll_cost)
    l_suc = (P.PPDU_V_LO + P.PPDU_V_HI) / 2 + oh
    return a0, a1, pc, a0 + a1 * l_suc + pc * l_col


def x_end(eps_i: float, eps_c: float, n_vis: int, n_nat: int, w_eff: int,
          access: str, steps: int = STEPS) -> float:
    """ln tau at the end of the window, integrating the drift in SLOTS.

    The window is consumed at a uniform rate, so the remaining window is known
    at every step and the collision cost can shrink with it the way the engine's
    does."""
    x = math.log(1.0 / w_eff)
    for i in range(steps):
        tau = math.exp(x)
        w_rem = w_eff * (1.0 - i / steps)
        a0, _a1, pc, lam = _rates(tau, n_vis, n_nat, access, w_rem)
        dn = (w_eff / steps) / lam              # epochs inside this slot slice
        x += dn * (1.0 - tau) * (eps_i * a0 - eps_c * pc)
        x = min(x, 0.0)
    return x


def c_idle(n_vis: int, n_nat: int = None, w_eff: int = None,
           access: str = "rts", c_coll: float = None,
           target: float = None) -> float:
    """The up step that lands the ramp on `target` by the end of the window.

    target defaults to Eq. (9)'s throughput-optimal rate for the whole
    contending set, 1 / (N_vis + N_nat). It is the operating-point knob: larger
    targets walk up the efficiency-proportionality frontier (section 4.5.19).
    """
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    c_coll = C_COLL if c_coll is None else c_coll
    target = 1.0 / (n_vis + n_nat) if target is None else target
    eps_c = math.log(c_coll)

    def g(e):
        return x_end(e, eps_c, n_vis, n_nat, w_eff, access) - math.log(target)
    lo, hi = 1e-4, 3.0
    if g(lo) > 0.0 or g(hi) < 0.0:
        raise ValueError(f"target {target:.4g} unreachable for "
                         f"N_vis={n_vis} N_nat={n_nat} W={w_eff} {access}")
    return math.exp(brentq(g, lo, hi, xtol=1e-6))


def coefficients(n_vis: int, n_nat: int = None, w_eff: int = None,
                 access: str = "rts", **kw) -> tuple[float, float]:
    """(c_idle, c_coll) for one deployment. The whole design rule."""
    cc = kw.pop("c_coll", None) or C_COLL
    return c_idle(n_vis, n_nat, w_eff, access, c_coll=cc, **kw), cc


def ramp_feasible(c_coll: float, eps_idle: float, n_vis: int, n_nat: int,
                  w_eff: int) -> bool:
    """Proposition 2, D(tau_0) > 0. Any solution of the ramp equation satisfies
    it by construction, since the walk could not climb otherwise, so this is a
    check rather than a design step."""
    tau0 = 1.0 / w_eff
    return c_coll < math.exp(eps_idle / EQ.r_star(tau0, n_vis, n_nat=n_nat))


def _main() -> None:
    print(f"c_coll = {C_COLL}  (a feasible choice; J does not identify it)\n")
    for access in ("rts", "basic"):
        print(f"{access}: c_idle  (c_coll = {C_COLL})")
        ws = (105, 210, 420, 840, 1680)
        print("  N_vis " + "".join(f"{'W='+str(w):>9}" for w in ws))
        for nv in (5, 10, 20, 50):
            print(f"  {nv:>5} " + "".join(
                f"{c_idle(nv, 10, w, access):9.2f}" for w in ws))
        print()


def _self_check() -> None:
    # the ramp must land where it was asked to
    for access in ("rts", "basic"):
        for nv in (5, 20):
            for w in (210, 420, 1680):
                ci = c_idle(nv, 10, w, access)
                got = x_end(math.log(ci), math.log(C_COLL), nv, 10, w, access)
                assert abs(got - math.log(1.0 / (nv + 10))) < 1e-4, (
                    access, nv, w, got)
                # and Proposition 2 is satisfied for free
                assert ramp_feasible(C_COLL, math.log(ci), nv, 10, w)

    # c_idle falls with the window and every value stays a real coefficient
    for access in ("rts", "basic"):
        cs = [c_idle(10, 10, w, access) for w in (105, 210, 420, 840, 1680)]
        assert all(a > b for a, b in zip(cs, cs[1:])), (access, cs)
        assert all(c > 1.0 for c in cs)

    # the integrator has converged: the answer must not depend on the step count
    a = c_idle(10, 10, 420, "rts")
    e = math.log(C_COLL)
    x200 = x_end(math.log(a), e, 10, 10, 420, "rts", steps=200)
    x2000 = x_end(math.log(a), e, 10, 10, 420, "rts", steps=2000)
    assert abs(x200 - x2000) < 5e-3, (x200, x2000)   # 0.5% in tau

    # a larger target is a more aggressive operating point, so it needs a
    # larger up step. This monotonicity is what makes the target a usable knob.
    prev = 0.0
    for th in (0.5, 0.8, 1.0, 1.5, 2.0):
        c = c_idle(10, 10, 420, "rts", target=th / 20)
        assert c > prev, (th, c, prev)
        prev = c

    print("design.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
