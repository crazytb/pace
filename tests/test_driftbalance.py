# -*- coding: utf-8 -*-
"""Section 4.5.26: the drift-balance rule and what it does and does not give."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pace-analysis"))

import drift as D            # noqa: E402
import driftbalance as DB    # noqa: E402


@pytest.mark.parametrize("n", [5, 10, 20, 50])
@pytest.mark.parametrize("n_nat", [0, 10])
@pytest.mark.parametrize("tau", [0.002, 0.02, 0.1])
def test_listener_conditioned_drift_matches_drift_py_at_r_one(n, n_nat, tau):
    """The general-r drift must collapse onto the shipped r = 1 equation, or
    the two halves of the analysis are describing different algorithms."""
    mine = DB.drift(math.log(tau), n, 1.0, 1.0, n_nat)
    theirs = D.drift(tau, n, DB.TAU_NAT if n_nat else 0.0)
    assert mine == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("access", ["rts", "basic"])
@pytest.mark.parametrize("n", [5, 20, 50])
def test_airtime_optimum_is_well_below_one_over_n(access, n):
    """tau* = 1/N is the zero-cost-collision answer. Once a collision burns 12
    slots (rts) or a whole frame (basic) the optimum backs off well below it."""
    t = DB.tau_target(n, access)
    assert t is not None and t < 1.0 / n
    assert (1.0 / n) / t > 2.0


@pytest.mark.parametrize("access", ["rts", "basic"])
@pytest.mark.parametrize("n", [5, 20, 50])
def test_r_star_round_trips_through_the_equilibrium(access, n):
    """A pair built from r* must put the drift zero back at the target it was
    built from; otherwise the rule is not solving its own equation."""
    t = DB.tau_target(n, access)
    r = DB.r_star(n, access)
    eps_i = 0.05
    back = DB.tau_eq(math.exp(eps_i), math.exp(r * eps_i), n)
    assert math.log(back / t) == pytest.approx(0.0, abs=1e-6)


def test_basic_access_with_natives_degenerates_to_silence():
    """Section 4.5.4 (6): with natives present and no fairness term, the pure
    airtime optimum under basic access is for the visitors to stay quiet. The
    rule must return None there rather than inventing a coefficient."""
    assert DB.tau_target(20, "basic", 10) is None
    assert DB.r_star(20, "basic", 10) is None
    assert DB.design_hat(20, 420, "basic", 9.27, 10) is None


@pytest.mark.parametrize("access,lo,hi", [("rts", 14.0, 22.4),
                                          ("basic", 83.0, 132.0)])
def test_r_star_is_large_because_collisions_are_rare_at_the_optimum(
        access, lo, hi):
    """The mechanism behind Proposition 1's degeneracy: at tau_target the
    listener-conditioned collision probability is a few percent, so eps_coll
    enters the drift multiplied by almost nothing and the balance demands a
    ratio in the tens. Measured range over N_vis in {5,10,20,50}."""
    rs = [DB.r_star(n, access) for n in (5, 10, 20, 50)]
    assert lo <= min(rs) and max(rs) <= hi
    for n in (5, 20, 50):
        assert DB.Pc_lis(DB.tau_target(n, access), n) < 0.05


@pytest.mark.parametrize("n", [5, 20, 50])
def test_shipped_equilibrium_sits_above_the_airtime_optimum(n):
    """The r = 1 drift equilibrium is 3-8x above tau_target, so a visit that
    ran to equilibrium would over-transmit. Theorem 2's early stop is what
    keeps the engine near the optimum instead."""
    for access, lo, hi in (("rts", 3.0, 3.5), ("basic", 7.0, 8.0)):
        ratio = DB.tau_eq(1.2, 1.2, n) / DB.tau_target(n, access)
        assert lo < ratio < hi


# ── section 4.5.27: the objective-targeted rule ──────────────────────────────

@pytest.mark.parametrize("access", ["rts", "basic"])
@pytest.mark.parametrize("n", [5, 20, 50])
def test_alpha_lifts_the_target_out_of_the_silence_degeneracy(access, n):
    """With natives and alpha = 0 under basic access the objective wants the
    visitors quiet; any alpha > 0 pulls the target back to a real value. This
    is what makes the J-targeted rule defined where the T-targeted one is not."""
    if access == "basic":
        assert DB.tau_J(n, 10, "basic", 0.0) is None
    for a in (0.25, 0.5, 1.0):
        t = DB.tau_J(n, 10, access, a)
        assert t is not None and 1e-4 < t < 0.2


@pytest.mark.parametrize("n", [5, 20, 50])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_r_J_lands_inside_the_swept_grid(n, alpha):
    """The whole point of targeting J rather than T: the ratio comes back into
    a range a sweep can resolve, instead of the 14-132 of section 4.5.26."""
    for access in ("rts", "basic"):
        r = DB.r_J(n, 10, access, alpha)
        assert 0.5 < r < 5.0, (access, n, alpha, r)


@pytest.mark.parametrize("n", [5, 20, 50])
def test_r_J_is_robust_to_the_measured_rho_bias(n):
    """The analytic rho runs a median 0.874 of the engine's. If r_J moved much
    under that, the rule would be untestable until the native model is fixed."""
    for access in ("rts", "basic"):
        vals = [DB.r_J(n, 10, access, 0.5, rho_cal=k)
                for k in (1.0, 0.874, 0.842)]
        assert max(vals) / min(vals) < 1.5


def test_rho_model_is_one_without_natives():
    for n in (5, 20):
        assert DB.rho_model(0.02, n, 0, "rts") == 1.0
