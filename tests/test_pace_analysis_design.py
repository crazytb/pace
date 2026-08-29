"""The coefficient design rule (pace-analysis/design.py).

Pins the two claims the manuscript will make about how c_idle and c_coll are
chosen, and the properties that make the rule usable at all:

  c_idle  the ramp equation lands on Eq. (9)'s target, has no fitted constant,
          and depends on W_eff through the integration rather than an exponent
  c_coll  is NOT identified by J once c_idle is free, so it is reported as a
          feasible choice rather than an optimum

The engine-backed tests are slow by this suite's standards. They are here
because these are the numbers the paper quotes.
"""
import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "pace-analysis"), os.path.join(_ROOT, "harq_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coeff_oracle as CO      # noqa: E402
import design as D             # noqa: E402
import equilibrium as EQ       # noqa: E402
import params as P             # noqa: E402


# ── the ramp equation does what it says ──────────────────────────────────────

@pytest.mark.parametrize("access", ["rts", "basic"])
@pytest.mark.parametrize("n_vis,w_eff", [(5, 210), (10, 420), (20, 1680)])
def test_ramp_lands_on_the_target(access, n_vis, w_eff):
    ci = D.c_idle(n_vis, 10, w_eff, access)
    got = D.x_end(math.log(ci), math.log(D.C_COLL), n_vis, 10,
                  w_eff, access)
    assert got == pytest.approx(math.log(1.0 / (n_vis + 10)), abs=1e-4)


def test_target_is_the_operating_point_knob():
    """A more aggressive target must need a larger up step, monotonically, or
    the target cannot be used to traverse the frontier."""
    cs = [D.c_idle(10, 10, 420, "rts", target=t / 20)
          for t in (0.5, 0.8, 1.0, 1.5, 2.0)]
    assert all(a < b for a, b in zip(cs, cs[1:])), cs
    assert all(c > 1.0 for c in cs)


def test_c_idle_falls_with_the_window():
    """The W dependence comes out of the integration, not out of a fitted
    exponent. It must still be monotone, which is what the exponent asserted."""
    for access in ("rts", "basic"):
        cs = [D.c_idle(10, 10, w, access)
              for w in (105, 210, 420, 840, 1680)]
        assert all(a > b for a, b in zip(cs, cs[1:])), (access, cs)


def _slopes(access, ws):
    e = [math.log(D.c_idle(10, 10, w, access)) for w in ws]
    return [math.log(e[i + 1] / e[i]) / math.log(ws[i + 1] / ws[i])
            for i in range(len(ws) - 1)]


@pytest.mark.parametrize("access", ["rts", "basic"])
def test_ramp_equation_reproduces_the_measured_half_power(access):
    """The -1/2 scaling was fitted in section 4.5.4 and its constant was then
    calibrated. The ramp equation is told nothing about either, so if it puts
    the local slope near -1/2 across the windows that were measured, the
    exponent has been derived rather than assumed. basic runs steeper than rts
    because its smaller gain pushes the saturation further out (below)."""
    s = _slopes(access, [105, 210, 420, 840, 1680])
    assert all(-0.80 < x < -0.40 for x in s), (access, s)


@pytest.mark.parametrize("access", ["rts", "basic"])
def test_the_half_power_is_a_crossover_not_a_law(access):
    """Widen the range and the exponent goes away: the walk becomes
    equilibrium-limited and eps_idle stops falling. The measured -1/2 is the
    middle of that crossover, which is why a fitted exponent drifted with the
    scenario (section 4.5.5) and its constant with it."""
    ws = [52, 105, 420, 1680, 3360, 6720, 13440]
    s = _slopes(access, ws)
    assert s[0] > -0.45, (access, s)               # shallow at very short W
    assert min(s) < -0.5, (access, s)              # steepest in the middle
    assert s[-1] > -0.05, (access, s)              # flat once equilibrium wins
    assert D.c_idle(10, 10, 13440, access) == pytest.approx(
        D.c_idle(10, 10, 6720, access), rel=0.01)


def test_unreachable_target_raises_rather_than_returning_nonsense():
    with pytest.raises(ValueError):
        D.c_idle(10, 10, 420, "rts", target=0.99)


# ── Proposition 2 is a check, not a design step ──────────────────────────────

@pytest.mark.parametrize("access", ["rts", "basic"])
@pytest.mark.parametrize("w_eff", [210, 420, 1680])
def test_ramp_solution_satisfies_proposition_2_for_free(access, w_eff):
    """The walk cannot climb unless D(tau_0) > 0, so any solution of the ramp
    equation satisfies the feasibility bound by construction. This is why the
    coupling between the two coefficients is not circular."""
    ci = D.c_idle(10, 10, w_eff, access)
    assert D.ramp_feasible(D.C_COLL, math.log(ci), 10, 10, w_eff)


def test_proposition_2_bound_tightens_with_native_load():
    """A0 ~ (1-tau_nat)^N_nat, so idle observations become exponentially rare
    and the feasible c_coll collapses. This is what makes the bound bite at
    heavy native load and be inert at light load."""
    tau0 = 1.0 / 420
    rs = [EQ.r_star(tau0, 10, n_nat=n) for n in (5, 10, 20, 30)]
    assert all(a < b for a, b in zip(rs, rs[1:])), rs
    assert rs[0] < 0.1 and rs[-1] > 2.0


# ── c_coll: a choice, not an optimum ────────────────────────────────────────

def test_c_coll_is_a_plain_feasible_choice():
    """It is an input. Pin the value only so that changing it is deliberate."""
    assert isinstance(D.C_COLL, float)
    assert 1.0 < D.C_COLL < 1.5


def test_J_does_not_identify_c_coll_once_c_idle_is_free():
    """The claim that replaced the withdrawn minimax. Sweeping the coefficients
    TOGETHER makes c_coll look decisive, but that couples it to the ramp
    equation's c_idle. Re-optimise c_idle at each c_coll and the best attainable
    J barely moves, so c_coll cannot be chosen by performance.

    Coarse on purpose: the point is the flatness, not the argmax."""
    scn = CO.Scn(10, 10, 420, "rts")
    seeds, visits = CO.EVAL_SEEDS[:20], 20
    best = {}
    for cc in (1.05, 1.40):
        js = []
        for e in (0.25, 0.40, 0.55, 0.70):
            m = CO.aggregate(CO.batch(scn, math.exp(e), cc, seeds, visits),
                             scn, 0.2)
            js.append(m["J"])
        best[cc] = max(js)
    assert math.exp(min(best.values()) - max(best.values())) > 0.95, best


# ── the rule as a whole ──────────────────────────────────────────────────────

def test_coefficients_returns_a_usable_pair():
    for access in ("rts", "basic"):
        ci, cc = D.coefficients(10, 10, 420, access)
        assert ci > 1.0 and cc > 1.0
        assert ci == pytest.approx(D.c_idle(10, 10, 420, access))
        assert cc == D.C_COLL


def test_rule_beats_the_shipped_constants_on_the_primitives():
    """Against (1.2, 1.2): the design rule must not lose total airtime while it
    raises the visitor share. Reported on T and rho rather than on J, since J
    was defined in this work and a comparison made only in it is circular."""
    scn = CO.Scn(10, 10, 420, "rts")
    ci, cc = D.coefficients(scn.n_vis, scn.n_nat, scn.w_eff, scn.access)
    new = CO.aggregate(CO.batch(scn, ci, cc, CO.EVAL_SEEDS[:20], 20), scn, 0.2)
    old = CO.aggregate(CO.batch(scn, 1.2, 1.2, CO.EVAL_SEEDS[:20], 20), scn, 0.2)
    assert new["rho"] > old["rho"] + 0.3
    assert new["T"] > old["T"] * 0.98
