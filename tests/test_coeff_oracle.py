"""Phase 5: analytical coefficients against a simulation oracle.

One test per item of the brief's section 14 checklist, plus the pieces of the
harness whose failure would be silent: pooled-then-J aggregation, seed
separation, and the identity G_J = exp(J_A - J_O).

Kept cheap: dev seeds and short sequences everywhere. These check that the
machinery computes what it claims, not that any particular scenario's numbers
have converged.
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

import coeff_oracle as CO       # noqa: E402
import dp                       # noqa: E402
import params as P              # noqa: E402

SCN = CO.Scn(10, 10, 420, "rts")


# ── 14.1  the shipped coefficients reproduce the published figure ────────────

@pytest.mark.parametrize("access,useful,vis", [("rts", 0.7195, 0.1046),
                                               ("basic", 0.6240, 0.0771)])
def test_shipped_coefficients_reproduce_fig29(access, useful, vis):
    """results/step9/fig29/data.csv, method=pace, N_visitor=10, averaged over
    its five seeds. This harness runs cold-start visits where fig29 carries tau
    across a sequence, so the tolerance is loose enough to allow that and tight
    enough to catch a unit or accounting error."""
    scn = CO.Scn(10, 10, 420, access)
    a = CO.aggregate(CO.batch(scn, 1.2, 1.2, CO.DEV_SEEDS, 40), scn, 0.2)
    assert a["T"] == pytest.approx(useful, rel=0.03)
    assert a["A_vis"] == pytest.approx(vis, rel=0.12)


# ── 14.2  coefficients are multiplicative factors above one ──────────────────

def test_batch_rejects_coefficients_below_one():
    """Below 1 inverts the rule (an idle would DECREASE tau), which is never a
    control, only a bug. Exactly 1 is allowed: it disables that update."""
    for bad in ((0.9, 1.2), (1.2, 0.95)):
        with pytest.raises(AssertionError):
            CO.batch(SCN, *bad, CO.DEV_SEEDS[:1], 1)
    CO.batch(SCN, 1.0, 1.2, CO.DEV_SEEDS[:1], 1)        # allowed


def test_every_candidate_stays_above_one():
    res = CO.run_scenario((10, 10, 420, "rts"), alphas=(0.2,), tune_visits=6,
                          eval_seeds=tuple(range(101, 107)), eval_visits=6,
                          n_boot=50)
    r = res["rows"][0]
    for k in ("c_idle_A", "c_coll_A", "c_idle_O", "c_coll_O",
              "c_idle_current", "c_coll_current", "c_idle_equilibrium_ratio",
              "c_coll_equilibrium_ratio"):
        assert r[k] > 1.0, (k, r[k])


# ── 14.3  the log transform is consistent in both directions ─────────────────

def test_eps_and_c_are_inverse_on_the_grid():
    g = CO.SimGrid(SCN, CO.DEV_SEEDS[:2], 3)
    p = g(0.3, 0.45)
    assert p["c_idle"] == pytest.approx(math.exp(p["eps_idle"]))
    assert p["c_coll"] == pytest.approx(math.exp(p["eps_coll"]))
    assert math.log(p["c_idle"]) == pytest.approx(0.3, abs=1e-6)
    assert math.log(p["c_coll"]) == pytest.approx(0.45, abs=1e-6)


# ── 14.4  the same evaluation seeds give the same answer ─────────────────────

def test_evaluation_is_deterministic():
    a = CO.batch(SCN, 1.35, 1.2, CO.DEV_SEEDS[:3], 5)
    b = CO.batch(SCN, 1.35, 1.2, CO.DEV_SEEDS[:3], 5)
    assert a == b


def test_common_random_numbers_are_shared_across_candidates():
    """Different coefficients must see the same exogenous workload. The first
    visit's PPDU draw is the part that is exactly shared, so compare that."""
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = SCN.n_vis, SCN.n_nat
    try:
        p1, _ = CO._rngs(SCN, 101)
        p2, _ = CO._rngs(SCN, 101)
        assert np.array_equal(f25._sample_ppdus25(p1), f25._sample_ppdus25(p2))
        p3, _ = CO._rngs(SCN, 102)
        assert not np.array_equal(f25._sample_ppdus25(p1),
                                  f25._sample_ppdus25(p3))
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old


# ── 14.5  tuning and evaluation seeds are disjoint ───────────────────────────

def test_seed_sets_are_disjoint():
    assert not set(CO.TUNE_SEEDS) & set(CO.EVAL_SEEDS)
    assert not set(CO.DEV_SEEDS) & (set(CO.TUNE_SEEDS) | set(CO.EVAL_SEEDS))


def test_run_scenario_refuses_overlapping_seeds():
    with pytest.raises(AssertionError):
        CO.run_scenario((10, 10, 420, "rts"), alphas=(0.2,), tune_visits=1,
                        tune_seeds=(1, 2), eval_seeds=(2, 3), eval_visits=1,
                        n_boot=10)


# ── 14.6  G_J equals the direct utility ratio ────────────────────────────────

def test_G_J_matches_the_direct_utility_ratio():
    res = CO.run_scenario((10, 10, 420, "rts"), alphas=(0.2,), tune_visits=6,
                          eval_seeds=tuple(range(101, 109)), eval_visits=6,
                          n_boot=50)
    r = res["rows"][0]
    direct = ((r["T_analytic"] * math.exp(-0.2 * math.log(r["rho_analytic"]) ** 2))
              / (r["T_oracle"] * math.exp(-0.2 * math.log(r["rho_oracle"]) ** 2)))
    assert r["G_J"] == pytest.approx(direct, rel=1e-9)
    assert r["G_J"] == pytest.approx(
        math.exp(r["J_analytic"] - r["J_oracle"]), rel=1e-12)


# ── 14.7  a boundary optimum is announced, not hidden ────────────────────────

def test_boundary_flag_fires_when_the_optimum_is_pinned():
    """A box that excludes the peak must come back flagged. Force it by
    searching a sliver at the very bottom of the feasible range, where J still
    increases with eps_idle."""
    g = CO.SimGrid(SCN, CO.DEV_SEEDS[:3], 6)
    out = CO.search(g, (0.2,), lo=0.011, hi=0.02, n_coarse=3, max_expand=0)
    assert out[0.2]["boundary"]


def test_interior_optimum_is_not_flagged():
    g = CO.SimGrid(SCN, CO.DEV_SEEDS[:3], 8)
    out = CO.search(g, (0.2,), lo=0.02, hi=1.3, n_coarse=7, max_expand=0)
    assert not out[0.2]["boundary"]


# ── 14.8  J is formed after pooling, never averaged over transitions ─────────

def test_J_is_not_the_mean_of_per_sequence_J():
    rows = CO.batch(SCN, 1.5, 1.2, CO.DEV_SEEDS, 8)
    pooled = CO.aggregate(rows, SCN, 0.2)
    per_seq = float(np.mean([CO.aggregate([r], SCN, 0.2)["J"] for r in rows]))
    # Jensen: ln is concave, so the mean of logs sits strictly below the log of
    # the mean. If these ever coincide the pooling has been undone.
    assert per_seq < pooled["J"]
    assert pooled["T"] == pytest.approx(
        sum(r["A_vis"] + r["A_nat"] for r in rows)
        / (sum(r["visits"] for r in rows) * SCN.w_eff))


def test_bootstrap_pools_the_same_way_as_aggregate():
    """With every resample forced to the identity the bootstrap must reproduce
    the point estimate exactly, which is the check that it pools rather than
    averages."""
    rows = CO.batch(SCN, 1.4, 1.2, CO.DEV_SEEDS, 6)
    same = CO.bootstrap(rows, rows, SCN, 0.2, n_boot=64)
    assert same["G_J_ci"] == pytest.approx((1.0, 1.0))
    assert same["dT_ci"] == pytest.approx((0.0, 0.0), abs=1e-12)


# ── 14.9  the analytical candidate is scored by the engine, not by the DP ────

def test_analytic_row_reports_simulation_J_not_model_J():
    res = CO.run_scenario((10, 10, 420, "rts"), alphas=(0.2,), tune_visits=6,
                          eval_seeds=tuple(range(101, 109)), eval_visits=6,
                          n_boot=50)
    r = res["rows"][0]
    # the model's own value is kept, in its own column, and differs
    assert r["J_model_A"] != pytest.approx(r["J_analytic"], abs=1e-6)
    # and the reported one is reproducible by re-running the engine
    again = CO.aggregate(
        CO.batch(CO.Scn(10, 10, 420, "rts"), r["c_idle_A"], r["c_coll_A"],
                 tuple(range(101, 109)), 6), CO.Scn(10, 10, 420, "rts"), 0.2)
    assert again["J"] == pytest.approx(r["J_analytic"], rel=1e-12)


# ── harness invariants ───────────────────────────────────────────────────────

def test_rho_is_one_when_there_are_no_natives():
    scn = CO.Scn(10, 0, 420, "rts")
    a = CO.aggregate(CO.batch(scn, 1.2, 1.2, CO.DEV_SEEDS[:3], 5), scn, 0.2)
    assert a["A_nat"] == 0.0
    assert a["rho"] == pytest.approx(1.0)
    assert a["J"] == pytest.approx(math.log(a["T"]))


def test_dp_point_reports_the_achieved_ratio_not_the_requested_one():
    """The DP quantises r onto a rational lattice. Whatever it actually ran
    must be what the table prints, or the analytic column names coefficients
    that were never evaluated."""
    p = CO.dp_point(SCN, 0.37, 0.19)
    assert p["eps_coll"] == pytest.approx(0.19)
    assert p["eps_idle"] == pytest.approx(p["r_eff"] * 0.19)
    assert p["c_idle"] == pytest.approx(math.exp(p["eps_idle"]))
    assert abs(p["r_eff"] - 0.37 / 0.19) / (0.37 / 0.19) < 0.06


# ── the Pareto view ──────────────────────────────────────────────────────────

def test_F_folds_over_and_under_share_together():
    assert CO.proportionality(1.0) == pytest.approx(1.0)
    assert CO.proportionality(2.0) == pytest.approx(CO.proportionality(0.5))
    assert CO.proportionality(0.3) < CO.proportionality(0.9) < 1.0


def test_objective_is_a_weighted_geometric_mean_of_T_and_F():
    """exp(J) = T * F^alpha. This is what makes J Pareto-consistent, so if the
    identity ever breaks the frontier figure stops meaning what it says."""
    for T, rho, al in ((0.72, 0.90, 0.2), (0.5, 1.4, 0.5), (0.63, 0.26, 0.05)):
        assert (math.exp(CO.objective(T, rho, al))
                == pytest.approx(T * CO.proportionality(rho) ** al))


def test_pareto_front_keeps_only_non_dominated_points():
    pts = [{"T": 0.72, "F": 0.99},        # dominates the next one on both axes
           {"T": 0.68, "F": 0.78},
           {"T": 0.73, "F": 0.62},        # trades T for F against the first
           {"T": 0.60, "F": 1.00}]
    front = CO.pareto_front(pts)
    assert {(p["T"], p["F"]) for p in front} == {(0.72, 0.99), (0.73, 0.62),
                                                 (0.60, 1.00)}
    assert [p["T"] for p in front] == sorted(p["T"] for p in front)


def test_every_alpha_optimum_is_non_dominated():
    """The Pareto-consistency of J, checked on real measurements rather than
    on the algebra: an argmax of ln T + alpha ln F can never be dominated."""
    scn = CO.Scn(10, 10, 420, "rts")
    g = CO.SimGrid(scn, CO.DEV_SEEDS[:3], 6)
    pts = [dict(g(ei, ec), F=CO.proportionality(g(ei, ec)["rho"]))
           for ei in np.geomspace(0.08, 0.7, 5)
           for ec in np.geomspace(0.08, 0.7, 5)]
    front = CO.pareto_front(pts)
    for al in (0.0, 0.05, 0.2, 0.5, 2.0):
        b = max(pts, key=lambda p: CO.objective(p["T"], p["rho"], al))
        assert any(q is b for q in front), al


def test_objective_floors_rather_than_diverging():
    assert CO.objective(0.0, 0.0, 0.2) == pytest.approx(
        math.log(CO.DELTA) - 0.2 * math.log(CO.DELTA) ** 2)
    assert CO.objective(0.5, 1.0, 0.5) == pytest.approx(math.log(0.5))
