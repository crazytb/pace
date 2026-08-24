"""Phase 2: the viability closed form and the finite-horizon DP.

Pins the accuracy claims the manuscript will make, so a retune of the engine or
a "simplification" of the recursion cannot quietly invalidate them.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "pace-analysis"), os.path.join(_ROOT, "harq_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dp                       # noqa: E402
import params as P              # noqa: E402
import viability as V           # noqa: E402


# ── viability ────────────────────────────────────────────────────────────────

def test_frame_cdf_endpoints():
    assert V.f_len(P.PPDU_V_LO - 1) == 0.0
    assert V.f_len(P.PPDU_V_HI) == 1.0
    assert abs(V.f_len(P.PPDU_V_LO) - 1 / V.N_LEN) < 1e-12


def test_dead_tail_is_the_corrected_value():
    """8.3%, not the 27% the plan's first draft claimed from L_hs = 88 slots."""
    start, frac = V.dead_tail()
    assert start == 35
    assert abs(frac - 35 / 420) < 1e-12
    assert 0.08 < frac < 0.09


def test_viability_is_monotone_so_the_target_rises():
    """The premise the section 4.4 lemma rests on."""
    ws = list(range(P.W_EFF, 0, -1))
    vs = [V.expected_viable(w, 20) for w in ws]
    ts = [V.fs_target(w, 20) for w in ws]
    assert all(a >= b - 1e-12 for a, b in zip(vs, vs[1:]))
    assert all(a <= b + 1e-12 for a, b in zip(ts, ts[1:]))


@pytest.mark.parametrize("n_vis", (20, 50))
def test_closed_form_tracks_engine(n_vis):
    obs = V.measure_viable(n_vis)
    worst = max(abs(V.expected_viable(w + 10, n_vis) - obs[w]) for w in obs)
    assert worst / n_vis < 0.12, worst / n_vis


# ── DP ───────────────────────────────────────────────────────────────────────

def test_lattice_bounds_cover_the_reachable_range():
    assert dp.K_LO < 0 < dp.K_HI
    assert abs(dp.tau_of(0) - P.TAU_0) < 1e-15
    assert dp.tau_of(dp.K_LO - 5) == P.TAU_FLOOR


def test_dp_table_is_a_bounded_probability_weighted_sum():
    tab = dp.solve(20)
    assert np.all(np.isfinite(tab))
    assert np.all(tab >= 0.0)
    assert np.all(tab[0] == 0.0)
    # expected airtime can never exceed the window it is drawn from
    for w in range(0, P.W_EFF + 1):
        assert tab[w].max() <= w + 1e-9, (w, tab[w].max())


@pytest.mark.parametrize("access,tol", [("rts", 0.04), ("basic", 0.05)])
def test_dp_matches_engine(access, tol):
    for n_vis in (5, 10, 20, 50):
        pred = dp.total_airtime(n_vis, access=access)
        obs = dp.measured(n_vis, access=access)["total"]
        assert abs(pred - obs) < tol, (access, n_vis, pred, obs)


def test_dp_meets_the_plan_acceptance_criterion():
    """Section 5.2 of the plan: total airtime within 0.72 +/- 0.03."""
    pred = dp.total_airtime(20, access="rts")
    assert 0.69 <= pred <= 0.75, pred


def test_dp_residual_is_an_overestimate_from_native_solo_successes():
    """The residual is localised, not diffuse.

    Natives decrement backoff only on idle epochs, so their attempts cluster
    into collisions. Modelling them as independent Bernoulli per epoch turns
    some of those collisions into solo wins, which is where the DP's optimism
    comes from. Native airtime dominates the total, so this shows up directly.
    """
    import measure_engine as ME
    mix = dp.outcome_mix(20)
    obs = ME.measure(20, P.N_NATIVE, *P.ACCESS["rts"])
    assert mix["solo_nat"] > obs["f_solo_nat"], (mix["solo_nat"], obs["f_solo_nat"])
    assert mix["coll"] < obs["f_coll"], (mix["coll"], obs["f_coll"])
    # and the visitor side, which the analysis actually models, is close
    assert abs(mix["solo_vis"] - obs["f_solo_vis"]) < 0.02


def test_homogeneity_assumption_holds():
    """Section 3.2 was to be verified, not assumed. Solo-copy resynchronises."""
    f25 = P.engine()
    f25.N_VISITOR, f25.N_NATIVE = 20, P.N_NATIVE
    try:
        st = {}
        for r in range(3):
            rp = np.random.default_rng(10001 + r * 71 + 7)
            rg = np.random.default_rng(200003 + r * 3163 + 20 * 211)
            for _ in range(30):
                f25._run_visit25(f25._sample_ppdus25(rp), rg, "pace",
                                 np.full(20, P.TAU_0), *P.ACCESS["rts"],
                                 stats=st)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    cv = st["tau_cv_sum"] / st["tau_cv_cnt"]
    assert cv < 0.05, f"population is not homogeneous: CV={cv:.4f}"
