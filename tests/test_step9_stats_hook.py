"""The `stats` hook added to _run_visit25 must be purely passive.

It exists for the drift analysis (pace-analysis/), which needs per-epoch
tallies the return value does not carry. Because it lives inside the slot loop,
a careless edit that consumes rng would silently shift every figure generated
from fig25/26/27/29. These tests pin that down.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harq_sim"))
import run_step9_fig25 as f25  # noqa: E402

MODES = ("pace", "dcf_excl", "dcf_conv", "oracle", "pace_noexcl")
ACCESS = (("rts", f25.COLL_RTS_24M, f25.OH_SUCC_24M), ("basic", "nocd", 0))


def _run(mode, coll_cost, succ_oh, seed, use_stats, visits=15, n_vis=20,
         trace=False):
    f25.N_VISITOR, f25.N_NATIVE = n_vis, 10
    try:
        rng_p = np.random.default_rng(seed * 10001)
        rng = np.random.default_rng(seed * 200003)
        st = ({"trace": []} if trace else {}) if use_stats else None
        out = []
        for _ in range(visits):
            ppdus = f25._sample_ppdus25(rng_p)
            tau0 = (np.full(n_vis, 1.0 / f25.W_REF)
                    if mode.startswith(("pace", "oracle")) else None)
            air, coll, idle, oh, _carry = f25._run_visit25(
                ppdus, rng, mode, tau0, coll_cost, succ_oh, stats=st)
            out.append((air.tolist(), coll, idle, oh))
        return out, st
    finally:
        f25.N_VISITOR, f25.N_NATIVE = 10, 10


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("label,coll_cost,succ_oh", ACCESS)
@pytest.mark.parametrize("seed", (1, 7, 42))
def test_stats_hook_does_not_perturb(mode, label, coll_cost, succ_oh, seed):
    """Identical results with and without the hook — it must not touch rng."""
    with_stats, _ = _run(mode, coll_cost, succ_oh, seed, True)
    without, none_st = _run(mode, coll_cost, succ_oh, seed, False)
    assert with_stats == without
    assert none_st is None


@pytest.mark.parametrize("mode", MODES)
def test_stats_tallies_are_consistent(mode):
    """Outcome counts partition the epochs; native tallies stay in range."""
    _out, st = _run(mode, f25.COLL_RTS_24M, f25.OH_SUCC_24M, 3, True)
    epochs = st["epochs"]
    assert epochs > 0
    outcomes = sum(st.get(k, 0) for k in
                   ("idle", "coll", "solo_vis", "solo_nat"))
    assert outcomes == epochs
    # natives are always viable, so nat_slots == epochs * N_NATIVE
    assert st["nat_slots"] == epochs * 10
    assert 0 <= st["nat_tx"] <= st["nat_slots"]


def test_measured_tau_nat_matches_bianchi():
    """The constant-tau_nat approximation in pace-analysis/drift.py.

    Natives decrement backoff only on idle epochs, so it is not a given that
    the per-epoch rate stays near the saturated-DCF value of 0.0525.
    """
    _out, st = _run("pace", f25.COLL_RTS_24M, f25.OH_SUCC_24M, 11, True,
                    visits=60)
    tau_nat = st["nat_tx"] / st["nat_slots"]
    assert 0.045 < tau_nat < 0.060, tau_nat


def test_epoch_budget_is_small():
    """Theorem 2's binding constraint: ~20 decisions, not 420."""
    _out, st = _run("pace", f25.COLL_RTS_24M, f25.OH_SUCC_24M, 5, True,
                    visits=60)
    assert 10 < st["epochs"] / 60 < 40, st["epochs"] / 60


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", (2, 13))
def test_trace_does_not_perturb(mode, seed):
    """The opt-in per-epoch trace must also stay passive."""
    traced, st = _run(mode, f25.COLL_RTS_24M, f25.OH_SUCC_24M, seed, True,
                      trace=True)
    plain, _ = _run(mode, f25.COLL_RTS_24M, f25.OH_SUCC_24M, seed, False)
    assert traced == plain
    assert len(st["trace"]) == st["epochs"]


def test_trace_rate_and_target_are_sane():
    """BEB enters above the FS target, PACE below it (section 4.4 lemma)."""
    _o, d = _run("dcf_excl", f25.COLL_RTS_24M, f25.OH_SUCC_24M, 4, True,
                 visits=40, trace=True)
    _o, p = _run("pace", f25.COLL_RTS_24M, f25.OH_SUCC_24M, 4, True,
                 visits=40, trace=True)
    for _wr, nvv, k, rate in d["trace"]:
        # nvv == 0 is legitimate: once every visitor has self-excluded the
        # natives still hold the channel, so the epoch is recorded with rate 0
        assert 0 <= nvv <= 20 and k >= nvv
        assert (rate == 0.0) if nvv == 0 else (0.0 < rate <= 2 / 17 + 1e-12)
    # first epoch of a visit: BEB starts at 2/(CW_min+1), PACE at 1/W_eff
    assert abs(d["trace"][0][3] - 2 / 17) < 1e-12
    assert abs(p["trace"][0][3] - 1 / f25.W_REF) < 1e-12
    # and those straddle the fair-share target 1/k
    k0 = d["trace"][0][2]
    assert d["trace"][0][3] > 1 / k0 > p["trace"][0][3]
