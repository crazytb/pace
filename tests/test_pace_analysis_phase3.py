"""Phase 3: model validation.

Pins the two claims the validation figures make, plus the bucket-comparison
logic behind them, which is where the first attempt went wrong twice.
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
import validate as VA           # noqa: E402


def test_rel_err_aligns_on_shared_buckets_not_positions():
    """The simulation reports no tau once every visitor has self-excluded, so
    its bucket list is a subset. Zipping positionally silently compares
    different points in the visit."""
    mod = {"w_rem": np.array([50.0, 70.0, 90.0]),
           "tau": np.array([1.0, 2.0, 3.0]),
           "mass": np.array([0.2, 0.3, 0.5])}
    sim = {"w_rem": np.array([70.0, 90.0]), "tau": np.array([2.0, 3.0])}
    err, wt = VA._rel_err(mod, sim)
    assert len(err) == 2
    assert np.allclose(err, 0.0)            # matched buckets are identical
    assert abs(wt.sum() - 1.0) < 1e-12


def test_rel_err_weights_by_occupancy():
    """A near-empty bucket must not dominate the mean."""
    mod = {"w_rem": np.array([50.0, 70.0]),
           "tau": np.array([10.0, 1.0]),
           "mass": np.array([0.001, 0.999])}
    sim = {"w_rem": np.array([50.0, 70.0]), "tau": np.array([1.0, 1.0])}
    err, wt = VA._rel_err(mod, sim)
    assert err.max() == pytest.approx(9.0)          # the outlier is still there
    assert float((err * wt).sum()) < 0.02           # but it carries no weight


def test_trajectory_buckets_drop_unreachable_ranges():
    """Some W_rem ranges are reachable only by an improbably long idle run;
    their conditional tau is huge and meaningless."""
    traj = dp.tau_trajectory(20, bin_w=20, access="basic")
    assert traj["mass"].min() >= 0.005
    assert abs(traj["mass"].sum() - 1.0) < 0.5      # most mass is retained
    assert len(traj["w_rem"]) > 10


@pytest.mark.parametrize("access", ("rts", "basic"))
def test_analysis_tracks_simulated_trajectory(access):
    mod = dp.tau_trajectory(20, bin_w=VA.BIN_W, access=access)
    sim = VA.sim_trajectory(20, access=access)
    err, wt = VA._rel_err(mod, sim)
    assert float((err * wt).sum()) < 0.20, float((err * wt).sum())
    assert err.max() < 0.60, err.max()


@pytest.mark.parametrize("access", ("rts", "basic"))
def test_tau_climbs_but_never_reaches_the_target(access):
    """Theorems 1 and 2 together, read off the trajectory."""
    import viability as V
    mod = dp.tau_trajectory(20, bin_w=VA.BIN_W, access=access)
    order = np.argsort(-mod["w_rem"])           # earliest epoch first
    tau = mod["tau"][order]
    assert tau[0] < tau[-1], "tau should climb over the visit"
    assert abs(tau[0] - P.TAU_0) / P.TAU_0 < 0.1, "must start at tau_0"
    for w, t in zip(mod["w_rem"], mod["tau"]):
        assert t < V.fs_target(int(w), 20), (w, t)


def test_model_is_insensitive_to_the_one_measured_input():
    """tau_nat is measured, not derived, so the model must not hinge on it."""
    rows = VA.sensitivity()
    vals = [v for _tn, v, _r in rows]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals
    # a 75% swing in tau_nat moves the answer by under 3 points
    assert max(abs(r) for _tn, _v, r in rows) < 0.03, rows
