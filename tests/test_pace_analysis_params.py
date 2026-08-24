"""params.py must stay derived from the engine, never a second source of truth.

The analysis quotes numbers in the manuscript. If someone retunes the simulator
and params.py keeps stale copies, the paper silently ships wrong values. These
tests fail loudly instead.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "pace-analysis"), os.path.join(_ROOT, "harq_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params as P              # noqa: E402
import run_step9_fig17 as f17   # noqa: E402
import run_step9_fig24 as f24   # noqa: E402
import run_step9_fig25 as f25   # noqa: E402


@pytest.mark.parametrize("name,engine_value", [
    ("W_EFF", f25.W_REF),
    ("N_NATIVE", f25.N_NATIVE),
    ("N_VISITOR", f25.N_VISITOR),
    ("PPDU_V_LO", f25.PPDU_V_LO),
    ("PPDU_V_HI", f25.PPDU_V_HI),
    ("PPDU_NATIVE", f25.PPDU_NATIVE_SLOTS),
    ("L_HS", f25.OH_SUCC_24M),
    ("L_COL", f25.COLL_RTS_24M),
    ("C_MIMD", f17.PND_C_COLL),
    ("CW_MIN", f24.DCF_CW_MIN_STD),
    ("CW_MAX", f17.DCF_CW_MAX),
])
def test_param_tracks_engine(name, engine_value):
    assert getattr(P, name) == engine_value


def test_lattice_invariant():
    """c_idle == c_coll, or the 1-D lattice argument collapses."""
    assert f17.PND_C_COLL == f17.PND_C_IDLE == P.C_MIMD


def test_overheads_are_slots_not_microseconds():
    """The unit error that shipped in the first draft of the analysis plan."""
    assert P.L_HS == 10 and P.L_COL == 12
    assert P.L_HS < P.PPDU_V_LO, "handshake must be shorter than the shortest frame"


def test_derived_quantities():
    assert P.TAU_0 == 1.0 / f25.W_REF
    assert P.BEB_ENTRY == 2.0 / (f24.DCF_CW_MIN_STD + 1.0)
    assert P.min_start() == f25.PPDU_V_LO + f25.OH_SUCC_24M == 35
    assert abs(P.dead_fraction() - 35 / 420) < 1e-12


def test_access_presets_drive_the_engine():
    """ACCESS entries must be accepted by _run_visit25 unchanged."""
    import numpy as np
    for mode, (coll_cost, succ_oh) in P.ACCESS.items():
        rng = np.random.default_rng(1)
        ppdus = f25._sample_ppdus25(np.random.default_rng(2))
        air, _c, _i, _o, _carry = f25._run_visit25(
            ppdus, rng, "pace", np.full(f25.N_VISITOR, P.TAU_0),
            coll_cost, succ_oh)
        assert air.sum() > 0, mode


def test_measured_tau_nat_is_documented_value():
    """TAU_NAT is measured, not derived, so pin it against measure_engine."""
    import measure_engine
    m = measure_engine.measure(20, P.N_NATIVE, *P.ACCESS["rts"])
    assert abs(m["tau_nat"] - P.TAU_NAT) < 0.006, (m["tau_nat"], P.TAU_NAT)
