"""Single source of truth for the analysis parameters.

Every constant here is DERIVED from the simulator modules rather than copied
from the manuscript's Table 2. Copying would create a second source of truth
that silently drifts the moment someone retunes the engine, which is exactly
how the L_hs = 88 slot error got into PACE_TWC_ANALYSIS.md in the first place
(88 us is 10 slots).

This module also owns the sys.path wiring, so the analysis modules import the
engine through `params.engine()` instead of repeating the hack.

Invariants the analysis depends on are asserted at import time. If the engine
is retuned in a way that breaks one, every analysis module fails loudly on
import rather than quietly reporting wrong numbers.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HARQ = os.path.join(_ROOT, "harq_sim")
if _HARQ not in sys.path:
    sys.path.insert(0, _HARQ)

import run_step9_fig17 as _f17        # noqa: E402  MIMD factors, CW_MAX
import run_step9_fig24 as _f24        # noqa: E402  standard CW_min
import run_step9_fig25 as _f25        # noqa: E402  visit engine, timings

# ── Window and populations ───────────────────────────────────────────────────
W_EFF = _f25.W_REF                          # 420 slots = 3.78 ms
N_NATIVE = _f25.N_NATIVE                    # 10
N_VISITOR = _f25.N_VISITOR                  # 10 (engine default; swept)

# ── Frame and overhead durations, in SLOTS (sigma = 9 us) ────────────────────
# The engine stores these already converted via _us2slot = ceil(us / 9).
PPDU_V_LO = _f25.PPDU_V_LO                  # 25 slots (225 us)
PPDU_V_HI = _f25.PPDU_V_HI                  # 100 slots (900 us)
PPDU_NATIVE = _f25.PPDU_NATIVE_SLOTS        # 50 slots (450 us)
L_HS = _f25.OH_SUCC_24M                     # 10 slots (88 us), RTS/CTS success
L_COL = _f25.COLL_RTS_24M                   # 12 slots (106 us), RTS collision
SLOT_US = 9

# ── PACE MIMD adaptation ─────────────────────────────────────────────────────
C_MIMD = _f17.PND_C_COLL                    # 1.2, both directions
TAU_0 = 1.0 / W_EFF                         # deadline-scaled initialisation
TAU_FLOOR = 1e-4                            # run_step9_fig25.py:303 (unnamed)

# ── Standard BEB baseline ────────────────────────────────────────────────────
CW_MIN = _f24.DCF_CW_MIN_STD                # 16
CW_MAX = _f17.DCF_CW_MAX                    # 1023
BEB_ENTRY = 2.0 / (CW_MIN + 1.0)            # 0.1176, population-blind

# ── Measured, not derivable ──────────────────────────────────────────────────
# Native attempt probability PER CONTENTION EPOCH. Measured 0.049-0.056 by
# measure_engine.py, matching the saturated-DCF Bianchi value for 10 stations.
# drift.py treats it as exogenous; the self-consistent coupling with the
# natives' frozen-backoff dynamics is future work.
TAU_NAT = 0.052

# ── Access modes: (collision cost, success overhead) for the engine ──────────
ACCESS = {
    "rts": (L_COL, L_HS),                   # mandatory RTS/CTS at 24 Mbps
    "basic": ("nocd", 0),                   # collision costs max L_i
}

# ── Measurement sampling (shared by the measuring modules) ───────────────────
VISITS = 60                                 # per repetition; second half = steady state
REPS = 12
MAX_EPOCH = 24                              # trace horizon; budget is 17-24


def engine():
    """The visit engine module (run_step9_fig25)."""
    return _f25


def _check() -> None:
    # The log-domain lattice argument (section 3.1) needs a single ratio: with
    # c_idle != c_coll the reachable set becomes a 2-D lattice and the drift
    # equation, Theorem 1 and Theorem 2 all lose their meaning.
    assert _f17.PND_C_COLL == _f17.PND_C_IDLE, (
        f"c_coll={_f17.PND_C_COLL} != c_idle={_f17.PND_C_IDLE}: the 1-D "
        "lattice argument in PACE_TWC_ANALYSIS.md section 3.1 no longer holds")
    assert C_MIMD > 1.0

    # Unit sanity: the overheads must be slot counts, not microseconds. This is
    # the exact error the manuscript plan shipped with.
    assert L_HS == -(-88 // SLOT_US) == 10, L_HS
    assert L_COL == -(-106 // SLOT_US) == 12, L_COL
    assert PPDU_V_LO < PPDU_V_HI <= W_EFF

    # A visitor needs L_min + L_HS slots to start, so the tail of the window is
    # structurally dead (section 5.2).
    assert 0 < min_start() < W_EFF
    assert 0.05 < dead_fraction() < 0.15, dead_fraction()

    # BEB enters above the fair-share target for every population of interest.
    assert BEB_ENTRY * (N_NATIVE + 10) > 1.0


def min_start() -> int:
    """Smallest W_rem at which a visitor can still begin an exchange."""
    return PPDU_V_LO + L_HS


def dead_fraction() -> float:
    """Fraction of the window in which no visitor can start (section 5.2)."""
    return min_start() / W_EFF


_check()


if __name__ == "__main__":
    print(f"W_eff            {W_EFF} slots ({W_EFF * SLOT_US / 1000:.2f} ms)")
    print(f"visitor PPDU     U[{PPDU_V_LO}, {PPDU_V_HI}] slots")
    print(f"native PPDU      {PPDU_NATIVE} slots")
    print(f"L_hs / L_col     {L_HS} / {L_COL} slots "
          f"(= ceil(88/{SLOT_US}) / ceil(106/{SLOT_US}))")
    print(f"c_mimd           {C_MIMD}  (c_idle == c_coll, lattice holds)")
    print(f"tau_0            {TAU_0:.5f} = 1/W_eff")
    print(f"tau_floor        {TAU_FLOOR}")
    print(f"CW_min / CW_max  {CW_MIN} / {CW_MAX}")
    print(f"BEB entry rate   {BEB_ENTRY:.4f} = 2/(CW_min+1)")
    print(f"tau_nat          {TAU_NAT} (measured, per epoch)")
    print(f"min start W_rem  {min_start()} slots")
    print(f"dead tail        {dead_fraction() * 100:.1f}% of W_eff")
    print("\nparams.py invariants: OK")
