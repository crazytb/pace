"""
Step 4 검증 테스트 — NPCA-HARQ Action Policy

T1 : Action enum — 필수 값 모두 존재
T2 : Delay estimators — primary_delay = obss_remain + backoff, npca_delay = switch + cw//2
T3 : HARQ_RETX_NPCA 선택 — HARQ 버퍼 유효 + 긴 OBSS → npca_delay < primary_delay
T4 : HARQ_RETX_PRIMARY 선택 — HARQ 버퍼 유효 + 짧은 OBSS → primary_delay < npca_delay
T5 : TX_NEW_NPCA 선택 — 버퍼 없음 + 긴 OBSS → npca_delay < primary_delay
T6 : 항상 primary 선택 — NPCA 조건 불충족(NPCA 채널 busy) → HARQ_RETX_PRIMARY 등
T7 : STA + 짧은 OBSS → policy가 primary 선택 → npca_transitions == 0
T8 : STA + 긴 OBSS  → policy가 NPCA 선택  → npca_transitions > 0
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.enums import Action, NPCA_ACTIONS
from harq_sim.policy import (
    NPCAHARQPolicy,
    estimate_primary_access_delay,
    estimate_npca_access_delay,
)
from harq_sim import phy
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_obss(channel: Channel, slot: int, duration: int) -> None:
    """Manually add an OBSS event and update the channel state."""
    channel.obss_traffic.append((f"test_obss_s{slot}", slot, duration, -1))
    channel.update(slot)


def _make_sta(
    *,
    obss_duration: int,
    switching_delay: int = 1,
    npca_qsrc: int = 0,
    npca_threshold: int = 0,
    npca_enabled: bool = True,
    harq_enabled: bool = True,
    snr_db_mean: float = 10.0,
    policy: NPCAHARQPolicy | None = None,
) -> tuple[STA, Channel, Channel]:
    """Create a minimal STA + channels for unit testing."""
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    # Inject OBSS starting at slot 0
    _inject_obss(primary, slot=0, duration=obss_duration)
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=npca_enabled,
        switching_delay=switching_delay,
        switch_back_delay=1,
        npca_min_duration_threshold=npca_threshold,
        npca_initial_qsrc=npca_qsrc,
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=snr_db_mean,
        snr_db_std=0.0,
        harq_enabled=harq_enabled,
        harq_validity_horizon=200,
        policy=policy,
    )
    return sta, primary, npca


def _activate_harq_buffer(sta: STA, slot: int = 0) -> None:
    """Force the HARQ buffer into active state for the head-of-line packet."""
    pkt = sta._peek_head()
    assert pkt is not None
    snr_linear = phy.snr_db_to_linear(sta.snr_db_mean)
    sta.harq_buffer.store(pkt, snr_linear, slot)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_t1_action_enum_completeness():
    """T1: Action enum contains all required values from guidelines §10."""
    required = {
        "STAY_PRIMARY", "TX_NEW_PRIMARY", "TX_NEW_NPCA",
        "ARQ_RETX_PRIMARY", "ARQ_RETX_NPCA",
        "HARQ_RETX_PRIMARY", "HARQ_RETX_NPCA", "FLUSH_HARQ",
    }
    actual = {a.name for a in Action}
    assert required.issubset(actual), f"Missing: {required - actual}"
    # NPCA_ACTIONS frozenset must cover all NPCA transitions
    npca_names = {a.name for a in NPCA_ACTIONS}
    assert "HARQ_RETX_NPCA" in npca_names
    assert "ARQ_RETX_NPCA"  in npca_names
    assert "TX_NEW_NPCA"    in npca_names
    assert "STAY_PRIMARY"   not in npca_names
    print("  T1 PASS: Action enum complete, NPCA_ACTIONS correct")


def test_t2_delay_estimators():
    """T2: estimate_primary_access_delay and estimate_npca_access_delay."""
    sta, primary, _ = _make_sta(obss_duration=50)

    # primary_delay = obss_remain + backoff_counter
    primary.obss_remain = 40        # override after inject
    sta.primary_backoff_counter = 7
    assert estimate_primary_access_delay(sta) == 40 + 7

    # npca_delay = switching_delay + npca_cw_init // 2
    # npca_qsrc=0 → cw_init = 2^0 × 16 − 1 = 15 → expected_backoff = 7
    sta.switching_delay = 1
    expected = 1 + 15 // 2   # = 8
    assert estimate_npca_access_delay(sta) == expected, (
        f"Expected {expected}, got {estimate_npca_access_delay(sta)}"
    )

    # Larger qsrc → larger expected backoff
    sta2, _, _ = _make_sta(obss_duration=10, npca_qsrc=2)
    # npca_qsrc=2 → cw = 2^2 × 16 − 1 = 63 → expected_backoff = 31
    assert estimate_npca_access_delay(sta2) == 1 + 63 // 2

    print(f"  T2 PASS: primary_delay=47, npca_delay={expected}")


def test_t3_harq_retx_npca_chosen():
    """T3: long OBSS + valid HARQ buffer → policy returns HARQ_RETX_NPCA."""
    sta, primary, npca = _make_sta(obss_duration=50, harq_enabled=True)
    _activate_harq_buffer(sta, slot=0)

    policy = NPCAHARQPolicy()
    # primary_delay = 50 + 0 = 50 >> npca_delay = 1 + 7 = 8
    action = policy.select_action(sta, slot=0)
    assert action == Action.HARQ_RETX_NPCA, f"Expected HARQ_RETX_NPCA, got {action}"
    assert action in NPCA_ACTIONS
    print(f"  T3 PASS: HARQ buffer active, long OBSS → {action.value}")


def test_t4_harq_retx_primary_chosen():
    """T4: short OBSS + valid HARQ buffer → policy returns HARQ_RETX_PRIMARY."""
    # switching_delay=1, npca_qsrc=0 → npca_delay = 1 + 7 = 8
    # OBSS duration = 5 → primary_delay = 5 + 0 = 5  <  8
    sta, primary, npca = _make_sta(
        obss_duration=5, harq_enabled=True, npca_threshold=0,
    )
    _activate_harq_buffer(sta, slot=0)
    sta.primary_backoff_counter = 0   # no additional wait after OBSS

    policy = NPCAHARQPolicy()
    action = policy.select_action(sta, slot=0)
    assert action == Action.HARQ_RETX_PRIMARY, f"Expected HARQ_RETX_PRIMARY, got {action}"
    assert action not in NPCA_ACTIONS
    print(f"  T4 PASS: HARQ buffer active, short OBSS → {action.value}")


def test_t5_tx_new_npca_chosen():
    """T5: no HARQ buffer + long OBSS + fresh packet → TX_NEW_NPCA."""
    sta, primary, npca = _make_sta(obss_duration=50, harq_enabled=False)
    # No buffer, fresh packet (retry_count == 0)
    policy = NPCAHARQPolicy()
    action = policy.select_action(sta, slot=0)
    assert action == Action.TX_NEW_NPCA, f"Expected TX_NEW_NPCA, got {action}"
    assert action in NPCA_ACTIONS
    print(f"  T5 PASS: no HARQ, long OBSS, fresh packet → {action.value}")


def test_t6_stays_primary_npca_unavailable():
    """T6: NPCA channel busy → can_transition_to_npca() False → stay on primary."""
    sta, primary, npca = _make_sta(obss_duration=50, harq_enabled=True)
    _activate_harq_buffer(sta, slot=0)

    # Make NPCA channel busy (blocks NPCA transition via overlaps_with_obss_ppdu is always False,
    # but we can make NPCA obss_remain > 0 to mimic it being unusable)
    # Actually the real check is via can_transition_to_npca() — let's disable npca directly.
    sta.npca_enabled = False   # simplest way to block NPCA transition

    policy = NPCAHARQPolicy()
    action = policy.select_action(sta, slot=0)
    assert action not in NPCA_ACTIONS, f"Expected non-NPCA action, got {action}"
    assert action == Action.HARQ_RETX_PRIMARY
    print(f"  T6 PASS: NPCA unavailable → {action.value}")


def test_t7_sta_stays_primary_short_obss():
    """T7: STA + policy + short OBSS → stays frozen, npca_transitions == 0."""
    # npca_delay = 1 (switching) + 7 (backoff mean) = 8
    # OBSS = 6 slots → primary_delay = 6 + backoff_counter
    # If backoff_counter is small (e.g. 0), primary_delay = 6 < 8 → STAY PRIMARY
    random.seed(99)
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)

    policy = NPCAHARQPolicy()
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=True,
        switching_delay=1,
        switch_back_delay=1,
        npca_min_duration_threshold=0,
        npca_initial_qsrc=0,   # cw=15, expected_backoff=7 → npca_delay=8
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=25.0,
        snr_db_std=0.0,
        harq_enabled=False,
        policy=policy,
    )
    # Force backoff_counter = 0 so primary_delay = obss_remain only
    sta.primary_backoff_counter = 0

    sim = Simulator(num_slots=30, stas=[sta], channels=[primary, npca], enable_trace=True)

    # Inject a short OBSS (6 slots) before the simulation starts
    _inject_obss(primary, slot=0, duration=6)

    sim.run()
    metrics = sim.compute_metrics()

    transitions = metrics[0]["npca_transitions"]
    pol_npca    = metrics[0]["policy_npca_chosen"]
    pol_primary = metrics[0]["policy_primary_chosen"]

    # Short OBSS → npca_delay(8) > primary_delay(≤6) → policy keeps STA on primary
    assert transitions == 0, f"Expected 0 NPCA transitions, got {transitions}"
    assert pol_npca == 0,    f"Expected 0 policy NPCA chosen, got {pol_npca}"
    assert pol_primary >= 1, f"Expected ≥1 policy primary chosen, got {pol_primary}"
    print(f"  T7 PASS: short OBSS (6 slots), npca_transitions={transitions}, "
          f"pol_primary={pol_primary}")


def test_t8_sta_transitions_long_obss():
    """T8: STA + policy + long OBSS → transitions to NPCA, npca_transitions > 0."""
    random.seed(99)
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)

    policy = NPCAHARQPolicy()
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=True,
        switching_delay=1,
        switch_back_delay=1,
        npca_min_duration_threshold=0,
        npca_initial_qsrc=0,   # cw=15 → npca_delay=8
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=25.0,
        snr_db_std=0.0,
        harq_enabled=False,
        policy=policy,
    )
    sta.primary_backoff_counter = 0   # primary_delay = obss_remain only

    sim = Simulator(num_slots=100, stas=[sta], channels=[primary, npca], enable_trace=True)

    # Inject a long OBSS (60 slots) → primary_delay=60 >> npca_delay=8 → go NPCA
    _inject_obss(primary, slot=0, duration=60)

    sim.run()
    metrics = sim.compute_metrics()

    transitions = metrics[0]["npca_transitions"]
    pol_npca    = metrics[0]["policy_npca_chosen"]

    assert transitions > 0, f"Expected NPCA transitions, got 0"
    assert pol_npca > 0,    f"Expected policy_npca_chosen > 0, got {pol_npca}"
    print(f"  T8 PASS: long OBSS (60 slots), npca_transitions={transitions}, "
          f"pol_npca={pol_npca}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_t1_action_enum_completeness,
    test_t2_delay_estimators,
    test_t3_harq_retx_npca_chosen,
    test_t4_harq_retx_primary_chosen,
    test_t5_tx_new_npca_chosen,
    test_t6_stays_primary_npca_unavailable,
    test_t7_sta_stays_primary_short_obss,
    test_t8_sta_transitions_long_obss,
]


def main():
    print("\n=== Step 4 NPCA-HARQ Policy Verification Tests ===")
    passed = failed = 0
    for t in TESTS:
        label = t.__name__.replace("test_", "").upper().replace("_", " ", 1)
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {label} FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"Results: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
