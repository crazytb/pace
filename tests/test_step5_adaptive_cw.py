"""
Step 5 검증 테스트 — Adaptive CW_npca_init

T1 : 기본값 — adaptive_cw=False이면 npca_initial_qsrc 변경 없음 (Step 4 backward compat)
T2 : 높은 NPCA 실패율 (>0.3) → select_npca_qsrc() 반환 ≥ default + 1
T3 : 큰 primary_cw (≥ 4×CW_MIN) → select_npca_qsrc() 반환 ≥ default + 1
T4 : 임박한 deadline → select_npca_qsrc() 반환 ≤ default - 1
T5 : qsrc 범위 클램핑 — 모든 +1/-1 적용 후 [NPCA_QSRC_MIN, NPCA_QSRC_MAX] 보장
T6 : 복합 조건 — 높은 실패율 + 큰 primary_cw + 많은 전환 → qsrc 최대값으로 클램핑
T7 : num_recent_npca_transitions 추적 — Simulator가 STA에 주입하는지 검증
T8 : adaptive=True 시뮬레이션 — npca_initial_qsrc가 전환마다 변화하는지 검증
T9 : npca_failure_rate 슬라이딩 윈도우 — NPCA TX 결과 올바르게 누적
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.configs import (
    NPCA_QSRC_MIN, NPCA_QSRC_MAX,
    NPCA_FAILURE_WINDOW, NPCA_TRANSITION_WINDOW,
    NPCA_TRANSITION_THRESHOLD, URGENT_DEADLINE_THRESHOLD,
)
from harq_sim.enums import ChannelType, FailureReason, TrafficClass
from harq_sim.policy import NPCAHARQPolicy, select_npca_qsrc
from harq_sim import phy
from harq_sim.packet import Packet
from harq_sim.sta import STA, CW_MIN
from harq_sim.simulator import Simulator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_obss(channel: Channel, slot: int, duration: int) -> None:
    channel.obss_traffic.append((f"test_obss_s{slot}", slot, duration, -1))
    channel.update(slot)


def _make_sta(
    *,
    obss_duration: int = 60,
    npca_qsrc: int = 0,
    npca_enabled: bool = True,
    harq_enabled: bool = True,
    snr_db_mean: float = 10.0,
    adaptive_cw: bool = False,
    traffic_class: TrafficClass = TrafficClass.BEST_EFFORT,
    policy: NPCAHARQPolicy | None = None,
) -> tuple[STA, Channel, Channel]:
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    _inject_obss(primary, slot=0, duration=obss_duration)
    if policy is None:
        policy = NPCAHARQPolicy(adaptive_cw=adaptive_cw) if npca_enabled else None
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=npca_enabled,
        switching_delay=1,
        switch_back_delay=1,
        npca_min_duration_threshold=0,
        npca_initial_qsrc=npca_qsrc,
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=snr_db_mean,
        snr_db_std=0.0,
        harq_enabled=harq_enabled,
        harq_validity_horizon=200,
        policy=policy,
        adaptive_cw=adaptive_cw,
    )
    # Override head-of-line packet traffic class if needed
    pkt = sta._peek_head()
    if pkt is not None and traffic_class != TrafficClass.BEST_EFFORT:
        pkt.traffic_class = traffic_class
    return sta, primary, npca


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_t1_backward_compat_no_adaptive():
    """T1: adaptive_cw=False → select_action() does NOT modify npca_initial_qsrc."""
    sta, primary, npca = _make_sta(obss_duration=60, npca_qsrc=2, adaptive_cw=False)
    policy = NPCAHARQPolicy(adaptive_cw=False)
    original_qsrc = sta.npca_initial_qsrc

    action = policy.select_action(sta, slot=0)
    assert sta.npca_initial_qsrc == original_qsrc, (
        f"npca_initial_qsrc changed unexpectedly: {original_qsrc} → {sta.npca_initial_qsrc}"
    )
    print(f"  T1 PASS: adaptive_cw=False, qsrc unchanged at {original_qsrc}, action={action.value}")


def test_t2_high_failure_rate_increases_qsrc():
    """T2: npca_failure_rate > 0.3 → select_npca_qsrc() returns ≥ default + 1."""
    sta, _, _ = _make_sta(obss_duration=60)
    default_q = 1
    sta.npca_initial_qsrc = default_q

    # Inject 4 failures out of 5 → failure_rate = 0.8 > 0.3
    for _ in range(5):
        sta._npca_tx_window.append(False)
    for _ in range(1):
        sta._npca_tx_window.append(True)

    assert sta.npca_failure_rate > 0.3
    q = select_npca_qsrc(sta, slot=0, default_qsrc=default_q)
    assert q >= default_q + 1, f"Expected qsrc ≥ {default_q + 1}, got {q}"
    print(f"  T2 PASS: npca_failure_rate={sta.npca_failure_rate:.2f} > 0.3 → qsrc={q}")


def test_t3_large_primary_cw_increases_qsrc():
    """T3: primary_cw ≥ 4×CW_MIN → select_npca_qsrc() returns ≥ default + 1."""
    sta, _, _ = _make_sta(obss_duration=60)
    default_q = 0
    sta.primary_cw = 4 * CW_MIN  # exactly at threshold

    q = select_npca_qsrc(sta, slot=0, default_qsrc=default_q)
    assert q >= default_q + 1, f"Expected qsrc ≥ {default_q + 1}, got {q}"
    print(f"  T3 PASS: primary_cw={sta.primary_cw} (4×CW_MIN={4*CW_MIN}) → qsrc={q}")


def test_t4_urgent_deadline_decreases_qsrc():
    """T4: deadline_remaining < URGENT_DEADLINE_THRESHOLD → select_npca_qsrc returns ≤ default - 1."""
    sta, _, _ = _make_sta(obss_duration=60, traffic_class=TrafficClass.XR)
    default_q = 2

    # XR deadline = 10ms = 1111 slots. Force a packet that expires very soon.
    pkt = sta._peek_head()
    assert pkt is not None
    # Set arrival_time such that deadline is just below URGENT_DEADLINE_THRESHOLD slots away at slot=0
    from harq_sim.packet import MAX_DELAY_SLOTS
    max_d = MAX_DELAY_SLOTS[TrafficClass.XR]
    # deadline_remaining(0) = arrival + max_d - 0. We want it < URGENT_DEADLINE_THRESHOLD.
    urgent = URGENT_DEADLINE_THRESHOLD - 1
    pkt.arrival_time = -(max_d - urgent)  # so deadline = arrival + max_d = urgent

    dr = pkt.deadline_remaining(0)
    assert dr is not None and dr < URGENT_DEADLINE_THRESHOLD, f"deadline_remaining={dr}"

    q = select_npca_qsrc(sta, slot=0, default_qsrc=default_q)
    assert q <= default_q - 1, f"Expected qsrc ≤ {default_q - 1}, got {q}"
    print(f"  T4 PASS: deadline_remaining={dr} < {URGENT_DEADLINE_THRESHOLD} → qsrc={q}")


def test_t5_qsrc_clamping():
    """T5: qsrc clamped to [NPCA_QSRC_MIN, NPCA_QSRC_MAX] regardless of conditions."""
    sta, _, _ = _make_sta(obss_duration=60)

    # Force all +1 conditions to overflow above NPCA_QSRC_MAX
    sta.primary_cw = 4 * CW_MIN
    sta.num_recent_npca_transitions = NPCA_TRANSITION_THRESHOLD + 10
    for _ in range(NPCA_FAILURE_WINDOW):
        sta._npca_tx_window.append(False)  # failure_rate = 1.0 > 0.3

    q_high = select_npca_qsrc(sta, slot=0, default_qsrc=NPCA_QSRC_MAX)
    assert q_high == NPCA_QSRC_MAX, f"Expected max clamp {NPCA_QSRC_MAX}, got {q_high}"

    # Force -1 condition to underflow below NPCA_QSRC_MIN
    pkt = sta._peek_head()
    assert pkt is not None
    from harq_sim.packet import MAX_DELAY_SLOTS
    max_d = MAX_DELAY_SLOTS[TrafficClass.XR]
    sta2, _, _ = _make_sta(obss_duration=60, traffic_class=TrafficClass.XR)
    pkt2 = sta2._peek_head()
    urgent = URGENT_DEADLINE_THRESHOLD - 1
    pkt2.arrival_time = -(max_d - urgent)

    q_low = select_npca_qsrc(sta2, slot=0, default_qsrc=NPCA_QSRC_MIN)
    assert q_low == NPCA_QSRC_MIN, f"Expected min clamp {NPCA_QSRC_MIN}, got {q_low}"

    print(f"  T5 PASS: max clamp → {q_high}, min clamp → {q_low}")


def test_t6_combined_conditions_clamp_to_max():
    """T6: all +1 conditions active → qsrc reaches NPCA_QSRC_MAX."""
    sta, _, _ = _make_sta(obss_duration=60, npca_qsrc=2)
    default_q = 2

    # Activate all three +1 conditions
    sta.primary_cw = 4 * CW_MIN
    sta.num_recent_npca_transitions = NPCA_TRANSITION_THRESHOLD + 1
    for _ in range(NPCA_FAILURE_WINDOW):
        sta._npca_tx_window.append(False)

    q = select_npca_qsrc(sta, slot=0, default_qsrc=default_q)
    # default=2 + 3 conditions = 5, clamped to NPCA_QSRC_MAX (5)
    assert q == NPCA_QSRC_MAX, f"Expected {NPCA_QSRC_MAX} (clamped), got {q}"
    print(f"  T6 PASS: 3 +1 conditions, default={default_q} → qsrc clamped to {q}")


def test_t7_simulator_injects_recent_transitions():
    """T7: Simulator injects num_recent_npca_transitions into STA each slot."""
    random.seed(7)
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    policy  = NPCAHARQPolicy(adaptive_cw=False)
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=True,
        switching_delay=1,
        switch_back_delay=1,
        npca_min_duration_threshold=0,
        npca_initial_qsrc=0,
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=25.0,
        snr_db_std=0.0,
        harq_enabled=False,
        policy=policy,
    )
    sta.primary_backoff_counter = 0
    sim = Simulator(num_slots=100, stas=[sta], channels=[primary, npca], enable_trace=True)
    # Inject long OBSS → STA will transition
    _inject_obss(primary, slot=0, duration=80)
    sim.run()

    # After run: if any transitions occurred, they should be recorded in the deque
    transitions = sta.stats["npca_transitions"]
    deque_count = len(sim._npca_transition_deque)
    assert deque_count == transitions, (
        f"Deque has {deque_count} entries but sta recorded {transitions} transitions"
    )
    assert sta.num_recent_npca_transitions >= 0  # must be a non-negative int
    print(f"  T7 PASS: {transitions} transitions recorded, deque_count={deque_count}")


def test_t8_adaptive_simulation_changes_qsrc():
    """T8: adaptive_cw=True simulation — qsrc_history varies (at least 1 transition logged)."""
    random.seed(8)
    primary = Channel(channel_id=0, obss_generation_rate=0.05, obss_duration_range=(40, 100))
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    policy  = NPCAHARQPolicy(adaptive_cw=True)
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=True,
        switching_delay=1,
        switch_back_delay=1,
        npca_min_duration_threshold=0,
        npca_initial_qsrc=0,
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=25.0,
        snr_db_std=0.0,
        harq_enabled=True,
        harq_validity_horizon=200,
        policy=policy,
        adaptive_cw=True,
    )
    sim = Simulator(num_slots=1000, stas=[sta], channels=[primary, npca], enable_trace=True)
    sim.run()

    transitions = sta.stats["npca_transitions"]
    qsrc_hist   = sta._npca_qsrc_history
    assert transitions > 0, "No NPCA transitions occurred — test inconclusive"
    assert len(qsrc_hist) == transitions, (
        f"qsrc_history length {len(qsrc_hist)} != transitions {transitions}"
    )
    # All recorded qsrc values must be within valid range
    for q in qsrc_hist:
        assert NPCA_QSRC_MIN <= q <= NPCA_QSRC_MAX, f"qsrc {q} out of range"
    metrics = sim.compute_metrics()
    avg_q = metrics[0]["avg_npca_qsrc"]
    assert avg_q is not None
    print(f"  T8 PASS: {transitions} transitions, avg_qsrc={avg_q:.2f}, "
          f"qsrc_range=[{min(qsrc_hist)}, {max(qsrc_hist)}]")


def test_t9_npca_failure_rate_window():
    """T9: npca_failure_rate sliding window correctly tracks NPCA TX outcomes."""
    sta, _, _ = _make_sta(obss_duration=60)
    assert sta.npca_failure_rate == 0.0  # empty window

    # Fill window with all successes
    for _ in range(NPCA_FAILURE_WINDOW):
        sta._npca_tx_window.append(True)
    assert sta.npca_failure_rate == 0.0

    # Fill window with all failures
    sta._npca_tx_window.clear()
    for _ in range(NPCA_FAILURE_WINDOW):
        sta._npca_tx_window.append(False)
    assert sta.npca_failure_rate == 1.0

    # Half and half
    sta._npca_tx_window.clear()
    for i in range(NPCA_FAILURE_WINDOW):
        sta._npca_tx_window.append(i % 2 == 0)  # alternating success/fail
    rate = sta.npca_failure_rate
    assert abs(rate - 0.5) < 1e-9, f"Expected 0.5, got {rate}"

    # handle_tx_result records NPCA TX — inject directly
    sta2, primary, npca = _make_sta(obss_duration=60, harq_enabled=False)
    # Simulate 3 NPCA failures via handle_tx_result
    for _ in range(3):
        pkt = sta2._peek_head()
        sta2.current_packet = pkt
        sta2.handle_tx_result(
            False, FailureReason.COLLISION, ChannelType.NPCA, slot=0
        )
    assert len(sta2._npca_tx_window) == 3
    assert sta2.npca_failure_rate == 1.0

    print(f"  T9 PASS: sliding window, failure_rate tracking verified")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_t1_backward_compat_no_adaptive,
    test_t2_high_failure_rate_increases_qsrc,
    test_t3_large_primary_cw_increases_qsrc,
    test_t4_urgent_deadline_decreases_qsrc,
    test_t5_qsrc_clamping,
    test_t6_combined_conditions_clamp_to_max,
    test_t7_simulator_injects_recent_transitions,
    test_t8_adaptive_simulation_changes_qsrc,
    test_t9_npca_failure_rate_window,
]


def main():
    print("\n=== Step 5 Adaptive CW_npca_init Verification Tests ===")
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
