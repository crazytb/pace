"""
Step 1 검증 테스트: HARQ 없이 NPCA 동작 확인

검증 항목:
  T1. Primary state save/restore — NPCA 전환 시 primary CW/backoff 저장, 복귀 후 복원
  T2. NPCA EDCA state 분리 — NPCA 실패 시 npca_cw 증가, primary_cw 불변
  T3. NPCA_TIMER / switch-back — OBSS 종료 후 primary로 복귀
  T4. NPCA 전환 조건 (D1.2 §37.18.3 Condition 1) — threshold 미달 시 전환 안 함
  T5. AP absence 실패 (guidelines §7)

실행:
  python -m pytest tests/test_step1_npca.py -v
  또는
  python tests/test_step1_npca.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
random.seed(0)

from harq_sim.channel import Channel
from harq_sim.enums import ChannelType, FailureReason, STAMode
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_channels(primary_obss_rate=0.1, obss_duration=(50, 50)):
    primary = Channel(channel_id=0, obss_generation_rate=primary_obss_rate,
                      obss_duration_range=obss_duration)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    return primary, npca


def make_sta(primary, npca, npca_enabled=True, qsrc=0, min_threshold=0,
             switching_delay=1, switch_back_delay=1):
    return STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=npca_enabled,
        ppdu_duration=20,
        switching_delay=switching_delay,
        switch_back_delay=switch_back_delay,
        npca_min_duration_threshold=min_threshold,
        npca_initial_qsrc=qsrc,
        retry_limit=7,
        infinite_queue=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# T1: Primary state save/restore
# ─────────────────────────────────────────────────────────────────────────────
def test_primary_state_save_restore():
    """
    NPCA 전환 직전의 primary CW/backoff/stage/retry가
    switch-back 후 정확히 복원되어야 한다.
    """
    primary, npca = make_channels(primary_obss_rate=0.0, obss_duration=(100, 100))
    sta = make_sta(primary, npca)

    # Manually drive primary CW to a non-default state
    sta.primary_cw              = 63
    sta.primary_backoff_counter = 17
    sta.primary_backoff_stage   = 2
    sta.primary_retry_counter   = 3

    snapshot_cw      = sta.primary_cw
    snapshot_backoff = sta.primary_backoff_counter
    snapshot_stage   = sta.primary_backoff_stage
    snapshot_retry   = sta.primary_retry_counter

    # Inject OBSS so transition condition is met
    primary.obss_traffic.append(("obss_manual", 0, 100, -1))
    primary.update(0)

    # Force transition
    assert sta.can_transition_to_npca(0), "NPCA transition should be possible"
    sta._start_npca_transition(0)

    # Verify saved state
    assert sta.saved_primary_state is not None
    assert sta.saved_primary_state["cw"]              == snapshot_cw
    assert sta.saved_primary_state["backoff_counter"] == snapshot_backoff
    assert sta.saved_primary_state["backoff_stage"]   == snapshot_stage
    assert sta.saved_primary_state["retry_counter"]   == snapshot_retry

    # Verify NPCA state was initialized (fresh CW, not inheriting primary CW)
    expected_npca_cw = 2 ** sta.npca_initial_qsrc * (15 + 1) - 1  # qsrc=0 → 15
    assert sta.npca_cw == expected_npca_cw, \
        f"NPCA CW should be {expected_npca_cw}, got {sta.npca_cw}"

    # Simulate switch-back restore
    sta._restore_primary_state()
    assert sta.primary_cw              == snapshot_cw,      "primary_cw not restored"
    assert sta.primary_backoff_counter == snapshot_backoff, "backoff_counter not restored"
    assert sta.primary_backoff_stage   == snapshot_stage,   "backoff_stage not restored"
    assert sta.primary_retry_counter   == snapshot_retry,   "retry_counter not restored"
    assert sta.saved_primary_state is None, "saved state should be cleared after restore"

    print("  T1 PASS: primary state save/restore correct")


# ─────────────────────────────────────────────────────────────────────────────
# T2: NPCA/Primary CW independence
# ─────────────────────────────────────────────────────────────────────────────
def test_npca_primary_cw_independence():
    """
    NPCA 채널에서 실패해도 primary_cw는 변하지 않아야 한다.
    primary 채널에서 실패해도 npca_cw는 변하지 않아야 한다.
    """
    primary, npca = make_channels()
    sta = make_sta(primary, npca, qsrc=0)

    # Set known initial states
    sta.primary_cw = 15
    sta.npca_cw    = 15

    # NPCA failure → only npca_cw should increase
    before_primary_cw = sta.primary_cw
    sta.increase_backoff_after_failure(ChannelType.NPCA)
    assert sta.primary_cw == before_primary_cw, \
        f"primary_cw should be unchanged after NPCA failure: {sta.primary_cw} != {before_primary_cw}"
    assert sta.npca_cw == 31, f"npca_cw should double to 31, got {sta.npca_cw}"

    # Primary failure → only primary_cw should increase
    sta.primary_cw = 15
    sta.npca_cw    = 15
    before_npca_cw = sta.npca_cw
    sta.increase_backoff_after_failure(ChannelType.PRIMARY)
    assert sta.npca_cw == before_npca_cw, \
        f"npca_cw should be unchanged after primary failure: {sta.npca_cw} != {before_npca_cw}"
    assert sta.primary_cw == 31, f"primary_cw should double to 31, got {sta.primary_cw}"

    print("  T2 PASS: primary/NPCA CW are independent")


# ─────────────────────────────────────────────────────────────────────────────
# T3: NPCA_TIMER / switch-back (simulation loop)
# ─────────────────────────────────────────────────────────────────────────────
def test_switch_back_after_obss_ends():
    """
    OBSS가 끝나면 STA가 primary로 복귀(SWITCH_BACK → PRIMARY_BACKOFF)해야 한다.
    복귀 후 primary state가 복원되어야 한다.
    """
    random.seed(1)
    primary, npca = make_channels(primary_obss_rate=0.0, obss_duration=(30, 30))
    sta = make_sta(primary, npca, switching_delay=1, switch_back_delay=1)

    # Set non-default primary state so we can verify restoration
    sta.primary_cw              = 63
    sta.primary_backoff_counter = 5
    sta.primary_backoff_stage   = 2
    sta.primary_retry_counter   = 1

    # Inject single OBSS burst on primary
    primary.obss_traffic.append(("obss_test", 0, 30, -1))
    primary.update(0)

    sim = Simulator(num_slots=80, stas=[sta], channels=[primary, npca], enable_trace=True)
    sim.run()

    # After simulation, STA should be back on primary
    assert sta.mode in (STAMode.PRIMARY_BACKOFF, STAMode.PRIMARY_FROZEN, STAMode.PRIMARY_TX), \
        f"STA should be on primary channel after OBSS ends, got {sta.mode}"
    assert sta.saved_primary_state is None, \
        "saved_primary_state should be cleared after switch-back"
    assert sta.stats["npca_transitions"] > 0, "Should have transitioned to NPCA at least once"
    assert sta.stats["switch_backs"] > 0, "Should have switched back at least once"

    print(f"  T3 PASS: switch-back works — transitions={sta.stats['npca_transitions']}, "
          f"switch_backs={sta.stats['switch_backs']}, final_mode={sta.mode.name}")


# ─────────────────────────────────────────────────────────────────────────────
# T4: NPCA transition condition — minimum duration threshold (D1.2 §37.18.3.1.c.i)
# ─────────────────────────────────────────────────────────────────────────────
def test_npca_min_duration_threshold():
    """
    OBSS 잔여시간 < min_duration_threshold 이면 NPCA 전환 안 함.
    OBSS 잔여시간 >= threshold 이면 전환 가능.
    """
    primary, npca = make_channels(primary_obss_rate=0.0)
    threshold = 40

    sta = make_sta(primary, npca, min_threshold=threshold)

    # Short OBSS (duration < threshold) → no transition
    primary.obss_traffic = [("short_obss", 0, 20, -1)]
    primary.update(0)
    assert not sta.can_transition_to_npca(0), \
        "Should NOT transition: obss_remain(20) < threshold(40)"

    # Long OBSS (duration >= threshold) → transition allowed
    primary.obss_traffic = [("long_obss", 0, 100, -1)]
    primary.update(0)
    assert sta.can_transition_to_npca(0), \
        "Should transition: obss_remain(100) >= threshold(40)"

    print(f"  T4 PASS: NPCA min duration threshold (D1.2 §37.18.3.1.c.i) enforced correctly")


# ─────────────────────────────────────────────────────────────────────────────
# T5: AP absence failure (guidelines §7)
# ─────────────────────────────────────────────────────────────────────────────
def test_ap_absence_failure():
    """
    AP가 NPCA 채널에 있을 때(ap_on_primary=False),
    primary에서 전송 시도는 AP_ABSENCE_DUE_TO_NPCA로 실패해야 한다.
    """
    primary, npca = make_channels(primary_obss_rate=0.0)
    sta = make_sta(primary, npca)
    sta.ap_on_primary = False  # AP is on NPCA

    # Force STA into PRIMARY_TX state and trigger result
    from harq_sim.packet import Packet
    pkt = Packet(arrival_time=0)
    sta.current_packet = pkt

    before_cw = sta.primary_cw

    # Simulate AP-absence result (as Simulator would dispatch)
    sta.handle_tx_result(
        success=False,
        failure_reason=FailureReason.AP_ABSENCE_DUE_TO_NPCA,
        channel_type=ChannelType.PRIMARY,
        slot=5,
    )

    assert sta.stats["ap_absence_failures"] == 1, "ap_absence_failures counter should increment"
    assert sta.primary_cw >= before_cw, "primary CW should increase after failure"
    assert pkt.retry_count == 1, "retry_count should increment"

    print("  T5 PASS: AP absence failure handled correctly")


# ─────────────────────────────────────────────────────────────────────────────
# T6: NPCA CW initialization from qsrc (guidelines §5.3)
# ─────────────────────────────────────────────────────────────────────────────
def test_npca_cw_from_qsrc():
    """
    npca_cw = 2^qsrc × (CW_MIN + 1) − 1
    qsrc=0 → 15, qsrc=1 → 31, qsrc=2 → 63, qsrc=3 → 127
    """
    primary, npca = make_channels()
    expected = {0: 15, 1: 31, 2: 63, 3: 127}
    for qsrc, exp_cw in expected.items():
        sta = make_sta(primary, npca, qsrc=qsrc)
        sta._init_npca_state()
        assert sta.npca_cw == exp_cw, f"qsrc={qsrc}: expected npca_cw={exp_cw}, got {sta.npca_cw}"

    print("  T6 PASS: NPCA CW correctly computed from npca_initial_qsrc")


# ─────────────────────────────────────────────────────────────────────────────
# T7: Full simulation smoke test — NPCA enabled vs disabled
# ─────────────────────────────────────────────────────────────────────────────
def test_smoke_npca_vs_no_npca():
    """
    NPCA 활성 시 primary OBSS 동안 NPCA 전환이 발생해야 한다.
    NPCA 비활성 시 전환이 0이어야 한다.
    """
    random.seed(42)

    # NPCA enabled
    primary_en, npca_en = make_channels(primary_obss_rate=0.05, obss_duration=(40, 80))
    sta_en = make_sta(primary_en, npca_en, npca_enabled=True)
    sim_en = Simulator(1000, [sta_en], [primary_en, npca_en], enable_trace=False)
    sim_en.run()

    # NPCA disabled
    random.seed(42)
    primary_dis, npca_dis = make_channels(primary_obss_rate=0.05, obss_duration=(40, 80))
    sta_dis = make_sta(primary_dis, npca_dis, npca_enabled=False)
    sim_dis = Simulator(1000, [sta_dis], [primary_dis, npca_dis], enable_trace=False)
    sim_dis.run()

    assert sta_en.stats["npca_transitions"] > 0, \
        "NPCA-enabled STA should have transitioned to NPCA at least once"
    assert sta_dis.stats["npca_transitions"] == 0, \
        "NPCA-disabled STA should have 0 transitions"
    # switch_backs ≤ npca_transitions (마지막 transition이 시뮬 종료 시 미완일 수 있음)
    assert sta_en.stats["switch_backs"] <= sta_en.stats["npca_transitions"], \
        "switch_backs should not exceed npca_transitions"
    assert sta_en.stats["switch_backs"] >= sta_en.stats["npca_transitions"] - 1, \
        "At most one unfinished NPCA transition allowed at simulation end"

    print(f"  T7 PASS: NPCA enabled → {sta_en.stats['npca_transitions']} transitions, "
          f"disabled → {sta_dis.stats['npca_transitions']} transitions")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    tests = [
        test_primary_state_save_restore,
        test_npca_primary_cw_independence,
        test_switch_back_after_obss_ends,
        test_npca_min_duration_threshold,
        test_ap_absence_failure,
        test_npca_cw_from_qsrc,
        test_smoke_npca_vs_no_npca,
    ]
    passed = failed = 0
    print("\n=== Step 1 NPCA Verification Tests ===\n")
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL [{t.__name__}]: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR [{t.__name__}]: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
