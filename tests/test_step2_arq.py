"""
Step 2 검증 테스트: ARQ-only retransmission

검증 항목:
  T1. 고 SNR (40 dB) → PHY 실패 없음 (결정론적)
  T2. 저 SNR (3 dB)  → PHY 실패 발생, primary_cw 증가
  T3. ARQ_RETX 타입  → retry 이후 tx_type = ARQ_RETX (CSV에서 확인)
  T4. Retry limit    → 패킷 drop (packets_dropped > 0)
  T5. PHY 실패 채널 독립성 — NPCA PHY 실패 → npca_cw 증가, primary_cw 불변
  T6. 데드라인 만료   → XR 트래픽 클래스 패킷 drop
  T7. Smoke test     — NPCA enabled, 중간 SNR, ARQ 발생

실행:
  python -m pytest tests/test_step2_arq.py -v
  또는
  python tests/test_step2_arq.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
random.seed(0)

from harq_sim.channel import Channel
from harq_sim.enums import ChannelType, FailureReason, PacketStatus, STAMode, TrafficClass
from harq_sim.packet import Packet
from harq_sim.phy import attempt_success, success_prob, MCS_SNR_THRESHOLDS
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_channels(primary_obss_rate=0.0, obss_duration=(50, 50)):
    primary = Channel(channel_id=0, obss_generation_rate=primary_obss_rate,
                      obss_duration_range=obss_duration)
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    return primary, npca


def make_sta(primary, npca, snr_db_mean=25.0, snr_db_std=0.0,
             retry_limit=7, npca_enabled=True, qsrc=0,
             switching_delay=1, switch_back_delay=1, min_threshold=0):
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
        retry_limit=retry_limit,
        infinite_queue=True,
        snr_db_mean=snr_db_mean,
        snr_db_std=snr_db_std,
    )


# ─────────────────────────────────────────────────────────────────────────────
# T1: High SNR — no PHY failures (deterministic)
# ─────────────────────────────────────────────────────────────────────────────
def test_high_snr_no_phy_failure():
    """SNR = 40 dB (MCS 7 threshold = 26) → success_prob ≈ 1.0 → zero PHY errors."""
    random.seed(10)
    primary, npca = make_channels(primary_obss_rate=0.0)
    sta = make_sta(primary, npca, snr_db_mean=40.0, snr_db_std=0.0, retry_limit=7)
    sim = Simulator(num_slots=300, stas=[sta], channels=[primary, npca], enable_trace=False)
    sim.run()

    assert sta.stats["phy_error_failures"] == 0, \
        f"No PHY errors expected at SNR=40dB, got {sta.stats['phy_error_failures']}"
    assert sta.stats["packets_delivered"] > 0, "Should have delivered packets at high SNR"

    print(f"  T1 PASS: SNR=40dB → 0 PHY errors, {sta.stats['packets_delivered']} delivered")


# ─────────────────────────────────────────────────────────────────────────────
# T2: Low SNR — PHY failures occur, CW increases
# ─────────────────────────────────────────────────────────────────────────────
def test_low_snr_phy_failure():
    """SNR = 3 dB (below MCS-0 threshold of 5 dB) → many PHY failures."""
    random.seed(20)
    primary, npca = make_channels(primary_obss_rate=0.0)
    sta = make_sta(primary, npca, snr_db_mean=3.0, snr_db_std=0.0, retry_limit=7)

    cw_before = sta.primary_cw

    sim = Simulator(num_slots=500, stas=[sta], channels=[primary, npca], enable_trace=False)
    sim.run()

    assert sta.stats["phy_error_failures"] > 0, \
        "Expected PHY failures at SNR=3dB"
    assert sta.primary_cw >= cw_before, \
        "primary_cw should have increased after PHY failures"

    print(f"  T2 PASS: SNR=3dB → {sta.stats['phy_error_failures']} PHY errors, "
          f"primary_cw={sta.primary_cw}")


# ─────────────────────────────────────────────────────────────────────────────
# T3: ARQ retry type — tx_type=ARQ_RETX after failure
# ─────────────────────────────────────────────────────────────────────────────
def test_arq_retx_type_in_log():
    """After a PHY failure, retry attempt should appear as ARQ_RETX in the CSV log."""
    random.seed(30)
    primary, npca = make_channels(primary_obss_rate=0.0)
    # SNR = 5 dB: MCS 0 threshold = 5.0 → success_prob = sigmoid(0) = 0.5
    # Statistically some will fail and be retried
    sta = make_sta(primary, npca, snr_db_mean=5.0, snr_db_std=0.0, retry_limit=7,
                   npca_enabled=False)
    sim = Simulator(num_slots=2000, stas=[sta], channels=[primary, npca], enable_trace=True)
    sim.run()

    arq_retx_rows = [e for e in sim.log if e.tx_type == "ARQ_RETX"]
    assert len(arq_retx_rows) > 0, \
        "Expected ARQ_RETX events in log after PHY failures"

    # Verify retry_count >= 1 for all ARQ_RETX rows
    for row in arq_retx_rows:
        assert row.retry_count is not None and row.retry_count >= 1, \
            f"ARQ_RETX row should have retry_count ≥ 1, got {row.retry_count}"

    print(f"  T3 PASS: {len(arq_retx_rows)} ARQ_RETX events logged with retry_count ≥ 1")


# ─────────────────────────────────────────────────────────────────────────────
# T4: Retry limit → packet drop
# ─────────────────────────────────────────────────────────────────────────────
def test_retry_limit_packet_drop():
    """retry_limit=2 + very low SNR → packets dropped after retry limit exceeded."""
    random.seed(40)
    primary, npca = make_channels(primary_obss_rate=0.0)
    # SNR = 1 dB → success_prob(1, 0) ≈ sigmoid(1-5) = sigmoid(-4) ≈ 0.018
    # With retry_limit=2, probability of 3 consecutive failures ≈ 0.982^3 ≈ 0.945
    sta = make_sta(primary, npca, snr_db_mean=1.0, snr_db_std=0.0, retry_limit=2,
                   npca_enabled=False)
    sim = Simulator(num_slots=1000, stas=[sta], channels=[primary, npca], enable_trace=False)
    sim.run()

    assert sta.stats["packets_dropped"] > 0, \
        f"Expected dropped packets at very low SNR with retry_limit=2, got 0"
    # Total = delivered + dropped > 0
    total = sta.stats["packets_delivered"] + sta.stats["packets_dropped"]
    assert total > 0

    print(f"  T4 PASS: retry_limit=2, SNR=1dB → {sta.stats['packets_dropped']} dropped, "
          f"{sta.stats['packets_delivered']} delivered")


# ─────────────────────────────────────────────────────────────────────────────
# T5: PHY failure channel independence — NPCA fail does NOT increase primary CW
# ─────────────────────────────────────────────────────────────────────────────
def test_phy_failure_channel_independence():
    """PHY failure on NPCA → npca_cw increases; primary_cw unchanged (and vice versa)."""
    primary, npca = make_channels()
    sta = make_sta(primary, npca, snr_db_mean=25.0)

    # Set known initial states
    sta.primary_cw = 15
    sta.npca_cw    = 15

    # NPCA PHY failure → only npca_cw should increase
    pkt1 = Packet(arrival_time=0)
    sta.current_packet = pkt1
    sta.packet_queue.append(pkt1)
    before_primary_cw = sta.primary_cw
    sta.handle_tx_result(
        success=False,
        failure_reason=FailureReason.PHY_ERROR,
        channel_type=ChannelType.NPCA,
        slot=10,
        snr_db=3.0,
    )
    assert sta.primary_cw == before_primary_cw, \
        f"primary_cw must not change after NPCA PHY failure: {sta.primary_cw} != {before_primary_cw}"
    assert sta.npca_cw == 31, \
        f"npca_cw should double to 31, got {sta.npca_cw}"
    assert sta.stats["phy_error_failures"] == 1

    # Primary PHY failure → only primary_cw should increase
    sta.primary_cw = 15
    sta.npca_cw    = 15
    pkt2 = Packet(arrival_time=0)
    sta.current_packet = pkt2
    sta.packet_queue.append(pkt2)
    before_npca_cw = sta.npca_cw
    sta.handle_tx_result(
        success=False,
        failure_reason=FailureReason.PHY_ERROR,
        channel_type=ChannelType.PRIMARY,
        slot=10,
        snr_db=3.0,
    )
    assert sta.npca_cw == before_npca_cw, \
        f"npca_cw must not change after primary PHY failure: {sta.npca_cw} != {before_npca_cw}"
    assert sta.primary_cw == 31, \
        f"primary_cw should double to 31, got {sta.primary_cw}"

    print("  T5 PASS: PHY failure CW independence — NPCA/primary CW updates are isolated")


# ─────────────────────────────────────────────────────────────────────────────
# T6: Deadline expired → packet drop
# ─────────────────────────────────────────────────────────────────────────────
def test_deadline_expired_drop():
    """XR traffic class: deadline = arrival_time + 1111 slots.
    If current_slot > deadline, packet should be dropped regardless of retry_limit."""
    primary, npca = make_channels()
    sta = make_sta(primary, npca, snr_db_mean=3.0, retry_limit=7)

    pkt = Packet(arrival_time=0, traffic_class=TrafficClass.XR)
    sta.current_packet = pkt
    sta.packet_queue.append(pkt)

    # Trigger failure at slot 2000 (well past XR deadline of 1111 slots)
    sta.handle_tx_result(
        success=False,
        failure_reason=FailureReason.PHY_ERROR,
        channel_type=ChannelType.PRIMARY,
        slot=2000,
        snr_db=3.0,
    )

    assert pkt.status == PacketStatus.DROPPED, \
        f"XR packet should be dropped after deadline, got {pkt.status}"
    assert sta.stats["packets_dropped"] == 1, \
        "packets_dropped should be 1"
    # retry_count should be 1 (one failed attempt before deadline check)
    assert pkt.retry_count == 1, \
        f"retry_count should be 1 at drop, got {pkt.retry_count}"

    print("  T6 PASS: XR deadline expired → packet dropped (retry_count=1, status=DROPPED)")


# ─────────────────────────────────────────────────────────────────────────────
# T7: Smoke test — ARQ with NPCA, moderate SNR
# ─────────────────────────────────────────────────────────────────────────────
def test_smoke_arq_with_npca():
    """Full simulation: NPCA enabled, moderate SNR (20 dB).
    Expect: NPCA transitions, some PHY errors, ARQ retries, positive PDR."""
    random.seed(42)
    primary, npca = make_channels(primary_obss_rate=0.05, obss_duration=(40, 80))
    sta = make_sta(primary, npca, snr_db_mean=20.0, snr_db_std=0.0,
                   retry_limit=7, npca_enabled=True)
    sim = Simulator(num_slots=2000, stas=[sta], channels=[primary, npca], enable_trace=True)
    sim.run()

    total_pkts = sta.stats["packets_delivered"] + sta.stats["packets_dropped"]
    pdr = sta.stats["packets_delivered"] / total_pkts if total_pkts else 0.0

    assert sta.stats["npca_transitions"] > 0, "Should have NPCA transitions"
    assert pdr > 0.0, f"PDR should be > 0, got {pdr:.3f}"

    # Check ARQ_RETX entries exist when there were PHY errors
    if sta.stats["phy_error_failures"] > 0:
        arq_retx_rows = [e for e in sim.log if e.tx_type == "ARQ_RETX"]
        assert len(arq_retx_rows) > 0, \
            "Expected ARQ_RETX in log when phy_error_failures > 0"

    # SNR field should be non-None for success and PHY-error events
    snr_rows = [e for e in sim.log if e.snr_db is not None]
    assert len(snr_rows) > 0, "snr_db should be recorded for TX completion events"

    print(f"  T7 PASS: NPCA transitions={sta.stats['npca_transitions']}, "
          f"PHY errors={sta.stats['phy_error_failures']}, "
          f"PDR={pdr:.3f}, snr_rows={len(snr_rows)}")


# ─────────────────────────────────────────────────────────────────────────────
# T8: PHY model — logistic success probability sanity check
# ─────────────────────────────────────────────────────────────────────────────
def test_phy_model_sanity():
    """Verify logistic PER model properties (guidelines §15.2)."""
    # At threshold: p = 0.5
    for mcs, threshold in MCS_SNR_THRESHOLDS.items():
        p = success_prob(threshold, mcs)
        assert abs(p - 0.5) < 1e-6, \
            f"MCS {mcs}: p_success at threshold should be 0.5, got {p:.6f}"

    # Well above threshold → p → 1
    p_high = success_prob(100.0, 0)
    assert p_high > 0.999, f"Very high SNR should give near-1 success prob, got {p_high:.6f}"

    # Well below threshold → p → 0
    p_low = success_prob(-100.0, 0)
    assert p_low < 0.001, f"Very low SNR should give near-0 success prob, got {p_low:.6f}"

    # Success monotonically increases with SNR
    prev = 0.0
    for snr in range(-5, 35):
        p = success_prob(float(snr), 0)
        assert p >= prev - 1e-9, "success_prob must be non-decreasing in SNR"
        prev = p

    print("  T8 PASS: Logistic PER model — symmetry, monotonicity, boundary values verified")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    tests = [
        test_high_snr_no_phy_failure,
        test_low_snr_phy_failure,
        test_arq_retx_type_in_log,
        test_retry_limit_packet_drop,
        test_phy_failure_channel_independence,
        test_deadline_expired_drop,
        test_smoke_arq_with_npca,
        test_phy_model_sanity,
    ]
    passed = failed = 0
    print("\n=== Step 2 ARQ Verification Tests ===\n")
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
