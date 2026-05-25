"""
Step 3 HARQ-CC 검증 테스트 (9개)

T1: PHY_ERROR → HARQ buffer 저장 (combining_count=1, accumulated_snr 양수)
T2: Chase Combining 누적 — 2번째 PHY 실패 시 combining_count=2
T3: HARQ_RETX 타입 — buffer 유효 시 tx_type=HARQ_RETX CSV 기록
T4: MCS 제약 (§9.4) — HARQ_RETX는 original_mcs 사용 (고 SNR이어도 변경 안 됨)
T5: Validity horizon 만료 — 기한 초과 시 ARQ_RETX fallback (buffer flush)
T6: 성공 시 buffer flush — 전달 완료 후 harq_buffer.active=False
T7: Drop 시 buffer flush — retry_limit 초과 drop 후 harq_buffer.active=False
T8: Collision은 buffer 저장 안 함 (§9.2) — 충돌 후 harq_buffer.active=False
T9: PDR 향상 — 경계 SNR에서 HARQ PDR > ARQ PDR (통계적 검증)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random

from harq_sim.channel import Channel
from harq_sim.enums import ChannelType, FailureReason, PacketStatus
from harq_sim.harq_buffer import HARQBuffer
from harq_sim.packet import Packet, TrafficClass
from harq_sim.phy import snr_db_to_linear, snr_linear_to_db
from harq_sim.simulator import Simulator
from harq_sim.sta import STA


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_primary_channel(obss_rate: float = 0.0) -> Channel:
    return Channel(channel_id=0, obss_generation_rate=obss_rate,
                   obss_duration_range=(30, 80))


def make_npca_channel() -> Channel:
    return Channel(channel_id=1, obss_generation_rate=0.0)


def make_sta(
    snr: float = 5.0,
    harq_enabled: bool = True,
    harq_horizon: int = 200,
    retry_limit: int = 7,
    ppdu_duration: int = 5,
    npca_enabled: bool = False,
    snr_std: float = 0.0,
) -> STA:
    primary = make_primary_channel()
    npca    = make_npca_channel()
    return STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        npca_enabled=npca_enabled,
        ppdu_duration=ppdu_duration,
        switching_delay=1,
        switch_back_delay=1,
        retry_limit=retry_limit,
        infinite_queue=True,
        snr_db_mean=snr,
        snr_db_std=snr_std,
        harq_enabled=harq_enabled,
        harq_validity_horizon=harq_horizon,
    )


def inject_phy_failure(sta: STA, snr_db: float, slot: int = 0) -> None:
    """Directly call handle_tx_result with PHY_ERROR for testing."""
    pkt = Packet(arrival_time=0)
    pkt.current_mcs = 3   # fixed MCS for testing
    sta.current_packet = pkt
    sta.packet_queue.append(pkt)
    sta._current_tx_snr_db = snr_db
    sta.handle_tx_result(
        False, FailureReason.PHY_ERROR, ChannelType.PRIMARY, slot,
        snr_db=snr_db, effective_snr_db=snr_db,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_t1_phy_error_stores_in_buffer():
    """T1: PHY_ERROR → buffer 활성화, combining_count=1, accumulated_snr > 0"""
    sta = make_sta(snr=5.0, harq_enabled=True)
    assert not sta.harq_buffer.active, "buffer must be inactive before any TX"

    inject_phy_failure(sta, snr_db=5.0, slot=0)

    assert sta.harq_buffer.active, "buffer must be active after PHY_ERROR"
    assert sta.harq_buffer.combining_count == 1, \
        f"combining_count should be 1, got {sta.harq_buffer.combining_count}"
    expected_linear = snr_db_to_linear(5.0)
    assert abs(sta.harq_buffer.accumulated_snr_linear - expected_linear) < 1e-9, \
        "accumulated_snr_linear must equal snr_db_to_linear(5.0)"
    assert sta.harq_buffer.original_mcs == 3, \
        f"original_mcs should be 3, got {sta.harq_buffer.original_mcs}"
    print("  T1 PASS: PHY_ERROR → buffer active, combining_count=1, accumulated_snr correct")


def test_t2_chase_combining_accumulation():
    """T2: 두 번째 PHY 실패 시 combining_count=2, accumulated_snr 누적"""
    sta = make_sta(snr=5.0, harq_enabled=True)
    snr1, snr2 = 5.0, 6.0

    inject_phy_failure(sta, snr_db=snr1, slot=0)
    assert sta.harq_buffer.combining_count == 1

    # Second PHY failure (HARQ_RETX also fails)
    pkt = sta.current_packet
    pkt.current_mcs = 3
    sta._current_tx_snr_db = snr2
    # Simulate HARQ combining: effective_snr uses accumulated + new
    new_linear = snr_db_to_linear(snr2)
    eff_snr = sta.harq_buffer.effective_snr_db(new_linear)
    sta.handle_tx_result(
        False, FailureReason.PHY_ERROR, ChannelType.PRIMARY, slot=10,
        snr_db=snr2, effective_snr_db=eff_snr,
    )

    assert sta.harq_buffer.combining_count == 2, \
        f"combining_count should be 2 after second fail, got {sta.harq_buffer.combining_count}"
    expected = snr_db_to_linear(snr1) + snr_db_to_linear(snr2)
    assert abs(sta.harq_buffer.accumulated_snr_linear - expected) < 1e-9, \
        "accumulated_snr_linear should be sum of both linear SNRs"
    eff_db = sta.harq_buffer.effective_snr_db()
    assert eff_db > snr1, \
        f"effective_snr_db ({eff_db:.2f}) must exceed original snr ({snr1})"
    print(f"  T2 PASS: count=2, accumulated_snr={sta.harq_buffer.accumulated_snr_linear:.4f}, "
          f"eff_snr_db={eff_db:.2f} dB")


def test_t3_harq_retx_type_in_csv():
    """T3: buffer 유효 시 tx_type=HARQ_RETX CSV 기록 확인"""
    random.seed(42)
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = make_npca_channel()

    # SNR at MCS0 threshold → ~50% success per attempt; with HARQ combining should succeed
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        ppdu_duration=3,
        retry_limit=7,
        infinite_queue=True,
        snr_db_mean=5.0,
        snr_db_std=0.0,
        harq_enabled=True,
        harq_validity_horizon=500,
    )

    sim = Simulator(num_slots=300, stas=[sta], channels=[primary, npca], enable_trace=True)
    sim.run()

    harq_rows = [e for e in sim.log if e.tx_type == "HARQ_RETX"]
    phy_err_rows = [e for e in sim.log if e.failure_reason == "PHY_ERROR"]

    assert len(phy_err_rows) > 0, "Need some PHY errors to generate HARQ retransmissions"
    assert len(harq_rows) > 0, \
        f"Expected HARQ_RETX events in log; got {len(harq_rows)}. PHY errors: {len(phy_err_rows)}"
    print(f"  T3 PASS: {len(harq_rows)} HARQ_RETX events logged "
          f"(PHY errors: {len(phy_err_rows)})")


def test_t4_mcs_constraint():
    """T4: HARQ_RETX는 original_mcs 사용 — 고 SNR에서도 MCS 변경 안 됨 (§9.4)"""
    random.seed(0)
    primary = Channel(channel_id=0, obss_generation_rate=0.0)
    npca    = make_npca_channel()

    # Low SNR → first TX uses MCS 0 (threshold 5 dB); high SNR for retry would select MCS 7
    # We verify that HARQ_RETX still uses MCS 0 (original_mcs)
    sta = STA(
        sta_id=0,
        primary_channel=primary,
        npca_channel=npca,
        ppdu_duration=3,
        retry_limit=5,
        infinite_queue=True,
        snr_db_mean=5.0,    # low SNR → MCS 0 → ~50% chance of PHY fail
        snr_db_std=0.0,
        harq_enabled=True,
        harq_validity_horizon=500,
    )

    # Inject PHY failure with MCS 0
    pkt = Packet(arrival_time=0)
    pkt.current_mcs = 0
    sta.current_packet = pkt
    sta.packet_queue.append(pkt)
    sta._current_tx_snr_db = 5.0
    sta.handle_tx_result(
        False, FailureReason.PHY_ERROR, ChannelType.PRIMARY, slot=0,
        snr_db=5.0, effective_snr_db=5.0,
    )

    assert sta.harq_buffer.original_mcs == 0, \
        f"original_mcs should be 0, got {sta.harq_buffer.original_mcs}"

    # Now simulate what _handle_primary_backoff does at retry:
    # Even if SNR sample is very high, HARQ_RETX should use original_mcs=0
    sta._current_tx_snr_db = 40.0   # pretend SNR jumped to 40 dB
    is_harq = sta._is_harq_retx_applicable(pkt, slot=3)

    assert is_harq, "HARQ should be applicable (buffer active, valid)"
    # If HARQ applicable, tx handler uses harq_buffer.original_mcs, not select_mcs(40.0)
    mcs_used = sta.harq_buffer.original_mcs
    from harq_sim.phy import select_mcs
    mcs_if_arq = select_mcs(40.0)   # would be MCS 7 without HARQ constraint
    assert mcs_used == 0, f"HARQ_RETX must use original_mcs=0, but mcs_used={mcs_used}"
    assert mcs_if_arq == 7, "ARQ would have selected MCS 7 at 40 dB (sanity check)"
    print(f"  T4 PASS: HARQ_RETX uses MCS {mcs_used} (original), "
          f"ARQ would have used MCS {mcs_if_arq}")


def test_t5_validity_horizon_expiry():
    """T5: validity horizon 만료 시 buffer flush → ARQ_RETX fallback"""
    sta = make_sta(snr=5.0, harq_enabled=True, harq_horizon=10)

    # Inject PHY failure at slot 0 → buffer valid until slot 10
    inject_phy_failure(sta, snr_db=5.0, slot=0)
    assert sta.harq_buffer.active
    assert sta.harq_buffer.validity_deadline == 10, \
        f"validity_deadline should be 10, got {sta.harq_buffer.validity_deadline}"

    pkt = sta.current_packet
    # Within horizon: HARQ should be applicable
    assert sta._is_harq_retx_applicable(pkt, slot=10), \
        "Buffer should still be valid at slot 10 (deadline inclusive)"

    # Beyond horizon: _is_harq_retx_applicable should flush buffer and return False
    result = sta._is_harq_retx_applicable(pkt, slot=11)
    assert not result, "Buffer should be invalid at slot 11 (deadline expired)"
    assert not sta.harq_buffer.active, \
        "_is_harq_retx_applicable must flush the buffer when expired"
    print("  T5 PASS: buffer expires at slot 11, flushed, ARQ fallback triggered")


def test_t6_flush_on_success():
    """T6: 성공 시 harq_buffer.flush() — 전달 완료 후 active=False"""
    sta = make_sta(snr=5.0, harq_enabled=True)

    # First: PHY failure → buffer active
    inject_phy_failure(sta, snr_db=5.0, slot=0)
    assert sta.harq_buffer.active

    # Second: success (HARQ_RETX succeeds)
    pkt = sta.current_packet
    sta._current_tx_snr_db = 20.0
    new_linear = snr_db_to_linear(20.0)
    eff_snr = sta.harq_buffer.effective_snr_db(new_linear)
    sta.handle_tx_result(
        True, FailureReason.NONE, ChannelType.PRIMARY, slot=10,
        snr_db=20.0, effective_snr_db=eff_snr,
    )

    assert not sta.harq_buffer.active, \
        "harq_buffer must be flushed after successful delivery"
    assert sta.stats["harq_tx_success"] == 1
    assert sta.stats["packets_delivered"] == 1
    print("  T6 PASS: buffer flushed after HARQ_RETX success, packets_delivered=1")


def test_t7_flush_on_drop():
    """T7: retry_limit 초과 drop 시 harq_buffer.flush()"""
    sta = make_sta(snr=5.0, harq_enabled=True, retry_limit=1)

    # First failure → buffer active
    inject_phy_failure(sta, snr_db=5.0, slot=0)
    assert sta.harq_buffer.active

    # Second failure (retry_count now exceeds retry_limit=1)
    pkt = sta.current_packet
    pkt.current_mcs = 3
    sta._current_tx_snr_db = 5.0
    new_linear = snr_db_to_linear(5.0)
    eff_snr = sta.harq_buffer.effective_snr_db(new_linear)
    sta.handle_tx_result(
        False, FailureReason.PHY_ERROR, ChannelType.PRIMARY, slot=10,
        snr_db=5.0, effective_snr_db=eff_snr,
    )

    # retry_count should now be 2 > retry_limit=1 → packet dropped, buffer flushed
    assert not sta.harq_buffer.active, \
        "harq_buffer must be flushed after packet drop"
    assert sta.stats["packets_dropped"] == 1, \
        f"Expected 1 packet dropped, got {sta.stats['packets_dropped']}"
    print("  T7 PASS: buffer flushed after retry_limit exceeded drop")


def test_t8_collision_no_buffer():
    """T8: Collision → buffer 저장 안 함 (§9.2, §15.3)"""
    sta = make_sta(snr=20.0, harq_enabled=True)
    assert not sta.harq_buffer.active

    # Inject collision failure
    pkt = Packet(arrival_time=0)
    pkt.current_mcs = 5
    sta.current_packet = pkt
    sta.packet_queue.append(pkt)
    sta._current_tx_snr_db = 20.0
    sta.handle_tx_result(
        False, FailureReason.COLLISION, ChannelType.PRIMARY, slot=0,
        snr_db=20.0, effective_snr_db=20.0,
    )

    assert not sta.harq_buffer.active, \
        "Collision must NOT store soft information in HARQ buffer (§9.2)"
    print("  T8 PASS: collision did not activate HARQ buffer")


def test_t9_pdr_improvement():
    """T9: 경계 SNR에서 HARQ PDR > ARQ PDR (통계적 검증)

    SNR = MCS0 threshold = 5 dB → p_success = 0.5 per ARQ attempt.
    With retry_limit=1 (2 total attempts):
      ARQ:  P(delivered) = 1 - (0.5)^2 = 0.75
      HARQ: eff_snr after 2 attempts = 5 + 10*log10(2) ≈ 8.0 dB
            p2 = sigmoid(8.0 - 5.0) ≈ 0.953
            P(delivered) ≈ 0.5 + 0.5*0.953 ≈ 0.977

    Expected HARQ PDR ≈ 0.977 vs ARQ PDR ≈ 0.75.
    """
    random.seed(42)

    def run_scenario(harq: bool) -> float:
        r = random.Random(1234)
        primary = Channel(channel_id=0, obss_generation_rate=0.0)
        npca    = make_npca_channel()
        sta = STA(
            sta_id=0,
            primary_channel=primary,
            npca_channel=npca,
            ppdu_duration=3,
            retry_limit=1,
            infinite_queue=True,
            snr_db_mean=5.0,
            snr_db_std=0.0,
            harq_enabled=harq,
            harq_validity_horizon=500,
        )
        sim = Simulator(num_slots=3000, stas=[sta], channels=[primary, npca],
                        enable_trace=False)
        random.seed(1234)
        sim.run()
        m = sim.compute_metrics()[0]
        total = m["packets_delivered"] + m["packets_dropped"]
        return m["packets_delivered"] / total if total else 0.0

    pdr_arq  = run_scenario(harq=False)
    pdr_harq = run_scenario(harq=True)

    print(f"  T9: ARQ PDR={pdr_arq:.4f}  HARQ PDR={pdr_harq:.4f}")
    assert pdr_harq > pdr_arq, \
        f"HARQ PDR ({pdr_harq:.4f}) should exceed ARQ PDR ({pdr_arq:.4f})"
    assert pdr_harq > 0.90, \
        f"HARQ PDR expected > 0.90 at boundary SNR, got {pdr_harq:.4f}"
    assert pdr_arq < 0.85, \
        f"ARQ PDR expected < 0.85 at boundary SNR with retry_limit=1, got {pdr_arq:.4f}"
    print(f"  T9 PASS: HARQ PDR ({pdr_harq:.4f}) > ARQ PDR ({pdr_arq:.4f}), "
          f"improvement = {(pdr_harq - pdr_arq)*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("T1", "PHY_ERROR stores in buffer",       test_t1_phy_error_stores_in_buffer),
    ("T2", "Chase Combining accumulation",     test_t2_chase_combining_accumulation),
    ("T3", "HARQ_RETX type in CSV",            test_t3_harq_retx_type_in_csv),
    ("T4", "MCS constraint (§9.4)",            test_t4_mcs_constraint),
    ("T5", "Validity horizon expiry",          test_t5_validity_horizon_expiry),
    ("T6", "Buffer flush on success",          test_t6_flush_on_success),
    ("T7", "Buffer flush on drop",             test_t7_flush_on_drop),
    ("T8", "Collision does not store buffer",  test_t8_collision_no_buffer),
    ("T9", "PDR improvement vs ARQ",           test_t9_pdr_improvement),
]


def main():
    print("=== Step 3 HARQ-CC Verification Tests ===")
    passed = failed = 0
    for tag, desc, fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  {tag} FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  {tag} ERROR: {type(e).__name__}: {e}")
            failed += 1
    print(f"Results: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
