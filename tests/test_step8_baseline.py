"""
Step 8 검증 테스트 — Baseline 비교

T1 : Legacy EDCA      — npca_transitions == 0, harq_tx_success == 0
T2 : Legacy EDCA      — harq_tx_fail == 0 (HARQ buffer 비활성)
T3 : ARQ-only NPCA    — npca_transitions > 0, harq_tx_success == 0
T4 : HARQ-only        — npca_transitions == 0, harq 활동 > 0
T5 : Fixed-CW NPCA-HARQ  — npca_transitions > 0, harq 활동 > 0
T6 : Adaptive-CW NPCA-HARQ — avg_npca_qsrc != None (adaptive 기록 확인)
T7 : grid_best_reward_profile() — 반환된 profile name ∈ INTENT_PROFILES 키
T8 : LLM-reward profile — validate_reward_profile() → True
T9 : 7개 baseline 전체 완료 — "aggregate" 키 및 필수 지표 존재
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.llm_reward_designer import LLMRewardDesigner, validate_reward_profile
from harq_sim.reward import INTENT_PROFILES, normalize_metrics, compute_reward
from harq_sim.run_step8 import (
    BASELINE_NAMES,
    build_and_run,
    grid_best_reward_profile,
    run_baseline,
)

# ── 공통 테스트 파라미터 ──────────────────────────────────────────────────────
# SNR=14dB → MCS3 경계 → ~50% PHY 성공률 → HARQ retransmission 활성화
_BASE = dict(
    num_slots=300,
    num_stas=2,
    obss_rate=0.2,
    obss_min=20,
    obss_max=60,
    npca_qsrc=0,
    npca_threshold=0,
    ppdu_duration=20,
    harq_horizon=200,
    snr_db_mean=14.0,
    snr_db_std=0.0,
    seed=42,
    enable_trace=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# T1: Legacy EDCA — NPCA 전환 없음, HARQ 성공 없음
# ─────────────────────────────────────────────────────────────────────────────
def test_t1_legacy_edca_no_npca_no_harq_success():
    sim = build_and_run(npca_enabled=False, harq_enabled=False, adaptive_cw=False, **_BASE)
    m   = sim.compute_metrics()
    for sta_id, s in m.items():
        if sta_id == "aggregate":
            continue
        assert s["npca_transitions"] == 0, (
            f"STA {sta_id}: expected 0 NPCA transitions for Legacy EDCA, got {s['npca_transitions']}"
        )
        assert s["harq_tx_success"] == 0, (
            f"STA {sta_id}: expected 0 harq_tx_success for Legacy EDCA, got {s['harq_tx_success']}"
        )
    print("T1 PASS: Legacy EDCA → npca_transitions=0, harq_tx_success=0 (all STAs)")


# ─────────────────────────────────────────────────────────────────────────────
# T2: Legacy EDCA — HARQ buffer 비활성 → harq_tx_fail == 0
# ─────────────────────────────────────────────────────────────────────────────
def test_t2_legacy_edca_no_harq_fail():
    sim = build_and_run(npca_enabled=False, harq_enabled=False, adaptive_cw=False, **_BASE)
    m   = sim.compute_metrics()
    for sta_id, s in m.items():
        if sta_id == "aggregate":
            continue
        assert s["harq_tx_fail"] == 0, (
            f"STA {sta_id}: expected 0 harq_tx_fail for Legacy EDCA, got {s['harq_tx_fail']}"
        )
    print("T2 PASS: Legacy EDCA → harq_tx_fail=0 (all STAs)")


# ─────────────────────────────────────────────────────────────────────────────
# T3: ARQ-only NPCA — NPCA 전환 있음, HARQ 성공 없음
# ─────────────────────────────────────────────────────────────────────────────
def test_t3_arq_only_npca_has_transitions_no_harq():
    sim = build_and_run(npca_enabled=True, harq_enabled=False, adaptive_cw=False, **_BASE)
    m   = sim.compute_metrics()
    agg = m["aggregate"]

    assert agg["npca_transition_count"] > 0, (
        f"ARQ-only NPCA: expected npca_transitions > 0, got {agg['npca_transition_count']}"
    )
    for sta_id, s in m.items():
        if sta_id == "aggregate":
            continue
        assert s["harq_tx_success"] == 0, (
            f"STA {sta_id}: expected 0 harq_tx_success for ARQ-only NPCA, got {s['harq_tx_success']}"
        )
    print(
        f"T3 PASS: ARQ-only NPCA → npca_transitions={agg['npca_transition_count']} > 0, "
        f"harq_tx_success=0 (all STAs)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T4: HARQ-only — NPCA 전환 없음, HARQ 활동(성공+실패) > 0
# NPCA 없이도 HARQ TX 기회를 충분히 확보하기 위해 낮은 OBSS rate 사용
# ─────────────────────────────────────────────────────────────────────────────
def test_t4_harq_only_no_npca_harq_active():
    # Low OBSS rate → STA gets many TX opportunities without NPCA
    # SNR=14dB → ~50% PHY failure → HARQ combining triggered reliably
    params = {**_BASE, "obss_rate": 0.02, "num_slots": 600}
    sim = build_and_run(npca_enabled=False, harq_enabled=True, adaptive_cw=False, **params)
    m   = sim.compute_metrics()
    agg = m["aggregate"]

    assert agg["npca_transition_count"] == 0, (
        f"HARQ-only: expected 0 NPCA transitions, got {agg['npca_transition_count']}"
    )

    total_harq = sum(
        s["harq_tx_success"] + s["harq_tx_fail"]
        for sta_id, s in m.items()
        if sta_id != "aggregate"
    )
    assert total_harq > 0, (
        f"HARQ-only: expected harq_tx_success + harq_tx_fail > 0, got {total_harq}"
    )
    print(f"T4 PASS: HARQ-only → npca_transitions=0, harq_activity={total_harq} > 0")


# ─────────────────────────────────────────────────────────────────────────────
# T5: Fixed-CW NPCA-HARQ — NPCA 전환 있음, HARQ 활동 > 0
# ─────────────────────────────────────────────────────────────────────────────
def test_t5_fixed_cw_npca_harq_both_active():
    sim = build_and_run(npca_enabled=True, harq_enabled=True, adaptive_cw=False, **_BASE)
    m   = sim.compute_metrics()
    agg = m["aggregate"]

    assert agg["npca_transition_count"] > 0, (
        f"Fixed-CW NPCA-HARQ: expected npca_transitions > 0, got {agg['npca_transition_count']}"
    )

    total_harq = sum(
        s["harq_tx_success"] + s["harq_tx_fail"]
        for sta_id, s in m.items()
        if sta_id != "aggregate"
    )
    assert total_harq > 0, (
        f"Fixed-CW NPCA-HARQ: expected harq activity > 0, got {total_harq}"
    )
    print(
        f"T5 PASS: Fixed-CW NPCA-HARQ → npca_transitions={agg['npca_transition_count']}, "
        f"harq_activity={total_harq}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T6: Adaptive-CW NPCA-HARQ — avg_npca_qsrc 기록됨 (not None)
# ─────────────────────────────────────────────────────────────────────────────
def test_t6_adaptive_cw_qsrc_recorded():
    sim = build_and_run(npca_enabled=True, harq_enabled=True, adaptive_cw=True, **_BASE)
    m   = sim.compute_metrics()

    recorded = False
    for sta_id, s in m.items():
        if sta_id == "aggregate":
            continue
        if s["avg_npca_qsrc"] is not None:
            recorded = True
            break

    assert recorded, "Adaptive-CW NPCA-HARQ: avg_npca_qsrc should be recorded for at least one STA"
    print("T6 PASS: Adaptive-CW NPCA-HARQ → avg_npca_qsrc recorded for at least one STA")


# ─────────────────────────────────────────────────────────────────────────────
# T7: grid_best_reward_profile() — 반환된 profile name ∈ INTENT_PROFILES 키
# ─────────────────────────────────────────────────────────────────────────────
def test_t7_grid_best_profile_valid_name():
    sim = build_and_run(npca_enabled=True, harq_enabled=True, adaptive_cw=True, **_BASE)
    best_name, best_profile, best_reward = grid_best_reward_profile(sim)

    assert best_name in INTENT_PROFILES, (
        f"grid_best_reward_profile() returned '{best_name}', not in INTENT_PROFILES"
    )
    assert isinstance(best_reward, float), (
        f"grid_best_reward_profile() reward must be float, got {type(best_reward)}"
    )
    # Verify it is actually the best among all profiles
    agg  = sim.compute_metrics()["aggregate"]
    norm = normalize_metrics(agg)
    for name, profile in INTENT_PROFILES.items():
        r = compute_reward(norm, profile["weights"], profile.get("constraints"), agg)
        assert r <= best_reward + 1e-9, (
            f"Profile '{name}' reward {r:.6f} > claimed best {best_reward:.6f}"
        )
    print(
        f"T7 PASS: grid_best_reward_profile() → '{best_name}' "
        f"(reward={best_reward:.4f}, verified best)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T8: LLM-reward (mock) — validate_reward_profile() → True
# ─────────────────────────────────────────────────────────────────────────────
def test_t8_llm_reward_profile_valid():
    designer = LLMRewardDesigner(use_mock=True)
    for intent_str in [
        "balanced throughput and latency",
        "delay sensitive XR",
        "maximize throughput",
        "energy saving mode",
        "fair coexistence",
    ]:
        profile = designer.design_reward(intent_str)
        result  = validate_reward_profile(profile)
        assert result is True, (
            f"validate_reward_profile() returned {result} for intent='{intent_str}'"
        )
    print("T8 PASS: LLM-reward (mock) → validate_reward_profile()=True for all 5 intents")


# ─────────────────────────────────────────────────────────────────────────────
# T9: 7개 baseline 전체 완료 — "aggregate" 키 및 필수 지표 존재
# ─────────────────────────────────────────────────────────────────────────────
def test_t9_all_baselines_complete_with_required_metrics():
    designer    = LLMRewardDesigner(use_mock=True)
    llm_profile = designer.design_reward("balanced throughput and latency")

    common = {**_BASE}

    required_keys = [
        "aggregate_throughput", "mean_access_delay", "p95_access_delay",
        "packet_delivery_ratio", "packet_loss_probability",
        "collision_probability", "jain_fairness_index",
        "total_energy_uj", "npca_transition_count",
    ]

    for name in BASELINE_NAMES:
        r = run_baseline(name, common, llm_profile)

        assert "aggregate" in r["metrics"] or all(k in r["metrics"] for k in required_keys), (
            f"Baseline '{name}': metrics missing required keys"
        )
        agg = r["metrics"]
        for key in required_keys:
            assert key in agg, f"Baseline '{name}': missing key '{key}' in aggregate metrics"

        assert isinstance(r["reward"], float), (
            f"Baseline '{name}': reward must be float, got {type(r['reward'])}"
        )
        assert "intent_name" in r["profile"], (
            f"Baseline '{name}': profile missing 'intent_name'"
        )
        print(
            f"  [{name}] TP={agg['aggregate_throughput']} "
            f"PDR={agg['packet_delivery_ratio']:.3f} "
            f"Reward={r['reward']:.4f} "
            f"Profile={r['profile']['intent_name']}"
        )

    print("T9 PASS: all 7 baselines completed with required aggregate metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_t1_legacy_edca_no_npca_no_harq_success,
        test_t2_legacy_edca_no_harq_fail,
        test_t3_arq_only_npca_has_transitions_no_harq,
        test_t4_harq_only_no_npca_harq_active,
        test_t5_fixed_cw_npca_harq_both_active,
        test_t6_adaptive_cw_qsrc_recorded,
        test_t7_grid_best_profile_valid_name,
        test_t8_llm_reward_profile_valid,
        test_t9_all_baselines_complete_with_required_metrics,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL [{t.__name__}]: {e}")
        except Exception as e:
            print(f"ERROR [{t.__name__}]: {type(e).__name__}: {e}")
    print(f"\n{'='*50}")
    print(f"Step 8 결과: {passed}/{len(tests)} passed")
