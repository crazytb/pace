"""
Step 7 검증 테스트 — LLM Reward Designer

T1 : LLMRewardDesigner(mock) — "delay" 키워드 → delay_sensitive profile 반환
T2 : LLMRewardDesigner(mock) — "throughput" 키워드 → throughput profile 반환
T3 : LLMRewardDesigner(mock) — "energy" 키워드 → energy_aware profile 반환
T4 : validate_reward_profile() — 유효한 profile → True 반환
T5 : validate_reward_profile() — weights 합 != 1.0 → ValueError 발생
T6 : validate_reward_profile() — 음수 weight → ValueError 발생
T7 : validate_reward_profile() — constraints 키 없음 → ValueError 발생
T8 : delay intent → delay + tail_delay weight 합 > 0.5
T9 : design_reward() profile → compute_reward() 결과가 valid float
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.llm_reward_designer import LLMRewardDesigner, validate_reward_profile
from harq_sim.reward import normalize_metrics, compute_reward, INTENT_PROFILES


# ─────────────────────────────────────────────────────────────────────────────
# T1: delay 키워드 → delay_sensitive
# ─────────────────────────────────────────────────────────────────────────────
def test_t1_mock_delay_intent():
    designer = LLMRewardDesigner(use_mock=True)
    profile  = designer.design_reward("delay sensitive XR traffic")
    assert profile["intent_name"] == "delay_sensitive", (
        f"Expected 'delay_sensitive', got '{profile['intent_name']}'"
    )
    print(f"T1 PASS: delay intent → intent_name='{profile['intent_name']}'")


# ─────────────────────────────────────────────────────────────────────────────
# T2: throughput 키워드 → throughput
# ─────────────────────────────────────────────────────────────────────────────
def test_t2_mock_throughput_intent():
    designer = LLMRewardDesigner(use_mock=True)
    profile  = designer.design_reward("maximize download throughput")
    assert profile["intent_name"] == "throughput", (
        f"Expected 'throughput', got '{profile['intent_name']}'"
    )
    print(f"T2 PASS: throughput intent → intent_name='{profile['intent_name']}'")


# ─────────────────────────────────────────────────────────────────────────────
# T3: energy 키워드 → energy_aware
# ─────────────────────────────────────────────────────────────────────────────
def test_t3_mock_energy_intent():
    designer = LLMRewardDesigner(use_mock=True)
    profile  = designer.design_reward("battery saving low-power device")
    assert profile["intent_name"] == "energy_aware", (
        f"Expected 'energy_aware', got '{profile['intent_name']}'"
    )
    print(f"T3 PASS: energy intent → intent_name='{profile['intent_name']}'")


# ─────────────────────────────────────────────────────────────────────────────
# T4: validate_reward_profile() — valid profile → True
# ─────────────────────────────────────────────────────────────────────────────
def test_t4_validate_valid_profile():
    profile = {
        "intent_name": "throughput",
        "weights": {
            "throughput":        0.45,
            "delay":             0.10,
            "tail_delay":        0.05,
            "packet_loss":       0.10,
            "collision":         0.10,
            "fairness":          0.10,
            "energy":            0.05,
            "legacy_protection": 0.05,
        },
        "constraints": {
            "packet_loss_max":        0.10,
            "p95_delay_max":          500.0,
            "legacy_degradation_max": 0.30,
        },
    }
    result = validate_reward_profile(profile)
    assert result is True, "validate_reward_profile() should return True for valid profile"
    print("T4 PASS: validate_reward_profile() returns True for valid profile")


# ─────────────────────────────────────────────────────────────────────────────
# T5: weights 합 != 1.0 → ValueError
# ─────────────────────────────────────────────────────────────────────────────
def test_t5_validate_weights_sum_not_one():
    profile = {
        "weights": {
            "throughput":  0.5,
            "delay":       0.5,
            "tail_delay":  0.5,   # sum = 1.5 ≠ 1.0
        },
        "constraints": {},
    }
    try:
        validate_reward_profile(profile)
        assert False, "Expected ValueError for weights sum != 1.0"
    except ValueError as e:
        print(f"T5 PASS: weights sum != 1.0 → ValueError: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# T6: 음수 weight → ValueError
# ─────────────────────────────────────────────────────────────────────────────
def test_t6_validate_negative_weight():
    profile = {
        "weights": {
            "throughput":  1.1,
            "delay":       -0.1,  # negative
        },
        "constraints": {},
    }
    try:
        validate_reward_profile(profile)
        assert False, "Expected ValueError for negative weight"
    except ValueError as e:
        print(f"T6 PASS: negative weight → ValueError: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# T7: constraints 키 없음 → ValueError
# ─────────────────────────────────────────────────────────────────────────────
def test_t7_validate_missing_constraints():
    profile = {
        "weights": {"throughput": 1.0},
        # no "constraints" key
    }
    try:
        validate_reward_profile(profile)
        assert False, "Expected ValueError for missing 'constraints'"
    except ValueError as e:
        print(f"T7 PASS: missing constraints → ValueError: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# T8: delay intent → delay + tail_delay weight 합 > 0.5
# ─────────────────────────────────────────────────────────────────────────────
def test_t8_delay_intent_high_delay_weight():
    designer = LLMRewardDesigner(use_mock=True)
    profile  = designer.design_reward("real-time video call with low latency")
    w        = profile["weights"]
    delay_total = w.get("delay", 0.0) + w.get("tail_delay", 0.0)
    assert delay_total > 0.5, (
        f"delay+tail_delay = {delay_total:.3f}, expected > 0.5 for delay intent"
    )
    print(f"T8 PASS: delay intent → delay+tail_delay = {delay_total:.3f} > 0.5")


# ─────────────────────────────────────────────────────────────────────────────
# T9: design_reward() → compute_reward() 결과가 valid float
# ─────────────────────────────────────────────────────────────────────────────
def test_t9_design_reward_compute_reward_pipeline():
    designer = LLMRewardDesigner(use_mock=True)

    raw_metrics = {
        "aggregate_throughput":          30,
        "mean_access_delay":             80.0,
        "p95_access_delay":              200.0,
        "packet_loss_probability":       0.05,
        "collision_probability":         0.08,
        "jain_fairness_index":           0.92,
        "legacy_throughput_degradation": 0.05,
        "total_energy_uj":               1500.0,
    }

    for intent_str in [
        "delay sensitive XR",
        "maximize throughput",
        "battery saving",
        "fair coexistence",
        "balanced quality of service",
    ]:
        profile = designer.design_reward(intent_str)
        norm    = normalize_metrics(raw_metrics)
        reward  = compute_reward(
            norm,
            profile["weights"],
            profile.get("constraints"),
            raw_metrics,
        )
        assert isinstance(reward, float), f"compute_reward() must return float, got {type(reward)}"
        print(f"  intent='{intent_str}' → reward={reward:.4f}")

    print("T9 PASS: all intents → design_reward() + compute_reward() pipeline works")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_t1_mock_delay_intent,
        test_t2_mock_throughput_intent,
        test_t3_mock_energy_intent,
        test_t4_validate_valid_profile,
        test_t5_validate_weights_sum_not_one,
        test_t6_validate_negative_weight,
        test_t7_validate_missing_constraints,
        test_t8_delay_intent_high_delay_weight,
        test_t9_design_reward_compute_reward_pipeline,
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
    print(f"Step 7 결과: {passed}/{len(tests)} passed")
