"""
Step 6 검증 테스트 — Reward Module

T1 : INTENT_PROFILES — 모든 프로파일 weights 합 == 1.0
T2 : throughput 프로파일 — 'throughput' 키 가중치가 최대
T3 : delay_sensitive 프로파일 — 'delay' 키 가중치가 최대
T4 : normalize_metrics() — 모든 출력 값 ∈ [0, 1]
T5 : normalize_metrics() — zero metrics 입력 시 적절한 경계값 반환
T6 : compute_reward() — constraint 위반 시 reward 감소
T7 : compute_reward() — throughput intent > fair_coexistence intent (처리량 높은 케이스)
T8 : compute_metrics() — 시뮬레이션 후 "aggregate" 키 존재 + 필수 필드 검증
T9 : energy tracking — 시뮬레이션 후 sta.total_energy_uj > 0
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.reward import INTENT_PROFILES, DEFAULT_REFS, normalize_metrics, compute_reward
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inject_obss(channel: Channel, slot: int, duration: int) -> None:
    channel.obss_traffic.append((f"test_obss_s{slot}", slot, duration, -1))
    channel.update(slot)


def _make_sim(num_slots: int = 300, num_stas: int = 2, seed: int = 0) -> Simulator:
    random.seed(seed)
    primary = Channel(channel_id=0, obss_generation_rate=0.05,
                      obss_duration_range=(30, 80))
    npca    = Channel(channel_id=1, obss_generation_rate=0.0)
    policy  = NPCAHARQPolicy(adaptive_cw=False)
    stas    = [
        STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca,
            npca_enabled=True,
            ppdu_duration=20,
            switching_delay=1,
            switch_back_delay=1,
            npca_initial_qsrc=0,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=25.0,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            policy=policy,
        )
        for i in range(num_stas)
    ]
    sim = Simulator(num_slots=num_slots, stas=stas, channels=[primary, npca],
                    enable_trace=True)
    sim.run()
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# T1: 모든 프로파일 weights 합 == 1.0
# ─────────────────────────────────────────────────────────────────────────────
def test_t1_intent_weights_sum_to_one():
    for name, profile in INTENT_PROFILES.items():
        total = sum(profile["weights"].values())
        assert abs(total - 1.0) < 1e-9, (
            f"Profile '{name}': weights sum = {total:.6f}, expected 1.0"
        )
    print("T1 PASS: all INTENT_PROFILES weights sum to 1.0")


# ─────────────────────────────────────────────────────────────────────────────
# T2: throughput 프로파일 — 'throughput' 키가 최대 가중치
# ─────────────────────────────────────────────────────────────────────────────
def test_t2_throughput_weight_is_max():
    w = INTENT_PROFILES["throughput"]["weights"]
    max_key = max(w, key=lambda k: w[k])
    assert max_key == "throughput", (
        f"throughput profile: max weight key = '{max_key}', expected 'throughput'"
    )
    print("T2 PASS: throughput intent has 'throughput' as max weight")


# ─────────────────────────────────────────────────────────────────────────────
# T3: delay_sensitive 프로파일 — 'delay' 키가 최대 가중치
# ─────────────────────────────────────────────────────────────────────────────
def test_t3_delay_weight_is_max():
    w = INTENT_PROFILES["delay_sensitive"]["weights"]
    max_key = max(w, key=lambda k: w[k])
    assert max_key == "delay", (
        f"delay_sensitive profile: max weight key = '{max_key}', expected 'delay'"
    )
    print("T3 PASS: delay_sensitive intent has 'delay' as max weight")


# ─────────────────────────────────────────────────────────────────────────────
# T4: normalize_metrics() — 모든 출력 ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def test_t4_normalize_in_unit_range():
    raw = {
        "aggregate_throughput":          30,
        "mean_access_delay":             80.0,
        "p95_access_delay":              200.0,
        "p99_access_delay":              250.0,
        "packet_delivery_ratio":         0.95,
        "packet_loss_probability":       0.05,
        "collision_probability":         0.10,
        "jain_fairness_index":           0.92,
        "legacy_throughput_degradation": 0.05,
        "total_energy_uj":               1500.0,
    }
    norm = normalize_metrics(raw)
    for key, val in norm.items():
        assert 0.0 <= val <= 1.0, (
            f"normalize_metrics(): {key} = {val:.4f} out of [0, 1]"
        )
    print("T4 PASS: normalize_metrics() all values in [0, 1]")


# ─────────────────────────────────────────────────────────────────────────────
# T5: normalize_metrics() — zero 입력 시 경계값 반환
# ─────────────────────────────────────────────────────────────────────────────
def test_t5_normalize_zero_metrics():
    raw = {
        "aggregate_throughput":          0,
        "mean_access_delay":             0.0,
        "p95_access_delay":              0.0,
        "packet_loss_probability":       0.0,
        "collision_probability":         0.0,
        "jain_fairness_index":           1.0,
        "legacy_throughput_degradation": 0.0,
        "total_energy_uj":               0.0,
    }
    norm = normalize_metrics(raw)
    assert norm["T_hat"] == 0.0, "zero throughput → T_hat must be 0.0"
    assert norm["D_hat"] == 1.0, "zero delay → D_hat must be 1.0"
    assert norm["loss_hat"] == 1.0, "zero loss → loss_hat must be 1.0"
    assert norm["fairness_hat"] == 1.0, "max fairness → fairness_hat must be 1.0"
    print("T5 PASS: normalize_metrics() zero-input boundary values correct")


# ─────────────────────────────────────────────────────────────────────────────
# T6: compute_reward() — constraint 위반 시 reward 감소
# ─────────────────────────────────────────────────────────────────────────────
def test_t6_constraint_violation_reduces_reward():
    weights = INTENT_PROFILES["throughput"]["weights"]
    constraints = INTENT_PROFILES["throughput"]["constraints"]

    # 정상 케이스 (loss < max)
    raw_ok = {
        "aggregate_throughput":          40,
        "mean_access_delay":             80.0,
        "p95_access_delay":              200.0,
        "packet_loss_probability":       0.05,  # ≤ 0.10 → no penalty
        "collision_probability":         0.05,
        "jain_fairness_index":           0.90,
        "legacy_throughput_degradation": 0.10,
        "total_energy_uj":               1000.0,
    }
    norm_ok  = normalize_metrics(raw_ok)
    reward_ok = compute_reward(norm_ok, weights, constraints, raw_ok)

    # 위반 케이스 (loss > max)
    raw_viol = dict(raw_ok)
    raw_viol["packet_loss_probability"] = 0.30   # >> 0.10
    norm_viol  = normalize_metrics(raw_viol)
    reward_viol = compute_reward(norm_viol, weights, constraints, raw_viol)

    assert reward_viol < reward_ok, (
        f"Constraint violation should reduce reward: "
        f"ok={reward_ok:.4f}, violated={reward_viol:.4f}"
    )
    print(f"T6 PASS: constraint violation reduces reward "
          f"({reward_ok:.4f} → {reward_viol:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# T7: throughput intent > fair_coexistence intent (처리량 높은 케이스)
# ─────────────────────────────────────────────────────────────────────────────
def test_t7_throughput_intent_higher_than_fairness_for_high_tp():
    raw = {
        "aggregate_throughput":          80,    # high throughput
        "mean_access_delay":             50.0,
        "p95_access_delay":              120.0,
        "packet_loss_probability":       0.02,
        "collision_probability":         0.03,
        "jain_fairness_index":           0.60,  # low fairness
        "legacy_throughput_degradation": 0.05,
        "total_energy_uj":               2000.0,
    }
    norm = normalize_metrics(raw, refs={"throughput_ref": 50})

    w_tp   = INTENT_PROFILES["throughput"]["weights"]
    w_fair = INTENT_PROFILES["fair_coexistence"]["weights"]

    r_tp   = compute_reward(norm, w_tp)
    r_fair = compute_reward(norm, w_fair)

    assert r_tp > r_fair, (
        f"High-throughput, low-fairness scenario: "
        f"throughput reward ({r_tp:.4f}) should exceed "
        f"fair_coexistence reward ({r_fair:.4f})"
    )
    print(f"T7 PASS: throughput intent ({r_tp:.4f}) > "
          f"fair_coexistence intent ({r_fair:.4f}) for high-TP scenario")


# ─────────────────────────────────────────────────────────────────────────────
# T8: compute_metrics() aggregate 키 존재 + 필수 필드 검증
# ─────────────────────────────────────────────────────────────────────────────
def test_t8_aggregate_metrics_present():
    sim = _make_sim()
    metrics = sim.compute_metrics()

    assert "aggregate" in metrics, "compute_metrics() must return an 'aggregate' key"

    agg = metrics["aggregate"]
    required = [
        "aggregate_throughput", "mean_access_delay", "p95_access_delay",
        "packet_delivery_ratio", "packet_loss_probability",
        "collision_probability", "collision_probability_primary",
        "collision_probability_npca", "jain_fairness_index",
        "total_energy_uj", "npca_transition_count",
    ]
    for field in required:
        assert field in agg, f"aggregate missing field '{field}'"

    assert agg["aggregate_throughput"] >= 0
    assert 0.0 <= agg["packet_delivery_ratio"] <= 1.0
    assert 0.0 <= agg["jain_fairness_index"] <= 1.0
    assert agg["total_energy_uj"] > 0.0

    print(f"T8 PASS: aggregate metrics present — "
          f"delivered={agg['aggregate_throughput']}, "
          f"pdr={agg['packet_delivery_ratio']:.3f}, "
          f"energy={agg['total_energy_uj']:.1f} μJ")


# ─────────────────────────────────────────────────────────────────────────────
# T9: energy tracking — sta.total_energy_uj > 0 after simulation
# ─────────────────────────────────────────────────────────────────────────────
def test_t9_energy_tracking():
    sim = _make_sim(num_slots=100)
    for sta in sim.stas:
        assert sta.total_energy_uj > 0.0, (
            f"STA {sta.sta_id}: total_energy_uj = {sta.total_energy_uj:.2f}, expected > 0"
        )
    # Energy should be roughly: 100 slots × (mix of TX/LISTEN) per STA
    # LISTEN: 0.495 μJ/slot — so 100 slots × 0.495 = 49.5 μJ minimum
    for sta in sim.stas:
        assert sta.total_energy_uj >= 49.0, (
            f"STA {sta.sta_id}: energy {sta.total_energy_uj:.1f} μJ seems too low"
        )
    print(f"T9 PASS: energy tracked — "
          f"STA0={sim.stas[0].total_energy_uj:.1f} μJ, "
          f"STA1={sim.stas[1].total_energy_uj:.1f} μJ")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_t1_intent_weights_sum_to_one,
        test_t2_throughput_weight_is_max,
        test_t3_delay_weight_is_max,
        test_t4_normalize_in_unit_range,
        test_t5_normalize_zero_metrics,
        test_t6_constraint_violation_reduces_reward,
        test_t7_throughput_intent_higher_than_fairness_for_high_tp,
        test_t8_aggregate_metrics_present,
        test_t9_energy_tracking,
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
    print(f"Step 6 결과: {passed}/{len(tests)} passed")
