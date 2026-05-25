"""
Step 6 시뮬레이션 실행 스크립트 — Reward Module

출력: results/step6/sim_trace.csv
       results/step6/summary.txt

사용법:
  python harq_sim/run_step6.py [OPTIONS]

  --slots         N  : 슬롯 수 (기본 500)
  --stas          N  : STA 수 (기본 3)
  --obss-rate     R  : OBSS 발생 확률 (기본 0.05)
  --obss-min      N  : OBSS 최소 지속 슬롯 (기본 30)
  --obss-max      N  : OBSS 최대 지속 슬롯 (기본 80)
  --qsrc          Q  : NPCA initial qsrc 0~5 (기본 0, CW=15)
  --threshold     T  : NPCA min duration threshold (기본 0)
  --ppdu          N  : PPDU 전송 슬롯 수 (기본 20)
  --snr           V  : 평균 SNR (dB, 기본 25.0)
  --snr-std       V  : SNR 표준편차 (기본 0.0, 0=결정론적)
  --harq-horizon  N  : HARQ buffer validity horizon (슬롯, 기본 200)
  --seed          S  : random seed (기본 42)
  --no-npca          : NPCA 비활성화 (비교용)
  --no-harq          : HARQ 비활성화 → ARQ-only
  --adaptive-cw      : Adaptive CW_npca_init 활성화 (Step 5)
  --intent        I  : reward profile — throughput | delay | qos | fair | energy
                       (기본 throughput)
  --reward-weights W : 커스텀 가중치 오버라이드, 예: 'throughput=0.5,energy=0.3,...'
  --out-dir      PATH : 출력 디렉토리 (기본 results/step6)
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.phy import select_mcs
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.reward import INTENT_PROFILES, normalize_metrics, compute_reward
from harq_sim.sta import STA
from harq_sim.simulator import Simulator

_INTENT_ALIASES = {
    "throughput": "throughput",
    "delay":      "delay_sensitive",
    "qos":        "qos_aware",
    "fair":       "fair_coexistence",
    "energy":     "energy_aware",
}


def _parse_weights(raw: str) -> dict:
    """Parse 'k=v,k=v,...' string into float dict."""
    weights = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        weights[k.strip()] = float(v.strip())
    return weights


def build_and_run(
    num_slots:      int,
    num_stas:       int,
    obss_rate:      float,
    obss_min:       int,
    obss_max:       int,
    npca_qsrc:      int,
    npca_threshold: int,
    ppdu_duration:  int,
    npca_enabled:   bool,
    harq_enabled:   bool,
    harq_horizon:   int,
    snr_db_mean:    float,
    snr_db_std:     float,
    adaptive_cw:    bool,
    seed:           int,
) -> Simulator:
    random.seed(seed)

    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(obss_min, obss_max),
    )
    npca = Channel(channel_id=1, obss_generation_rate=0.0)

    policy = NPCAHARQPolicy(adaptive_cw=adaptive_cw) if npca_enabled else None

    stas = [
        STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca,
            npca_enabled=npca_enabled,
            ppdu_duration=ppdu_duration,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=npca_threshold,
            npca_initial_qsrc=npca_qsrc,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=snr_db_mean,
            snr_db_std=snr_db_std,
            harq_enabled=harq_enabled,
            harq_validity_horizon=harq_horizon,
            policy=policy,
            adaptive_cw=adaptive_cw,
        )
        for i in range(num_stas)
    ]

    sim = Simulator(num_slots=num_slots, stas=stas, channels=[primary, npca], enable_trace=True)
    sim.run()
    return sim


def print_summary(
    sim:     Simulator,
    args:    argparse.Namespace,
    profile: dict,
    weights: dict,
) -> str:
    all_metrics = sim.compute_metrics()
    agg         = all_metrics["aggregate"]
    norm        = normalize_metrics(agg)
    reward      = compute_reward(
        norm, weights, profile.get("constraints"), agg,
    )
    mcs_at_snr  = select_mcs(args.snr)
    intent_name = _INTENT_ALIASES.get(args.intent, args.intent)
    cw_mode     = "adaptive (Step 5)" if args.adaptive_cw else "fixed qsrc"

    lines = []
    lines.append("=" * 110)
    lines.append("Step 6 Simulation Summary — Reward Module")
    lines.append("=" * 110)
    lines.append(f"  Slots         : {args.slots}")
    lines.append(f"  STAs          : {args.stas}")
    lines.append(f"  OBSS rate     : {args.obss_rate}")
    lines.append(f"  OBSS duration : {args.obss_min}~{args.obss_max} slots")
    lines.append(f"  PPDU duration : {args.ppdu} slots")
    lines.append(f"  NPCA enabled  : {not args.no_npca}")
    lines.append(f"  HARQ enabled  : {not args.no_harq}")
    lines.append(f"  HARQ horizon  : {args.harq_horizon} slots")
    lines.append(f"  CW mode       : {cw_mode}")
    lines.append(f"  SNR (mean)    : {args.snr:.1f} dB  → MCS {mcs_at_snr}")
    lines.append(f"  SNR std dev   : {args.snr_std:.1f} dB")
    lines.append(f"  Random seed   : {args.seed}")
    lines.append(f"  Intent        : {intent_name}")
    lines.append("")

    lines.append(
        f"{'STA':>4} | {'P_succ':>7} {'P_fail':>7} {'N_succ':>7} {'N_fail':>7} "
        f"{'HARQ_ok':>8} {'PDR':>6} {'Col_p':>7} {'Energy_uJ':>10}"
    )
    lines.append("-" * 80)
    for sta_id, m in all_metrics.items():
        if sta_id == "aggregate":
            continue
        lines.append(
            f"{sta_id:>4} | "
            f"{m['primary_tx_success']:>7} "
            f"{m['primary_tx_fail']:>7} "
            f"{m['npca_tx_success']:>7} "
            f"{m['npca_tx_fail']:>7} "
            f"{m['harq_tx_success']:>8} "
            f"{m['pdr']:>6.3f} "
            f"{m['collision_prob']:>7.3f} "
            f"{m['total_energy_uj']:>10.1f}"
        )
    lines.append("")
    lines.append("Aggregate metrics:")
    lines.append(f"  Total delivered   : {agg['aggregate_throughput']}")
    lines.append(f"  Mean delay        : {agg['mean_access_delay']:.1f} slots")
    lines.append(f"  P95 delay         : {agg['p95_access_delay']:.1f} slots")
    lines.append(f"  P99 delay         : {agg['p99_access_delay']:.1f} slots")
    lines.append(f"  PDR               : {agg['packet_delivery_ratio']:.4f}")
    lines.append(f"  Packet loss       : {agg['packet_loss_probability']:.4f}")
    lines.append(f"  Col prob (primary): {agg['collision_probability_primary']:.4f}")
    lines.append(f"  Col prob (NPCA)   : {agg['collision_probability_npca']:.4f}")
    lines.append(f"  Jain fairness     : {agg['jain_fairness_index']:.4f}")
    lines.append(f"  Total energy      : {agg['total_energy_uj']:.1f} μJ")
    lines.append(f"  NPCA transitions  : {agg['npca_transition_count']}")
    lines.append("")
    lines.append("Normalized metrics:")
    lines.append(f"  T_hat={norm['T_hat']:.4f}  D_hat={norm['D_hat']:.4f}  "
                 f"D95_hat={norm['D95_hat']:.4f}  loss_hat={norm['loss_hat']:.4f}")
    lines.append(f"  col_hat={norm['collision_hat']:.4f}  fair_hat={norm['fairness_hat']:.4f}  "
                 f"E_hat={norm['energy_hat']:.4f}  leg_hat={norm['legacy_hat']:.4f}")
    lines.append("")
    lines.append(f"  Reward ({intent_name}): {reward:.6f}")
    lines.append("")

    summary = "\n".join(lines)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HARQ-NPCA Step 6 — Reward Module")
    parser.add_argument("--slots",           type=int,   default=500)
    parser.add_argument("--stas",            type=int,   default=3)
    parser.add_argument("--obss-rate",       type=float, default=0.05)
    parser.add_argument("--obss-min",        type=int,   default=30)
    parser.add_argument("--obss-max",        type=int,   default=80)
    parser.add_argument("--qsrc",            type=int,   default=0)
    parser.add_argument("--threshold",       type=int,   default=0)
    parser.add_argument("--ppdu",            type=int,   default=20)
    parser.add_argument("--snr",             type=float, default=25.0)
    parser.add_argument("--snr-std",         type=float, default=0.0)
    parser.add_argument("--harq-horizon",    type=int,   default=200)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--no-npca",         action="store_true")
    parser.add_argument("--no-harq",         action="store_true")
    parser.add_argument("--adaptive-cw",     action="store_true")
    parser.add_argument("--intent",          type=str,   default="throughput",
                        choices=list(_INTENT_ALIASES.keys()),
                        help="Reward profile intent")
    parser.add_argument("--reward-weights",  type=str,   default="",
                        help="Custom weight overrides: 'throughput=0.5,energy=0.3,...'")
    parser.add_argument("--out-dir",         type=str,   default="results/step6")
    args = parser.parse_args()

    intent_name = _INTENT_ALIASES[args.intent]
    profile     = INTENT_PROFILES[intent_name]
    weights     = dict(profile["weights"])   # copy
    if args.reward_weights:
        overrides = _parse_weights(args.reward_weights)
        weights.update(overrides)

    sim = build_and_run(
        num_slots=args.slots,
        num_stas=args.stas,
        obss_rate=args.obss_rate,
        obss_min=args.obss_min,
        obss_max=args.obss_max,
        npca_qsrc=args.qsrc,
        npca_threshold=args.threshold,
        ppdu_duration=args.ppdu,
        npca_enabled=not args.no_npca,
        harq_enabled=not args.no_harq,
        harq_horizon=args.harq_horizon,
        snr_db_mean=args.snr,
        snr_db_std=args.snr_std,
        adaptive_cw=args.adaptive_cw,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "sim_trace.csv")
    sim.to_csv(csv_path)

    summary = print_summary(sim, args, profile, weights)
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
