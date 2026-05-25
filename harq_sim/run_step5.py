"""
Step 5 시뮬레이션 실행 스크립트 — Adaptive CW_npca_init

출력: results/step5/sim_trace.csv
       results/step5/summary.txt

사용법:
  python harq_sim/run_step5.py [OPTIONS]

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
  --adaptive-cw      : Adaptive CW_npca_init 활성화 (Step 5 핵심 기능)
  --no-adaptive-cw   : 고정 qsrc 사용 (Step 4 동작, 기본값)
  --out-dir      PATH : 출력 디렉토리 (기본 results/step5)
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.phy import select_mcs
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


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


def print_summary(sim: Simulator, args: argparse.Namespace) -> str:
    metrics = sim.compute_metrics()
    mcs_at_snr = select_mcs(args.snr)
    cw_mode = "adaptive (Step 5)" if args.adaptive_cw else "fixed qsrc (Step 4)"
    lines = []
    lines.append("=" * 100)
    lines.append("Step 5 Simulation Summary — Adaptive CW_npca_init")
    lines.append("=" * 100)
    lines.append(f"  Slots         : {args.slots}")
    lines.append(f"  STAs          : {args.stas}")
    lines.append(f"  OBSS rate     : {args.obss_rate}")
    lines.append(f"  OBSS duration : {args.obss_min}~{args.obss_max} slots")
    lines.append(f"  PPDU duration : {args.ppdu} slots")
    lines.append(f"  NPCA enabled  : {not args.no_npca}")
    lines.append(f"  HARQ enabled  : {not args.no_harq}")
    lines.append(f"  HARQ horizon  : {args.harq_horizon} slots (~{args.harq_horizon * 9 / 1000:.1f} ms)")
    lines.append(f"  CW mode       : {cw_mode}")
    lines.append(f"  NPCA qsrc     : {args.qsrc}  (CW_init = {2**args.qsrc * 16 - 1})")
    lines.append(f"  Min threshold : {args.threshold} slots")
    lines.append(f"  SNR (mean)    : {args.snr:.1f} dB  → MCS {mcs_at_snr}")
    lines.append(f"  SNR std dev   : {args.snr_std:.1f} dB  ({'deterministic' if args.snr_std == 0 else 'variable'})")
    lines.append(f"  Random seed   : {args.seed}")
    lines.append("")
    lines.append(
        f"{'STA':>4} | {'P_succ':>7} {'P_fail':>7} {'N_succ':>7} {'N_fail':>7} "
        f"{'HARQ_ok':>8} {'Fail%':>6} {'PDR':>6} {'Col_p':>7} "
        f"{'Pol_N':>6} {'Pol_P':>6} {'AvgQsrc':>8}"
    )
    lines.append("-" * 105)
    for sta_id, m in metrics.items():
        sta = sim.stas[sta_id]
        npca_total = m["npca_tx_success"] + m["npca_tx_fail"]
        npca_fail_rate = m["npca_tx_fail"] / npca_total if npca_total > 0 else 0.0
        avg_q = m["avg_npca_qsrc"]
        avg_q_str = f"{avg_q:.2f}" if avg_q is not None else "  N/A"
        lines.append(
            f"{sta_id:>4} | "
            f"{m['primary_tx_success']:>7} "
            f"{m['primary_tx_fail']:>7} "
            f"{m['npca_tx_success']:>7} "
            f"{m['npca_tx_fail']:>7} "
            f"{m['harq_tx_success']:>8} "
            f"{npca_fail_rate:>6.3f} "
            f"{m['pdr']:>6.3f} "
            f"{m['collision_prob']:>7.3f} "
            f"{m['policy_npca_chosen']:>6} "
            f"{m['policy_primary_chosen']:>6} "
            f"{avg_q_str:>8}"
        )
    lines.append("")
    summary = "\n".join(lines)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HARQ-NPCA Step 5 — Adaptive CW_npca_init")
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
    parser.add_argument("--adaptive-cw",     action="store_true",
                        help="Enable adaptive CW_npca_init (Step 5 feature)")
    parser.add_argument("--no-adaptive-cw",  action="store_true",
                        help="Force fixed qsrc (Step 4 behavior, default)")
    parser.add_argument("--out-dir",         type=str,   default="results/step5")
    args = parser.parse_args()

    use_adaptive = args.adaptive_cw and not args.no_adaptive_cw

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
        adaptive_cw=use_adaptive,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "sim_trace.csv")
    sim.to_csv(csv_path)

    summary = print_summary(sim, args)
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
