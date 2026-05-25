"""
Step 1 시뮬레이션 실행 스크립트

출력: results/step1/sim_trace.csv
       results/step1/summary.txt

사용법:
  python harq_sim/run_step1.py [OPTIONS]

  --slots    N     : 슬롯 수 (기본 500)
  --stas     N     : STA 수 (기본 3)
  --obss-rate R    : OBSS 발생 확률 (기본 0.05)
  --obss-min  N    : OBSS 최소 지속 슬롯 (기본 30)
  --obss-max  N    : OBSS 최대 지속 슬롯 (기본 80)
  --qsrc     Q     : NPCA initial qsrc 0~5 (기본 0, CW=15)
  --threshold T    : NPCA min duration threshold (기본 0)
  --ppdu     N     : PPDU 전송 슬롯 수 (기본 20)
  --seed     S     : random seed (기본 42)
  --no-npca        : NPCA 비활성화 (비교용)
  --out-dir  PATH  : 출력 디렉토리 (기본 results/step1)
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


def build_and_run(
    num_slots: int,
    num_stas: int,
    obss_rate: float,
    obss_min: int,
    obss_max: int,
    npca_qsrc: int,
    npca_threshold: int,
    ppdu_duration: int,
    npca_enabled: bool,
    seed: int,
) -> Simulator:
    random.seed(seed)

    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(obss_min, obss_max),
    )
    npca = Channel(channel_id=1, obss_generation_rate=0.0)

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
        )
        for i in range(num_stas)
    ]

    sim = Simulator(num_slots=num_slots, stas=stas, channels=[primary, npca], enable_trace=True)
    sim.run()
    return sim


def print_summary(sim: Simulator, args: argparse.Namespace) -> str:
    metrics = sim.compute_metrics()
    lines = []
    lines.append("=" * 60)
    lines.append("Step 1 Simulation Summary")
    lines.append("=" * 60)
    lines.append(f"  Slots         : {args.slots}")
    lines.append(f"  STAs          : {args.stas}")
    lines.append(f"  OBSS rate     : {args.obss_rate}")
    lines.append(f"  OBSS duration : {args.obss_min}~{args.obss_max} slots")
    lines.append(f"  PPDU duration : {args.ppdu} slots")
    lines.append(f"  NPCA enabled  : {not args.no_npca}")
    lines.append(f"  NPCA qsrc     : {args.qsrc}  (CW_init = {2**args.qsrc * 16 - 1})")
    lines.append(f"  Min threshold : {args.threshold} slots")
    lines.append(f"  Random seed   : {args.seed}")
    lines.append("")
    lines.append(f"{'STA':>4} | {'P_succ':>7} {'P_fail':>7} {'N_succ':>7} {'N_fail':>7} "
                 f"{'N_trans':>8} {'PDR':>6} {'Col_prob':>9}")
    lines.append("-" * 65)
    for sta_id, m in metrics.items():
        lines.append(
            f"{sta_id:>4} | "
            f"{m['primary_tx_success']:>7} "
            f"{m['primary_tx_fail']:>7} "
            f"{m['npca_tx_success']:>7} "
            f"{m['npca_tx_fail']:>7} "
            f"{m['npca_transitions']:>8} "
            f"{m['pdr']:>6.3f} "
            f"{m['collision_prob']:>9.3f}"
        )
    lines.append("")
    summary = "\n".join(lines)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="HARQ-NPCA Step 1 simulator")
    parser.add_argument("--slots",      type=int,   default=500)
    parser.add_argument("--stas",       type=int,   default=3)
    parser.add_argument("--obss-rate",  type=float, default=0.05)
    parser.add_argument("--obss-min",   type=int,   default=30)
    parser.add_argument("--obss-max",   type=int,   default=80)
    parser.add_argument("--qsrc",       type=int,   default=0)
    parser.add_argument("--threshold",  type=int,   default=0)
    parser.add_argument("--ppdu",       type=int,   default=20)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-npca",    action="store_true")
    parser.add_argument("--out-dir",    type=str,   default="results/step1")
    args = parser.parse_args()

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
