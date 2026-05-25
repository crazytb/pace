"""
Step 8 시뮬레이션 실행 스크립트 — 7개 Baseline 비교

출력: results/step8/
       comparison.csv       ← 7 baselines × 주요 지표 행
       summary.txt          ← 텍스트 비교 테이블

사용법:
  python harq_sim/run_step8.py [OPTIONS]

  --slots         N  : 슬롯 수 (기본 500)
  --stas          N  : STA 수 (기본 3)
  --obss-rate     R  : OBSS 발생 확률 (기본 0.05)
  --obss-min      N  : OBSS 최소 지속 슬롯 (기본 30)
  --obss-max      N  : OBSS 최대 지속 슬롯 (기본 80)
  --qsrc          Q  : NPCA initial qsrc 0~5 (기본 0)
  --threshold     T  : NPCA min duration threshold (기본 0)
  --ppdu          N  : PPDU 전송 슬롯 수 (기본 20)
  --snr           V  : 평균 SNR (dB, 기본 25.0)
  --snr-std       V  : SNR 표준편차 (기본 0.0)
  --harq-horizon  N  : HARQ buffer validity horizon (슬롯, 기본 200)
  --seed          S  : random seed (기본 42)
  --intent        S  : reward 계산 기준 intent (기본 "balanced throughput and latency")
  --mock             : LLM API 미사용 (기본)
  --no-mock          : 실제 Anthropic API 사용 (ANTHROPIC_API_KEY 필요)
  --no-trace         : CSV per-slot trace 비활성화 (속도↑)
  --out-dir      PATH : 출력 디렉토리 (기본 results/step8)
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.llm_reward_designer import LLMRewardDesigner, validate_reward_profile
from harq_sim.phy import select_mcs
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.reward import INTENT_PROFILES, normalize_metrics, compute_reward
from harq_sim.sta import STA
from harq_sim.simulator import Simulator


# ── 7 Baseline 메타데이터 ──────────────────────────────────────────────────────

BASELINE_NAMES = [
    "legacy_edca",
    "arq_only_npca",
    "harq_only",
    "fixed_cw_npca_harq",
    "adaptive_cw_npca_harq",
    "llm_reward_npca_harq",
    "grid_best_reward_npca_harq",
]

BASELINE_DESCRIPTIONS = {
    "legacy_edca":                "Legacy EDCA (NPCA 없음, HARQ 없음)",
    "arq_only_npca":              "ARQ-only NPCA (NPCA 있음, HARQ 없음)",
    "harq_only":                  "HARQ-only (NPCA 없음, HARQ 있음)",
    "fixed_cw_npca_harq":         "Fixed-CW NPCA-HARQ",
    "adaptive_cw_npca_harq":      "Adaptive-CW NPCA-HARQ",
    "llm_reward_npca_harq":       "LLM-reward NPCA-HARQ",
    "grid_best_reward_npca_harq": "Grid-best reward NPCA-HARQ",
}


# ── Core simulation builder (guidelines §28) ──────────────────────────────────

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
    enable_trace:   bool = True,
) -> Simulator:
    """Build channels + STAs, run simulation, return Simulator."""
    random.seed(seed)

    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(obss_min, obss_max),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)

    policy = NPCAHARQPolicy(adaptive_cw=adaptive_cw) if npca_enabled else None

    stas = [
        STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca_ch,
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

    sim = Simulator(
        num_slots=num_slots,
        stas=stas,
        channels=[primary, npca_ch],
        enable_trace=enable_trace,
    )
    sim.run()
    return sim


# ── Grid-best reward (guidelines §23 항목 7) ──────────────────────────────────

def grid_best_reward_profile(sim: Simulator) -> tuple:
    """Find the INTENT_PROFILE with highest reward for given simulation result.

    Returns
    -------
    tuple of (profile_name: str, profile: dict, best_reward: float)
    """
    agg  = sim.compute_metrics()["aggregate"]
    norm = normalize_metrics(agg)

    best_name   = None
    best_reward = -float("inf")
    for name, profile in INTENT_PROFILES.items():
        r = compute_reward(
            norm,
            profile["weights"],
            profile.get("constraints"),
            agg,
        )
        if r > best_reward:
            best_reward = r
            best_name   = name

    return best_name, INTENT_PROFILES[best_name], best_reward


# ── Per-baseline runner ────────────────────────────────────────────────────────

def run_baseline(
    name:          str,
    common_kwargs: dict,
    reward_profile: dict,
) -> dict:
    """Run one baseline configuration and return a results dict.

    Parameters
    ----------
    name : str
        One of BASELINE_NAMES.
    common_kwargs : dict
        Shared simulation parameters (slots, stas, snr, …).
    reward_profile : dict
        {intent_name, weights, constraints} — used only for reward computation.

    Returns
    -------
    dict with keys: name, description, metrics (aggregate), reward, profile
    """
    npca_enabled = name not in ("legacy_edca", "harq_only")
    harq_enabled = name not in ("legacy_edca", "arq_only_npca")
    adaptive_cw  = name in ("adaptive_cw_npca_harq", "llm_reward_npca_harq",
                             "grid_best_reward_npca_harq")

    sim = build_and_run(
        npca_enabled=npca_enabled,
        harq_enabled=harq_enabled,
        adaptive_cw=adaptive_cw,
        **common_kwargs,
    )

    metrics = sim.compute_metrics()
    agg     = metrics["aggregate"]

    if name == "grid_best_reward_npca_harq":
        grid_name, grid_profile, reward = grid_best_reward_profile(sim)
        profile_used = {**grid_profile, "intent_name": grid_name}
    else:
        norm    = normalize_metrics(agg)
        reward  = compute_reward(
            norm,
            reward_profile["weights"],
            reward_profile.get("constraints"),
            agg,
        )
        profile_used = reward_profile

    return {
        "name":        name,
        "description": BASELINE_DESCRIPTIONS[name],
        "metrics":     agg,
        "per_sta":     {k: v for k, v in metrics.items() if k != "aggregate"},
        "reward":      reward,
        "profile":     profile_used,
    }


# ── Summary formatter ─────────────────────────────────────────────────────────

def print_summary(results: list[dict], args: argparse.Namespace, mcs_at_snr: int) -> str:
    lines = []
    lines.append("=" * 120)
    lines.append("Step 8 Baseline Comparison Summary")
    lines.append("=" * 120)
    lines.append(f"  Slots         : {args.slots}")
    lines.append(f"  STAs          : {args.stas}")
    lines.append(f"  OBSS rate     : {args.obss_rate}")
    lines.append(f"  OBSS duration : {args.obss_min}~{args.obss_max} slots")
    lines.append(f"  PPDU duration : {args.ppdu} slots")
    lines.append(f"  SNR (mean)    : {args.snr:.1f} dB  → MCS {mcs_at_snr}")
    lines.append(f"  SNR std dev   : {args.snr_std:.1f} dB")
    lines.append(f"  Random seed   : {args.seed}")
    lines.append("")

    # Header
    col_w = 30
    hdr = (
        f"{'Baseline':<{col_w}} | {'TP':>5} {'PDR':>6} {'Delay':>7} {'P95':>7} "
        f"{'ColP':>6} {'Fair':>6} {'E(μJ)':>8} {'Ntr':>5} {'Reward':>8} {'Profile'}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in results:
        m = r["metrics"]
        row = (
            f"{r['name']:<{col_w}} | "
            f"{m['aggregate_throughput']:>5} "
            f"{m['packet_delivery_ratio']:>6.3f} "
            f"{m['mean_access_delay']:>7.1f} "
            f"{m['p95_access_delay']:>7.1f} "
            f"{m['collision_probability']:>6.3f} "
            f"{m['jain_fairness_index']:>6.3f} "
            f"{m['total_energy_uj']:>8.1f} "
            f"{m['npca_transition_count']:>5} "
            f"{r['reward']:>8.4f} "
            f"{r['profile'].get('intent_name', 'N/A')}"
        )
        lines.append(row)

    lines.append("")
    lines.append("Columns: TP=packets delivered, PDR, Delay=mean slots, P95=p95 slots,")
    lines.append("         ColP=collision prob, Fair=Jain fairness, E=total energy,")
    lines.append("         Ntr=NPCA transitions, Reward=profile score")

    summary = "\n".join(lines)
    print(summary)
    return summary


# ── CSV comparison writer ─────────────────────────────────────────────────────

def save_comparison_csv(results: list[dict], path: str) -> None:
    if not results:
        return

    fieldnames = [
        "baseline", "description",
        "aggregate_throughput", "packet_delivery_ratio",
        "packet_loss_probability",
        "mean_access_delay", "p95_access_delay", "p99_access_delay",
        "collision_probability", "collision_probability_primary", "collision_probability_npca",
        "jain_fairness_index", "total_energy_uj",
        "npca_transition_count", "npca_transition_rate",
        "reward", "intent_name",
    ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            m = r["metrics"]
            writer.writerow({
                "baseline":                        r["name"],
                "description":                     r["description"],
                "aggregate_throughput":             m["aggregate_throughput"],
                "packet_delivery_ratio":            round(m["packet_delivery_ratio"], 6),
                "packet_loss_probability":          round(m["packet_loss_probability"], 6),
                "mean_access_delay":                round(m["mean_access_delay"], 3),
                "p95_access_delay":                 round(m["p95_access_delay"], 3),
                "p99_access_delay":                 round(m["p99_access_delay"], 3),
                "collision_probability":            round(m["collision_probability"], 6),
                "collision_probability_primary":    round(m["collision_probability_primary"], 6),
                "collision_probability_npca":       round(m["collision_probability_npca"], 6),
                "jain_fairness_index":              round(m["jain_fairness_index"], 6),
                "total_energy_uj":                  round(m["total_energy_uj"], 3),
                "npca_transition_count":            m["npca_transition_count"],
                "npca_transition_rate":             round(m["npca_transition_rate"], 6),
                "reward":                           round(r["reward"], 6),
                "intent_name":                      r["profile"].get("intent_name", ""),
            })
    print(f"Comparison CSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HARQ-NPCA Step 8 — Baseline Comparison")
    parser.add_argument("--slots",         type=int,   default=500)
    parser.add_argument("--stas",          type=int,   default=3)
    parser.add_argument("--obss-rate",     type=float, default=0.05)
    parser.add_argument("--obss-min",      type=int,   default=30)
    parser.add_argument("--obss-max",      type=int,   default=80)
    parser.add_argument("--qsrc",          type=int,   default=0)
    parser.add_argument("--threshold",     type=int,   default=0)
    parser.add_argument("--ppdu",          type=int,   default=20)
    parser.add_argument("--snr",           type=float, default=25.0)
    parser.add_argument("--snr-std",       type=float, default=0.0)
    parser.add_argument("--harq-horizon",  type=int,   default=200)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--intent",        type=str,
                        default="balanced throughput and latency")
    parser.add_argument("--mock",          dest="no_mock", action="store_false",
                        default=True,
                        help="Use mock LLM (default, no API key needed)")
    parser.add_argument("--no-mock",       dest="no_mock", action="store_true",
                        help="Use real Anthropic API (ANTHROPIC_API_KEY required)")
    parser.add_argument("--no-trace",      action="store_true",
                        help="Disable per-slot CSV trace (faster)")
    parser.add_argument("--out-dir",       type=str, default="results/step8")
    args = parser.parse_args()

    # ── LLM reward profile (used for LLM-reward baseline; others use same profile too) ──
    use_mock = not args.no_mock
    designer = LLMRewardDesigner(use_mock=use_mock)
    print(f"[LLM] Designing reward for intent: \"{args.intent}\" "
          f"({'mock' if use_mock else 'API'})")
    llm_profile = designer.design_reward(args.intent)
    print(f"[LLM] Profile → {llm_profile['intent_name']}")

    # ── Shared kwargs for build_and_run ───────────────────────────────────────
    common = dict(
        num_slots=args.slots,
        num_stas=args.stas,
        obss_rate=args.obss_rate,
        obss_min=args.obss_min,
        obss_max=args.obss_max,
        npca_qsrc=args.qsrc,
        npca_threshold=args.threshold,
        ppdu_duration=args.ppdu,
        harq_horizon=args.harq_horizon,
        snr_db_mean=args.snr,
        snr_db_std=args.snr_std,
        seed=args.seed,
        enable_trace=not args.no_trace,
    )

    # ── Run all 7 baselines ───────────────────────────────────────────────────
    results = []
    for name in BASELINE_NAMES:
        print(f"  Running [{name}] …", end=" ", flush=True)
        r = run_baseline(name, common, llm_profile)
        results.append(r)
        m = r["metrics"]
        print(
            f"TP={m['aggregate_throughput']} "
            f"PDR={m['packet_delivery_ratio']:.3f} "
            f"Reward={r['reward']:.4f}"
        )

    # ── Output ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    mcs_at_snr  = select_mcs(args.snr)
    summary     = print_summary(results, args, mcs_at_snr)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved → {summary_path}")

    csv_path = os.path.join(args.out_dir, "comparison.csv")
    save_comparison_csv(results, csv_path)

    profile_path = os.path.join(args.out_dir, "llm_profile.json")
    with open(profile_path, "w") as f:
        json.dump(llm_profile, f, indent=2)
    print(f"LLM profile saved → {profile_path}")


if __name__ == "__main__":
    main()
