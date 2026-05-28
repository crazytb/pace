"""
Figure 8: Cross-Channel HARQ Combining in NPCA Systems (RQ8)

3 methods:
  no_harq       — ARQ only (HARQ disabled)
  same_ch_harq  — HARQ enabled, buffer flushed on channel switch (same-channel only)
  cross_ch_harq — HARQ enabled with cross-channel combining (current behavior)

Sweeps:
  Primary:   snr_db_mean ∈ {8,10,12,14,16,18,20,22,24,26} dB, num_stas=20 fixed
  Secondary: num_stas ∈ {5,10,20,30,50}, snr_db_mean=20 fixed

env: obss_occupancy=50%, obss_max=500, ppdu=20, 50000 slots × 3 seeds

계획: guidelines/step9/fig8.md
출력:
  manuscript/figure/fig8_cross_channel_harq.{eps,png,pdf}
  results/step9/fig8/snr_sweep.csv
  results/step9/fig8/nstas_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harq_sim.channel import Channel
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.sta import STA
from harq_sim.simulator import Simulator

# ─────────────────────────────────────────────────────────────────────────────
# 실험 상수
# ─────────────────────────────────────────────────────────────────────────────

SNR_LIST   = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # dB
NSTAS_LIST = [5, 10, 20, 30, 50]
SEEDS      = [42, 123, 456]

SNR_FIXED   = 20     # secondary sweep (num_stas) 시 고정
NSTAS_FIXED = 20     # primary sweep (SNR) 시 고정

OBSS_OCCUPANCY = 0.50
OBSS_MIN       = 20
OBSS_MAX       = 500
PPDU_DURATION  = 20
HARQ_HORIZON   = 200
NPCA_THRESHOLD = 0
NPCA_QSRC      = 0

FULL_SLOTS = 50_000
FAST_SLOTS = 5_000

# MCS 임계점 (수직 점선용)
MCS_THRESHOLDS = [5, 8, 11, 14, 17, 20, 23, 26]

METHODS = [
    {"name": "no_harq",       "harq_enabled": False, "cross_channel_harq": False,
     "label": "No HARQ (ARQ only)",              "color": "#aaaaaa", "ls": ":"},
    {"name": "same_ch_harq",  "harq_enabled": True,  "cross_channel_harq": False,
     "label": "Same-channel HARQ",               "color": "#ff7f0e", "ls": "--"},
    {"name": "cross_ch_harq", "harq_enabled": True,  "cross_channel_harq": True,
     "label": "Cross-channel HARQ (proposed)",   "color": "#1f77b4", "ls": "-"},
]


def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


# ─────────────────────────────────────────────────────────────────────────────
# Simulation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_and_run(
    num_slots:          int,
    num_stas:           int,
    snr_db_mean:        float,
    harq_enabled:       bool,
    cross_channel_harq: bool,
    seed:               int,
) -> Simulator:
    random.seed(seed)
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)
    policy = NPCAHARQPolicy(adaptive_cw=False)
    stas = [
        STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=PPDU_DURATION,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=NPCA_THRESHOLD,
            npca_initial_qsrc=NPCA_QSRC,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=snr_db_mean,
            snr_db_std=0.0,
            harq_enabled=harq_enabled,
            harq_validity_horizon=HARQ_HORIZON,
            policy=policy,
            adaptive_cw=False,
            ppdu_truncation=True,
            ppdu_min_tx_slots=3,
            cross_channel_harq=cross_channel_harq,
        )
        for i in range(num_stas)
    ]
    sim = Simulator(
        num_slots=num_slots,
        stas=stas,
        channels=[primary, npca_ch],
        enable_trace=False,
    )
    sim.run()
    return sim


def _extract_row(sim: Simulator, method: str, num_stas: int, snr: float, seed: int) -> dict:
    metrics = sim.compute_metrics()
    agg     = metrics["aggregate"]

    # Per-STA aggregate for HARQ stats
    total_phy_errors  = sum(metrics[i].get("phy_error_failures", 0) for i in range(len(sim.stas)))
    total_harq_succ   = sum(metrics[i].get("harq_tx_success", 0)    for i in range(len(sim.stas)))
    total_harq_fail   = sum(metrics[i].get("harq_tx_fail", 0)       for i in range(len(sim.stas)))
    total_tx          = sum(
        metrics[i]["primary_tx_success"] + metrics[i]["primary_tx_fail"]
        + metrics[i]["npca_tx_success"]  + metrics[i]["npca_tx_fail"]
        for i in range(len(sim.stas))
    )

    return {
        "method":                method,
        "num_stas":              num_stas,
        "snr_db_mean":           snr,
        "seed":                  seed,
        "aggregate_throughput":  agg["aggregate_throughput"],
        "packet_delivery_ratio": agg["packet_delivery_ratio"],
        "collision_prob_npca":   agg.get("collision_probability_npca", 0.0),
        "npca_transition_count": agg["npca_transition_count"],
        "phy_error_count":       total_phy_errors,
        "phy_error_rate":        total_phy_errors / total_tx if total_tx > 0 else 0.0,
        "harq_success_count":    total_harq_succ,
        "harq_fail_count":       total_harq_fail,
        "total_tx":              total_tx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────────────────────────────────────

def run_snr_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    total = len(SNR_LIST) * len(METHODS) * len(SEEDS)
    done  = 0
    print(f"\n=== SNR Sweep (num_stas={NSTAS_FIXED}) ===")
    for snr in SNR_LIST:
        for method in METHODS:
            for seed in SEEDS:
                done += 1
                sim = build_and_run(
                    num_slots=num_slots,
                    num_stas=NSTAS_FIXED,
                    snr_db_mean=snr,
                    harq_enabled=method["harq_enabled"],
                    cross_channel_harq=method["cross_channel_harq"],
                    seed=seed,
                )
                row = _extract_row(sim, method["name"], NSTAS_FIXED, snr, seed)
                rows.append(row)
                print(
                    f"  [{done:3d}/{total}] snr={snr:2d}dB  {method['name']:<15}"
                    f"  seed={seed}  TP={row['aggregate_throughput']:5d}"
                    f"  phy_err={row['phy_error_rate']:.3f}"
                    f"  harq_succ={row['harq_success_count']:4d}",
                    flush=True,
                )
    return rows


def run_nstas_sweep(num_slots: int, snr_sweep_rows: list[dict]) -> list[dict]:
    """num_stas sweep at fixed SNR=20. Reuse snr_sweep rows where num_stas==NSTAS_FIXED."""
    rows: list[dict] = []
    # Reuse already-computed num_stas=NSTAS_FIXED rows from SNR sweep
    rows.extend([r for r in snr_sweep_rows if r["snr_db_mean"] == SNR_FIXED])

    # Run new rows for other num_stas values
    new_nstas = [n for n in NSTAS_LIST if n != NSTAS_FIXED]
    total = len(new_nstas) * len(METHODS) * len(SEEDS)
    done  = 0
    print(f"\n=== num_stas Sweep (SNR={SNR_FIXED} dB) ===")
    for num_stas in new_nstas:
        for method in METHODS:
            for seed in SEEDS:
                done += 1
                sim = build_and_run(
                    num_slots=num_slots,
                    num_stas=num_stas,
                    snr_db_mean=SNR_FIXED,
                    harq_enabled=method["harq_enabled"],
                    cross_channel_harq=method["cross_channel_harq"],
                    seed=seed,
                )
                row = _extract_row(sim, method["name"], num_stas, SNR_FIXED, seed)
                rows.append(row)
                print(
                    f"  [{done:3d}/{total}] N={num_stas:2d}  {method['name']:<15}"
                    f"  seed={seed}  TP={row['aggregate_throughput']:5d}"
                    f"  phy_err={row['phy_error_rate']:.3f}",
                    flush=True,
                )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    "method", "num_stas", "snr_db_mean", "seed",
    "aggregate_throughput", "packet_delivery_ratio",
    "collision_prob_npca", "npca_transition_count",
    "phy_error_count", "phy_error_rate",
    "harq_success_count", "harq_fail_count", "total_tx",
]


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows, method, x_key, x_val, metric):
    vals = [r[metric] for r in rows if r["method"] == method and r[x_key] == x_val]
    if not vals:
        return 0.0, 0.0
    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(snr_rows: list[dict], nstas_rows: list[dict], fig_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 10), sharex=False)
    fig.subplots_adjust(hspace=0.32)

    # ── Panel (a): Throughput vs SNR ─────────────────────────────────────────
    ax = axes[0]
    for m in METHODS:
        means, stds = zip(*[_stats(snr_rows, m["name"], "snr_db_mean", snr, "aggregate_throughput")
                            for snr in SNR_LIST])
        ax.plot(SNR_LIST, means, color=m["color"], ls=m["ls"], lw=2.0,
                marker="o", markersize=5, label=m["label"])
        ax.fill_between(SNR_LIST,
                        [v - s for v, s in zip(means, stds)],
                        [v + s for v, s in zip(means, stds)],
                        color=m["color"], alpha=0.12)
    for thr in MCS_THRESHOLDS:
        if SNR_LIST[0] <= thr <= SNR_LIST[-1]:
            ax.axvline(thr, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_ylabel("Aggregate Throughput\n(delivered packets)", fontsize=10)
    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_xticks(SNR_LIST)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=9, va="top")

    # ── Panel (b): HARQ Gain (%) vs SNR ──────────────────────────────────────
    ax = axes[1]
    no_harq_tp  = {snr: _stats(snr_rows, "no_harq", "snr_db_mean", snr, "aggregate_throughput")[0]
                   for snr in SNR_LIST}
    for m in METHODS[1:]:  # skip no_harq
        gains = []
        for snr in SNR_LIST:
            base = no_harq_tp[snr]
            tp, _ = _stats(snr_rows, m["name"], "snr_db_mean", snr, "aggregate_throughput")
            gains.append(100.0 * (tp - base) / base if base > 0 else 0.0)
        ax.plot(SNR_LIST, gains, color=m["color"], ls=m["ls"], lw=2.0,
                marker="s", markersize=5, label=m["label"])
    ax.axhline(0, color="gray", lw=0.8)
    for thr in MCS_THRESHOLDS:
        if SNR_LIST[0] <= thr <= SNR_LIST[-1]:
            ax.axvline(thr, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_ylabel("HARQ Gain vs. No-HARQ (%)", fontsize=10)
    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_xticks(SNR_LIST)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.97, "(b) MCS thresholds: vertical dotted lines",
            transform=ax.transAxes, fontsize=8, va="top")

    # ── Panel (c): Throughput vs num_stas at SNR=20 ───────────────────────────
    ax = axes[2]
    for m in METHODS:
        means, stds = zip(*[_stats(nstas_rows, m["name"], "num_stas", n, "aggregate_throughput")
                            for n in NSTAS_LIST])
        ax.plot(NSTAS_LIST, means, color=m["color"], ls=m["ls"], lw=2.0,
                marker="^", markersize=5, label=m["label"])
        ax.fill_between(NSTAS_LIST,
                        [v - s for v, s in zip(means, stds)],
                        [v + s for v, s in zip(means, stds)],
                        color=m["color"], alpha=0.12)
    ax.set_ylabel("Aggregate Throughput\n(delivered packets)", fontsize=10)
    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_xticks(NSTAS_LIST)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.97, f"(c) SNR = {SNR_FIXED} dB (p_success ≈ 0.5 per attempt)",
            transform=ax.transAxes, fontsize=8, va="top")

    fig.suptitle(
        "Fig. 8  Cross-Channel HARQ Combining in NPCA Systems\n"
        f"(OBSS occupancy={int(OBSS_OCCUPANCY*100)}%, obss_max={OBSS_MAX})",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig8_cross_channel_harq")
    plt.close(fig)


def _save_figure(fig, fig_dir: str, name: str) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ms_fig    = os.path.join(repo_root, "manuscript", "figure")
    os.makedirs(ms_fig, exist_ok=True)
    for ext, kwargs in [
        ("eps", dict(format="eps",  bbox_inches="tight")),
        ("png", dict(format="png",  bbox_inches="tight", dpi=300)),
        ("pdf", dict(format="pdf",  bbox_inches="tight")),
    ]:
        dest = os.path.join(ms_fig, f"{name}.{ext}")
        fig.savefig(dest, **kwargs)
        print(f"  Figure → {dest}")
    preview = os.path.join(fig_dir, f"{name}_preview.png")
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print(f"  Preview → {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 8 — Cross-Channel HARQ Combining in NPCA")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig8")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode = "FAST" if args.fast else "FULL"
    total_snr   = len(SNR_LIST) * len(METHODS) * len(SEEDS)
    new_nstas   = [n for n in NSTAS_LIST if n != NSTAS_FIXED]
    total_nstas = len(new_nstas) * len(METHODS) * len(SEEDS)

    print(f"=== Figure 8 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")
    print(f"    SNR sweep:   {len(SNR_LIST)} SNR pts × {len(METHODS)} methods × {len(SEEDS)} seeds = {total_snr} runs")
    print(f"    N sweep:     {len(new_nstas)} N pts × {len(METHODS)} methods × {len(SEEDS)} seeds = {total_nstas} new runs")
    print(f"    Total runs:  {total_snr + total_nstas} (incl. reuse)")

    snr_rows   = run_snr_sweep(num_slots)
    nstas_rows = run_nstas_sweep(num_slots, snr_rows)

    snr_csv = os.path.join(out_dir, "snr_sweep.csv")
    nst_csv = os.path.join(out_dir, "nstas_sweep.csv")
    save_csv(snr_rows,   snr_csv)
    save_csv(nstas_rows, nst_csv)

    print("\nPlotting...")
    plot(snr_rows, nstas_rows, out_dir)

    # 요약 출력
    print("\n=== Summary (SNR sweep, num_stas=20) ===")
    print(f"{'SNR':>4}  {'no_harq':>8}  {'same_ch':>8}  {'cross_ch':>8}  "
          f"{'same gain%':>10}  {'cross gain%':>11}  {'Δ%':>6}")
    for snr in SNR_LIST:
        tp_no, _   = _stats(snr_rows, "no_harq",       "snr_db_mean", snr, "aggregate_throughput")
        tp_sa, _   = _stats(snr_rows, "same_ch_harq",  "snr_db_mean", snr, "aggregate_throughput")
        tp_cr, _   = _stats(snr_rows, "cross_ch_harq", "snr_db_mean", snr, "aggregate_throughput")
        g_sa = 100.0 * (tp_sa - tp_no) / tp_no if tp_no > 0 else 0.0
        g_cr = 100.0 * (tp_cr - tp_no) / tp_no if tp_no > 0 else 0.0
        print(f"{snr:4d}  {tp_no:8.0f}  {tp_sa:8.0f}  {tp_cr:8.0f}  "
              f"{g_sa:10.2f}%  {g_cr:11.2f}%  {g_cr - g_sa:+.2f}%")

    print(f"\nFigure 8 완료")
    print(f"  SNR 데이터  : {snr_csv}")
    print(f"  N 데이터    : {nst_csv}")
    print(f"  논문용      : manuscript/figure/fig8_cross_channel_harq.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
