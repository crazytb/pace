"""
W_eff × N 2차원 스윕: qsrc*(N, W_eff) 의존성 검증

목적: qsrc*가 N에만 의존하는지, 아니면 W_eff(OBSS 지속시간)에도 의존하는지 확인.
     OBSS_MIN=OBSS_MAX=D로 고정해 W_eff = D - ppdu_duration을 결정론적으로 제어.

출력:
  results/weff_check/data.csv
  results/weff_check/qsrc_star_heatmap.png  ← qsrc*(N, W_eff) 등고선 지도
  results/weff_check/qsrc_vs_weff.png       ← W_eff별 throughput 곡선 (N 고정)

실행:
  python harq_sim/run_weff_check.py [--fast] [--out-dir results/weff_check]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harq_sim.run_step8 import build_and_run

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

PPDU_DURATION = 20

# OBSS 지속시간 고정값 → W_eff = D - PPDU_DURATION
# OBSS_MIN = OBSS_MAX = D (결정론적 W_eff 제어)
OBSS_D_LIST = [40, 80, 150, 300, 500]   # → W_eff = [20, 60, 130, 280, 480]

NUM_STAS_LIST = [5, 10, 20, 30, 50]
QSRC_LIST     = [0, 1, 2, 3, 4, 5]
SEEDS         = [42, 123, 456]

OBSS_OCCUPANCY = 0.50   # 각 D에서 동일한 채널 점유율 유지

FULL_SLOTS = 30_000
FAST_SLOTS =  5_000


def _weff(D: int) -> int:
    return D - PPDU_DURATION

def _rate(D: int) -> float:
    """고정 OBSS duration D, 목표 occupancy → rate = occ / D"""
    return OBSS_OCCUPANCY / D


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    total = len(OBSS_D_LIST) * len(NUM_STAS_LIST) * len(QSRC_LIST) * len(SEEDS)
    done  = 0

    for D in OBSS_D_LIST:
        rate = _rate(D)
        weff = _weff(D)
        for num_stas in NUM_STAS_LIST:
            for qsrc in QSRC_LIST:
                for seed in SEEDS:
                    done += 1
                    print(
                        f"  [{done:4d}/{total}] W_eff={weff:3d}  N={num_stas:2d}"
                        f"  q={qsrc}  seed={seed}",
                        flush=True,
                    )
                    sim = build_and_run(
                        num_slots       = num_slots,
                        num_stas        = num_stas,
                        obss_rate       = rate,
                        obss_min        = D,
                        obss_max        = D,   # 고정 → W_eff 결정론적
                        npca_qsrc       = qsrc,
                        npca_threshold  = 0,
                        ppdu_duration   = PPDU_DURATION,
                        harq_horizon    = 200,
                        snr_db_mean     = 20.0,
                        snr_db_std      = 0.0,
                        npca_enabled    = True,
                        harq_enabled    = True,
                        adaptive_cw     = False,
                        seed            = seed,
                        enable_trace    = False,
                    )
                    agg = sim.compute_metrics()["aggregate"]
                    rows.append({
                        "obss_duration":              D,
                        "weff":                       weff,
                        "num_stas":                   num_stas,
                        "npca_qsrc":                  qsrc,
                        "seed":                       seed,
                        "aggregate_throughput":        agg["aggregate_throughput"],
                        "collision_probability_npca":  agg["collision_probability_npca"],
                        "npca_transition_count":       agg["npca_transition_count"],
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["obss_duration", "weff", "num_stas", "npca_qsrc", "seed",
          "aggregate_throughput", "collision_probability_npca",
          "npca_transition_count"]

def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────

def _mean(rows, D, num_stas, qsrc, metric):
    vals = [r[metric] for r in rows
            if r["obss_duration"] == D
            and r["num_stas"] == num_stas
            and r["npca_qsrc"] == qsrc]
    return statistics.mean(vals) if vals else 0.0

def _optimal_qsrc(rows, D, num_stas):
    best_q, best_tp = 0, -1.0
    for q in QSRC_LIST:
        tp = _mean(rows, D, num_stas, q, "aggregate_throughput")
        if tp > best_tp:
            best_tp, best_q = tp, q
    return best_q, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

WEFF_LIST = [_weff(D) for D in OBSS_D_LIST]

def plot_all(rows: list[dict], out_dir: str) -> None:
    _plot_heatmap(rows, out_dir)
    _plot_curves(rows, out_dir)


def _plot_heatmap(rows: list[dict], out_dir: str) -> None:
    """qsrc*(N, W_eff) 등고선 지도"""
    # 행: N (y), 열: W_eff (x)
    grid = np.zeros((len(NUM_STAS_LIST), len(OBSS_D_LIST)), dtype=float)
    for i, n in enumerate(NUM_STAS_LIST):
        for j, D in enumerate(OBSS_D_LIST):
            q_opt, _ = _optimal_qsrc(rows, D, n)
            grid[i, j] = q_opt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd",
                   origin="lower", vmin=0, vmax=5)

    ax.set_xticks(range(len(OBSS_D_LIST)))
    ax.set_xticklabels([f"W={w}" for w in WEFF_LIST], fontsize=9)
    ax.set_yticks(range(len(NUM_STAS_LIST)))
    ax.set_yticklabels([f"N={n}" for n in NUM_STAS_LIST], fontsize=9)
    ax.set_xlabel("Effective NPCA Window W_eff (slots)", fontsize=10)
    ax.set_ylabel("Number of STAs", fontsize=10)

    # 셀에 qsrc* 숫자 표기
    for i in range(len(NUM_STAS_LIST)):
        for j in range(len(OBSS_D_LIST)):
            val = int(grid[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val >= 3 else "black")

    plt.colorbar(im, ax=ax, label="Optimal qsrc*")
    ax.set_title("qsrc*(N, W_eff) — Optimal NPCA Initial CW Exponent\n"
                 "(OBSS occ=50%, SNR=20dB, PPDU=20 slots)", fontsize=11)
    fig.tight_layout()

    path = os.path.join(out_dir, "qsrc_star_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Heatmap → {path}")
    plt.close(fig)


def _plot_curves(rows: list[dict], out_dir: str) -> None:
    """W_eff별 throughput vs qsrc 곡선 — N=20 고정"""
    N_FIXED = 20
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): Throughput vs qsrc, lines per W_eff
    ax = axes[0]
    for idx, D in enumerate(OBSS_D_LIST):
        means = [_mean(rows, D, N_FIXED, q, "aggregate_throughput") for q in QSRC_LIST]
        q_opt, _ = _optimal_qsrc(rows, D, N_FIXED)
        ax.plot(QSRC_LIST, means, color=colors[idx], marker="o", lw=1.8,
                label=f"W_eff={_weff(D)}")
        ax.plot(q_opt, means[q_opt], marker="*", color=colors[idx],
                markersize=14, zorder=5)

    ax.set_xlabel("qsrc", fontsize=10)
    ax.set_ylabel("Aggregate Throughput", fontsize=10)
    ax.set_xticks(QSRC_LIST)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=9, title="W_eff (slots)")
    ax.set_title(f"(a) Throughput vs qsrc  [N={N_FIXED}]\n★ = qsrc*", fontsize=10)

    # Panel (b): qsrc*(W_eff) per N
    ax = axes[1]
    n_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#9467bd"]
    for idx, n in enumerate(NUM_STAS_LIST):
        qsrcs = [_optimal_qsrc(rows, D, n)[0] for D in OBSS_D_LIST]
        ax.plot(WEFF_LIST, qsrcs, color=n_colors[idx], marker="o", lw=1.8,
                label=f"N={n}")

    ax.set_xlabel("W_eff (slots)", fontsize=10)
    ax.set_ylabel("Optimal qsrc*", fontsize=10)
    ax.set_yticks(QSRC_LIST)
    ax.set_yticklabels([f"q={q} (CW={2**q*16-1})" for q in QSRC_LIST], fontsize=8)
    ax.set_xticks(WEFF_LIST)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=9)
    ax.set_title("(b) qsrc*(W_eff) per N — W_eff 의존성 확인", fontsize=10)

    fig.suptitle("qsrc*(N, W_eff) 검증: NPCA W_eff 의존성\n"
                 "(OBSS_MIN=OBSS_MAX, occ=50%, SNR=20dB)", fontsize=11)
    fig.tight_layout()

    path = os.path.join(out_dir, "qsrc_vs_weff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Curves → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    print("\n=== qsrc*(N, W_eff) 요약 ===")
    header = "N\\W_eff" + "".join(f"  W={w:4d}" for w in WEFF_LIST)
    print(header)
    print("-" * len(header))
    for n in NUM_STAS_LIST:
        row_str = f"N={n:2d}  "
        for D in OBSS_D_LIST:
            q_opt, _ = _optimal_qsrc(rows, D, n)
            row_str += f"  q*={q_opt}      "
        print(row_str)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="W_eff × N 2D sweep: qsrc*(N, W_eff) 의존성 검증")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/weff_check")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode  = "FAST" if args.fast else "FULL"
    total = len(OBSS_D_LIST) * len(NUM_STAS_LIST) * len(QSRC_LIST) * len(SEEDS)
    print(f"=== W_eff × N qsrc* 검증 [{mode}] — {num_slots} slots ===")
    print(f"    W_eff: {WEFF_LIST}")
    print(f"    N:     {NUM_STAS_LIST}")
    print(f"    qsrc:  {QSRC_LIST}")
    print(f"    Total: {len(OBSS_D_LIST)}×{len(NUM_STAS_LIST)}×{len(QSRC_LIST)}×{len(SEEDS)} = {total} runs")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print_summary(rows)
    print("\nPlotting...")
    plot_all(rows, out_dir)

    print(f"\n완료: {out_dir}/")


if __name__ == "__main__":
    main()
