"""
Figure 3: Fixed CW_npca_init(qsrc) × num_stas → 최적 CW 도출 (RQ3)

Guidelines §30 RQ3: fixed small CW_npca_init은 NPCA collision burst를 유발하는가?
확장: num_stas별로 throughput을 최대화하는 최적 qsrc* 궤적 도출.

계획: guidelines/step9/fig3.md
출력:
  manuscript/figure/fig3_qsrc_sweep.{eps,png,pdf}
  results/step9/fig3/data.csv

실행:
  python harq_sim/run_step9_fig3.py [--fast] [--out-dir results/step9/fig3]
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
import matplotlib.ticker as mticker
import numpy as np

from harq_sim.run_step8 import build_and_run

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

QSRC_LIST     = [0, 1, 2, 3, 4, 5]
NUM_STAS_LIST = [5, 10, 20, 30, 50]
SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 500
OBSS_OCCUPANCY = 0.50

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

CW_MIN = 15

# num_stas별 선 색상
_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#9467bd"]

def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))

def _qsrc_to_cw(qsrc: int) -> int:
    return 2 ** qsrc * (CW_MIN + 1) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Sweep: num_stas × qsrc × seed
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    total = len(NUM_STAS_LIST) * len(QSRC_LIST) * len(SEEDS)
    done  = 0

    for num_stas in NUM_STAS_LIST:
        for qsrc in QSRC_LIST:
            cw = _qsrc_to_cw(qsrc)
            for seed in SEEDS:
                done += 1
                print(f"  [{done:3d}/{total}] num_stas={num_stas:2d}  "
                      f"qsrc={qsrc}  CW={cw:<4}  seed={seed}", flush=True)

                sim = build_and_run(
                    num_slots       = num_slots,
                    num_stas        = num_stas,
                    obss_rate       = obss_rate,
                    obss_min        = OBSS_MIN,
                    obss_max        = OBSS_MAX,
                    npca_qsrc       = qsrc,
                    npca_threshold  = 0,
                    ppdu_duration   = 20,
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
                    "num_stas":                   num_stas,
                    "npca_qsrc":                  qsrc,
                    "npca_cw":                    cw,
                    "seed":                       seed,
                    "collision_probability_npca":  agg["collision_probability_npca"],
                    "aggregate_throughput":        agg["aggregate_throughput"],
                    "mean_access_delay":           agg["mean_access_delay"],
                    "npca_transition_count":       agg["npca_transition_count"],
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["num_stas", "npca_qsrc", "npca_cw", "seed",
          "collision_probability_npca", "aggregate_throughput",
          "mean_access_delay", "npca_transition_count"]

def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows: list[dict], num_stas: int, qsrc: int,
           metric: str) -> tuple[float, float]:
    vals = [r[metric] for r in rows
            if r["num_stas"] == num_stas and r["npca_qsrc"] == qsrc]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std

def _optimal_qsrc(rows: list[dict], num_stas: int) -> tuple[int, float]:
    """throughput이 최대인 qsrc와 그 값을 반환."""
    best_qsrc, best_tp = 0, -1.0
    for q in QSRC_LIST:
        m, _ = _stats(rows, num_stas, q, "aggregate_throughput")
        if m > best_tp:
            best_tp, best_qsrc = m, q
    return best_qsrc, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.5), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    x_qsrc = QSRC_LIST
    x_labels = [f"q={q}\n(CW={_qsrc_to_cw(q)})" for q in x_qsrc]

    # ── Panel (a): Throughput vs qsrc, lines per num_stas ────────────────────
    ax_a = axes[0]
    for idx, num_stas in enumerate(NUM_STAS_LIST):
        means, stds = [], []
        for q in x_qsrc:
            m, s = _stats(rows, num_stas, q, "aggregate_throughput")
            means.append(m); stds.append(s)

        color = _COLORS[idx]
        ax_a.plot(x_qsrc, means, color=color, ls="-", marker="o",
                  markersize=5, linewidth=1.6, label=f"N={num_stas}")
        ax_a.fill_between(x_qsrc,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=color, alpha=0.12)

        # 최적 qsrc에 star 마커
        q_opt, tp_opt = _optimal_qsrc(rows, num_stas)
        ax_a.plot(q_opt, tp_opt, marker="*", color=color,
                  markersize=11, zorder=5)

    ax_a.set_ylabel("Aggregate Throughput\n(delivered packets)", fontsize=10)
    ax_a.set_xticks(x_qsrc)
    ax_a.set_xticklabels(x_labels, fontsize=8)
    ax_a.set_xlabel("NPCA Initial CW Exponent (qsrc)", fontsize=9)
    ax_a.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_a.legend(fontsize=9, loc="lower left", ncol=2,
                frameon=True, edgecolor="#cccccc",
                title="Num STAs", title_fontsize=8)
    ax_a.text(0.02, 0.97, "(a) ★ = optimal qsrc per N", transform=ax_a.transAxes,
              fontsize=9, va="top")

    # ── Panel (b): Collision Prob vs qsrc, lines per num_stas ────────────────
    ax_b = axes[1]
    for idx, num_stas in enumerate(NUM_STAS_LIST):
        means, stds = [], []
        for q in x_qsrc:
            m, s = _stats(rows, num_stas, q, "collision_probability_npca")
            means.append(m); stds.append(s)

        color = _COLORS[idx]
        ax_b.plot(x_qsrc, means, color=color, ls="-", marker="s",
                  markersize=5, linewidth=1.6, label=f"N={num_stas}")
        ax_b.fill_between(x_qsrc,
                          [m - s for m, s in zip(means, stds)],
                          [m + s for m, s in zip(means, stds)],
                          color=color, alpha=0.12)

    ax_b.set_ylabel("NPCA Collision Probability", fontsize=10)
    ax_b.set_xticks(x_qsrc)
    ax_b.set_xticklabels(x_labels, fontsize=8)
    ax_b.set_xlabel("NPCA Initial CW Exponent (qsrc)", fontsize=9)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.legend(fontsize=9, loc="upper right", ncol=2,
                frameon=True, edgecolor="#cccccc",
                title="Num STAs", title_fontsize=8)
    ax_b.text(0.02, 0.97, "(b)", transform=ax_b.transAxes, fontsize=9, va="top")

    # ── Panel (c): Optimal qsrc* vs num_stas ─────────────────────────────────
    ax_c = axes[2]

    opt_qsrcs, opt_tps, base_tps = [], [], []
    for num_stas in NUM_STAS_LIST:
        q_opt, tp_opt = _optimal_qsrc(rows, num_stas)
        tp_base, _    = _stats(rows, num_stas, 0, "aggregate_throughput")
        opt_qsrcs.append(q_opt)
        opt_tps.append(tp_opt)
        base_tps.append(tp_base)

    gains = [(o - b) / b * 100 if b > 0 else 0
             for o, b in zip(opt_tps, base_tps)]

    color_opt  = "#4c72b0"
    color_gain = "#dd8452"
    ax_c2 = ax_c.twinx()

    bar_w = 0.5
    bars = ax_c.bar(NUM_STAS_LIST, opt_qsrcs, width=bar_w,
                    color=color_opt, alpha=0.7, label="Optimal qsrc*")
    ax_c.set_yticks(QSRC_LIST)
    ax_c.set_yticklabels(
        [f"q={q} (CW={_qsrc_to_cw(q)})" for q in QSRC_LIST],
        fontsize=8)
    ax_c.set_ylabel("Optimal qsrc*", fontsize=10, color=color_opt)
    ax_c.tick_params(axis="y", labelcolor=color_opt)

    l_gain, = ax_c2.plot(NUM_STAS_LIST, gains, color=color_gain,
                         ls="--", marker="^", markersize=7, linewidth=1.6,
                         label="TP gain over qsrc=0 (%)")
    ax_c2.set_ylabel("Throughput Gain over\nqsrc=0 (%)", fontsize=10,
                     color=color_gain)
    ax_c2.tick_params(axis="y", labelcolor=color_gain, labelsize=9)
    ax_c2.axhline(0, color=color_gain, lw=0.8, ls=":")

    ax_c.set_xlabel("Number of STAs", fontsize=10)
    ax_c.set_xticks(NUM_STAS_LIST)
    ax_c.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)

    lines_c = [bars, l_gain]
    labels_c = ["Optimal qsrc*", "TP gain over qsrc=0 (%)"]
    ax_c.legend([bars.patches[0], l_gain], labels_c,
                fontsize=9, loc="upper left",
                frameon=True, edgecolor="#cccccc")
    ax_c.text(0.02, 0.97, "(c)", transform=ax_c.transAxes, fontsize=9, va="top")

    fig.suptitle(
        "Fig. 3  Fixed CW$_{\\mathrm{npca\\_init}}$ Sweep: Collision vs. Throughput\n"
        "and Optimal qsrc* per Number of STAs",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig3_qsrc_sweep")
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
        description="Figure 3 — qsrc × num_stas sweep, optimal CW derivation (RQ3)")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig3")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode = "FAST" if args.fast else "FULL"
    rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    print(f"=== Figure 3 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")
    print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate={rate:.5f})")
    print(f"    num_stas × qsrc: {len(NUM_STAS_LIST)} × {len(QSRC_LIST)} = "
          f"{len(NUM_STAS_LIST)*len(QSRC_LIST)} conditions")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 3 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig3_qsrc_sweep.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
