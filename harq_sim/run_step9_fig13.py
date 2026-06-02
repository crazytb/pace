"""
Figure 13: Frame Delivery Delay — qsrc × HARQ 교차 분석

계획: guidelines/step9/fig13.md
출력:
  manuscript/figure/fig13_delay_qsrc_harq.{eps,png,pdf}
  results/step9/fig13/data.csv

실행:
  python harq_sim/run_step9_fig13.py [--fast] [--out-dir results/step9/fig13]
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

NUM_STAS_LIST = [5, 10, 20, 30, 50]
SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 500
OBSS_OCCUPANCY = 0.50

# SNR=14dB: MCS3 임계값에 위치 → 첫 TX 성공률 ≈ 50%, HARQ combining 시 ≈ 95%
SNR_DB_MEAN = 14.0
SNR_DB_STD  = 2.0

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

# 10가지 비교 조건
METHODS = [
    {"name": "fixed_q0_arq",  "harq": False, "adaptive": False, "qsrc": 0,
     "label": "Fixed q=0, ARQ"},
    {"name": "fixed_q1_arq",  "harq": False, "adaptive": False, "qsrc": 1,
     "label": "Fixed q=1, ARQ"},
    {"name": "fixed_q2_arq",  "harq": False, "adaptive": False, "qsrc": 2,
     "label": "Fixed q=2, ARQ"},
    {"name": "fixed_q3_arq",  "harq": False, "adaptive": False, "qsrc": 3,
     "label": "Fixed q=3, ARQ"},
    {"name": "adaptive_arq",  "harq": False, "adaptive": True,  "qsrc": 0,
     "label": "Adaptive, ARQ"},
    {"name": "fixed_q0_harq", "harq": True,  "adaptive": False, "qsrc": 0,
     "label": "Fixed q=0, HARQ"},
    {"name": "fixed_q1_harq", "harq": True,  "adaptive": False, "qsrc": 1,
     "label": "Fixed q=1, HARQ"},
    {"name": "fixed_q2_harq", "harq": True,  "adaptive": False, "qsrc": 2,
     "label": "Fixed q=2, HARQ"},
    {"name": "fixed_q3_harq", "harq": True,  "adaptive": False, "qsrc": 3,
     "label": "Fixed q=3, HARQ"},
    {"name": "adaptive_harq", "harq": True,  "adaptive": True,  "qsrc": 0,
     "label": "Adaptive, HARQ"},
]

# qsrc별 색상, adaptive는 파란색 강조
_QSRC_COLORS = {0: "#999999", 1: "#666666", 2: "#333333", 3: "#111111"}
_ADAP_COLOR  = "#1f77b4"

_LS_FIXED = {0: ":", 1: "--", 2: "-.", 3: (0, (5, 1))}
_LS_ADAP  = "-"


def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)
    total = len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    done  = 0

    for num_stas in NUM_STAS_LIST:
        for method in METHODS:
            for seed in SEEDS:
                done += 1
                print(
                    f"  [{done:3d}/{total}] N={num_stas:2d}  method={method['name']:<18}"
                    f"  seed={seed}",
                    flush=True,
                )

                sim = build_and_run(
                    num_slots       = num_slots,
                    num_stas        = num_stas,
                    obss_rate       = obss_rate,
                    obss_min        = OBSS_MIN,
                    obss_max        = OBSS_MAX,
                    npca_qsrc       = method["qsrc"],
                    npca_threshold  = 0,
                    ppdu_duration   = 20,
                    harq_horizon    = 200,
                    snr_db_mean     = SNR_DB_MEAN,
                    snr_db_std      = SNR_DB_STD,
                    npca_enabled    = True,
                    harq_enabled    = method["harq"],
                    adaptive_cw     = method["adaptive"],
                    seed            = seed,
                    enable_trace    = False,
                )

                agg = sim.compute_metrics()["aggregate"]

                # mean_retx_count = total_tx / total_delivered
                total_tx = sum(
                    s.stats["primary_tx_success"] + s.stats["primary_tx_fail"]
                    + s.stats["npca_tx_success"]  + s.stats["npca_tx_fail"]
                    for s in sim.stas
                )
                total_del = agg["aggregate_throughput"]
                mean_retx = total_tx / total_del if total_del > 0 else 0.0

                rows.append({
                    "num_stas":                  num_stas,
                    "method":                    method["name"],
                    "harq_enabled":              method["harq"],
                    "adaptive_cw":               method["adaptive"],
                    "npca_qsrc":                 method["qsrc"],
                    "seed":                      seed,
                    "mean_access_delay":         agg["mean_access_delay"],
                    "p95_access_delay":          agg["p95_access_delay"],
                    "p99_access_delay":          agg["p99_access_delay"],
                    "mean_retx_count":           mean_retx,
                    "aggregate_throughput":      total_del,
                    "collision_probability_npca": agg["collision_probability_npca"],
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    "num_stas", "method", "harq_enabled", "adaptive_cw", "npca_qsrc", "seed",
    "mean_access_delay", "p95_access_delay", "p99_access_delay",
    "mean_retx_count", "aggregate_throughput", "collision_probability_npca",
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

def _stats(rows, num_stas, method_name, metric):
    vals = [r[metric] for r in rows
            if r["num_stas"] == num_stas and r["method"] == method_name]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _arq_name(harq_name: str) -> str:
    """'fixed_q0_harq' → 'fixed_q0_arq', 'adaptive_harq' → 'adaptive_arq'"""
    return harq_name.replace("_harq", "_arq")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _method_style(method: dict) -> tuple:
    """Returns (color, linestyle, linewidth, zorder)."""
    if method["adaptive"]:
        return _ADAP_COLOR, _LS_ADAP, 2.2, 4
    q = method["qsrc"]
    return _QSRC_COLORS[q], _LS_FIXED[q], 1.4, 2


def plot(rows: list[dict], fig_dir: str) -> None:
    x = NUM_STAS_LIST

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.35)

    # ── determine common y-axis limits for panels (a) and (b) ────────────────
    all_delays = [r["mean_access_delay"] for r in rows]
    y_max = max(all_delays) * 1.12 if all_delays else 1.0
    y_min = 0.0

    # ── Panel (a): HARQ=off ───────────────────────────────────────────────────
    ax = axes[0]
    arq_methods = [m for m in METHODS if not m["harq"]]
    for method in arq_methods:
        color, ls, lw, zord = _method_style(method)
        means, stds = zip(*[_stats(rows, n, method["name"], "mean_access_delay") for n in x])
        ax.plot(x, means, color=color, ls=ls, lw=lw, marker="o",
                markersize=5, label=method["label"], zorder=zord)
        ax.fill_between(x,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.12, zorder=zord - 1)

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Mean Access Delay (slots)", fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="upper left", frameon=True, edgecolor="#cccccc")
    ax.text(0.02, 0.97, "(a) HARQ = off",
            transform=ax.transAxes, fontsize=9, va="top")

    # ── Panel (b): HARQ=on ────────────────────────────────────────────────────
    ax = axes[1]
    harq_methods = [m for m in METHODS if m["harq"]]
    for method in harq_methods:
        color, ls, lw, zord = _method_style(method)
        means, stds = zip(*[_stats(rows, n, method["name"], "mean_access_delay") for n in x])
        ax.plot(x, means, color=color, ls=ls, lw=lw, marker="o",
                markersize=5, label=method["label"], zorder=zord)
        ax.fill_between(x,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.12, zorder=zord - 1)

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Mean Access Delay (slots)", fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="upper left", frameon=True, edgecolor="#cccccc")
    ax.text(0.02, 0.97, "(b) HARQ = on",
            transform=ax.transAxes, fontsize=9, va="top")

    # ── Panel (c): HARQ delay reduction % ────────────────────────────────────
    ax = axes[2]
    # Draw for each HARQ method → compare with its ARQ counterpart
    for method in harq_methods:
        color, ls, lw, zord = _method_style(method)
        arq_name = _arq_name(method["name"])
        reduction = []
        for n in x:
            arq_mean, _  = _stats(rows, n, arq_name,       "mean_access_delay")
            hrq_mean, _  = _stats(rows, n, method["name"], "mean_access_delay")
            if arq_mean > 0:
                pct = (arq_mean - hrq_mean) / arq_mean * 100.0
            else:
                pct = 0.0
            reduction.append(pct)

        label = method["label"].replace(", HARQ", "")
        ax.plot(x, reduction, color=color, ls=ls, lw=lw, marker="^",
                markersize=5, label=label, zorder=zord)

    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("HARQ Delay Reduction (%)", fontsize=10)
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="upper right", frameon=True, edgecolor="#cccccc")
    ax.text(0.02, 0.97, r"(c) $\Delta$delay = (ARQ$-$HARQ)/ARQ",
            transform=ax.transAxes, fontsize=9, va="top")

    fig.suptitle(
        "Fig. 13  Frame Delivery Delay: qsrc × HARQ\n"
        f"(OBSS occ={int(OBSS_OCCUPANCY*100)}%, SNR={SNR_DB_MEAN}dB±{SNR_DB_STD}dB)",
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig13_delay_qsrc_harq")
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
        description="Figure 13 — Frame Delivery Delay: qsrc × HARQ")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig13")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode  = "FAST" if args.fast else "FULL"
    rate  = _occupancy_to_rate(OBSS_OCCUPANCY)
    total = len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    print(f"=== Figure 13 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")
    print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate={rate:.5f})")
    print(f"    SNR={SNR_DB_MEAN}dB ± {SNR_DB_STD}dB")
    print(f"    {len(NUM_STAS_LIST)} num_stas × {len(METHODS)} methods × {len(SEEDS)} seeds = {total} runs")

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 13 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig13_delay_qsrc_harq.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
