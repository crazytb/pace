"""
Figure 9: qsrc Optimization Under Native NPCA Contention (RQ9)

NPCA 채널을 primary로 사용하는 native STA N_n명이 상주할 때
optimal qsrc*가 어떻게 달라지는지 관찰.
Trans STA와 Native STA의 throughput을 모두 시각화 (4-panel).

Guidelines: guidelines/step9/fig9.md
스크립트: harq_sim/run_step9_fig9.py
출력: manuscript/figure/fig9_native_npca.{eps,png,pdf}

실행:
  python harq_sim/run_step9_fig9.py [--fast] [--out-dir results/step9/fig9]
  python harq_sim/run_step9_fig9.py --replot --csv results/step9/fig9/data.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harq_sim.channel import Channel
from harq_sim.simulator import Simulator
from harq_sim.sta import STA

# ─────────────────────────────────────────────────────────────────────────────
# 실험 설정
# ─────────────────────────────────────────────────────────────────────────────

N_TRANSITION  = 20
N_NATIVE_LIST = [0, 5, 10, 20, 30]
QSRC_LIST     = [0, 1, 2, 3, 4, 5]
SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 200
OBSS_OCCUPANCY = 0.30
SNR_DB_MEAN    = 20.0

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#9467bd"]

CW_MIN = 15


def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


def _qsrc_to_cw(qsrc: int) -> int:
    return 2 ** qsrc * (CW_MIN + 1) - 1


def _formula_qsrc_star(n_total: int) -> int:
    """Fig 3 closed-form: qsrc* = max(0, round(log2(N/16)))"""
    if n_total <= 0:
        return 0
    return max(0, round(math.log2(n_total / 16)))


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(num_slots: int, qsrc: int, n_native: int, seed: int,
            adaptive_cw: bool = False) -> dict:
    random.seed(seed)
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)

    primary_ch = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)

    trans_stas = [
        STA(
            sta_id=i,
            primary_channel=primary_ch,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=20,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=0,
            npca_initial_qsrc=qsrc,
            adaptive_cw=adaptive_cw,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=SNR_DB_MEAN,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
        )
        for i in range(N_TRANSITION)
    ]

    native_stas = [
        STA(
            sta_id=N_TRANSITION + i,
            primary_channel=npca_ch,    # NPCA 채널이 native STA의 primary
            npca_channel=None,
            npca_enabled=False,
            ppdu_duration=20,
            switching_delay=1,
            switch_back_delay=1,
            npca_initial_qsrc=0,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=SNR_DB_MEAN,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            ap_on_primary=True,
        )
        for i in range(n_native)
    ]

    sim = Simulator(
        num_slots=num_slots,
        stas=trans_stas + native_stas,
        channels=[primary_ch, npca_ch],
        enable_trace=False,
    )
    sim.run()

    metrics = sim.compute_metrics()

    # Trans STA 집계
    trans_delivered = sum(metrics[i]["packets_delivered"] for i in range(N_TRANSITION))

    trans_npca_tx = sum(
        sim.stas[i].stats["npca_tx_success"] + sim.stas[i].stats["npca_tx_fail"]
        for i in range(N_TRANSITION)
    )
    trans_npca_col = sum(
        sim.stas[i].stats.get("npca_collision_count", 0)
        for i in range(N_TRANSITION)
    )
    trans_col_prob = trans_npca_col / trans_npca_tx if trans_npca_tx > 0 else 0.0
    trans_transitions = sum(metrics[i]["npca_transitions"] for i in range(N_TRANSITION))

    # Native STA 집계
    native_delivered = 0
    native_col_prob  = 0.0
    if n_native > 0:
        native_delivered = sum(
            metrics[N_TRANSITION + i]["packets_delivered"] for i in range(n_native)
        )
        native_tx_total = sum(
            sim.stas[N_TRANSITION + i].stats["primary_tx_success"]
            + sim.stas[N_TRANSITION + i].stats["primary_tx_fail"]
            for i in range(n_native)
        )
        native_col = sum(
            sim.stas[N_TRANSITION + i].stats.get("primary_collision_count", 0)
            for i in range(n_native)
        )
        native_col_prob = native_col / native_tx_total if native_tx_total > 0 else 0.0

    # adaptive qsrc 사용 시 실제 적용된 평균 qsrc 기록
    mean_qsrc_used = qsrc
    if adaptive_cw:
        all_history = []
        for i in range(N_TRANSITION):
            all_history.extend(sim.stas[i]._npca_qsrc_history)
        mean_qsrc_used = (sum(all_history) / len(all_history)) if all_history else qsrc

    return {
        "qsrc":                        qsrc,
        "adaptive_cw":                 adaptive_cw,
        "mean_qsrc_used":              mean_qsrc_used,
        "N_native":                    n_native,
        "seed":                        seed,
        "trans_throughput":            trans_delivered,
        "trans_collision_prob":        trans_col_prob,
        "trans_npca_transition_count": trans_transitions,
        "native_throughput":           native_delivered,
        "native_collision_prob":       native_col_prob,
        "total_npca_throughput":       trans_delivered + native_delivered,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    total = len(N_NATIVE_LIST) * len(QSRC_LIST) * len(SEEDS)
    done  = 0

    for n_native in N_NATIVE_LIST:
        for qsrc in QSRC_LIST:
            for seed in SEEDS:
                done += 1
                print(f"  [{done:3d}/{total}] N_native={n_native:2d}  "
                      f"qsrc={qsrc}  seed={seed}", flush=True)
                row = run_one(num_slots, qsrc, n_native, seed)
                rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    "qsrc", "adaptive_cw", "mean_qsrc_used", "N_native", "seed",
    "trans_throughput", "trans_collision_prob", "trans_npca_transition_count",
    "native_throughput", "native_collision_prob",
    "total_npca_throughput",
]


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(rows: list[dict], n_native: int, qsrc: int,
              metric: str) -> tuple[float, float]:
    vals = [r[metric] for r in rows
            if r["N_native"] == n_native and r["qsrc"] == qsrc]
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _optimal_qsrc(rows: list[dict], n_native: int, metric: str = "trans_throughput") -> tuple[int, float]:
    best_q, best_tp = 0, -1.0
    for q in QSRC_LIST:
        m, _ = _mean_std(rows, n_native, q, metric)
        if m > best_tp:
            best_tp, best_q = m, q
    return best_q, best_tp


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_tp_panel(ax, rows: list[dict], metric: str, panel_label: str,
                   ylabel: str, skip_zero_native: bool = False) -> None:
    """Plot throughput vs qsrc with one line per N_native."""
    x_qsrc   = QSRC_LIST
    x_labels = [f"q={q}\n(CW={_qsrc_to_cw(q)})" for q in x_qsrc]

    for idx, n_native in enumerate(N_NATIVE_LIST):
        if skip_zero_native and n_native == 0:
            continue
        means, stds = [], []
        for q in x_qsrc:
            m, s = _mean_std(rows, n_native, q, metric)
            means.append(m)
            stds.append(s)

        color = _COLORS[idx]
        label = f"$N_n$={n_native}"
        ax.plot(x_qsrc, means, color=color, ls="-", marker="o",
                markersize=5, linewidth=1.6, label=label)
        ax.fill_between(x_qsrc,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.12)

        q_opt, tp_opt = _optimal_qsrc(rows, n_native, metric)
        ax.plot(q_opt, tp_opt, marker="*", color=color, markersize=12, zorder=5)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x_qsrc)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("NPCA Initial CW Exponent (qsrc)", fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="best", ncol=2, frameon=True, edgecolor="#cccccc",
              title=f"$N_n$,  $N_t$={N_TRANSITION}", title_fontsize=8)
    ax.text(0.02, 0.97, f"({panel_label})  ★ = qsrc* per $N_n$",
            transform=ax.transAxes, fontsize=9, va="top")


def plot(rows: list[dict], fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # ── Panel (a): Trans STA throughput vs qsrc ───────────────────────────────
    _plot_tp_panel(
        axes[0, 0], rows,
        metric="trans_throughput",
        panel_label="a",
        ylabel="Transitioning STA\nThroughput (delivered pkts)",
    )

    # ── Panel (b): Native STA throughput vs qsrc ─────────────────────────────
    _plot_tp_panel(
        axes[0, 1], rows,
        metric="native_throughput",
        panel_label="b",
        ylabel="Native STA\nThroughput (delivered pkts)",
        skip_zero_native=True,  # N_n=0 has no native STAs
    )
    axes[0, 1].text(0.98, 0.05,
                    "Native TP ↑ as qsrc ↑\n(trans STAs yield channel)",
                    transform=axes[0, 1].transAxes, fontsize=8, va="bottom",
                    ha="right", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Panel (c): Total NPCA throughput vs qsrc ─────────────────────────────
    _plot_tp_panel(
        axes[1, 0], rows,
        metric="total_npca_throughput",
        panel_label="c",
        ylabel="Total NPCA Throughput\n(trans + native, delivered pkts)",
    )
    axes[1, 0].text(0.98, 0.05,
                    "System-level optimum\nmay differ from trans-only",
                    transform=axes[1, 0].transAxes, fontsize=8, va="bottom",
                    ha="right", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Panel (d): qsrc* comparison — trans-optimal vs total-optimal ─────────
    ax_d = axes[1, 1]

    trans_opts  = [_optimal_qsrc(rows, n, "trans_throughput")[0]  for n in N_NATIVE_LIST]
    total_opts  = [_optimal_qsrc(rows, n, "total_npca_throughput")[0] for n in N_NATIVE_LIST]
    ref_qsrcs   = [_formula_qsrc_star(N_TRANSITION + n) for n in N_NATIVE_LIST]

    ax_d.plot(N_NATIVE_LIST, trans_opts, color="#4c72b0", ls="-", marker="o",
              markersize=8, linewidth=2.0, label="qsrc* for Trans TP (trans-only)")
    ax_d.plot(N_NATIVE_LIST, total_opts, color="#dd8452", ls="-", marker="s",
              markersize=8, linewidth=2.0, label="qsrc* for Total TP (system)")
    ax_d.plot(N_NATIVE_LIST, ref_qsrcs, color="#999999", ls="--", marker="^",
              markersize=6, linewidth=1.4,
              label=r"Formula qsrc$^*(N_t + N_n)$")

    # Annotate divergence region
    for idx, (n, qt, qs) in enumerate(zip(N_NATIVE_LIST, trans_opts, total_opts)):
        if qt != qs:
            ax_d.annotate(f"Δ={qs-qt:+d}",
                          xy=(n, (qt + qs) / 2),
                          xytext=(n + 1.5, (qt + qs) / 2 + 0.15),
                          fontsize=8, color="#c44e52",
                          arrowprops=dict(arrowstyle="-", color="#c44e52", lw=0.8))

    ax_d.set_ylabel("Optimal qsrc*", fontsize=10)
    ax_d.set_xlabel("Number of Native STAs ($N_n$)", fontsize=10)
    ax_d.set_xticks(N_NATIVE_LIST)
    ax_d.set_yticks(QSRC_LIST)
    ax_d.set_yticklabels([f"q={q}\n(CW={_qsrc_to_cw(q)})" for q in QSRC_LIST], fontsize=8)
    ax_d.set_ylim(-0.4, max(max(trans_opts), max(total_opts), max(ref_qsrcs)) + 0.6)
    ax_d.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_d.legend(fontsize=8, loc="upper left", frameon=True, edgecolor="#cccccc")
    ax_d.text(0.02, 0.97, "(d)  qsrc* trade-off: Trans vs. System",
              transform=ax_d.transAxes, fontsize=9, va="top")

    fig.suptitle(
        "Fig. 9  Trans & Native STA Throughput vs qsrc Under Native NPCA Contention\n"
        f"($N_t$={N_TRANSITION} transitioning STAs, OBSS occ.={OBSS_OCCUPANCY:.0%})",
        fontsize=11, y=1.01,
    )

    _save_figure(fig, fig_dir, "fig9_native_npca")
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

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive comparison sweep & plot
# ─────────────────────────────────────────────────────────────────────────────

COMPARE_METHODS = [
    {"name": "adaptive",  "adaptive_cw": True,  "qsrc": 0, "label": "Adaptive qsrc (proposed)", "color": "#2ca02c", "ls": "-",  "lw": 2.2, "marker": "D"},
    {"name": "fixed_q0",  "adaptive_cw": False, "qsrc": 0, "label": "Fixed qsrc=0 (CW=15)",     "color": "#4c72b0", "ls": "--", "lw": 1.6, "marker": "o"},
    {"name": "fixed_q1",  "adaptive_cw": False, "qsrc": 1, "label": "Fixed qsrc=1 (CW=31)",     "color": "#dd8452", "ls": "--", "lw": 1.4, "marker": "s"},
    {"name": "fixed_q2",  "adaptive_cw": False, "qsrc": 2, "label": "Fixed qsrc=2 (CW=63)",     "color": "#c44e52", "ls": "--", "lw": 1.2, "marker": "^"},
]


def run_compare_sweep(num_slots: int) -> list[dict]:
    rows: list[dict] = []
    total = len(N_NATIVE_LIST) * len(COMPARE_METHODS) * len(SEEDS)
    done  = 0
    for n_native in N_NATIVE_LIST:
        for method in COMPARE_METHODS:
            for seed in SEEDS:
                done += 1
                print(f"  [{done:3d}/{total}] N_native={n_native:2d}  "
                      f"method={method['name']:<12}  seed={seed}", flush=True)
                row = run_one(num_slots, method["qsrc"], n_native, seed,
                              adaptive_cw=method["adaptive_cw"])
                row["method"] = method["name"]
                rows.append(row)
    return rows


def _cstats(rows: list[dict], n_native: int, method: str, metric: str):
    vals = [r[metric] for r in rows
            if r["N_native"] == n_native and r["method"] == method]
    if not vals:
        return 0.0, 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def plot_compare(rows: list[dict], fig_dir: str) -> None:
    metrics = [
        ("trans_throughput",      "Trans STA Throughput\n(delivered pkts)"),
        ("native_throughput",     "Native STA Throughput\n(delivered pkts)"),
        ("total_npca_throughput", "Total NPCA Throughput\n(trans + native, pkts)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.32)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in COMPARE_METHODS:
            means, stds = [], []
            for nn in N_NATIVE_LIST:
                m, s = _cstats(rows, nn, method["name"], metric)
                means.append(m)
                stds.append(s)
            ax.plot(N_NATIVE_LIST, means,
                    color=method["color"], ls=method["ls"],
                    linewidth=method["lw"], marker=method["marker"],
                    markersize=7, label=method["label"])
            ax.fill_between(N_NATIVE_LIST,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=method["color"], alpha=0.10)

        ax.set_xlabel("Number of Native STAs ($N_n$)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(N_NATIVE_LIST)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.legend(fontsize=8, frameon=True, edgecolor="#cccccc")

    # Panel labels
    for ax, label in zip(axes, ["(a) Trans STA TP", "(b) Native STA TP", "(c) Total NPCA TP"]):
        ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=9, va="top")

    # Mean qsrc used by adaptive
    adaptive_mean_q = []
    for nn in N_NATIVE_LIST:
        vals = [r["mean_qsrc_used"] for r in rows
                if r["N_native"] == nn and r["method"] == "adaptive"]
        adaptive_mean_q.append(statistics.mean(vals) if vals else 0)

    ax2 = axes[2].twinx()
    ax2.plot(N_NATIVE_LIST, adaptive_mean_q, color="#2ca02c", ls=":", lw=1.4,
             marker="x", markersize=8, label="Adaptive mean qsrc")
    ax2.set_ylabel("Adaptive mean qsrc used", fontsize=9, color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(-0.2, 5.5)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Fig. 9 Ext — Adaptive vs Fixed qsrc: Trans / Native / Total Throughput\n"
        f"($N_t$={N_TRANSITION}, OBSS occ.={OBSS_OCCUPANCY:.0%})",
        fontsize=11,
    )
    _save_figure(fig, fig_dir, "fig9_adaptive_compare")
    plt.close(fig)


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "qsrc":                        int(row["qsrc"]),
                "N_native":                    int(row["N_native"]),
                "seed":                        int(row["seed"]),
                "trans_throughput":            float(row["trans_throughput"]),
                "trans_collision_prob":        float(row["trans_collision_prob"]),
                "trans_npca_transition_count": float(row["trans_npca_transition_count"]),
                "native_throughput":           float(row["native_throughput"]),
                "native_collision_prob":       float(row["native_collision_prob"]),
                "total_npca_throughput":       float(row["total_npca_throughput"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 9 — qsrc* under native NPCA contention (RQ9)")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_SLOTS} slots")
    parser.add_argument("--out-dir", default="results/step9/fig9")
    parser.add_argument("--replot",  action="store_true",
                        help="Skip simulation; reload CSV from --out-dir and replot")
    parser.add_argument("--compare-adaptive", action="store_true",
                        help="Run adaptive vs fixed comparison (trans+native total TP)")
    args = parser.parse_args()

    out_dir  = args.out_dir
    csv_path = os.path.join(out_dir, "data.csv")
    os.makedirs(out_dir, exist_ok=True)

    if args.compare_adaptive:
        # ── Adaptive vs Fixed comparison mode ─────────────────────────────────
        num_slots  = FAST_SLOTS if args.fast else FULL_SLOTS
        mode       = "FAST" if args.fast else "FULL"
        cmp_dir    = os.path.join(os.path.dirname(out_dir), "fig9_adaptive")
        os.makedirs(cmp_dir, exist_ok=True)
        cmp_csv    = os.path.join(cmp_dir, "data.csv")

        rate = _occupancy_to_rate(OBSS_OCCUPANCY)
        total = len(N_NATIVE_LIST) * len(COMPARE_METHODS) * len(SEEDS)
        print(f"=== Figure 9 Adaptive Comparison [{mode}] — {num_slots} slots ===")
        print(f"    N_transition={N_TRANSITION},  N_native={N_NATIVE_LIST}")
        print(f"    Methods: {[m['name'] for m in COMPARE_METHODS]}")
        print(f"    {len(N_NATIVE_LIST)} × {len(COMPARE_METHODS)} methods × {len(SEEDS)} seeds = {total} runs")

        cmp_rows = run_compare_sweep(num_slots)

        cmp_fields = FIELDS + ["method"]
        os.makedirs(os.path.dirname(os.path.abspath(cmp_csv)), exist_ok=True)
        with open(cmp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cmp_fields,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"  CSV → {cmp_csv}")

        print("Plotting comparison...")
        plot_compare(cmp_rows, cmp_dir)
        print(f"\n완료: {cmp_csv}")
        print(f"  논문용: manuscript/figure/fig9_adaptive_compare.{{eps,png,pdf}}")
        return

    if args.replot:
        print(f"=== Figure 9 [REPLOT] — loading {csv_path} ===")
        rows = load_csv(csv_path)
        print(f"    Loaded {len(rows)} rows")
    else:
        num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
        mode = "FAST" if args.fast else "FULL"
        rate = _occupancy_to_rate(OBSS_OCCUPANCY)
        print(f"=== Figure 9 [{mode}] — {num_slots} slots × {len(SEEDS)} seeds ===")
        print(f"    N_transition={N_TRANSITION},  N_native sweep={N_NATIVE_LIST}")
        print(f"    OBSS occupancy={OBSS_OCCUPANCY:.0%}  (rate={rate:.6f})")
        total = len(N_NATIVE_LIST) * len(QSRC_LIST) * len(SEEDS)
        print(f"    {len(N_NATIVE_LIST)} × {len(QSRC_LIST)} × {len(SEEDS)} = {total} runs")

        rows = run_sweep(num_slots)
        save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print(f"\nFigure 9 완료")
    print(f"  데이터  : {csv_path}")
    print(f"  논문용  : manuscript/figure/fig9_native_npca.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
