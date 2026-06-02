"""
Figure 14: Per-STA PPDU-Aware Threshold — Heterogeneous PPDU Environment

Plan: guidelines/step9/fig14.md
Output:
  manuscript/figure/fig14_ppdu_aware_threshold.{eps,png,pdf}
  results/step9/fig14/data.csv

Run:
  python harq_sim/run_step9_fig14.py [--fast] [--out-dir results/step9/fig14]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
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
# Experiment parameters
# ─────────────────────────────────────────────────────────────────────────────

NUM_STAS_LIST = [10, 20, 30, 50]
SEEDS         = [42, 123, 456]

OBSS_MIN       = 20
OBSS_MAX       = 100
OBSS_OCCUPANCY = 0.50

FULL_SLOTS = 50_000
FAST_SLOTS =  5_000

# Oracle qsrc* from weff_check_full sweep (OBSS 20-100 → avg W_eff ~40 for PPDU=20)
ORACLE_QSRC = {10: 0, 20: 1, 30: 1, 50: 2}

PPDU_CONFIGS = ["homo_short", "homo_medium", "homo_long", "hetero"]

METHODS = [
    {"name": "fixed_q0_t0",    "adaptive": False, "qsrc": 0,
     "threshold_mode": "zero", "label": "Fixed q=0, thr=0"},
    {"name": "fixed_q0_tppdu", "adaptive": False, "qsrc": 0,
     "threshold_mode": "ppdu", "label": "Fixed q=0, thr=ppdu"},
    {"name": "fixed_q1_t0",    "adaptive": False, "qsrc": 1,
     "threshold_mode": "zero", "label": "Fixed q=1, thr=0"},
    {"name": "fixed_q1_tppdu", "adaptive": False, "qsrc": 1,
     "threshold_mode": "ppdu", "label": "Fixed q=1, thr=ppdu"},
    {"name": "adaptive_t0",    "adaptive": True,  "qsrc": 0,
     "threshold_mode": "zero", "label": "Adaptive, thr=0"},
    {"name": "adaptive_tppdu", "adaptive": True,  "qsrc": 0,
     "threshold_mode": "ppdu", "label": "Adaptive, thr=ppdu (proposed)"},
    {"name": "oracle_t0",      "adaptive": False, "qsrc": None,
     "threshold_mode": "zero", "label": "Oracle q*, thr=0"},
    {"name": "oracle_tppdu",   "adaptive": False, "qsrc": None,
     "threshold_mode": "ppdu", "label": "Oracle q*, thr=ppdu"},
]

# Methods to show in panel (a)
PANEL_A_METHODS = {"fixed_q0_t0", "fixed_q0_tppdu", "adaptive_t0",
                   "adaptive_tppdu", "oracle_tppdu"}

_COLORS = {
    "fixed_q0_t0":    "#aaaaaa",
    "fixed_q0_tppdu": "#888888",
    "fixed_q1_t0":    "#777777",
    "fixed_q1_tppdu": "#555555",
    "adaptive_t0":    "#ff7f0e",
    "adaptive_tppdu": "#1f77b4",
    "oracle_t0":      "#aabb55",
    "oracle_tppdu":   "#2ca02c",
}
_LS = {
    "fixed_q0_t0":    ":",
    "fixed_q0_tppdu": ":",
    "fixed_q1_t0":    "--",
    "fixed_q1_tppdu": "--",
    "adaptive_t0":    "-",
    "adaptive_tppdu": "-",
    "oracle_t0":      "--",
    "oracle_tppdu":   "--",
}

GROUP_COLORS = {"short": "#e41a1c", "medium": "#377eb8", "long": "#4daf4a"}
PPDU_VALS    = {"short": 10, "medium": 20, "long": 40}


# ─────────────────────────────────────────────────────────────────────────────
# PPDU list builder
# ─────────────────────────────────────────────────────────────────────────────

def _ppdu_list(config: str, n: int) -> list:
    if config == "homo_short":
        return [10] * n
    if config == "homo_medium":
        return [20] * n
    if config == "homo_long":
        return [40] * n
    # hetero: short / medium / long, remainder goes to long group
    n_short  = n // 3
    n_medium = n // 3
    n_long   = n - 2 * (n // 3)
    return [10] * n_short + [20] * n_medium + [40] * n_long


def _group_bounds(n: int) -> tuple:
    """Return (short_end, medium_end) index bounds for hetero config."""
    n_short  = n // 3
    n_medium = n // 3
    return n_short, n_short + n_medium


def _occupancy_to_rate(occ: float) -> float:
    mean_dur = (OBSS_MIN + OBSS_MAX) / 2.0
    return occ / (mean_dur * (1.0 - occ))


# ─────────────────────────────────────────────────────────────────────────────
# Core: per-STA heterogeneous PPDU simulation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_and_run_hetero(
    num_slots:       int,
    ppdu_list:       list,
    threshold_mode:  str,    # "zero" | "ppdu"
    npca_qsrc:       int,
    adaptive_cw:     bool,
    obss_rate:       float,
    seed:            int,
) -> Simulator:
    """Build channels + per-STA heterogeneous STAs, run simulation."""
    random.seed(seed)

    primary = Channel(
        channel_id=0,
        obss_generation_rate=obss_rate,
        obss_duration_range=(OBSS_MIN, OBSS_MAX),
    )
    npca_ch = Channel(channel_id=1, obss_generation_rate=0.0)
    policy  = NPCAHARQPolicy(adaptive_cw=adaptive_cw)

    stas = []
    for i, ppdu_dur in enumerate(ppdu_list):
        threshold = ppdu_dur if threshold_mode == "ppdu" else 0
        stas.append(STA(
            sta_id=i,
            primary_channel=primary,
            npca_channel=npca_ch,
            npca_enabled=True,
            ppdu_duration=ppdu_dur,
            switching_delay=1,
            switch_back_delay=1,
            npca_min_duration_threshold=threshold,
            npca_initial_qsrc=npca_qsrc,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=20.0,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=200,
            policy=policy,
            adaptive_cw=adaptive_cw,
        ))

    sim = Simulator(
        num_slots=num_slots,
        stas=stas,
        channels=[primary, npca_ch],
        enable_trace=False,
    )
    sim.run()
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int) -> list:
    rows = []
    obss_rate = _occupancy_to_rate(OBSS_OCCUPANCY)

    total = len(PPDU_CONFIGS) * len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    done  = 0

    for ppdu_config in PPDU_CONFIGS:
        for num_stas in NUM_STAS_LIST:
            ppdu_lst = _ppdu_list(ppdu_config, num_stas)
            short_end, medium_end = _group_bounds(num_stas)

            for method in METHODS:
                qsrc = ORACLE_QSRC[num_stas] if method["qsrc"] is None else method["qsrc"]

                for seed in SEEDS:
                    done += 1
                    print(
                        f"  [{done:4d}/{total}] config={ppdu_config:<12}  N={num_stas:2d}"
                        f"  method={method['name']:<16}  qsrc={qsrc}  seed={seed}",
                        flush=True,
                    )

                    sim = build_and_run_hetero(
                        num_slots=num_slots,
                        ppdu_list=ppdu_lst,
                        threshold_mode=method["threshold_mode"],
                        npca_qsrc=qsrc,
                        adaptive_cw=method["adaptive"],
                        obss_rate=obss_rate,
                        seed=seed,
                    )

                    all_m = sim.compute_metrics()
                    agg   = all_m["aggregate"]

                    # Per-group throughput (only meaningful for hetero config)
                    if ppdu_config == "hetero":
                        tp_short  = sum(all_m[i]["packets_delivered"]
                                        for i in range(0, short_end))
                        tp_medium = sum(all_m[i]["packets_delivered"]
                                        for i in range(short_end, medium_end))
                        tp_long   = sum(all_m[i]["packets_delivered"]
                                        for i in range(medium_end, num_stas))
                    else:
                        tp_short  = ""
                        tp_medium = ""
                        tp_long   = ""

                    # Mean qsrc: average over all STA transition histories
                    if method["adaptive"]:
                        histories = [sta._npca_qsrc_history for sta in sim.stas
                                     if sta._npca_qsrc_history]
                        mean_qsrc = (float(np.mean([q for h in histories for q in h]))
                                     if histories else 0.0)
                    else:
                        mean_qsrc = float(qsrc)

                    rows.append({
                        "ppdu_config":              ppdu_config,
                        "num_stas":                 num_stas,
                        "method":                   method["name"],
                        "threshold_mode":           method["threshold_mode"],
                        "seed":                     seed,
                        "aggregate_throughput":     agg["aggregate_throughput"],
                        "collision_probability_npca": agg["collision_probability_npca"],
                        "npca_transition_count":    agg["npca_transition_count"],
                        "mean_qsrc":                mean_qsrc,
                        "tp_short":                 tp_short,
                        "tp_medium":                tp_medium,
                        "tp_long":                  tp_long,
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    "ppdu_config", "num_stas", "method", "threshold_mode", "seed",
    "aggregate_throughput", "collision_probability_npca", "npca_transition_count",
    "mean_qsrc", "tp_short", "tp_medium", "tp_long",
]


def save_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows, ppdu_config, num_stas, method, metric):
    vals = [float(r[metric]) for r in rows
            if (r["ppdu_config"] == ppdu_config
                and r["num_stas"] == num_stas
                and r["method"] == method
                and r[metric] != "")]
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _group_means(rows, ppdu_config, num_stas, method, metric):
    vals = [float(r[metric]) for r in rows
            if (r["ppdu_config"] == ppdu_config
                and r["num_stas"] == num_stas
                and r["method"] == method
                and r[metric] != "")]
    return statistics.mean(vals) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.38)

    x      = NUM_STAS_LIST
    config = "hetero"

    # ── Panel (a): Aggregate Throughput vs N ─────────────────────────────────
    ax = axes[0]
    for method in METHODS:
        name = method["name"]
        if name not in PANEL_A_METHODS:
            continue
        color = _COLORS[name]
        ls    = _LS[name]
        lw    = 2.0 if ("adaptive" in name or "oracle" in name) else 1.2

        means, stds = zip(*[_stats(rows, config, n, name, "aggregate_throughput")
                             for n in x])
        ax.plot(x, means, color=color, ls=ls, lw=lw,
                marker="o", markersize=5, label=method["label"])
        ax.fill_between(x,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.10)

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Aggregate Throughput (delivered packets)", fontsize=10)
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    ax.set_title("(a) Aggregate Throughput vs N\n(hetero PPDU: 10/20/40 mix)", fontsize=10)

    # ── Panel (b): Per-Group Throughput vs N ─────────────────────────────────
    ax = axes[1]
    group_metrics = {"short": "tp_short", "medium": "tp_medium", "long": "tp_long"}
    for method_name, ls, marker, suffix in [
        ("adaptive_t0",    "--", "s", "thr=0"),
        ("adaptive_tppdu", "-",  "o", "thr=ppdu"),
    ]:
        for group, metric in group_metrics.items():
            color    = GROUP_COLORS[group]
            ppdu_val = PPDU_VALS[group]
            means_g  = [_group_means(rows, config, n, method_name, metric) for n in x]
            ax.plot(x, means_g, color=color, ls=ls, lw=1.8,
                    marker=marker, markersize=5,
                    label="PPDU=%d, %s" % (ppdu_val, suffix))

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Group Throughput (packets)", fontsize=10)
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=7.5, frameon=True, ncol=1)
    ax.set_title("(b) Per-Group Throughput vs N\n(hetero PPDU, adaptive: solid=thr=ppdu, dashed=thr=0)",
                 fontsize=10)

    # ── Panel (c): Mean qsrc vs N — adaptive convergence ─────────────────────
    ax = axes[2]
    for method_name, color, ls, label in [
        ("adaptive_t0",    "#ff7f0e", "--", "Adaptive, thr=0"),
        ("adaptive_tppdu", "#1f77b4", "-",  "Adaptive, thr=ppdu (proposed)"),
        ("oracle_tppdu",   "#2ca02c", "--", "Oracle q*, thr=ppdu"),
    ]:
        means, stds = zip(*[_stats(rows, config, n, method_name, "mean_qsrc")
                             for n in x])
        ax.plot(x, means, color=color, ls=ls, lw=2.0,
                marker="^", markersize=6, label=label)
        if "adaptive" in method_name:
            ax.fill_between(x,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.15)

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Mean qsrc", fontsize=10)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=9, frameon=True)
    ax.set_title("(c) Mean qsrc vs N\n(hetero PPDU, adaptive qsrc convergence)", fontsize=10)

    fig.suptitle(
        "Fig. 14  Per-STA PPDU-Aware Threshold: Adaptive qsrc under Heterogeneous PPDU\n"
        "(OBSS U(%d,%d), occ=%d%%, SNR=20dB, HARQ on)" % (OBSS_MIN, OBSS_MAX,
                                                            int(OBSS_OCCUPANCY * 100)),
        fontsize=11,
    )

    _save_figure(fig, fig_dir, "fig14_ppdu_aware_threshold")
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
        dest = os.path.join(ms_fig, "%s.%s" % (name, ext))
        fig.savefig(dest, **kwargs)
        print("  Figure -> %s" % dest)

    preview = os.path.join(fig_dir, "%s_preview.png" % name)
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print("  Preview -> %s" % preview)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 14 -- Per-STA PPDU-aware threshold vs. threshold=0")
    parser.add_argument("--fast",    action="store_true",
                        help="Quick validation: %d slots" % FAST_SLOTS)
    parser.add_argument("--out-dir", default="results/step9/fig14")
    args = parser.parse_args()

    num_slots = FAST_SLOTS if args.fast else FULL_SLOTS
    out_dir   = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mode       = "FAST" if args.fast else "FULL"
    rate       = _occupancy_to_rate(OBSS_OCCUPANCY)
    total_runs = len(PPDU_CONFIGS) * len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    print("=== Figure 14 [%s] -- %d slots x %d seeds ===" % (mode, num_slots, len(SEEDS)))
    print("    OBSS U(%d,%d), occupancy=%d%%  (rate=%.5f)" % (
        OBSS_MIN, OBSS_MAX, int(OBSS_OCCUPANCY * 100), rate))
    print("    %d PPDU configs x %d N x %d methods x %d seeds = %d runs" % (
        len(PPDU_CONFIGS), len(NUM_STAS_LIST), len(METHODS), len(SEEDS), total_runs))

    rows = run_sweep(num_slots)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting...")
    plot(rows, out_dir)

    print("\nFigure 14 complete")
    print("  Data    : %s" % csv_path)
    print("  Figures : manuscript/figure/fig14_ppdu_aware_threshold.{eps,png,pdf}")


if __name__ == "__main__":
    main()
