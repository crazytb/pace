"""
Figure 7: PPDU Truncation × qsrc* 최적화 상호작용.

2×2 Factorial:
  truncation: {False (no_trunc), True (trunc)}
  qsrc:       {0 (q0), qsrc*(N) (qstar)}

4 methods:
  no_trunc_q0    — full PPDU required, qsrc=0
  no_trunc_qstar — full PPDU required, qsrc*(N)
  trunc_q0       — truncation enabled, qsrc=0
  trunc_qstar    — truncation enabled, qsrc*(N)

qsrc*(N) = max(0, round(log2(N/16)))  — from guidelines/step9/analysis_qsrc.md

env: obss_max=500, occ=50%, harq=True, ppdu=20, 50000 slots × 3 seeds
"""

import csv
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harq_sim.channel import Channel
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.sta import STA
from harq_sim.simulator import Simulator

# ── 실험 상수 ─────────────────────────────────────────────────────────────────
NUM_STAS_LIST    = [5, 10, 20, 30, 50, 70, 100, 150, 200]
SEEDS            = [42, 123, 456]
OBSS_RATE        = 0.50
OBSS_MIN         = 20
OBSS_MAX         = 500
PPDU_DURATION    = 20
PPDU_MIN_TX      = 5      # 최소 TX 창 (truncation 활성 시)
HARQ_HORIZON     = 200
NPCA_THRESHOLD   = 0
NUM_SLOTS        = 50_000

METHODS = ["no_trunc_q0", "no_trunc_qstar", "trunc_q0", "trunc_qstar"]


def _qsrc_star(n: int) -> int:
    """qsrc*(N) = max(0, round(log2(N/16)))"""
    return max(0, round(math.log2(n / 16))) if n > 0 else 0


def _method_params(method: str, num_stas: int) -> dict:
    truncation = method.startswith("trunc")
    qsrc = _qsrc_star(num_stas) if method.endswith("qstar") else 0
    return {"truncation": truncation, "qsrc": qsrc}


# ── Simulation builder ────────────────────────────────────────────────────────

def build_and_run(
    num_slots: int,
    num_stas: int,
    truncation: bool,
    qsrc: int,
    seed: int,
) -> Simulator:
    random.seed(seed)
    primary = Channel(
        channel_id=0,
        obss_generation_rate=OBSS_RATE,
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
            npca_initial_qsrc=qsrc,
            retry_limit=7,
            infinite_queue=True,
            snr_db_mean=25.0,
            snr_db_std=0.0,
            harq_enabled=True,
            harq_validity_horizon=HARQ_HORIZON,
            policy=policy,
            adaptive_cw=False,
            ppdu_truncation=truncation,
            ppdu_min_tx_slots=PPDU_MIN_TX,
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


def _extract_row(sim: Simulator, method: str, num_stas: int, seed: int) -> dict:
    metrics = sim.compute_metrics()
    agg = metrics["aggregate"]

    total_truncated = sum(
        v.get("npca_tx_truncated", 0)
        for k, v in metrics.items() if k != "aggregate"
    )
    total_npca_tx = sum(
        v.get("npca_tx_success", 0) + v.get("npca_tx_fail", 0)
        for k, v in metrics.items() if k != "aggregate"
    )
    trunc_frac = total_truncated / total_npca_tx if total_npca_tx > 0 else 0.0

    return {
        "method":               method,
        "num_stas":             num_stas,
        "seed":                 seed,
        "aggregate_throughput": agg.get("aggregate_throughput", 0),
        "collision_prob_npca":  agg.get("collision_probability_npca", 0.0),
        "npca_transitions":     agg.get("npca_transition_count", 0),
        "npca_tx_truncated":    total_truncated,
        "truncated_frac":       trunc_frac,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(num_slots: int = NUM_SLOTS) -> list[dict]:
    rows = []
    total = len(NUM_STAS_LIST) * len(METHODS) * len(SEEDS)
    done = 0
    for num_stas in NUM_STAS_LIST:
        for method in METHODS:
            p = _method_params(method, num_stas)
            for seed in SEEDS:
                sim = build_and_run(num_slots, num_stas, p["truncation"], p["qsrc"], seed)
                row = _extract_row(sim, method, num_stas, seed)
                rows.append(row)
                done += 1
                print(
                    f"  [{done:3d}/{total}] {method:<20s} N={num_stas:2d} "
                    f"seed={seed}  TP={row['aggregate_throughput']:5d}  "
                    f"trunc_frac={row['truncated_frac']:.2f}  "
                    f"col={row['collision_prob_npca']:.3f}"
                )
    return rows


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _mean_std_by_n(rows: list[dict], method: str, metric: str):
    ns = sorted({r["num_stas"] for r in rows})
    means, stds = [], []
    for n in ns:
        vals = [r[metric] for r in rows if r["method"] == method and r["num_stas"] == n]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return ns, means, stds


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], fig_dir: str, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    style = {
        "no_trunc_q0":    dict(color="#1f77b4", linestyle="--", marker="o",  label="No-trunc, q=0"),
        "no_trunc_qstar": dict(color="#1f77b4", linestyle="-",  marker="s",  label="No-trunc, q*"),
        "trunc_q0":       dict(color="#d62728", linestyle="--", marker="^",  label="Trunc, q=0"),
        "trunc_qstar":    dict(color="#d62728", linestyle="-",  marker="D",  label="Trunc, q*"),
    }

    # ── Panel (a): Throughput vs N ────────────────────────────────────────────
    ax = axes[0]
    for method in METHODS:
        ns, means, stds = _mean_std_by_n(rows, method, "aggregate_throughput")
        m = np.array(means)
        s = np.array(stds)
        st = style[method]
        ax.plot(ns, m, linestyle=st["linestyle"], color=st["color"],
                marker=st["marker"], label=st["label"])
        ax.fill_between(ns, m - s, m + s, color=st["color"], alpha=0.10)

    ax.set_ylabel("Aggregate Throughput (packets)")
    ax.set_title("(a) Throughput vs Number of STAs")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # ── Panel (b): qsrc* gain (%) vs N ───────────────────────────────────────
    ax = axes[1]
    for trunc_flag, label, color in [
        (False, "No-trunc: gain(q*) - gain(q=0)", "#1f77b4"),
        (True,  "Trunc:    gain(q*) - gain(q=0)", "#d62728"),
    ]:
        pfx = "trunc" if trunc_flag else "no_trunc"
        ns_q0,    m_q0,    _ = _mean_std_by_n(rows, f"{pfx}_q0",    "aggregate_throughput")
        ns_qstar, m_qstar, _ = _mean_std_by_n(rows, f"{pfx}_qstar", "aggregate_throughput")
        gains = [
            (qs - q0) / q0 * 100 if q0 > 0 else 0.0
            for q0, qs in zip(m_q0, m_qstar)
        ]
        ax.plot(ns_q0, gains, color=color,
                linestyle="-" if trunc_flag else "--",
                marker="D" if trunc_flag else "s",
                label=label)

    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Number of STAs (N)")
    ax.set_ylabel("qsrc* Gain over q=0 (%)")
    ax.set_title("(b) qsrc* Contribution: Truncation vs No-Truncation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()

    fig_name = "fig7_truncation_qsrc_massive"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for fmt, kw in [("eps", {}), ("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(os.path.join(fig_dir, f"{fig_name}.{fmt}"),
                    format=fmt, bbox_inches="tight", **kw)
    fig.savefig(os.path.join(out_dir, f"{fig_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {fig_dir}/{fig_name}.{{eps,png,pdf}}")


# ── CSV save ──────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "data.csv")
    fields = ["method", "num_stas", "seed",
              "aggregate_throughput", "collision_prob_npca",
              "npca_transitions", "npca_tx_truncated", "truncated_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved → {path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    print("\n=== Summary: Aggregate Throughput (mean ± std) ===")
    print(f"{'Method':<20s} " + " ".join(f"N={n:2d}" for n in NUM_STAS_LIST))
    for method in METHODS:
        row_str = f"{method:<20s} "
        for n in NUM_STAS_LIST:
            vals = [r["aggregate_throughput"] for r in rows
                    if r["method"] == method and r["num_stas"] == n]
            row_str += f"{np.mean(vals):6.0f} "
        print(row_str)

    print("\n=== qsrc* Gain (%) by Truncation Mode ===")
    print(f"{'N':>4s}  {'No-Trunc gain%':>16s}  {'Trunc gain%':>12s}  {'Δ(Trunc-NoTrunc)':>18s}")
    for n in NUM_STAS_LIST:
        def tp(method):
            return np.mean([r["aggregate_throughput"] for r in rows
                            if r["method"] == method and r["num_stas"] == n])
        gain_no = (tp("no_trunc_qstar") - tp("no_trunc_q0")) / tp("no_trunc_q0") * 100
        gain_tr = (tp("trunc_qstar")    - tp("trunc_q0"))    / tp("trunc_q0")    * 100
        print(f"{n:4d}  {gain_no:16.2f}%  {gain_tr:12.2f}%  {gain_tr - gain_no:18.2f}%")

    print("\n=== Truncated TX Fraction (trunc methods only) ===")
    print(f"{'N':>4s}  {'trunc_q0 frac':>14s}  {'trunc_qstar frac':>16s}")
    for n in NUM_STAS_LIST:
        def frac(method):
            return np.mean([r["truncated_frac"] for r in rows
                            if r["method"] == method and r["num_stas"] == n])
        print(f"{n:4d}  {frac('trunc_q0'):14.3f}  {frac('trunc_qstar'):16.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    out_dir     = os.path.join(project_dir, "results", "step9", "fig7_v2")
    fig_dir     = os.path.join(project_dir, "manuscript", "figure")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=out_dir)
    parser.add_argument("--slots",   type=int, default=NUM_SLOTS)
    args = parser.parse_args()

    print("=== Fig 7: PPDU Truncation × qsrc* Interaction ===")
    print(f"Methods: {METHODS}")
    print(f"N list:  {NUM_STAS_LIST}")
    print(f"Seeds:   {SEEDS},  slots: {args.slots}")
    print(f"PPDU duration={PPDU_DURATION}, min_tx_slots={PPDU_MIN_TX}")
    print()

    rows = run_sweep(args.slots)
    save_csv(rows, args.out_dir)
    print_summary(rows)
    plot(rows, fig_dir, args.out_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
