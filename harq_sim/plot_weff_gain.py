"""
W_eff × N qsrc 최적화 이득 + 처리량 시각화

results/weff_check_full/data.csv 를 사용해 두 개 figure 생성:
  fig_weff_gain  : (a) Heatmap qsrc* 이득(%)  (b) TP vs qsrc [N=50]  (c) 이득 vs W_eff
  fig_weff_tput  : (a) TP vs N [qsrc=0]       (b) TP vs N [qsrc*]    (c) TP vs N [qsrc* − qsrc=0 절대 차이]

출력:
  manuscript/figure/fig_weff_gain.{eps,png,pdf}
  manuscript/figure/fig_weff_tput.{eps,png,pdf}
"""
from __future__ import annotations

import csv
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

DATA_CSV = "results/weff_check_full/data.csv"

OBSS_D_LIST   = [40, 80, 150, 300, 500]
WEFF_LIST     = [D - 20 for D in OBSS_D_LIST]   # [20, 60, 130, 280, 480]
NUM_STAS_LIST = [5, 10, 20, 30, 50]
QSRC_LIST     = [0, 1, 2, 3, 4, 5]
PPDU          = 20

N_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#9467bd"]
W_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> list[dict]:
    return list(csv.DictReader(open(path)))

def tp_mean(rows, D, n, q):
    vals = [float(r["aggregate_throughput"]) for r in rows
            if r["obss_duration"] == str(D)
            and r["num_stas"] == str(n)
            and r["npca_qsrc"] == str(q)]
    return mean(vals) if vals else 0.0

def gain_pct(rows, D, n):
    """qsrc=0 대비 최적 qsrc* 이득(%)"""
    q0   = tp_mean(rows, D, n, 0)
    best = max(tp_mean(rows, D, n, q) for q in QSRC_LIST)
    return (best - q0) / q0 * 100 if q0 > 0 else 0.0

def opt_qsrc(rows, D, n):
    return max(QSRC_LIST, key=lambda q: tp_mean(rows, D, n, q))

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], out_dir: str) -> None:
    fig = plt.figure(figsize=(14, 4.8))
    gs  = fig.add_gridspec(1, 3, wspace=0.40)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Panel (a): Heatmap qsrc* 이득 (%) ────────────────────────────────────
    grid = np.array([[gain_pct(rows, D, n)
                      for D in OBSS_D_LIST]
                     for n in NUM_STAS_LIST], dtype=float)

    cmap = plt.cm.YlOrRd
    im   = ax_a.imshow(grid, aspect="auto", cmap=cmap,
                       origin="lower", vmin=0, vmax=16)

    ax_a.set_xticks(range(len(WEFF_LIST)))
    ax_a.set_xticklabels([str(w) for w in WEFF_LIST], fontsize=9)
    ax_a.set_yticks(range(len(NUM_STAS_LIST)))
    ax_a.set_yticklabels([f"N={n}" for n in NUM_STAS_LIST], fontsize=9)
    ax_a.set_xlabel("$W_{\\mathrm{eff}}$ (slots)", fontsize=10)
    ax_a.set_ylabel("Number of STAs", fontsize=10)

    for i, n in enumerate(NUM_STAS_LIST):
        for j, D in enumerate(OBSS_D_LIST):
            val    = grid[i, j]
            q_star = opt_qsrc(rows, D, n)
            text   = f"{val:.1f}%\n(q*={q_star})"
            color  = "white" if val > 9 else "black"
            ax_a.text(j, i, text, ha="center", va="center",
                      fontsize=8, color=color)

    plt.colorbar(im, ax=ax_a, label="Gain over qsrc=0 (%)", shrink=0.85)
    ax_a.set_title("(a) qsrc optimization gain\n"
                   r"$\Delta$TP = (TP$_{q^*}$ − TP$_{q=0}$) / TP$_{q=0}$",
                   fontsize=10)

    # ── Panel (b): Throughput vs qsrc, N=50, W_eff별 ─────────────────────────
    N_FIXED = 50
    for idx, (D, weff) in enumerate(zip(OBSS_D_LIST, WEFF_LIST)):
        means = [tp_mean(rows, D, N_FIXED, q) for q in QSRC_LIST]
        q_star = opt_qsrc(rows, D, N_FIXED)
        ax_b.plot(QSRC_LIST, means, color=W_COLORS[idx], marker="o", lw=1.8,
                  label=f"$W_{{\\mathrm{{eff}}}}$={weff}")
        ax_b.plot(q_star, means[q_star], marker="*", color=W_COLORS[idx],
                  markersize=13, zorder=5)

    ax_b.set_xlabel("qsrc", fontsize=10)
    ax_b.set_ylabel("Aggregate Throughput (packets)", fontsize=10)
    ax_b.set_xticks(QSRC_LIST)
    ax_b.set_xticklabels([f"q={q}\n(CW={2**q*16-1})" for q in QSRC_LIST], fontsize=8)
    ax_b.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_b.legend(fontsize=8, loc="upper right", frameon=True)
    ax_b.set_title(f"(b) Throughput vs qsrc  [N={N_FIXED}]\n"
                   r"★ = $q^*$, short $W_{\mathrm{eff}}$ widens q=0 penalty",
                   fontsize=10)

    # ── Panel (c): 이득(%) vs W_eff, N별 ─────────────────────────────────────
    for idx, n in enumerate(NUM_STAS_LIST):
        gains = [gain_pct(rows, D, n) for D in OBSS_D_LIST]
        ax_c.plot(WEFF_LIST, gains, color=N_COLORS[idx], marker="o", lw=1.8,
                  label=f"N={n}")

    ax_c.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax_c.set_xlabel("$W_{\\mathrm{eff}}$ (slots)", fontsize=10)
    ax_c.set_ylabel("qsrc* gain over qsrc=0 (%)", fontsize=10)
    ax_c.set_xticks(WEFF_LIST)
    ax_c.set_xticklabels([str(w) for w in WEFF_LIST], fontsize=9)
    ax_c.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax_c.legend(fontsize=9, frameon=True)
    ax_c.set_title("(c) CW Amnesia penalty amplification\n"
                   r"short $W_{\mathrm{eff}}$ + high $N$ → largest qsrc gain",
                   fontsize=10)

    fig.suptitle(
        "NPCA CW Amnesia: qsrc Optimization Gain vs. $W_{\\mathrm{eff}}$ and $N$\n"
        "(OBSS occ=50%, SNR=20 dB, PPDU=20 slots, 30k slots × 3 seeds)",
        fontsize=11,
    )

    _save(fig, out_dir, "fig_weff_gain")
    plt.close(fig)


def _save(fig, out_dir: str, name: str) -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ms   = os.path.join(repo, "manuscript", "figure")
    os.makedirs(ms, exist_ok=True)

    for ext, kw in [
        ("eps", dict(format="eps",  bbox_inches="tight")),
        ("png", dict(format="png",  bbox_inches="tight", dpi=300)),
        ("pdf", dict(format="pdf",  bbox_inches="tight")),
    ]:
        path = os.path.join(ms, f"{name}.{ext}")
        fig.savefig(path, **kw)
        print(f"  → {path}")

    preview = os.path.join(out_dir, f"{name}.png")
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print(f"  Preview → {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def plot_tput(rows: list[dict], out_dir: str) -> None:
    """Throughput vs N figure: qsrc=0 / qsrc* / 절대 이득"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel (a): Throughput vs N — qsrc=0 (suboptimal baseline) ────────────
    ax = axes[0]
    for idx, (D, weff) in enumerate(zip(OBSS_D_LIST, WEFF_LIST)):
        means = [tp_mean(rows, D, n, 0) for n in NUM_STAS_LIST]
        ax.plot(NUM_STAS_LIST, means, color=W_COLORS[idx], marker="o", lw=1.8,
                label=f"$W_{{\\mathrm{{eff}}}}$={weff}")

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Aggregate Throughput (packets)", fontsize=10)
    ax.set_xticks(NUM_STAS_LIST)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=9, frameon=True)
    ax.set_title("(a) Throughput vs N — qsrc = 0\n"
                 "short $W_{\\mathrm{eff}}$ hurts at high N", fontsize=10)

    # ── Panel (b): Throughput vs N — qsrc* (optimal) ─────────────────────────
    ax = axes[1]
    for idx, (D, weff) in enumerate(zip(OBSS_D_LIST, WEFF_LIST)):
        means = [tp_mean(rows, D, n, opt_qsrc(rows, D, n)) for n in NUM_STAS_LIST]
        ax.plot(NUM_STAS_LIST, means, color=W_COLORS[idx], marker="o", lw=1.8,
                label=f"$W_{{\\mathrm{{eff}}}}$={weff}")

    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("Aggregate Throughput (packets)", fontsize=10)
    ax.set_xticks(NUM_STAS_LIST)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=9, frameon=True)
    ax.set_title("(b) Throughput vs N — qsrc = qsrc*\n"
                 "$W_{\\mathrm{eff}}$ gap closes with optimal CW", fontsize=10)

    # ── Panel (c): 절대 처리량 차이 (qsrc* − qsrc=0) ─────────────────────────
    ax = axes[2]
    for idx, (D, weff) in enumerate(zip(OBSS_D_LIST, WEFF_LIST)):
        diffs = [tp_mean(rows, D, n, opt_qsrc(rows, D, n)) - tp_mean(rows, D, n, 0)
                 for n in NUM_STAS_LIST]
        ax.plot(NUM_STAS_LIST, diffs, color=W_COLORS[idx], marker="^", lw=1.8,
                label=f"$W_{{\\mathrm{{eff}}}}$={weff}")

    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlabel("Number of STAs", fontsize=10)
    ax.set_ylabel("TP gain: qsrc* − qsrc=0 (packets)", fontsize=10)
    ax.set_xticks(NUM_STAS_LIST)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=9, frameon=True)
    ax.set_title("(c) Absolute gain from qsrc optimization\n"
                 "largest at short $W_{\\mathrm{eff}}$ + high N", fontsize=10)

    fig.suptitle(
        "NPCA Throughput: W_eff × N  (OBSS occ=50%, SNR=20 dB, PPDU=20 slots)",
        fontsize=11,
    )

    _save(fig, out_dir, "fig_weff_tput")
    plt.close(fig)


def main() -> None:
    rows = load(DATA_CSV)
    out  = os.path.dirname(DATA_CSV)
    plot(rows, out)
    plot_tput(rows, out)
    print("완료")

if __name__ == "__main__":
    main()
