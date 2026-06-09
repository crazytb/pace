"""
Figure 15: MFG-Optimal NPCA Access Protocol vs DCF-Static Backoff

Standalone NPCA visit simulation (no harq_sim STA/Simulator dependencies).
Compares DCF-static vs MFG-optimal protocols in finite-horizon NPCA contention.

Output:
  manuscript/figure/fig15_mfg_npca.{eps,png,pdf}
  results/step9/fig15/data.csv

Run:
  python harq_sim/run_step9_fig15.py [--fast] [--out-dir results/step9/fig15]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

N_LIST    = [5, 10, 20, 30, 50]
WEFF_LIST = [20, 50, 100, 200, 500]
SEEDS     = [42, 123, 456]

FULL_VISITS   = 1000
FAST_VISITS   = 100
ORACLE_VISITS = 300


# ─────────────────────────────────────────────────────────────────────────────
# DCF batch simulation (NumPy vectorized over n_visits)
# ─────────────────────────────────────────────────────────────────────────────

def sim_dcf_batch(
    N: int, W_eff: int, CW0: int, n_visits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate n_visits NPCA DCF visits simultaneously (vectorized).

    Model:
    - N STAs, each starts with backoff ~ U{0, CW0-1}
    - Idle slot: all active STAs decrement backoff
    - Busy slot (solo or collision): non-participants freeze (standard DCF)
    - Solo TX → success, STA exits
    - Collision → BEB: CW doubles, new backoff drawn; STA stays

    Returns: int array shape (n_visits,) — success count per visit.
    """
    V = n_visits
    CW0 = max(CW0, 1)
    backoffs = rng.integers(0, CW0, size=(V, N))
    cws      = np.full((V, N), CW0, dtype=np.int32)
    active   = np.ones((V, N), dtype=bool)
    success  = np.zeros(V, dtype=np.int32)

    for _ in range(W_eff):
        if not active.any():
            break

        ready   = active & (backoffs == 0)   # (V, N)
        n_ready = ready.sum(axis=1)           # (V,)

        idle = n_ready == 0
        solo = n_ready == 1
        coll = n_ready > 1

        # Solo: success — deactivate the transmitting STA
        if solo.any():
            active  &= ~(ready & solo[:, None])
            success += solo.astype(np.int32)

        # Collision: BEB — double CW, draw new random backoff
        if coll.any():
            col_sta  = ready & coll[:, None]        # (V, N) mask of colliders
            new_cws  = np.minimum(2 * (cws + 1) - 1, 1023)
            cws      = np.where(col_sta, new_cws, cws)
            # Draw random integers in [0, 1023]; mod (cws+1) is exact since cws+1 divides 1024
            r        = rng.integers(0, 1024, size=(V, N), dtype=np.int32)
            new_bo   = r % (cws + 1)
            backoffs = np.where(col_sta, new_bo, backoffs)

        # Decrement only on idle slots (channel free → backoff runs)
        if idle.any():
            dec      = active & (backoffs > 0) & idle[:, None]
            backoffs = backoffs - dec.astype(np.int32)

    return success


# ─────────────────────────────────────────────────────────────────────────────
# MFG adaptive batch simulation
# ─────────────────────────────────────────────────────────────────────────────

def sim_mfg_adaptive_batch(
    N: int, W_eff: int, n_visits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    MFG adaptive: τ*(t) = 1/remaining per slot (Bernoulli TX).

    Each slot: every remaining STA transmits independently with prob τ*(t).
    - Exactly 1 TX → success, STA exits; remaining decreases
    - >1 TX → collision; all retry next slot (remaining unchanged)
    - 0 TX → idle

    This implements the fixed-point condition τ*(t) = 1/(N·n(t)) with
    n(t) = remaining/N → τ*(t) = 1/remaining.

    Returns: int array shape (n_visits,) — success count per visit.
    """
    V         = n_visits
    remaining = np.full(V, N, dtype=np.float64)
    success   = np.zeros(V, dtype=np.int32)

    for _ in range(W_eff):
        active = remaining > 0
        if not active.any():
            break

        tau   = np.where(active, 1.0 / np.maximum(remaining, 1.0), 0.0)
        n_rem = np.round(remaining).astype(np.int32)
        n_tx  = rng.binomial(n_rem, tau.clip(0.0, 1.0))

        solo      = (n_tx == 1) & active
        success  += solo.astype(np.int32)
        remaining = np.where(solo, remaining - 1.0, remaining)
        remaining = np.maximum(remaining, 0.0)
        # n_tx > 1: collision; STAs retry (remaining unchanged)

    return success


# ─────────────────────────────────────────────────────────────────────────────
# Oracle: best DCF CW0 by simulation sweep
# ─────────────────────────────────────────────────────────────────────────────

def _oracle_candidates(N: int, W_eff: int) -> list[int]:
    cands = set()
    v = max(1, N // 4)
    while v <= max(4 * N, W_eff):
        cands.add(v)
        v = int(v * 1.5) + 1
    cands.update([N // 2, N, 2 * N, 3 * N, W_eff])
    return sorted(c for c in cands if 1 <= c <= max(4 * N, W_eff))


def find_oracle(N: int, W_eff: int, rng: np.random.Generator,
                n_visits: int = ORACLE_VISITS) -> tuple[int, float]:
    best_cw, best_mean = 1, 0.0
    for cw in _oracle_candidates(N, W_eff):
        m = float(sim_dcf_batch(N, W_eff, cw, n_visits, rng).mean())
        if m > best_mean:
            best_mean, best_cw = m, cw
    return best_cw, best_mean


# ─────────────────────────────────────────────────────────────────────────────
# MFG analytical solution
# ─────────────────────────────────────────────────────────────────────────────

def mfg_tau_analytical(N: int, t: np.ndarray) -> np.ndarray:
    """τ*(t) = 1/(N−t) for t < N, NaN otherwise."""
    denom = np.where(t < N, (N - t).astype(float), 1.0)
    tau   = 1.0 / denom
    tau[t >= N] = np.nan
    return tau


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(n_visits: int) -> list[dict]:
    rows  = []
    total = len(N_LIST) * len(WEFF_LIST) * 4 * len(SEEDS)
    done  = 0

    for N in N_LIST:
        for W_eff in WEFF_LIST:
            for seed in SEEDS:
                # ── DCF qsrc* (CW0 = 2N) ──
                rng = np.random.default_rng(seed)
                s   = sim_dcf_batch(N, W_eff, 2 * N, n_visits, rng)
                rows.append(_mkrow(N, W_eff, "dcf_qsrc_star", seed, 2*N, s))
                done += 1
                _log(done, total, N, W_eff, "dcf_qsrc_star", seed)

                # ── MFG precommit (CW0 = N) ──
                rng = np.random.default_rng(seed)
                s   = sim_dcf_batch(N, W_eff, N, n_visits, rng)
                rows.append(_mkrow(N, W_eff, "mfg_precommit", seed, N, s))
                done += 1
                _log(done, total, N, W_eff, "mfg_precommit", seed)

                # ── MFG adaptive ──
                rng = np.random.default_rng(seed)
                s   = sim_mfg_adaptive_batch(N, W_eff, n_visits, rng)
                rows.append(_mkrow(N, W_eff, "mfg_adaptive", seed, None, s))
                done += 1
                _log(done, total, N, W_eff, "mfg_adaptive", seed)

                # ── Oracle ──
                rng = np.random.default_rng(seed + 99999)
                best_cw, best_mean = find_oracle(N, W_eff, rng)
                rows.append({
                    "N": N, "W_eff": W_eff, "protocol": "oracle",
                    "seed": seed, "CW0": best_cw,
                    "mean_success": best_mean, "std_success": 0.0,
                })
                done += 1
                _log(done, total, N, W_eff, "oracle", seed)

    return rows


def _mkrow(N, W_eff, proto, seed, cw0, s_arr):
    return {
        "N": N, "W_eff": W_eff, "protocol": proto, "seed": seed,
        "CW0": cw0,
        "mean_success": float(s_arr.mean()),
        "std_success":  float(s_arr.std()),
    }


def _log(done, total, N, W_eff, proto, seed):
    print(f"  [{done:3d}/{total}] N={N:2d}  W_eff={W_eff:3d}  {proto:<16}  seed={seed}",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def _mean_over_seeds(rows, N, W_eff, proto):
    vals = [r["mean_success"] for r in rows
            if r["N"] == N and r["W_eff"] == W_eff and r["protocol"] == proto]
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ["N", "W_eff", "protocol", "seed", "CW0", "mean_success", "std_success"]


def save_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = {
    "dcf_qsrc_star": dict(color="#888888", ls="--", lw=1.8, marker="s", ms=5),
    "mfg_precommit": dict(color="#1f77b4", ls="-.", lw=1.8, marker="^", ms=5),
    "mfg_adaptive":  dict(color="#d62728", ls="-",  lw=2.2, marker="o", ms=6),
    "oracle":        dict(color="#2ca02c", ls="--", lw=1.5, marker="D", ms=5),
}
_LABEL = {
    "dcf_qsrc_star": "DCF qsrc* (CW₀=2N)",
    "mfg_precommit": "MFG precommit (CW₀=N)",
    "mfg_adaptive":  "MFG adaptive",
    "oracle":        "Oracle (best CW₀)",
}


def plot(rows: list, fig_dir: str) -> None:
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    ax_a  = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[0, 2])
    ax_b3 = fig.add_subplot(gs[1, 0])
    ax_c  = fig.add_subplot(gs[1, 1:])

    _panel_a_analytical(ax_a)
    for ax, W_eff, tag in [
        (ax_b1, 50,  "(b1)"),
        (ax_b2, 100, "(b2)"),
        (ax_b3, 500, "(b3)"),
    ]:
        _panel_b_tput(ax, rows, W_eff, tag)
    _panel_c_heatmap(ax_c, rows)

    fig.suptitle(
        "Fig. 15  MFG-Optimal vs DCF-Static Backoff — Finite-Horizon NPCA Contention\n"
        "(NPCA visit model: N STAs × W_eff available slots, BEB retries within visit)",
        fontsize=11,
    )
    _save_figure(fig, fig_dir, "fig15_mfg_npca")
    plt.close(fig)


def _panel_a_analytical(ax) -> None:
    """τ*(t) = 1/(N−t) analytical curves for multiple N."""
    t_max = 55
    t     = np.arange(t_max)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for N, c in zip([10, 20, 30, 50], colors):
        tau   = mfg_tau_analytical(N, t)
        valid = t < N
        ax.plot(t[valid], tau[valid], color=c, lw=1.8, label=f"MFG N={N}")
        ax.axhline(1.0 / (2 * N), color=c, ls=":", lw=1.1, alpha=0.55)

    ax.set_xlabel("Slot t within NPCA visit", fontsize=10)
    ax.set_ylabel("TX probability τ(t)", fontsize=10)
    ax.set_xlim(-1, t_max)
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(a) Analytical τ*(t) = 1/(N−t) (solid)\nvs DCF τ_static = 1/(2N) (dotted)",
                 fontsize=10)


def _panel_b_tput(ax, rows: list, W_eff: int, tag: str) -> None:
    """Success TX per visit vs N for a fixed W_eff."""
    x = N_LIST
    for proto, style in _STYLE.items():
        means = [_mean_over_seeds(rows, N, W_eff, proto) for N in x]
        if not any(m > 0 for m in means):
            continue
        ax.plot(x, means, label=_LABEL[proto], **style)

    ax.set_xlabel("N (contending STAs)", fontsize=10)
    ax.set_ylabel("Mean success TX / visit", fontsize=10)
    ax.set_xticks(x)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"{tag} Throughput per visit vs N\n(W_eff={W_eff})", fontsize=10)


def _panel_c_heatmap(ax, rows: list) -> None:
    """Δgain heatmap: (mfg_adaptive − dcf_qsrc_star) / dcf_qsrc_star × 100%."""
    gain = np.zeros((len(N_LIST), len(WEFF_LIST)))

    for i, N in enumerate(N_LIST):
        for j, W_eff in enumerate(WEFF_LIST):
            dcf  = _mean_over_seeds(rows, N, W_eff, "dcf_qsrc_star")
            mfg  = _mean_over_seeds(rows, N, W_eff, "mfg_adaptive")
            gain[i, j] = (mfg - dcf) / max(dcf, 1e-9) * 100.0

    vmax = max(float(gain.max()), 1.0)
    vmin = min(float(gain.min()), -vmax / 4)
    im = ax.imshow(gain, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, origin="lower")

    ax.set_xticks(range(len(WEFF_LIST)))
    ax.set_xticklabels(WEFF_LIST)
    ax.set_yticks(range(len(N_LIST)))
    ax.set_yticklabels(N_LIST)
    ax.set_xlabel("W_eff (available NPCA slots)", fontsize=10)
    ax.set_ylabel("N (contending STAs)", fontsize=10)

    for i in range(len(N_LIST)):
        for j in range(len(WEFF_LIST)):
            ax.text(j, i, f"{gain[i, j]:+.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="black" if abs(gain[i, j]) < 0.6 * vmax else "white")

    # Diagonal W_eff = 2N reference line
    n_arr = np.array(N_LIST, dtype=float)
    w_arr = np.array(WEFF_LIST, dtype=float)
    for mult, ls, lbl in [(2, "--", "W_eff=2N"), (1, ":", "W_eff=N")]:
        x_pts, y_pts = [], []
        for j, W in enumerate(WEFF_LIST):
            for i, N in enumerate(N_LIST):
                if abs(W - mult * N) < (w_arr[1] - w_arr[0]) * 0.6:
                    x_pts.append(j)
                    y_pts.append(i)
        if x_pts:
            ax.plot(x_pts, y_pts, "k" + ls, lw=1.5, label=lbl)

    plt.colorbar(im, ax=ax, label="Gain % vs DCF qsrc*", fraction=0.03)
    ax.set_title("(c) Gain: MFG adaptive vs DCF qsrc* (CW₀=2N)\n"
                 "(dashed: W_eff=2N boundary; dotted: W_eff=N)",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=True, loc="lower right")


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
        description="Figure 15 — MFG-optimal vs DCF-static NPCA access")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_VISITS} visits/config")
    parser.add_argument("--out-dir", default="results/step9/fig15")
    args = parser.parse_args()

    n_visits = FAST_VISITS if args.fast else FULL_VISITS
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    total = len(N_LIST) * len(WEFF_LIST) * 4 * len(SEEDS)
    print(f"=== Figure 15 [{'FAST' if args.fast else 'FULL'}]  {n_visits} visits each ===")
    print(f"    N      ∈ {N_LIST}")
    print(f"    W_eff  ∈ {WEFF_LIST}")
    print(f"    seeds    {SEEDS}")
    print(f"    configs  {total}  (4 protocols × {len(N_LIST)}N × {len(WEFF_LIST)}W_eff × {len(SEEDS)} seeds)")

    rows = run_sweep(n_visits)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting ...")
    plot(rows, out_dir)

    print("\nFigure 15 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : manuscript/figure/fig15_mfg_npca.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
