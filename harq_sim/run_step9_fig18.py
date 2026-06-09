"""
Figure 18: PND MIMD Parameter Study — c_coll × c_idle Grid

Research question:
  What (c_coll, c_idle) maximizes PND efficiency in a finite NPCA window?
  How does the optimal pair shift with N (STA density)?

PND MIMD update rules:
  Solo win:   DW viable STAs copy sender's τ  (oracle-like sync)
  Collision:  DW viable STAs: τ /= c_coll    (1.0 = no penalty)
  Idle:       DW viable STAs: τ *= c_idle    (>1 = increase)

  c_coll ∈ {1.0, 1.2, 1.5, 2.0, 3.0}
  c_idle ∈ {1.2, 1.5, 2.0, 3.0, 5.0}

PPDU: uniform U[3,12] (same as fig17 v5)
N:    [10, 20, 30, 50]
W_eff:[20, 50, 100, 200]

Panels:
  (a) efficiency vs N  — c_coll sweep (c_idle=1.5 fixed), W_eff=50
  (b) efficiency vs N  — c_idle sweep (c_coll=1.0 fixed), W_eff=50
  (c) heatmap (c_coll × c_idle) at N=20, W_eff=50
  (d) heatmap (c_coll × c_idle) at N=50, W_eff=50

Output:
  manuscript/figure/fig18_pnd_parameter_study.{eps,png,pdf}
  results/step9/fig18/data.csv
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
from matplotlib.patches import Rectangle
import numpy as np

# ─── Parameters ───────────────────────────────────────────────────────────────

C_COLL_LIST = [1.0, 1.2, 1.5, 2.0, 3.0]
C_IDLE_LIST = [1.2, 1.5, 2.0, 3.0, 5.0]

N_LIST    = [10, 20, 30, 50]
WEFF_LIST = [20, 50, 100, 200]
SEEDS     = [42, 123, 456, 789, 1234]

FULL_VISITS = 1000
FAST_VISITS = 50

FAST_N_LIST    = [20, 50]
FAST_WEFF_LIST = [20, 50]
FAST_SEEDS     = [42]

_ORACLE_CC = -1.0   # sentinel for oracle rows
_ORACLE_CI = -1.0

FIELDS = ["c_coll", "c_idle", "N", "W_eff", "seed", "efficiency"]


# ─── PPDU sampler — uniform U[3,12] ───────────────────────────────────────────

def _sample_ppdu(N: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(3, 13, size=N).astype(np.int32)


# ─── Oracle ────────────────────────────────────────────────────────────────────

def _run_oracle(W_eff: int, ppdus: np.ndarray, rng: np.random.Generator) -> int:
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    successes = 0
    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k = int(viable.sum())
        if k == 0:
            break
        tau = 1.0 / k
        tx = rng.random(N) < (tau * viable)
        n_tx = int(tx.sum())
        if n_tx == 1:
            i = int(np.where(tx)[0][0])
            succeeded[i] = True
            W_rem -= int(ppdus[i])
            successes += 1
        else:
            W_rem -= 1
    return successes


# ─── PND visit ─────────────────────────────────────────────────────────────────

def _run_pnd(
    W_eff: int,
    ppdus: np.ndarray,
    rng: np.random.Generator,
    c_coll: float,
    c_idle: float,
    oracle_succs: int,
) -> float:
    """PND MIMD (no CD). Returns efficiency = successes / oracle_succs."""
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    tau = np.full(N, 1.0 / N)
    successes = 0
    _solo_tau = 0.0

    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        if viable.sum() == 0:
            break
        tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)
        n_tx = int(tx.sum())

        if n_tx == 1:
            i = int(np.where(tx)[0][0])
            _solo_tau = float(tau[i])
            succeeded[i] = True
            W_rem -= int(ppdus[i])
            successes += 1
            tau[i] = 0.0
            copy_mask = (~tx) & (~succeeded) & viable
            tau[copy_mask] = _solo_tau
        elif n_tx > 1:
            W_rem -= 1
            if c_coll > 1.0:
                tau[(~tx) & (~succeeded) & viable] /= c_coll
        else:
            W_rem -= 1
            tau[(~succeeded) & viable] *= c_idle

        tau = np.clip(tau, 1e-4, 1.0)

    return (successes / oracle_succs) if oracle_succs > 0 else 0.0


# ─── Main sweep ───────────────────────────────────────────────────────────────

def run_sweep(n_visits: int, n_list: list, weff_list: list, seeds: list) -> list[dict]:
    rows: list[dict] = []
    pnd_configs = [(cc, ci) for cc in C_COLL_LIST for ci in C_IDLE_LIST]
    total = len(n_list) * len(weff_list) * len(seeds) * (1 + len(pnd_configs))
    done = 0

    for N in n_list:
        for W_eff in weff_list:
            for seed in seeds:
                rng_ppdu = np.random.default_rng(seed * 10001 + 7)

                oracle_results: list[tuple[np.ndarray, int]] = []
                for v in range(n_visits):
                    ppdus = _sample_ppdu(N, rng_ppdu)
                    rng_v = np.random.default_rng(seed * 100003 + v)
                    os_v = _run_oracle(W_eff, ppdus, rng_v)
                    oracle_results.append((ppdus.copy(), os_v))

                rows.append({
                    "c_coll": _ORACLE_CC, "c_idle": _ORACLE_CI,
                    "N": N, "W_eff": W_eff, "seed": seed, "efficiency": 1.0,
                })
                done += 1
                print(f"  [{done:5d}/{total}]  oracle            N={N:2d}  W_eff={W_eff:3d}  seed={seed}",
                      flush=True)

                for idx, (cc, ci) in enumerate(pnd_configs):
                    eff_sum = 0.0
                    n_valid = 0
                    for v, (ppdus, os_v) in enumerate(oracle_results):
                        if os_v == 0:
                            continue
                        rng_v = np.random.default_rng(seed * 200003 + v * 31 + idx * 97)
                        eff_sum += _run_pnd(W_eff, ppdus, rng_v, cc, ci, os_v)
                        n_valid += 1
                    rows.append({
                        "c_coll": cc, "c_idle": ci,
                        "N": N, "W_eff": W_eff, "seed": seed,
                        "efficiency": eff_sum / max(n_valid, 1),
                    })
                    done += 1
                    print(f"  [{done:5d}/{total}]  pnd cc={cc:.1f} ci={ci:.1f}   "
                          f"N={N:2d}  W_eff={W_eff:3d}  seed={seed}", flush=True)

    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _eff(rows: list[dict], c_coll: float, c_idle: float, N: int, W_eff: int) -> float:
    vals = [r["efficiency"] for r in rows
            if r["c_coll"] == c_coll and r["c_idle"] == c_idle
            and r["N"] == N and r["W_eff"] == W_eff]
    return float(np.mean(vals)) if vals else float("nan")


def _oracle_eff(rows: list[dict], N: int, W_eff: int) -> float:
    return _eff(rows, _ORACLE_CC, _ORACLE_CI, N, W_eff)


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot(rows: list[dict], n_list: list, weff_list: list, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.50, wspace=0.40)
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    _panel_a(ax_a, rows, n_list)
    _panel_b(ax_b, rows, n_list)

    # heatmap panels: use largest available N
    avail_N = sorted({r["N"] for r in rows})
    n_mid  = avail_N[len(avail_N) // 2] if len(avail_N) >= 2 else avail_N[0]
    n_max  = avail_N[-1]
    w_ref  = 50 if 50 in {r["W_eff"] for r in rows} else weff_list[1]
    _panel_heatmap(ax_c, rows, N_ref=n_mid, W_ref=w_ref, panel_label="(c)")
    _panel_heatmap(ax_d, rows, N_ref=n_max, W_ref=w_ref, panel_label="(d)")

    fig.suptitle(
        "Fig. 18  PND MIMD Parameter Study — c_coll × c_idle Grid\n"
        "(uniform U[3,12], finite NPCA window W_eff slots)",
        fontsize=12,
    )
    _save_figure(fig, fig_dir, "fig18_pnd_parameter_study")
    plt.close(fig)


_COLL_COLORS = plt.cm.plasma(np.linspace(0.10, 0.85, len(C_COLL_LIST)))
_IDLE_COLORS = plt.cm.viridis(np.linspace(0.15, 0.90, len(C_IDLE_LIST)))


def _panel_a(ax, rows: list[dict], n_list: list) -> None:
    C_IDLE_REF = 1.5
    W_EFF_REF  = 50

    oracle = [_oracle_eff(rows, N, W_EFF_REF) for N in n_list]
    ax.plot(n_list, oracle, "k--", lw=1.5, marker="D", ms=5, label="Oracle", zorder=5)

    for c_coll, color in zip(C_COLL_LIST, _COLL_COLORS):
        means = [_eff(rows, c_coll, C_IDLE_REF, N, W_EFF_REF) for N in n_list]
        lbl = f"c_coll={c_coll:.1f}" + ("  [no penalty]" if c_coll == 1.0 else "")
        ax.plot(n_list, means, color=color, lw=2.0, marker="o", ms=5, label=lbl)

    ax.set_xlim(n_list[0] - 3, n_list[-1] + 3)
    ax.set_ylim(0.70, 1.15)
    ax.set_xticks(n_list)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("N (contending STAs)", fontsize=10)
    ax.set_ylabel("Efficiency (successes / oracle)", fontsize=10)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) c_coll sweep  (c_idle={C_IDLE_REF}, W_eff={W_EFF_REF})", fontsize=10)


def _panel_b(ax, rows: list[dict], n_list: list) -> None:
    C_COLL_REF = 1.0
    W_EFF_REF  = 50

    oracle = [_oracle_eff(rows, N, W_EFF_REF) for N in n_list]
    ax.plot(n_list, oracle, "k--", lw=1.5, marker="D", ms=5, label="Oracle", zorder=5)

    for c_idle, color in zip(C_IDLE_LIST, _IDLE_COLORS):
        means = [_eff(rows, C_COLL_REF, c_idle, N, W_EFF_REF) for N in n_list]
        ax.plot(n_list, means, color=color, lw=2.0, marker="s", ms=5,
                label=f"c_idle={c_idle:.1f}")

    ax.set_xlim(n_list[0] - 3, n_list[-1] + 3)
    ax.set_ylim(0.70, 1.15)
    ax.set_xticks(n_list)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("N (contending STAs)", fontsize=10)
    ax.set_ylabel("Efficiency (successes / oracle)", fontsize=10)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(b) c_idle sweep  (c_coll={C_COLL_REF:.1f}, W_eff={W_EFF_REF})", fontsize=10)


def _panel_heatmap(
    ax, rows: list[dict], N_ref: int, W_ref: int, panel_label: str,
) -> None:
    grid = np.full((len(C_IDLE_LIST), len(C_COLL_LIST)), float("nan"))
    for ci_idx, ci in enumerate(C_IDLE_LIST):
        for cc_idx, cc in enumerate(C_COLL_LIST):
            v = _eff(rows, cc, ci, N_ref, W_ref)
            if not np.isnan(v):
                grid[ci_idx, cc_idx] = v

    if np.all(np.isnan(grid)):
        ax.text(0.5, 0.5, f"N={N_ref} / W_eff={W_ref}\nnot in sweep",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{panel_label} (N={N_ref}, W_eff={W_ref})", fontsize=9)
        return

    vmin = max(0.70, float(np.nanmin(grid)) - 0.01)
    vmax = min(1.10, float(np.nanmax(grid)) + 0.01)
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="YlOrRd", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Efficiency", fontsize=9)

    ax.set_xticks(range(len(C_COLL_LIST)))
    ax.set_xticklabels([f"{cc:.1f}" for cc in C_COLL_LIST])
    ax.set_yticks(range(len(C_IDLE_LIST)))
    ax.set_yticklabels([f"{ci:.1f}" for ci in C_IDLE_LIST])
    ax.set_xlabel("c_coll", fontsize=10)
    ax.set_ylabel("c_idle", fontsize=10)

    for ci_idx in range(len(C_IDLE_LIST)):
        for cc_idx in range(len(C_COLL_LIST)):
            v = grid[ci_idx, cc_idx]
            if not np.isnan(v):
                ax.text(cc_idx, ci_idx, f"{v:.3f}",
                        ha="center", va="center", fontsize=7.5, color="black")

    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(Rectangle(
        (best[1] - 0.5, best[0] - 0.5), 1, 1,
        fill=False, edgecolor="#1f77b4", lw=2.5,
    ))

    oracle = _oracle_eff(rows, N_ref, W_ref)
    ax.set_title(
        f"{panel_label} (N={N_ref}, W_eff={W_ref})  oracle={oracle:.4f}\n"
        f"best cc={C_COLL_LIST[best[1]]:.1f} ci={C_IDLE_LIST[best[0]]:.1f} "
        f"eff={float(np.nanmax(grid)):.4f}",
        fontsize=9,
    )


def _save_figure(fig, fig_dir: str, name: str) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ms_fig = os.path.join(repo_root, "manuscript", "figure")
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


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    avail_N    = sorted({r["N"]    for r in rows if r["c_coll"] != _ORACLE_CC})
    avail_Weff = sorted({r["W_eff"] for r in rows if r["c_coll"] != _ORACLE_CC})

    print("\n=== Top-3 configs per (N, W_eff) ===")
    for N in avail_N:
        for W_eff in avail_Weff:
            results = [
                (_eff(rows, cc, ci, N, W_eff), cc, ci)
                for cc in C_COLL_LIST for ci in C_IDLE_LIST
            ]
            results = [(e, cc, ci) for e, cc, ci in results if not np.isnan(e)]
            results.sort(reverse=True)
            oracle = _oracle_eff(rows, N, W_eff)
            top = results[:3]
            print(f"  N={N:2d}  W_eff={W_eff:3d}  oracle={oracle:.4f}  |  "
                  + "  ".join(f"cc={cc:.1f}/ci={ci:.1f}→{e:.4f}" for e, cc, ci in top))


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 18 — PND parameter study")
    parser.add_argument("--fast", action="store_true",
                        help=f"Quick mode: {FAST_VISITS} visits, small N/W grid")
    parser.add_argument("--out-dir", default="results/step9/fig18")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    n_visits  = FAST_VISITS if args.fast else FULL_VISITS
    n_list    = FAST_N_LIST if args.fast else N_LIST
    weff_list = FAST_WEFF_LIST if args.fast else WEFF_LIST
    seeds     = FAST_SEEDS if args.fast else SEEDS

    pnd_configs = [(cc, ci) for cc in C_COLL_LIST for ci in C_IDLE_LIST]
    total = len(n_list) * len(weff_list) * len(seeds) * (1 + len(pnd_configs))

    print(f"=== Figure 18 [{'FAST' if args.fast else 'FULL'}]  {n_visits} visits each ===")
    print(f"    N          ∈ {n_list}")
    print(f"    W_eff      ∈ {weff_list}")
    print(f"    c_coll     ∈ {C_COLL_LIST}")
    print(f"    c_idle     ∈ {C_IDLE_LIST}")
    print(f"    PND configs: {len(pnd_configs)}")
    print(f"    total      : {total} configurations")

    rows = run_sweep(n_visits, n_list, weff_list, seeds)
    save_csv(rows, os.path.join(out_dir, "data.csv"))
    print_summary(rows)

    print("\nPlotting ...")
    plot(rows, n_list, weff_list, out_dir)
    print(f"\nFigure 18 complete.")
    print(f"  Data    : {out_dir}/data.csv")
    print(f"  Figures : manuscript/figure/fig18_pnd_parameter_study.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
