"""
Figure 19: DCF-Benchmark Comparison — Throughput + τ Trajectory

4 methods:
  dcf_self_excl : IEEE 802.11 DCF + self-exclusion (benchmark)
  ema_ad_low    : EMA adaptive β, α↓=0.10
  consec_L2     : Consecutive L=2 idle trigger
  and           : Phase-based open-loop (Vasudevan et al., MobiCom 2009)

2 panels:
  (a) Efficiency vs N  (W_eff=50, uniform U[3,12])
  (b) τ trajectory     (N=20, W_eff=100, seed=42, uniform)

Run:
  # Reuse fig17_v7 data (recommended)
  .venv/bin/python harq_sim/run_step9_fig19.py \\
      --base-csv results/step9/fig17_v7/data.csv

  # Fresh simulation (4 methods + oracle)
  .venv/bin/python harq_sim/run_step9_fig19.py

  # Quick test
  .venv/bin/python harq_sim/run_step9_fig19.py --fast
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_step9_fig17 as _f17

# ─── Methods ──────────────────────────────────────────────────────────────────

METHODS_19 = ["dcf_self_excl", "ema_ad_low", "consec_L2", "and", "pnd"]
METHODS_19_SIM = METHODS_19 + ["oracle"]   # oracle needed for ref line

# ─── Style ────────────────────────────────────────────────────────────────────

_STYLE = {
    "dcf_self_excl": dict(color="#08306b", ls="-",  lw=2.0, marker="8", ms=6),
    "ema_ad_low":    dict(color="#e6550d", ls="-",  lw=2.0, marker="p", ms=6),
    "consec_L2":     dict(color="#756bb1", ls="--", lw=2.0, marker="x", ms=7,
                          markeredgewidth=1.8),
    "and":           dict(color="#7f2704", ls=":",  lw=1.8, marker="d", ms=5),
    "pnd":           dict(color="#17becf", ls="-",  lw=2.0, marker="P", ms=6),
}
_LABEL = {
    "dcf_self_excl": "DCF + self-excl",
    "ema_ad_low":    "EMA adaptive (α↓=0.10)",
    "consec_L2":     "Consec L=2",
    "and":           "AND (open-loop)",
    "pnd":           f"PND MIMD (cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE})",
}

# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows: list) -> None:
    W_ref = 50
    P_ref = "uniform"

    # Oracle reference (thin gray)
    oracle_means = [_f17._mean_metric(rows, "W_eff_utilization",
                                      method="oracle", ppdu_dist=P_ref,
                                      N=N, W_eff=W_ref)
                    for N in _f17.N_LIST]
    ax.plot(_f17.N_LIST, oracle_means,
            color="gray", ls="--", lw=1.0, marker="D", ms=4,
            label="Oracle (reference)", zorder=1)

    for method in METHODS_19:
        means = [_f17._mean_metric(rows, "W_eff_utilization",
                                   method=method, ppdu_dist=P_ref,
                                   N=N, W_eff=W_ref)
                 for N in _f17.N_LIST]
        ax.plot(_f17.N_LIST, means, label=_LABEL[method], **_STYLE[method])

    ax.set_xlabel("N (contending STAs)", fontsize=11)
    ax.set_ylabel("NPCA Window Utilization  Σ(ppdu·succ) / W_eff", fontsize=10)
    ax.set_xticks(_f17.N_LIST)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, frameon=True, loc="lower right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) NPCA Window Utilization vs N\n(W_eff={W_ref}, uniform U[3,12])",
                 fontsize=10)


def _panel_b(ax, traj_data: dict) -> None:
    style_traj = {
        "dcf_self_excl": dict(color="#08306b", ls="-",  lw=2.0),
        "ema_ad_low":    dict(color="#e6550d", ls="-",  lw=2.0),
        "consec_L2":     dict(color="#756bb1", ls="--", lw=2.0),
        "and":           dict(color="#7f2704", ls=":",  lw=1.8),
        "pnd":           dict(color="#17becf", ls="-",  lw=2.0),
    }

    ppdu_thresholds = None
    for method in METHODS_19:
        if method not in traj_data:
            continue
        data  = traj_data[method]
        w_rem = data["w_rem"]
        tau   = data["tau"]
        if w_rem:
            ax.plot(w_rem, tau, label=_LABEL[method], **style_traj[method])
        if ppdu_thresholds is None:
            ppdu_thresholds = data.get("ppdu_thresholds", [])

    if ppdu_thresholds:
        first = True
        for thresh in sorted(set(ppdu_thresholds)):
            lbl = "STA ppdu_i thresholds" if first else None
            ax.axvline(thresh, color="gray", ls=":", lw=0.8, alpha=0.6, label=lbl)
            first = False

    ax.invert_xaxis()
    ax.set_xlabel("W_rem (remaining slots, left=high)", fontsize=11)
    ax.set_ylabel("τ (TX probability)", fontsize=11)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(b) τ trajectory — single visit\n(N=20, W_eff=100, seed=42, uniform)",
                 fontsize=10)


# ─── Plot + save ──────────────────────────────────────────────────────────────

def plot(rows: list, traj_data: dict, out_dir: str, fig_dir: str) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.subplots_adjust(wspace=0.35)

    _panel_a(ax_a, rows)
    _panel_b(ax_b, traj_data)

    fig.suptitle(
        f"Fig. 19  PND MIMD vs Alternatives — Throughput & τ Trajectory\n"
        f"(PND cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE}; uniform U[3,12])",
        fontsize=11,
    )

    fig_name = "fig19_dcf_benchmark"
    for ext, kwargs in [
        ("eps", dict(format="eps", bbox_inches="tight")),
        ("png", dict(format="png", bbox_inches="tight", dpi=300)),
        ("pdf", dict(format="pdf", bbox_inches="tight")),
    ]:
        dest = os.path.join(fig_dir, f"{fig_name}.{ext}")
        fig.savefig(dest, **kwargs)
        print(f"  Figure → {dest}")

    preview = os.path.join(out_dir, f"{fig_name}_preview.png")
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print(f"  Preview → {preview}")

    plt.close(fig)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows: list) -> None:
    print("\n=== Hypothesis Check ===")
    W, P, N = 50, "uniform", 20

    METRIC = "W_eff_utilization"

    dcf  = _f17._mean_metric(rows, METRIC, method="dcf_self_excl", ppdu_dist=P, N=N, W_eff=W)
    ema  = _f17._mean_metric(rows, METRIC, method="ema_ad_low",    ppdu_dist=P, N=N, W_eff=W)
    cl2  = _f17._mean_metric(rows, METRIC, method="consec_L2",     ppdu_dist=P, N=N, W_eff=W)
    and_ = _f17._mean_metric(rows, METRIC, method="and",           ppdu_dist=P, N=N, W_eff=W)
    pnd_ = _f17._mean_metric(rows, METRIC, method="pnd",           ppdu_dist=P, N=N, W_eff=W)

    print(f"\nH1: pnd        > dcf_self_excl   {pnd_:.4f} vs {dcf:.4f}  "
          + ("✅ PASS" if pnd_ > dcf else "❌ FAIL"))
    print(f"H2: ema_ad_low > dcf_self_excl   {ema:.4f} vs {dcf:.4f}  "
          + ("✅ PASS" if ema > dcf else "❌ FAIL"))
    print(f"H3: consec_L2  > dcf_self_excl   {cl2:.4f} vs {dcf:.4f}  "
          + ("✅ PASS" if cl2 > dcf else "❌ FAIL"))
    print(f"H4: and        < dcf_self_excl   {and_:.4f} vs {dcf:.4f}  "
          + ("✅ PASS" if and_ < dcf else "❌ FAIL"))
    print(f"H5: pnd        > ema_ad_low      {pnd_:.4f} vs {ema:.4f}  "
          + ("✅ PASS" if pnd_ > ema else "❌ FAIL"))

    # H6: DCF degrades faster with N
    avail_N = sorted({r["N"] for r in rows
                      if r["ppdu_dist"] == P and r["W_eff"] == W
                      and r["method"] == "dcf_self_excl"})
    print(f"\nH6: DCF degrades faster with N than PND  (available N={avail_N})")
    if len(avail_N) >= 2:
        dcf_vals = [_f17._mean_metric(rows, METRIC,
                                      method="dcf_self_excl", ppdu_dist=P, N=n, W_eff=W)
                    for n in avail_N]
        pnd_vals = [_f17._mean_metric(rows, METRIC,
                                      method="pnd", ppdu_dist=P, N=n, W_eff=W)
                    for n in avail_N]
        dcf_drop = dcf_vals[0] - dcf_vals[-1]
        pnd_drop = pnd_vals[0] - pnd_vals[-1]
        print(f"  DCF drop N={avail_N[0]}→{avail_N[-1]}: {dcf_drop:+.4f}")
        print(f"  PND drop N={avail_N[0]}→{avail_N[-1]}: {pnd_drop:+.4f}")
        print(f"  → {'✅ PASS' if dcf_drop > pnd_drop else '❌ FAIL'}")
    else:
        print("  ⚠️ Not enough N values — skip")

    # Summary table
    print(f"\n--- W_eff_utilization summary (uniform, W_eff={W}, N∈{avail_N}) ---")
    header = f"{'method':<20}" + "".join(f"  N={n:2d}" for n in avail_N)
    print(f"  {header}")
    for m in METHODS_19:
        vals = [_f17._mean_metric(rows, METRIC,
                                  method=m, ppdu_dist=P, N=n, W_eff=W)
                for n in avail_N]
        print(f"  {m:<20}" + "".join(f"  {v:.4f}" for v in vals))


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 19 — DCF-benchmark comparison (throughput + τ trajectory)")
    parser.add_argument("--fast",     action="store_true",
                        help=f"Quick mode: {_f17.FAST_VISITS} visits, small N/W grid")
    parser.add_argument("--out-dir",  default="results/step9/fig19")
    parser.add_argument("--base-csv", default=None, metavar="PATH",
                        help="Existing data CSV (e.g. fig17_v7/data.csv) to skip re-simulation")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir   = os.path.join(repo_root, "manuscript", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load or simulate ──────────────────────────────────────────────────────
    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = _f17.load_csv(args.base_csv)
        present = {r["method"] for r in rows}
        missing = [m for m in METHODS_19_SIM if m not in present]
        if missing:
            print(f"  Missing: {missing} — simulating ...")
            nv = _f17.FAST_VISITS if args.fast else _f17.FULL_VISITS
            nl = _f17.FAST_N_LIST if args.fast else _f17.N_LIST
            wl = _f17.FAST_WEFF_LIST if args.fast else _f17.WEFF_LIST
            sl = _f17.FAST_SEEDS if args.fast else _f17.SEEDS
            new_rows = _f17.run_sweep(nv, nl, wl, sl, methods_filter=missing)
            rows = _f17.merge_rows(rows, new_rows, missing)
        else:
            print(f"  All methods present.")
    else:
        nv = _f17.FAST_VISITS if args.fast else _f17.FULL_VISITS
        nl = _f17.FAST_N_LIST if args.fast else _f17.N_LIST
        wl = _f17.FAST_WEFF_LIST if args.fast else _f17.WEFF_LIST
        sl = _f17.FAST_SEEDS if args.fast else _f17.SEEDS
        total = len(_f17.PPDU_DIST_NAMES) * len(nl) * len(wl) * len(sl) * len(METHODS_19_SIM)
        print(f"=== Figure 19 [{'FAST' if args.fast else 'FULL'}]  {nv} visits ===")
        print(f"    methods : {METHODS_19_SIM}")
        print(f"    total   : {total} configurations")
        rows = _f17.run_sweep(nv, nl, wl, sl, methods_filter=METHODS_19_SIM)

    # Save filtered CSV (4 methods + oracle only)
    out_methods = set(METHODS_19_SIM)
    out_rows = [r for r in rows if r["method"] in out_methods]
    csv_path = os.path.join(out_dir, "data.csv")
    _f17.save_csv(out_rows, csv_path)

    # ── τ trajectory ──────────────────────────────────────────────────────────
    print("Computing τ trajectory for panel (b) ...")
    rng_ppdu   = np.random.default_rng(42 * 10001 + 7)
    ppdus_traj = _f17.sample_ppdu("uniform", 20, rng_ppdu)
    traj_data  = _f17.run_trajectory_visit(
        METHODS_19, W_eff=100, ppdus=ppdus_traj, rng_seed=42,
    )

    check_hypotheses(rows)

    print("\nPlotting ...")
    plot(rows, traj_data, out_dir, fig_dir)

    print("\nFigure 19 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig19_dcf_benchmark.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
