"""
Figure 27 (paper Fig. eval-3, fig3-1/fig3-2): Impact of Visiting Duration

Paper Subsection C. Same mixed standard-unit setting as fig26 (Subsections
A/B), but sweeping the NPCA visiting duration W_eff instead of the visitor
count: the OBSS transmission that opens the window varies widely in length,
so the window an NPCA visit gets is the environmental variable here.

  W_eff ∈ {100, 200, 420, 840, 1680} slots = 0.9, 1.8, 3.78, 7.56, 15.12 ms
  (E[L_visitor] = 62.5 slots → W/E[L] ≈ 1.6 … 27)

N visitors (fixed, chosen via --pick-n fast sweep), M=10 natives.
Methods: dcf_excl (standard NPCA, deferring impl.), pace (Algorithm 1, cold τ0=1/W_eff per
visit), oracle key = fair-share reference FS (τ*=1/|V(t)|).

Two single-panel figures for the LaTeX subfigure pair:
  fig3-1  basic access        fig3-2  mandatory RTS/CTS @24Mbps
y = total useful airtime / W_eff, x = visiting duration (ms, log scale).

Run:
  .venv/bin/python harq_sim/run_step9_fig27.py --pick-n     # fast N comparison
  .venv/bin/python harq_sim/run_step9_fig27.py              # full, N=N_REF
  .venv/bin/python harq_sim/run_step9_fig27.py --base-csv results/step9/fig27/data.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_step9_fig17 as _f17
import run_step9_fig25 as _f25
import run_step9_fig26 as _f26

# ─── Parameters ───────────────────────────────────────────────────────────────

METHODS_27 = ["dcf_excl", "pace", "pace_wrule", "oracle"]

# Window-scaled coefficient, plotted alongside the fixed one so the reader can
# see what the fixed value costs at the ends of the sweep. Section 4.5.40:
# the log step is bounded by the stationary spread of a stochastic
# approximation, eps <= delta / sqrt(E), and the epoch budget E is close to
# proportional to the window (E ~ W^0.85, since the mean slots per epoch move
# only 1.5x across a 16.8x sweep), which is where the square root comes from.
# C is calibrated, at alpha = 0.5.
C_WRULE = 10.16


def _c_of_w(w_eff: int) -> float:
    return math.exp(C_WRULE / math.sqrt(w_eff))

W_LIST   = [100, 200, 420, 840, 1680]        # slots (σ=9µs) → 0.9–15.12 ms
N_REF    = 20                                 # visitors (see --pick-n)
M_FIX    = 10                                 # natives

SEEDS_27    = [42, 123, 456, 789, 1234]
FULL_REPS   = 20
FULL_VISITS = 50

FAST_SEEDS  = [42]
FAST_REPS   = 5
FAST_VISITS = 30
FAST_WLIST  = [100, 420, 1680]

FIELDS_27 = ["access", "W_eff", "N", "method", "seed",
             "succ_v", "succ_n", "useful"]

_STYLE = dict(_f26._STYLE_26)
_LABEL = dict(_f26._LABEL_26)
_LABEL["pace"] = "PACE-static"
_STYLE["pace_wrule"] = dict(color="#d62728", ls=":", lw=2.0, marker="v", ms=5)
_LABEL["pace_wrule"] = "PACE-dynamic"


# ─── One config ───────────────────────────────────────────────────────────────

def run_config(method: str, n_visitor: int, w_eff: int, coll_cost, succ_oh: int,
               seed: int, reps: int, visits: int) -> dict:
    """PACE per Algorithm 1: cold τ0 = 1/W_eff each visit, no carry."""
    _f25.N_VISITOR = n_visitor
    _f25.N_NATIVE = M_FIX
    _f25.W_REF = w_eff
    m_idx = METHODS_27.index(method)
    saved_c = (_f17.PND_C_COLL, _f17.PND_C_IDLE)
    if method == "pace_wrule":
        _f17.PND_C_COLL = _f17.PND_C_IDLE = _c_of_w(w_eff)
    sv = sn = 0.0
    for r in range(reps):
        rng_p = np.random.default_rng(seed * 10001 + r * 71 + 7)
        rng = np.random.default_rng(seed * 200003 + r * 3163 + m_idx * 29
                                    + n_visitor * 211 + w_eff * 7)
        for v in range(visits):
            ppdus = _f25._sample_ppdus25(rng_p)
            tau0 = (np.full(n_visitor, 1.0 / w_eff)
                    if method.startswith("pace") else None)
            mode = {"dcf_excl": "dcf_excl", "pace": "pace",
                    "pace_wrule": "pace", "oracle": "oracle"}[method]
            air, _c, _i, _o, _carry = _f25._run_visit25(
                ppdus, rng, mode, tau0, coll_cost, succ_oh)
            if v >= visits // 2:
                sv += air[:n_visitor].sum()
                sn += air[n_visitor:].sum()
    norm = reps * (visits - visits // 2) * w_eff
    _f17.PND_C_COLL, _f17.PND_C_IDLE = saved_c
    _f25.N_VISITOR, _f25.N_NATIVE, _f25.W_REF = 10, 10, 420   # restore
    return {"succ_v": sv / norm, "succ_n": sn / norm, "useful": (sv + sn) / norm}


ACCESS_CONFIGS = [
    ("basic", "nocd", 0),
    ("rts",   _f25.COLL_RTS_24M, _f25.OH_SUCC_24M),
]


def run_sweep(n_visitor: int, w_list: list, seeds: list, reps: int,
              visits: int) -> list[dict]:
    rows = []
    total = len(ACCESS_CONFIGS) * len(w_list) * len(METHODS_27) * len(seeds)
    done = 0
    for access, cc, oh in ACCESS_CONFIGS:
        for w in w_list:
            for method in METHODS_27:
                for seed in seeds:
                    res = run_config(method, n_visitor, w, cc, oh,
                                     seed, reps, visits)
                    rows.append({"access": access, "W_eff": w, "N": n_visitor,
                                 "method": method, "seed": seed, **res})
                    done += 1
                    print(f"  [{done:3d}/{total}] {access:<6} W={w:4d} "
                          f"{method:<9} seed={seed}", flush=True)
    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean27(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


def summary(rows, w_list: list) -> None:
    for access in ["basic", "rts"]:
        print(f"\n--- {access}: total useful airtime ---")
        print(f"  {'method':<10}" + "".join(
            f"{w * 9 / 1000:>8.2f}ms" for w in w_list))
        for m in METHODS_27:
            print(f"  {m:<10}" + "".join(
                f"{_mean27(rows, 'useful', access=access, W_eff=w, method=m):>10.3f}"
                for w in w_list))
        print(f"  visitor share (pace):")
        print(f"  {'pace sv':<10}" + "".join(
            f"{_mean27(rows, 'succ_v', access=access, W_eff=w, method='pace'):>10.3f}"
            for w in w_list))
        print(f"  {'dcf sv':<10}" + "".join(
            f"{_mean27(rows, 'succ_v', access=access, W_eff=w, method='dcf_excl'):>10.3f}"
            for w in w_list))


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _plot_one(rows, access: str, w_list: list, ylim, fig_dir: str,
              out_dir: str, fig_name: str, leg_loc: str = "best") -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    xs = [w * 9 / 1000 for w in w_list]      # ms
    for m in METHODS_27:
        ys = [_mean27(rows, "useful", access=access, W_eff=w, method=m)
              for w in w_list]
        ax.plot(xs, ys, label=_LABEL[m], **_STYLE[m])
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.1f}".rstrip("0").rstrip(".") for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Visiting duration (ms)")
    ax.set_ylabel("Total airtime / $W_\\mathrm{eff}$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7.5, frameon=True, loc=leg_loc,
              handlelength=1.5, borderpad=0.3, labelspacing=0.3)
    ax.grid(color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

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
    plt.close(fig)


def plot(rows, w_list: list, out_dir: str, fig_dir: str) -> None:
    # shared y-range across both panels for direct comparability
    vals = [r["useful"] for r in rows]
    pad = (max(vals) - min(vals)) * 0.06
    ylim = (min(vals) - pad, max(vals) + pad)
    _plot_one(rows, "basic", w_list, ylim, fig_dir, out_dir, "fig3-1",
              leg_loc="upper right")
    _plot_one(rows, "rts",   w_list, ylim, fig_dir, out_dir, "fig3-2",
              leg_loc="lower right")


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_27)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in ("W_eff", "N", "seed"):
                row[k] = int(row[k])
            for k in ("succ_v", "succ_n", "useful"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Fig eval-3 (fig3-1/fig3-2) — visiting duration sweep")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--pick-n", action="store_true",
                        help="fast comparison of candidate N values")
    parser.add_argument("--n", type=int, default=N_REF)
    parser.add_argument("--out-dir", default="results/step9/fig27")
    parser.add_argument("--base-csv", default=None, metavar="PATH")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "results", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    if args.pick_n:
        for n in [10, 20]:
            print(f"\n===== N = {n} =====")
            rows = run_sweep(n, FAST_WLIST, FAST_SEEDS, FAST_REPS, FAST_VISITS)
            summary(rows, FAST_WLIST)
        return

    w_list = FAST_WLIST if args.fast else W_LIST
    if args.base_csv:
        rows = load_csv(args.base_csv)
        w_list = sorted({r["W_eff"] for r in rows})
    else:
        seeds = FAST_SEEDS if args.fast else SEEDS_27
        reps = FAST_REPS if args.fast else FULL_REPS
        visits = FAST_VISITS if args.fast else FULL_VISITS
        print(f"=== Figure 27 (fig3-1/fig3-2) [{'FAST' if args.fast else 'FULL'}] "
              f"N={args.n}, M={M_FIX} ===")
        rows = run_sweep(args.n, w_list, seeds, reps, visits)
        save_csv(rows, os.path.join(args.out_dir, "data.csv"))

    summary(rows, w_list)
    print("\nPlotting ...")
    plot(rows, w_list, args.out_dir, fig_dir)
    print("\nDone → results/figure/fig3-{1,2}.*")


if __name__ == "__main__":
    main()
