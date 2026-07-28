"""
Figure 29 (paper Fig. eval-5, fig5-1/fig5-2): Ablation Study

Paper Subsection E. Isolates the two mechanisms of Algorithm 1 in the
reference mixed native/visitor setting of Subsection A (M=10 natives,
visitor sweep, IEEE 802.11 standard units):

  pace          full PACE (τ0 = 1/W_eff, Phase b self-exclusion)
  pace_noexcl   PACE without PPDU-aware self-exclusion — unfittable frames
                keep contending and truncate at NPCA_TIMER expiry
  pace_high     PACE without the one-probe initialization — naive τ0 = 0.5
  dcf_excl      standard-compliant NPCA baseline (deferring impl.)

Two single-panel figures (basic / mandatory RTS/CTS) for the LaTeX
subfigure pair; y = total useful airtime / W_eff.

Run:
  .venv/bin/python harq_sim/run_step9_fig29.py
  .venv/bin/python harq_sim/run_step9_fig29.py --fast
  .venv/bin/python harq_sim/run_step9_fig29.py --base-csv results/step9/fig29/data.csv
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

import run_step9_fig25 as _f25

# ─── Parameters ───────────────────────────────────────────────────────────────

# variant → (engine mode, τ0); τ0 None = no probability vector (DCF)
VARIANTS = {
    "pace":        ("pace",        "one_probe"),
    "pace_noexcl": ("pace_noexcl", "one_probe"),
    "pace_high":   ("pace",        0.5),
    "pace_rand":   ("pace",        "rand"),      # τ0 ~ U(0,1) per STA per visit
    "dcf_excl":    ("dcf_excl",    None),
}
METHODS_29 = list(VARIANTS.keys())
# plotted subset (CSV keeps every variant)
PLOT_METHODS = ["pace", "pace_noexcl", "pace_rand", "dcf_excl"]

NV_LIST = [5, 10, 20, 50]
M_FIX   = 10

ACCESS_CONFIGS = [
    ("basic", "nocd", 0),
    ("rts",   _f25.COLL_RTS_24M, _f25.OH_SUCC_24M),
]

SEEDS_29    = [42, 123, 456, 789, 1234]
FULL_REPS   = 20
FULL_VISITS = 50

FAST_SEEDS  = [42]
FAST_REPS   = 5
FAST_VISITS = 30

_STYLE_29 = {
    "pace":        dict(color="#ff7f0e", ls="-",  lw=2.2, marker="^", ms=7),
    "pace_noexcl": dict(color="#1f77b4", ls="--", lw=1.9, marker="o", ms=6),
    "pace_high":   dict(color="#8c564b", ls=":",  lw=1.7, marker="v", ms=6),
    "pace_rand":   dict(color="#9467bd", ls=":",  lw=1.9, marker="s", ms=6),
    "dcf_excl":    dict(color="#525252", ls="-.", lw=1.9, marker="x", ms=7),
}
_LABEL_29 = {
    "pace":        "PACE (full)",
    "pace_noexcl": "PACE w/o self-exclusion",
    "pace_high":   "PACE w/ naive $\\tau_0{=}0.5$",
    "pace_rand":   "PACE w/ naive $\\tau_0{\\sim}\\mathcal{U}(0,1)$",
    "dcf_excl":    "Standard NPCA (CSMA/CA)",
}


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_config(variant: str, n_visitor: int, coll_cost, succ_oh: int,
               seed: int, reps: int, visits: int) -> dict:
    """Every visit starts cold from the variant's τ0 (Algorithm 1)."""
    mode, init = VARIANTS[variant]
    _f25.N_VISITOR = n_visitor
    _f25.N_NATIVE = M_FIX
    m_idx = METHODS_29.index(variant)
    sv = sn = 0.0
    for r in range(reps):
        rng_p = np.random.default_rng(seed * 10001 + r * 71 + 7)
        rng = np.random.default_rng(seed * 200003 + r * 3163 + m_idx * 29
                                    + n_visitor * 211)
        for v in range(visits):
            ppdus = _f25._sample_ppdus25(rng_p)
            if init is None:
                tau0 = None
            elif init == "one_probe":
                tau0 = np.full(n_visitor, 1.0 / _f25.W_REF)
            elif init == "rand":
                tau0 = rng.random(n_visitor)
            else:
                tau0 = np.full(n_visitor, float(init))
            air, _c, _i, _o, _carry = _f25._run_visit25(
                ppdus, rng, mode, tau0, coll_cost, succ_oh)
            if v >= visits // 2:
                sv += air[:n_visitor].sum()
                sn += air[n_visitor:].sum()
    norm = reps * (visits - visits // 2) * _f25.W_REF
    _f25.N_VISITOR = 10          # restore defaults
    _f25.N_NATIVE = 10
    return {"succ_v": sv / norm, "succ_n": sn / norm, "useful": (sv + sn) / norm}


def run_sweep(nv_list: list, seeds: list, reps: int, visits: int) -> list[dict]:
    rows = []
    total = len(ACCESS_CONFIGS) * len(nv_list) * len(METHODS_29) * len(seeds)
    done = 0
    for access, cc, oh in ACCESS_CONFIGS:
        for n_v in nv_list:
            for variant in METHODS_29:
                for seed in seeds:
                    res = run_config(variant, n_v, cc, oh, seed, reps, visits)
                    rows.append({"access": access, "N_visitor": n_v,
                                 "method": variant, "seed": seed, **res})
                    done += 1
                    print(f"  [{done:3d}/{total}] {access:<6} Nv={n_v:2d} "
                          f"{variant:<12} seed={seed}", flush=True)
    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean29(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _plot_one(rows, access: str, nv_list: list, ylim, fig_dir: str,
              out_dir: str, fig_name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for m in PLOT_METHODS:
        ys = [_mean29(rows, "useful", access=access, method=m, N_visitor=n)
              for n in nv_list]
        ax.plot(nv_list, ys, label=_LABEL_29[m], **_STYLE_29[m])
    ax.set_xticks(nv_list)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Number of visitor STAs $N_\\mathrm{vis}$", fontsize=9)
    ax.set_ylabel("Total airtime / $W_\\mathrm{eff}$", fontsize=9)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=7.5, frameon=True, loc="best",
              handlelength=2.0, borderpad=0.35, labelspacing=0.3)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
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


def plot(rows, nv_list: list, out_dir: str, fig_dir: str) -> None:
    ymax = max(r["useful"] for r in rows) * 1.12
    ylim = (0.0, ymax)
    _plot_one(rows, "basic", nv_list, ylim, fig_dir, out_dir, "fig5-1")
    _plot_one(rows, "rts",   nv_list, ylim, fig_dir, out_dir, "fig5-2")


# ─── Summary / CSV ────────────────────────────────────────────────────────────

def summary(rows, nv_list: list) -> None:
    for access in ["basic", "rts"]:
        for metric, tag in [("useful", "total useful airtime"),
                            ("succ_v", "visitor useful airtime")]:
            print(f"\n--- {access}: {tag} ---")
            print(f"  {'method':<13}"
                  + "".join(f"{'N=' + str(n):>9}" for n in nv_list))
            for m in METHODS_29:
                print(f"  {m:<13}" + "".join(
                    f"{_mean29(rows, metric, access=access, method=m, N_visitor=n):>9.3f}"
                    for n in nv_list))


def save_csv(rows, path) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["N_visitor"] = int(row["N_visitor"])
            row["seed"] = int(row["seed"])
            for k in ("succ_v", "succ_n", "useful"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Fig eval-5 (fig5-1/fig5-2) — PACE ablation study")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-dir", default="results/step9/fig29")
    parser.add_argument("--base-csv", default=None, metavar="PATH")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "results", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    seeds = FAST_SEEDS if args.fast else SEEDS_29
    reps = FAST_REPS if args.fast else FULL_REPS
    visits = FAST_VISITS if args.fast else FULL_VISITS
    nv_list = [5, 10, 20] if args.fast else NV_LIST

    if args.base_csv:
        rows = load_csv(args.base_csv)
        nv_list = sorted({r["N_visitor"] for r in rows})
    else:
        print(f"=== Figure 29 ablation [{'FAST' if args.fast else 'FULL'}] "
              f"(M={M_FIX} fixed) ===")
        rows = run_sweep(nv_list, seeds, reps, visits)

    save_csv(rows, os.path.join(args.out_dir, "data.csv"))
    summary(rows, nv_list)
    print("\nPlotting ...")
    plot(rows, nv_list, args.out_dir, fig_dir)
    print("\nDone → results/figure/fig5-{1,2}.*")


if __name__ == "__main__":
    main()
