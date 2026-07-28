"""
Figure 26 (paper Fig. eval-1): Why PACE? — Standard-Compliant NPCA vs PACE
under Native Load, Basic Access and Mandatory RTS/CTS

Paper subsection A figure. Mixed native/visitor NPCA channel in IEEE 802.11
standard units (fig25 conventions: slot ≡ aSlotTime σ=9µs, visitor PPDU
U[225,900]µs, native PPDU 450µs, W_eff=3.78ms, RTS/CTS@24Mbps: success OH
88µs=10σ, RTS collision 78µs=9σ).

Two single-panel figures for a LaTeX subfigure pair:
  fig1-1  basic access (no RTS/CTS; collision = max Lᵢ of colliders)
  fig1-2  mandatory RTS/CTS @ 24 Mbps control rate
Each: visitor useful airtime vs N_native for
  standard NPCA (CSMA/CA visitor, deferring implementation),
  PACE (warm, τ carried across transitions), oracle.

Run:
  .venv/bin/python harq_sim/run_step9_fig26.py
  .venv/bin/python harq_sim/run_step9_fig26.py --fast
  .venv/bin/python harq_sim/run_step9_fig26.py --base-csv results/step9/fig26/data.csv
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

METHODS_26 = ["dcf_excl", "pace", "oracle"]
N_NATIVE_LIST = [0, 5, 10, 20]

# --visitor-sweep mode: fix natives, sweep visitor group size
VIS_NV_LIST = [5, 10, 20, 50]
VIS_M_FIX   = 10

# (label, coll_cost, succ_oh) — fig25 standard-derived values
ACCESS_CONFIGS = [
    ("basic", "nocd", 0),
    ("rts",   _f25.COLL_RTS_24M, _f25.OH_SUCC_24M),
]

SEEDS_26    = [42, 123, 456, 789, 1234]
FULL_REPS   = 20
FULL_VISITS = 50

FAST_SEEDS  = [42]
FAST_REPS   = 5
FAST_VISITS = 30
FAST_NATS   = [0, 10, 20]

FIELDS_26 = ["access", "N_native", "method", "seed",
             "succ_v", "succ_n", "useful"]

_STYLE_26 = {
    "dcf_excl": dict(color="#525252", ls="-.", lw=1.9, marker="x", ms=7),
    "pace":     dict(color="#ff7f0e", ls="-",  lw=2.2, marker="^", ms=7),
    "oracle":   dict(color="#2ca02c", ls="--", lw=1.8, marker="D", ms=6),
}
_LABEL_26 = {
    "dcf_excl": "Standard NPCA (CSMA/CA)",
    "pace":     "PACE",
    "oracle":   "Fair share (FS)",
}


# ─── Sweep (reuses fig25 visit engine; N_NATIVE patched per config) ───────────

def run_config(method: str, n_native: int, coll_cost, succ_oh: int,
               seed: int, reps: int, visits: int) -> dict:
    _f25.N_NATIVE = n_native
    m_idx = METHODS_26.index(method)
    sv = sn = 0.0
    for r in range(reps):
        rng_p = np.random.default_rng(seed * 10001 + r * 71 + 7)
        rng = np.random.default_rng(seed * 200003 + r * 3163 + m_idx * 29
                                    + n_native * 101)
        tau_c = np.full(_f25.N_VISITOR, 1.0 / _f25.N_VISITOR) \
            if method == "pace" else None
        for v in range(visits):
            ppdus = _f25._sample_ppdus25(rng_p)
            air, _c, _i, _o, carry = _f25._run_visit25(
                ppdus, rng, method, tau_c, coll_cost, succ_oh)
            if method == "pace":
                tau_c = carry
            if v >= visits // 2:
                sv += air[:_f25.N_VISITOR].sum()
                sn += air[_f25.N_VISITOR:].sum()
    norm = reps * (visits - visits // 2) * _f25.W_REF
    return {"succ_v": sv / norm, "succ_n": sn / norm, "useful": (sv + sn) / norm}


def run_config_v(method: str, n_visitor: int, coll_cost, succ_oh: int,
                 seed: int, reps: int, visits: int) -> dict:
    """Visitor-sweep variant: patches N_VISITOR too (M fixed at VIS_M_FIX).

    PACE follows Algorithm alg:pace exactly: every visit starts cold from
    τ0 = 1/W_eff (Phase a), no state carried between visits.
    """
    _f25.N_VISITOR = n_visitor
    _f25.N_NATIVE = VIS_M_FIX
    m_idx = METHODS_26.index(method)
    sv = sn = 0.0
    for r in range(reps):
        rng_p = np.random.default_rng(seed * 10001 + r * 71 + 7)
        rng = np.random.default_rng(seed * 200003 + r * 3163 + m_idx * 29
                                    + n_visitor * 211)
        for v in range(visits):
            ppdus = _f25._sample_ppdus25(rng_p)
            tau0 = np.full(n_visitor, 1.0 / _f25.W_REF) \
                if method == "pace" else None
            air, _c, _i, _o, _carry = _f25._run_visit25(
                ppdus, rng, method, tau0, coll_cost, succ_oh)
            if v >= visits // 2:
                sv += air[:n_visitor].sum()
                sn += air[n_visitor:].sum()
    norm = reps * (visits - visits // 2) * _f25.W_REF
    _f25.N_VISITOR = 10          # restore defaults
    return {"succ_v": sv / norm, "succ_n": sn / norm, "useful": (sv + sn) / norm}


def run_sweep_v(nv_list: list, seeds: list, reps: int, visits: int) -> list[dict]:
    rows = []
    total = len(ACCESS_CONFIGS) * len(nv_list) * len(METHODS_26) * len(seeds)
    done = 0
    for access, cc, oh in ACCESS_CONFIGS:
        for n_v in nv_list:
            for method in METHODS_26:
                for seed in seeds:
                    res = run_config_v(method, n_v, cc, oh, seed, reps, visits)
                    rows.append({"access": access, "N_visitor": n_v,
                                 "method": method, "seed": seed, **res})
                    done += 1
                    print(f"  [{done:3d}/{total}] {access:<6} Nv={n_v:2d} "
                          f"{method:<9} seed={seed}", flush=True)
    return rows


def run_sweep(nat_list: list, seeds: list, reps: int, visits: int) -> list[dict]:
    rows = []
    total = len(ACCESS_CONFIGS) * len(nat_list) * len(METHODS_26) * len(seeds)
    done = 0
    for access, cc, oh in ACCESS_CONFIGS:
        for n_nat in nat_list:
            for method in METHODS_26:
                for seed in seeds:
                    res = run_config(method, n_nat, cc, oh, seed, reps, visits)
                    rows.append({"access": access, "N_native": n_nat,
                                 "method": method, "seed": seed, **res})
                    done += 1
                    print(f"  [{done:3d}/{total}] {access:<6} nat={n_nat:2d} "
                          f"{method:<9} seed={seed}", flush=True)
    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean26(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ─── Plot: two single-panel figures (LaTeX subfigure pair) ────────────────────

def _plot_one(rows, access: str, nat_list: list, ylim, fig_dir: str,
              out_dir: str, fig_name: str,
              xkey: str = "N_native",
              xlabel: str = "Number of native STAs $N_\\mathrm{nat}$",
              ykey: str = "succ_v",
              ylabel: str = "Visitor airtime / $W_\\mathrm{eff}$") -> None:
    # Sized for a full-\columnwidth IEEE subfigure (stacked vertically) —
    # rendered ≈1:1, so use print-scale fonts (~9-10pt).
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for m in METHODS_26:
        ys = [_mean26(rows, ykey, access=access, method=m, **{xkey: n})
              for n in nat_list]
        ax.plot(nat_list, ys, label=_LABEL_26[m], **_STYLE_26[m])
    ax.set_xticks(nat_list)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=True, loc="best",
              handlelength=2.0, borderpad=0.35, labelspacing=0.35)
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


def plot(rows, nat_list: list, out_dir: str, fig_dir: str) -> None:
    ymax = max(r["succ_v"] for r in rows) * 1.12
    ylim = (0.0, ymax)
    _plot_one(rows, "basic", nat_list, ylim, fig_dir, out_dir,
              "fig1-1_native_sweep")
    _plot_one(rows, "rts",   nat_list, ylim, fig_dir, out_dir,
              "fig1-2_native_sweep")


def _plot_fair_one(rows, access: str, nv_list: list, ylim, fig_dir: str,
                   out_dir: str, fig_name: str) -> None:
    """Airtime proportionality: (visitor airtime share) / (population share)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for m in METHODS_26:
        ys = []
        for n in nv_list:
            sv = _mean26(rows, "succ_v", access=access, method=m, N_visitor=n)
            tot = _mean26(rows, "useful", access=access, method=m, N_visitor=n)
            pop = n / (n + VIS_M_FIX)
            ys.append((sv / tot) / pop if tot > 0 else float("nan"))
        ax.plot(nv_list, ys, label=_LABEL_26[m], **_STYLE_26[m])
    ax.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax.set_xticks(nv_list)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Number of visitor STAs $N_\\mathrm{vis}$", fontsize=9)
    ax.set_ylabel("Visitor airtime proportionality $\\rho$", fontsize=9)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=True, loc="lower right",
              handlelength=2.0, borderpad=0.35, labelspacing=0.35)
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


def plot_v_fairness(rows, nv_list: list, out_dir: str, fig_dir: str) -> None:
    ylim = (0.0, 1.85)
    _plot_fair_one(rows, "basic", nv_list, ylim, fig_dir, out_dir, "fig2-1")
    _plot_fair_one(rows, "rts",   nv_list, ylim, fig_dir, out_dir, "fig2-2")


def plot_v(rows, nv_list: list, out_dir: str, fig_dir: str,
           total: bool = False) -> None:
    # Visitor sweep is the paper figure (Subsection A) → unsuffixed names.
    xl = "Number of visitor STAs $N_\\mathrm{vis}$"
    if total:
        ymax = max(r["useful"] for r in rows) * 1.12
        for access, name in [("basic", "fig1-1_total"), ("rts", "fig1-2_total")]:
            _plot_one(rows, access, nv_list, (0.0, ymax), fig_dir, out_dir,
                      name, xkey="N_visitor", xlabel=xl, ykey="useful",
                      ylabel="Total airtime / $W_\\mathrm{eff}$")
        return
    ymax = max(r["succ_v"] for r in rows) * 1.12
    ylim = (0.0, ymax)
    _plot_one(rows, "basic", nv_list, ylim, fig_dir, out_dir,
              "fig1-1", xkey="N_visitor", xlabel=xl)
    _plot_one(rows, "rts",   nv_list, ylim, fig_dir, out_dir,
              "fig1-2", xkey="N_visitor", xlabel=xl)


# ─── Summary ──────────────────────────────────────────────────────────────────

def summary(rows, nat_list: list, xkey: str = "N_native",
            xtag: str = "M") -> None:
    for access in ["basic", "rts"]:
        print(f"\n--- {access}: visitor useful airtime (succ_v) ---")
        print(f"  {'method':<10}"
              + "".join(f"{xtag + '=' + str(n):>9}" for n in nat_list))
        for m in METHODS_26:
            print(f"  {m:<10}" + "".join(
                f"{_mean26(rows, 'succ_v', access=access, method=m, **{xkey: n}):>9.3f}"
                for n in nat_list))
        print(f"  pace/dcf  " + "".join(
            f"{_mean26(rows, 'succ_v', access=access, method='pace', **{xkey: n}) / _mean26(rows, 'succ_v', access=access, method='dcf_excl', **{xkey: n}):>9.2f}"
            for n in nat_list))
        print(f"\n  channel useful (total):")
        for m in METHODS_26:
            print(f"  {m:<10}" + "".join(
                f"{_mean26(rows, 'useful', access=access, method=m, **{xkey: n}):>9.3f}"
                for n in nat_list))


# ─── CSV ──────────────────────────────────────────────────────────────────────

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
            for k in ("N_native", "N_visitor", "seed"):
                if k in row:
                    row[k] = int(row[k])
            for k in ("succ_v", "succ_n", "useful"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Fig eval-1 (fig1-1/fig1-2) — Why PACE vs standard NPCA")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--visitor-sweep", action="store_true",
                        help=f"fix M={VIS_M_FIX}, sweep N_visitor {VIS_NV_LIST}")
    parser.add_argument("--total", action="store_true",
                        help="visitor-sweep only: plot total (native+visitor) "
                             "airtime as fig1-{1,2}_total")
    parser.add_argument("--fairness", action="store_true",
                        help="visitor-sweep only: plot airtime proportionality "
                             "as fig2-{1,2}")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--base-csv", default=None, metavar="PATH")
    args = parser.parse_args()

    out_dir = args.out_dir or ("results/step9/fig26_visitor" if args.visitor_sweep
                               else "results/step9/fig26")
    os.makedirs(out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "results", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    seeds = FAST_SEEDS if args.fast else SEEDS_26
    reps = FAST_REPS if args.fast else FULL_REPS
    visits = FAST_VISITS if args.fast else FULL_VISITS

    if args.visitor_sweep:
        nv_list = [5, 10, 20] if args.fast else VIS_NV_LIST
        if args.base_csv:
            rows = load_csv(args.base_csv)
            nv_list = sorted({r["N_visitor"] for r in rows})
        else:
            print(f"=== Figure 26 visitor-sweep [{'FAST' if args.fast else 'FULL'}] "
                  f"(M={VIS_M_FIX} fixed) ===")
            rows = run_sweep_v(nv_list, seeds, reps, visits)
        save_csv(rows, os.path.join(out_dir, "data.csv"))
        summary(rows, nv_list, xkey="N_visitor", xtag="N")
        print("\nPlotting ...")
        if args.fairness:
            plot_v_fairness(rows, nv_list, out_dir, fig_dir)
        else:
            plot_v(rows, nv_list, out_dir, fig_dir, total=args.total)
        print("\nDone.")
        return

    nat_list = FAST_NATS if args.fast else N_NATIVE_LIST
    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = load_csv(args.base_csv)
        nat_list = sorted({r["N_native"] for r in rows})
    else:
        print(f"=== Figure 26 (fig1-1/fig1-2) [{'FAST' if args.fast else 'FULL'}] ===")
        print(f"    access  : {[a[0] for a in ACCESS_CONFIGS]}")
        print(f"    N_native: {nat_list}   methods: {METHODS_26}")
        print(f"    visits  : {visits}  reps/seed: {reps}  seeds: {seeds}")
        rows = run_sweep(nat_list, seeds, reps, visits)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    summary(rows, nat_list)

    print("\nPlotting ...")
    plot(rows, nat_list, out_dir, fig_dir)

    print("\nFigure 26 complete → results/figure/fig1-1.*, fig1-2.*")


if __name__ == "__main__":
    main()
