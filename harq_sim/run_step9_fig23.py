"""
Figure 23: Collision Cost Model — Ideal Collision Detection (CD) vs No CD

The finite-window simulator charges a collision a single slot. That is exact only
under an idealized collision detection (or an RTS/CTS-style short handshake) in
which colliding stations abort after one slot. Without such detection, colliding
frames occupy the medium for their full length, so a collision costs about as
much as a success: the busy period lasts until the longest colliding PPDU ends.

RQ23: How much does the 1-slot-collision idealization inflate the results, and
does PACE keep its advantage over DCF once collisions are charged their true cost?

Two collision-cost models, applied uniformly to every scheme:
  cd    (ideal)   : a collision advances the window by 1 slot   (W_rem -= 1)
  no_cd (real)    : a collision advances it by max_i L_i over the colliders
A success always advances the window by the winner's L_i; an idle slot by 1.

Methods: oracle (tau=1/|viable|), pnd (PACE), dcf_self_excl (BEB), and (open-loop).
Heterogeneous PPDU (bimodal {4,12}) so collisions among long frames are expensive.

4 panels:
  (a) W_eff utilization by method, cd vs no_cd bars   (N=20, W_eff=50)
  (b) W_eff utilization vs N, pnd/dcf, cd solid / no_cd dashed
  (c) PACE-over-DCF utilization ratio vs N, cd vs no_cd  (advantage growth)
  (d) collision rate vs N per method (no_cd)            (why the gap moves)

Run:
  .venv/bin/python harq_sim/run_step9_fig23.py
  .venv/bin/python harq_sim/run_step9_fig23.py --fast
  .venv/bin/python harq_sim/run_step9_fig23.py --base-csv results/step9/fig23/data.csv
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

# ─── Parameters ───────────────────────────────────────────────────────────────

METHODS_23 = ["oracle", "pnd", "dcf_self_excl", "and"]
CD_MODES   = [True, False]          # True = ideal CD (coll=1 slot); False = real (coll=max L)

PPDU_DIST = "bimodal"              # {4,12}: long frames make no-CD collisions costly

N_LIST_23    = [10, 20, 30, 50]
WEFF_LIST_23 = [50, 100]
SEEDS_23     = [42, 123, 456, 789, 1234]
FULL_VISITS_23 = 1000

FAST_N_LIST_23    = [10, 20]
FAST_WEFF_LIST_23 = [50]
FAST_SEEDS_23     = [42]
FAST_VISITS_23    = 50

FIELDS_23 = [
    "method", "cd", "N", "W_eff", "seed",
    "W_eff_utilization", "efficiency", "collision_rate", "successes",
]

_STYLE_23 = {m: _f17._METHOD_STYLE[m] for m in METHODS_23}
_LABEL_23 = {m: _f17._METHOD_LABEL[m] for m in METHODS_23}


# ─── Single-visit simulation with collision-cost model ────────────────────────

def _collision_cost(cd: bool, ppdus: np.ndarray, tx: np.ndarray) -> int:
    """Slots consumed by a collision. Ideal CD -> 1; else max colliding PPDU."""
    if cd:
        return 1
    idx = np.where(tx)[0]
    return int(ppdus[idx].max()) if idx.size else 1


def _run_visit_cd(
    method: str, W_eff: int, ppdus: np.ndarray, rng: np.random.Generator,
    oracle_successes: int, cd: bool,
) -> dict:
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff

    tau = np.full(N, 1.0 / N)
    dcf_cw = np.full(N, N, dtype=np.int64)
    dcf_bo = rng.integers(0, N, size=N).astype(np.int64)
    _solo_sender_tau = 0.0
    and_phase = 1
    and_slots = 0
    and_dur = int(math.ceil(2**and_phase * math.e * math.log(2**and_phase)))

    successes = 0
    useful_slots = 0
    total_epochs = 0
    collision_epochs = 0

    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k_viable = int(viable.sum())
        if k_viable == 0:
            break

        if method == "oracle":
            tau_o = 1.0 / k_viable
            tx = rng.random(N) < np.where(viable, tau_o, 0.0)
        elif method == "dcf_self_excl":
            tx = (dcf_bo == 0) & viable
        elif method == "and":
            and_p = max(1.0 / (2 ** and_phase), 1e-4)
            tx = rng.random(N) < np.where(viable, and_p, 0.0)
        else:   # pnd
            tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)

        n_tx = int(tx.sum())
        outcome_idle = (n_tx == 0)
        outcome_coll = (n_tx > 1)
        outcome_solo = (n_tx == 1)

        # ── Advance window by outcome-dependent epoch length ──────────────────
        if outcome_solo:
            i = int(np.where(tx)[0][0])
            if ppdus[i] <= W_rem:
                _solo_sender_tau = float(tau[i])
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                successes += 1
                useful_slots += int(ppdus[i])
                tau[i] = 0.0
                if method == "dcf_self_excl":
                    dcf_bo[i] = W_eff + 1
            else:
                W_rem -= 1
                outcome_solo = False
                outcome_coll = True
        elif outcome_coll:
            W_rem -= _collision_cost(cd, ppdus, tx)
        else:
            W_rem -= 1

        total_epochs += 1
        if outcome_coll:
            collision_epochs += 1

        # ── State updates ─────────────────────────────────────────────────────
        if method == "dcf_self_excl":
            if outcome_coll:
                for j in np.where(tx)[0]:
                    j = int(j)
                    dcf_cw[j] = min(int(dcf_cw[j]) * 2, _f17.DCF_CW_MAX)
                    dcf_bo[j] = int(rng.integers(0, max(int(dcf_cw[j]), 1)))
            elif outcome_idle:
                mask = (~succeeded) & viable & (dcf_bo > 0)
                dcf_bo[mask] -= 1

        elif method == "pnd":
            if outcome_solo:
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] = _solo_sender_tau
            elif outcome_coll:
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] /= _f17.PND_C_COLL
            elif outcome_idle:
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] *= _f17.PND_C_IDLE
            for k in range(N):
                if not succeeded[k]:
                    tau[k] = float(np.clip(tau[k], 1e-4, 1.0))

        if method == "and":
            and_slots += 1
            if and_slots >= and_dur and and_phase < 60:
                and_phase += 1
                and_slots = 0
                and_dur = int(math.ceil(2**and_phase * math.e * math.log(2**and_phase)))

    efficiency = (successes / oracle_successes) if oracle_successes > 0 else 0.0
    col_rate = (collision_epochs / total_epochs) if total_epochs > 0 else 0.0
    weff_util = useful_slots / W_eff if W_eff > 0 else 0.0
    return {
        "W_eff_utilization": weff_util,
        "efficiency":        efficiency,
        "collision_rate":    col_rate,
        "successes":         successes,
    }


def _oracle_successes_cd(W_eff: int, ppdus: np.ndarray, rng: np.random.Generator, cd: bool) -> int:
    """Oracle successes under the same collision-cost model (denominator for efficiency)."""
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    succ = 0
    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k = int(viable.sum())
        if k == 0:
            break
        tx = rng.random(N) < (viable * (1.0 / k))
        n_tx = int(tx.sum())
        if n_tx == 1:
            i = int(np.where(tx)[0][0])
            succeeded[i] = True
            W_rem -= int(ppdus[i])
            succ += 1
        elif n_tx > 1:
            W_rem -= _collision_cost(cd, ppdus, tx)
        else:
            W_rem -= 1
    return succ


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(n_visits: int, n_list: list, weff_list: list, seeds: list) -> list[dict]:
    method_idx = {m: i for i, m in enumerate(_f17.METHODS)}
    rows = []
    total = len(n_list) * len(weff_list) * len(seeds) * len(CD_MODES) * len(METHODS_23)
    done = 0

    for N in n_list:
        for W_eff in weff_list:
            for seed in seeds:
                rng_ppdu = np.random.default_rng(seed * 10001 + 7)
                ppdu_visits = [_f17.sample_ppdu(PPDU_DIST, N, rng_ppdu) for _ in range(n_visits)]

                for cd in CD_MODES:
                    # oracle baselines under this cd
                    os_list = []
                    for v, ppdus in enumerate(ppdu_visits):
                        rng_o = np.random.default_rng(seed * 100003 + v)
                        os_list.append(_oracle_successes_cd(W_eff, ppdus, rng_o, cd))

                    for method in METHODS_23:
                        m_idx = method_idx.get(method, 0)
                        util_l, eff_l, col_l, suc_l = [], [], [], []
                        for v, ppdus in enumerate(ppdu_visits):
                            rng_v = np.random.default_rng(seed * 200003 + v * 17 + m_idx)
                            res = _run_visit_cd(method, W_eff, ppdus, rng_v, os_list[v], cd)
                            util_l.append(res["W_eff_utilization"])
                            eff_l.append(res["efficiency"])
                            col_l.append(res["collision_rate"])
                            suc_l.append(res["successes"])
                        rows.append({
                            "method":            method,
                            "cd":                int(cd),
                            "N":                 N,
                            "W_eff":             W_eff,
                            "seed":              seed,
                            "W_eff_utilization": float(np.mean(util_l)),
                            "efficiency":        float(np.mean(eff_l)),
                            "collision_rate":    float(np.mean(col_l)),
                            "successes":         float(np.mean(suc_l)),
                        })
                        done += 1
                        print(f"  [{done:4d}/{total}] N={N:2d} W={W_eff:3d} "
                              f"cd={int(cd)} {method:<14} seed={seed}", flush=True)
    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean23(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows) -> None:
    N_ref, W_ref = 20, 50
    x = np.arange(len(METHODS_23))
    w = 0.38
    cd_vals = [_mean23(rows, "W_eff_utilization", method=m, cd=1, N=N_ref, W_eff=W_ref) for m in METHODS_23]
    nc_vals = [_mean23(rows, "W_eff_utilization", method=m, cd=0, N=N_ref, W_eff=W_ref) for m in METHODS_23]
    ax.bar(x - w/2, cd_vals, w, label="ideal CD (coll = 1 slot)", color="#4575b4", edgecolor="white")
    ax.bar(x + w/2, nc_vals, w, label="no CD (coll = max $L_i$)", color="#d73027", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_LABEL_23[m].split(" (")[0] for m in METHODS_23], fontsize=8, rotation=12)
    ax.set_ylabel("W_eff utilization  Σ(L·succ)/W_eff", fontsize=10)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7, axis="y")
    ax.set_title(f"(a) Collision-cost model effect  (N={N_ref}, W_eff={W_ref}, bimodal)", fontsize=10)


def _panel_b(ax, rows) -> None:
    W_ref = 50
    avail_n = sorted({r["N"] for r in rows if r["W_eff"] == W_ref})
    for method in ["oracle", "pnd", "dcf_self_excl"]:
        st = _STYLE_23[method]
        for cd, ls, tag in [(1, "-", "CD"), (0, "--", "no-CD")]:
            means = [_mean23(rows, "W_eff_utilization", method=method, cd=cd, W_eff=W_ref, N=n)
                     for n in avail_n]
            ax.plot(avail_n, means, ls=ls, color=st["color"], marker=st["marker"],
                    ms=st.get("ms", 5), lw=1.9,
                    label=f"{_LABEL_23[method].split(' (')[0]} [{tag}]")
    ax.set_xlabel("N (contending STAs)", fontsize=11)
    ax.set_ylabel("W_eff utilization", fontsize=10)
    ax.set_xticks(avail_n)
    ax.legend(fontsize=7, frameon=True, ncol=1, loc="best")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(b) Utilization vs N  (W_eff={W_ref}; solid=CD, dashed=no-CD)", fontsize=10)


def _panel_c(ax, rows) -> None:
    W_ref = 50
    avail_n = sorted({r["N"] for r in rows if r["W_eff"] == W_ref})
    for cd, ls, tag, col in [(1, "-", "CD", "#4575b4"), (0, "--", "no-CD", "#d73027")]:
        ratios = []
        for n in avail_n:
            p = _mean23(rows, "W_eff_utilization", method="pnd",           cd=cd, W_eff=W_ref, N=n)
            d = _mean23(rows, "W_eff_utilization", method="dcf_self_excl", cd=cd, W_eff=W_ref, N=n)
            ratios.append(p / d if d > 0 else float("nan"))
        ax.plot(avail_n, ratios, ls=ls, color=col, marker="o", lw=2.0,
                label=f"PACE / DCF  [{tag}]")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("N (contending STAs)", fontsize=11)
    ax.set_ylabel("PACE utilization / DCF utilization", fontsize=10)
    ax.set_xticks(avail_n)
    ax.legend(fontsize=8, frameon=True, loc="best")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(c) PACE advantage over DCF  (W_eff={W_ref})", fontsize=10)


def _panel_d(ax, rows) -> None:
    W_ref = 50
    avail_n = sorted({r["N"] for r in rows if r["W_eff"] == W_ref})
    for method in METHODS_23:
        st = _STYLE_23[method]
        means = [_mean23(rows, "collision_rate", method=method, cd=0, W_eff=W_ref, N=n)
                 for n in avail_n]
        ax.plot(avail_n, means, label=_LABEL_23[method].split(" (")[0], **st)
    ax.set_xlabel("N (contending STAs)", fontsize=11)
    ax.set_ylabel("Collision rate  (collided epochs / total)", fontsize=10)
    ax.set_xticks(avail_n)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(d) Collision rate vs N  (no-CD, W_eff={W_ref})", fontsize=10)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows) -> None:
    print("\n=== Hypothesis Check (W_eff=50, bimodal) ===")
    W = 50
    avail_n = sorted({r["N"] for r in rows if r["W_eff"] == W})

    print("\nH1: no-CD lowers utilization for every method (collisions now expensive)")
    for m in METHODS_23:
        for n in [avail_n[-1]]:
            cd = _mean23(rows, "W_eff_utilization", method=m, cd=1, W_eff=W, N=n)
            nc = _mean23(rows, "W_eff_utilization", method=m, cd=0, W_eff=W, N=n)
            print(f"  {m:<14} N={n}: CD={cd:.3f}  no-CD={nc:.3f}  drop={cd-nc:+.3f}")

    print("\nH2: PACE advantage over DCF grows without CD")
    for n in avail_n:
        for cd, tag in [(1, "CD"), (0, "no-CD")]:
            p = _mean23(rows, "W_eff_utilization", method="pnd",           cd=cd, W_eff=W, N=n)
            d = _mean23(rows, "W_eff_utilization", method="dcf_self_excl", cd=cd, W_eff=W, N=n)
            r = p / d if d > 0 else float("nan")
            print(f"  N={n:2d} [{tag:<5}]: PACE={p:.3f} DCF={d:.3f} ratio={r:.3f}")

    print("\n--- collision rate (no-CD, W_eff=50) ---")
    for m in METHODS_23:
        vals = [_mean23(rows, "collision_rate", method=m, cd=0, W_eff=W, N=n) for n in avail_n]
        print(f"  {m:<14}: " + " ".join(f"{v:.3f}" for v in vals))


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_23)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path) -> list[dict]:
    int_fields = {"cd", "N", "W_eff", "seed"}
    str_fields = {"method"}
    float_fields = {f for f in FIELDS_23 if f not in int_fields and f not in str_fields}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in int_fields:
                if k in row:
                    row[k] = int(row[k])
            for k in float_fields:
                if k in row:
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = float("nan")
            rows.append(row)
    return rows


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot(rows, out_dir, fig_dir) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.40, wspace=0.30)
    _panel_a(axes[0, 0], rows)
    _panel_b(axes[0, 1], rows)
    _panel_c(axes[1, 0], rows)
    _panel_d(axes[1, 1], rows)
    fig.suptitle(
        "Fig. 23  Collision-Cost Model — Ideal Collision Detection vs No CD\n"
        f"(bimodal PPDU {{4,12}}, PND cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE}; "
        "CD: collision = 1 slot, no-CD: collision = max colliding $L_i$)",
        fontsize=11,
    )
    fig_name = "fig23_collision_cost"
    for ext, kw in [("eps", dict(format="eps", bbox_inches="tight")),
                    ("png", dict(format="png", bbox_inches="tight", dpi=300)),
                    ("pdf", dict(format="pdf", bbox_inches="tight"))]:
        dest = os.path.join(fig_dir, f"{fig_name}.{ext}")
        fig.savefig(dest, **kw)
        print(f"  Figure → {dest}")
    preview = os.path.join(out_dir, f"{fig_name}_preview.png")
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight")
    print(f"  Preview → {preview}")
    plt.close(fig)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 23 — collision-cost model (CD vs no-CD)")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-dir", default="results/step9/fig23")
    parser.add_argument("--base-csv", default=None, metavar="PATH")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "results", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = load_csv(args.base_csv)
    else:
        nv = FAST_VISITS_23    if args.fast else FULL_VISITS_23
        nl = FAST_N_LIST_23    if args.fast else N_LIST_23
        wl = FAST_WEFF_LIST_23 if args.fast else WEFF_LIST_23
        sl = FAST_SEEDS_23     if args.fast else SEEDS_23
        total = len(nl) * len(wl) * len(sl) * len(CD_MODES) * len(METHODS_23)
        print(f"=== Figure 23 [{'FAST' if args.fast else 'FULL'}]  {nv} visits ===")
        print(f"    methods : {METHODS_23}")
        print(f"    cd modes: {CD_MODES}")
        print(f"    N       : {nl}")
        print(f"    W_eff   : {wl}")
        print(f"    configs : {total}")
        rows = run_sweep(nv, nl, wl, sl)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)
    check_hypotheses(rows)
    print("\nPlotting ...")
    plot(rows, out_dir, fig_dir)
    print("\nFigure 23 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig23_collision_cost.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
