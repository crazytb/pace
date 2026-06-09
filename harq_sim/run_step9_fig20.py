"""
Figure 20: Throughput–Fairness Tradeoff — PPDU-Class Fairness Analysis

RQ: Does PND's high NPCA window utilization come at a fairness cost?
Specifically, in bimodal {4,12} environments, do small PPDU STAs monopolize
the window at the expense of large PPDU STAs?

3 panels (bimodal PPDU dist only):
  (a) Jain's Fairness Index vs N   — W_eff=50
  (b) Per-class success rate bar   — N=20, W_eff=50
  (c) Throughput–Fairness scatter  — N=20, W_eff=50

Methods: oracle, pnd, ema_ad_low, consec_L2, dcf_self_excl, and

Run:
  .venv/bin/python harq_sim/run_step9_fig20.py
  .venv/bin/python harq_sim/run_step9_fig20.py --fast
  .venv/bin/python harq_sim/run_step9_fig20.py --base-csv results/step9/fig20/data.csv
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

# ─── Methods & schema ─────────────────────────────────────────────────────────

METHODS_20 = ["oracle", "pnd", "ema_ad_low", "consec_L2", "dcf_self_excl", "and"]

FIELDS_20 = [
    "method", "ppdu_dist", "N", "W_eff", "seed",
    "W_eff_utilization", "mean_jain",
    "p_success_small", "p_success_large", "class_gap",
    "p_cond_small", "p_cond_large",
]

# ─── Style ────────────────────────────────────────────────────────────────────

_STYLE_20 = {
    "oracle":        dict(color="#2ca02c", ls="--", lw=1.5, marker="D", ms=7),
    "pnd":           dict(color="#17becf", ls="-",  lw=2.2, marker="P", ms=8),
    "ema_ad_low":    dict(color="#e6550d", ls="-",  lw=2.0, marker="p", ms=7),
    "consec_L2":     dict(color="#756bb1", ls="--", lw=2.0, marker="x", ms=7,
                          markeredgewidth=1.8),
    "dcf_self_excl": dict(color="#08306b", ls="-",  lw=2.0, marker="8", ms=7),
    "and":           dict(color="#7f2704", ls=":",  lw=1.8, marker="d", ms=6),
}
_LABEL_20 = {
    "oracle":        "Oracle (reference)",
    "pnd":           f"PND MIMD (cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE})",
    "ema_ad_low":    "EMA adaptive (α↓=0.10)",
    "consec_L2":     "Consec L=2",
    "dcf_self_excl": "DCF + self-excl",
    "and":           "AND (open-loop)",
}

# ─── Per-STA-tracking simulation ──────────────────────────────────────────────

def _run_visit_fairness(
    method: str, W_eff: int, ppdus: np.ndarray, rng: np.random.Generator,
    oracle_successes: int,
) -> dict:
    """
    Replicates _f17._run_visit logic for methods in METHODS_20.
    Additionally returns per-STA tracking needed for fairness metrics:
      succeeded_mask : np.ndarray[bool, N]
      viable_rounds  : np.ndarray[int,  N]  — rounds each STA was viable
    """
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    remaining = N
    viable_rounds = np.zeros(N, dtype=np.int32)

    tau = np.full(N, 1.0 / N)
    ema_idle = np.full(N, _f17.IDLE_TARGET)
    consec_idle = np.zeros(N, dtype=np.int32)
    prev_viable_count = N

    dcf_cw = np.full(N, N, dtype=np.int64)
    dcf_bo = rng.integers(0, N, size=N).astype(np.int64)

    _solo_sender_tau = 0.0

    and_phase = 1
    and_phase_slots = 0
    and_phase_dur = int(math.ceil(2 ** and_phase * math.e * math.log(2 ** and_phase)))

    successes = 0
    useful_slots = 0

    _ema_methods = frozenset((
        "ema_fixed_low", "ema_fixed_high", "ema_adaptive",
        "ema_ad_low", "ema_ad_med", "ema_no_coll",
    ))

    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k_viable = int(viable.sum())
        if k_viable == 0:
            break

        viable_rounds[viable] += 1
        tau_oracle_val = 1.0 / k_viable

        # ── TX decision ───────────────────────────────────────────────────────
        if method == "oracle":
            tx = rng.random(N) < np.where(viable, tau_oracle_val, 0.0)
        elif method == "dcf_self_excl":
            tx = (dcf_bo == 0) & viable
        elif method == "and":
            and_p = max(1.0 / (2 ** and_phase), 1e-4)
            tx = rng.random(N) < np.where(viable, and_p, 0.0)
        else:
            # EMA / PND / consec variants
            tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)

        n_tx = int(tx.sum())
        outcome_idle = (n_tx == 0)
        outcome_coll = (n_tx > 1)
        outcome_solo = (n_tx == 1)

        # ── Outcome accounting ────────────────────────────────────────────────
        if outcome_solo:
            i = int(np.where(tx)[0][0])
            if ppdus[i] <= W_rem:
                _solo_sender_tau = float(tau[i])
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                successes += 1
                useful_slots += int(ppdus[i])
                remaining -= 1
                tau[i] = 0.0
                if method == "dcf_self_excl":
                    dcf_bo[i] = W_eff + 1
            else:
                W_rem -= 1
                outcome_solo = False
                outcome_coll = True
        else:
            W_rem -= 1

        cur_viable_count = k_viable
        viable_count_stable = (cur_viable_count == prev_viable_count)

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

        elif method in _ema_methods:
            alpha_down_val = _f17._ALPHA_DOWN_MAP.get(method, _f17.ALPHA_DOWN)
            for i in range(N):
                if succeeded[i] or (not viable[i] and not succeeded[i]):
                    continue
                beta = float(np.clip(_f17.BETA_BASE * N / max(W_rem, 1), 0.05, 0.50))
                ema_idle[i] = (1 - beta) * ema_idle[i] + beta * float(outcome_idle)
                gap = ema_idle[i] - _f17.IDLE_TARGET
                if gap > _f17.BAND and viable[i]:
                    tau[i] = min(tau[i] * (1 + _f17.ALPHA_UP * gap), 1.0)
                elif outcome_coll and viable[i] and alpha_down_val > 0.0:
                    tau[i] *= (1.0 - alpha_down_val)
                tau[i] = float(np.clip(tau[i], 1e-4, 1.0))

        elif method in ("consec_L2", "consec_L4"):
            L = 2 if method == "consec_L2" else 4
            for i in range(N):
                if succeeded[i] or (not viable[i] and not succeeded[i]):
                    continue
                if outcome_idle:
                    consec_idle[i] += 1
                    if consec_idle[i] >= L and viable[i]:
                        tau[i] = min(tau[i] * (1 + _f17.ALPHA_UP), 1.0)
                else:
                    consec_idle[i] = 0
                    if outcome_coll and viable[i]:
                        tau[i] *= (1 - _f17.ALPHA_DOWN)
                tau[i] = float(np.clip(tau[i], 1e-4, 1.0))

        elif method in ("pnd", "pnd_cd"):
            if outcome_solo:
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] = _solo_sender_tau
            elif outcome_coll:
                for k in range(N):
                    if succeeded[k]:
                        continue
                    if viable[k]:
                        if not tx[k]:
                            tau[k] /= _f17.PND_C_COLL
                        elif method == "pnd_cd":
                            tau[k] /= _f17.PND_C_COLL
            elif outcome_idle:
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] *= _f17.PND_C_IDLE
            for k in range(N):
                if not succeeded[k]:
                    tau[k] = float(np.clip(tau[k], 1e-4, 1.0))

        if method == "and":
            and_phase_slots += 1
            if and_phase_slots >= and_phase_dur and and_phase < 60:
                and_phase += 1
                and_phase_slots = 0
                and_phase_dur = int(math.ceil(2 ** and_phase * math.e * math.log(2 ** and_phase)))

        prev_viable_count = cur_viable_count

    weff_util = useful_slots / W_eff if W_eff > 0 else 0.0

    return {
        "W_eff_utilization": weff_util,
        "succeeded_mask":    succeeded,
        "viable_rounds":     viable_rounds,
    }


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep_fairness(
    n_visits: int, n_list: list, weff_list: list, seeds: list,
) -> list[dict]:
    """Bimodal {4,12} only — collects per-STA fairness metrics over n_visits."""
    rows = []
    method_idx = {m: i for i, m in enumerate(_f17.METHODS)}
    total = len(n_list) * len(weff_list) * len(seeds) * len(METHODS_20)
    done = 0

    for N in n_list:
        for W_eff in weff_list:
            for seed in seeds:
                rng_ppdu = np.random.default_rng(seed * 10001 + 7)

                # Pre-generate oracle successes (same RNG scheme as fig17)
                oracle_data: list[tuple[np.ndarray, int]] = []
                for v in range(n_visits):
                    rng_v = np.random.default_rng(seed * 100003 + v)
                    ppdus = _f17.sample_ppdu("bimodal", N, rng_ppdu)
                    os_v = _f17._run_oracle_visit(W_eff, ppdus, rng_v)
                    oracle_data.append((ppdus.copy(), os_v))

                for method in METHODS_20:
                    m_idx = method_idx.get(method, 0)

                    J_list: list[float] = []
                    weff_util_list: list[float] = []
                    succ_small: list[float] = []
                    succ_large: list[float] = []
                    vr_small:   list[int]   = []
                    vr_large:   list[int]   = []

                    for v, (ppdus, os_v) in enumerate(oracle_data):
                        rng_v = np.random.default_rng(seed * 200003 + v * 17 + m_idx)
                        res = _run_visit_fairness(method, W_eff, ppdus, rng_v, os_v)

                        # Jain's index: J = k^2 / (N * k) = k/N for binary success
                        s_vec = res["succeeded_mask"].astype(float)
                        n_succ = s_vec.sum()
                        J = float(n_succ ** 2 / (N * (s_vec ** 2).sum())) if n_succ > 0 else 1.0
                        J_list.append(J)
                        weff_util_list.append(res["W_eff_utilization"])

                        for i in range(N):
                            succ_i = float(res["succeeded_mask"][i])
                            vr_i   = int(res["viable_rounds"][i])
                            if ppdus[i] == 4:
                                succ_small.append(succ_i)
                                vr_small.append(vr_i)
                            elif ppdus[i] == 12:
                                succ_large.append(succ_i)
                                vr_large.append(vr_i)

                    p_s = float(np.mean(succ_small)) if succ_small else 0.0
                    p_l = float(np.mean(succ_large)) if succ_large else 0.0
                    sum_vrs = int(np.sum(vr_small))
                    sum_vrl = int(np.sum(vr_large))

                    row = {
                        "method":            method,
                        "ppdu_dist":         "bimodal",
                        "N":                 N,
                        "W_eff":             W_eff,
                        "seed":              seed,
                        "W_eff_utilization": float(np.mean(weff_util_list)),
                        "mean_jain":         float(np.mean(J_list)),
                        "p_success_small":   p_s,
                        "p_success_large":   p_l,
                        "class_gap":         p_s - p_l,
                        "p_cond_small":      float(np.sum(succ_small)) / sum_vrs if sum_vrs > 0 else 0.0,
                        "p_cond_large":      float(np.sum(succ_large)) / sum_vrl if sum_vrl > 0 else 0.0,
                    }
                    rows.append(row)
                    done += 1
                    print(f"  [{done:4d}/{total}] bimodal N={N:2d} W_eff={W_eff:3d} "
                          f"{method:<18} seed={seed}", flush=True)

    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean_m(rows: list, metric: str, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    return float(np.mean(vals)) if vals else 0.0


def _std_m(rows: list, metric: str, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    return float(np.std(vals)) if vals else 0.0


# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows: list) -> None:
    W_ref = 50
    avail_N = sorted({r["N"] for r in rows if r["W_eff"] == W_ref})

    for method in METHODS_20:
        means = [_mean_m(rows, "mean_jain", method=method, W_eff=W_ref, N=N)
                 for N in avail_N]
        ax.plot(avail_N, means, label=_LABEL_20[method], **_STYLE_20[method])

    ax.set_xlabel("N (contending STAs)", fontsize=11)
    ax.set_ylabel("Mean Jain's Fairness Index", fontsize=10)
    ax.set_xticks(avail_N)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) Jain's Fairness Index vs N\n(W_eff={W_ref}, bimodal {{4,12}})",
                 fontsize=10)


def _panel_b(ax, rows: list) -> None:
    N_ref = 20
    W_ref = 50

    # Fall back to smallest available N if 20 not present (fast mode)
    avail_N = sorted({r["N"] for r in rows})
    if N_ref not in avail_N:
        N_ref = avail_N[0]

    x = np.arange(len(METHODS_20))
    width = 0.35

    p_small = [_mean_m(rows, "p_success_small", method=m, N=N_ref, W_eff=W_ref)
               for m in METHODS_20]
    p_large = [_mean_m(rows, "p_success_large", method=m, N=N_ref, W_eff=W_ref)
               for m in METHODS_20]
    std_s   = [_std_m(rows,  "p_success_small", method=m, N=N_ref, W_eff=W_ref)
               for m in METHODS_20]
    std_l   = [_std_m(rows,  "p_success_large", method=m, N=N_ref, W_eff=W_ref)
               for m in METHODS_20]

    ax.bar(x - width / 2, p_small, width, yerr=std_s, capsize=3,
           label="Small (ppdu=4)", color="#2196F3", alpha=0.85, edgecolor="white",
           error_kw=dict(elinewidth=0.8))
    ax.bar(x + width / 2, p_large, width, yerr=std_l, capsize=3,
           label="Large (ppdu=12)", color="#FF5722", alpha=0.85, edgecolor="white",
           error_kw=dict(elinewidth=0.8))

    short_labels = [_LABEL_20[m].split(" (")[0] for m in METHODS_20]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Per-STA success rate", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7, axis="y")
    ax.set_title(f"(b) Per-class success rate\n(N={N_ref}, W_eff={W_ref}, bimodal {{4,12}})",
                 fontsize=10)


def _panel_c(ax, rows: list) -> None:
    N_ref = 20
    W_ref = 50

    avail_N = sorted({r["N"] for r in rows})
    if N_ref not in avail_N:
        N_ref = avail_N[0]

    for method in METHODS_20:
        x = _mean_m(rows, "W_eff_utilization", method=method, N=N_ref, W_eff=W_ref)
        y = _mean_m(rows, "mean_jain",          method=method, N=N_ref, W_eff=W_ref)
        if x == 0.0 and y == 0.0:
            continue
        st = _STYLE_20[method]
        ax.plot(x, y,
                linestyle="none",
                color=st["color"], marker=st["marker"],
                ms=st.get("ms", 8) + 2,
                markeredgecolor="white", markeredgewidth=0.6,
                label=_LABEL_20[method], zorder=5)
        short = _LABEL_20[method].split(" (")[0]
        ax.annotate(short, xy=(x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5,
                    color=st["color"])

    ax.set_xlabel("NPCA Window Utilization  Σ(ppdu·succ) / W_eff", fontsize=10)
    ax.set_ylabel("Mean Jain's Fairness Index", fontsize=10)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7.5, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(c) Throughput–Fairness Scatter\n(N={N_ref}, W_eff={W_ref}, bimodal {{4,12}})",
                 fontsize=10)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows: list) -> None:
    print("\n=== Hypothesis Check (bimodal, W_eff=50) ===")

    avail_N = sorted({r["N"] for r in rows if r["W_eff"] == 50})
    N = 20 if 20 in avail_N else avail_N[0]
    W = 50

    print(f"  Using N={N}, W_eff={W}\n")

    # H1: p_small > p_large for all methods
    print("H1: p_success_small > p_success_large (all methods)")
    all_pass = True
    for m in METHODS_20:
        ps = _mean_m(rows, "p_success_small", method=m, N=N, W_eff=W)
        pl = _mean_m(rows, "p_success_large", method=m, N=N, W_eff=W)
        ok = ps > pl
        all_pass = all_pass and ok
        print(f"  {m:<20}: small={ps:.4f}  large={pl:.4f}  gap={ps - pl:+.4f}  "
              + ("✅" if ok else "❌"))
    print(f"  → H1 overall {'✅ PASS' if all_pass else '❌ FAIL (some methods)'}")

    # H2: PND class_gap <= DCF class_gap
    pnd_gap = _mean_m(rows, "class_gap", method="pnd",           N=N, W_eff=W)
    dcf_gap = _mean_m(rows, "class_gap", method="dcf_self_excl", N=N, W_eff=W)
    print(f"\nH2: PND class_gap <= DCF class_gap")
    print(f"  PND gap={pnd_gap:.4f}  DCF gap={dcf_gap:.4f}  "
          + ("✅ PASS" if pnd_gap <= dcf_gap else "❌ FAIL"))

    # H3: PND Jain's J >= DCF Jain's J
    pnd_j = _mean_m(rows, "mean_jain", method="pnd",           N=N, W_eff=W)
    dcf_j = _mean_m(rows, "mean_jain", method="dcf_self_excl", N=N, W_eff=W)
    print(f"\nH3: PND Jain's J >= DCF Jain's J")
    print(f"  PND J={pnd_j:.4f}  DCF J={dcf_j:.4f}  "
          + ("✅ PASS" if pnd_j >= dcf_j else "❌ FAIL"))

    # H4: PND Pareto-dominant over DCF (higher TP AND higher/equal J)
    pnd_tp = _mean_m(rows, "W_eff_utilization", method="pnd",           N=N, W_eff=W)
    dcf_tp = _mean_m(rows, "W_eff_utilization", method="dcf_self_excl", N=N, W_eff=W)
    pareto = (pnd_tp > dcf_tp) and (pnd_j >= dcf_j)
    print(f"\nH4: PND Pareto-dominant (TP AND J both >= DCF)")
    print(f"  PND: TP={pnd_tp:.4f}  J={pnd_j:.4f}")
    print(f"  DCF: TP={dcf_tp:.4f}  J={dcf_j:.4f}")
    print(f"  → {'✅ PASS' if pareto else '❌ FAIL'}")

    # Summary table
    print(f"\n--- Fairness summary (bimodal, W_eff={W}, N={N}) ---")
    hdr = (f"  {'method':<20} {'TP_util':>8} {'Jain_J':>7} "
           f"{'p_s':>6} {'p_l':>6} {'gap':>6} {'cond_s':>7} {'cond_l':>7}")
    print(hdr)
    for m in METHODS_20:
        print(
            f"  {m:<20} "
            f"{_mean_m(rows, 'W_eff_utilization', method=m, N=N, W_eff=W):8.4f} "
            f"{_mean_m(rows, 'mean_jain',          method=m, N=N, W_eff=W):7.4f} "
            f"{_mean_m(rows, 'p_success_small',    method=m, N=N, W_eff=W):6.4f} "
            f"{_mean_m(rows, 'p_success_large',    method=m, N=N, W_eff=W):6.4f} "
            f"{_mean_m(rows, 'class_gap',          method=m, N=N, W_eff=W):6.4f} "
            f"{_mean_m(rows, 'p_cond_small',       method=m, N=N, W_eff=W):7.4f} "
            f"{_mean_m(rows, 'p_cond_large',       method=m, N=N, W_eff=W):7.4f}"
        )


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_20)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path: str) -> list[dict]:
    int_fields   = {"N", "W_eff", "seed"}
    str_fields   = {"method", "ppdu_dist"}
    float_fields = {f for f in FIELDS_20 if f not in int_fields and f not in str_fields}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in int_fields:
                if k in row:
                    row[k] = int(row[k])
            for k in float_fields:
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot(rows: list, out_dir: str, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.42)

    _panel_a(axes[0], rows)
    _panel_b(axes[1], rows)
    _panel_c(axes[2], rows)

    fig.suptitle(
        "Fig. 20  Throughput–Fairness Tradeoff — PPDU-Class Fairness Analysis\n"
        f"(bimodal {{4,12}}, N∈{sorted({r['N'] for r in rows})}, "
        f"PND cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE})",
        fontsize=11,
    )

    fig_name = "fig20_fairness"
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


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 20 — Throughput–Fairness analysis (PPDU-class fairness)")
    parser.add_argument("--fast",     action="store_true",
                        help=f"Quick mode: {_f17.FAST_VISITS} visits, small N/W grid")
    parser.add_argument("--out-dir",  default="results/step9/fig20")
    parser.add_argument("--base-csv", default=None, metavar="PATH",
                        help="Existing data CSV to skip re-simulation")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir   = os.path.join(repo_root, "manuscript", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = load_csv(args.base_csv)
    else:
        nv = _f17.FAST_VISITS if args.fast else _f17.FULL_VISITS
        nl = _f17.FAST_N_LIST if args.fast else _f17.N_LIST
        wl = _f17.FAST_WEFF_LIST if args.fast else _f17.WEFF_LIST
        sl = _f17.FAST_SEEDS if args.fast else _f17.SEEDS

        total = len(nl) * len(wl) * len(sl) * len(METHODS_20)
        print(f"=== Figure 20 [{'FAST' if args.fast else 'FULL'}]  {nv} visits ===")
        print(f"    methods : {METHODS_20}")
        print(f"    N       : {nl}")
        print(f"    W_eff   : {wl}")
        print(f"    seeds   : {sl}")
        print(f"    total   : {total} configurations")
        rows = run_sweep_fairness(nv, nl, wl, sl)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    check_hypotheses(rows)

    print("\nPlotting ...")
    plot(rows, out_dir, fig_dir)

    print("\nFigure 20 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig20_fairness.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
