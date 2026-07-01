"""
Figure 22: Initial Transmission Probability Sensitivity of PACE

RQ22: In the original PND (infinite-horizon neighbor discovery), the initial
transmission probability τ_0 barely matters — τ converges to the optimum quickly,
so randomized vs optimal init give near-identical discovery time
(Song et al., WCNC 2014, Fig. 2).

PACE runs the same MIMD feedback but inside a FINITE NPCA window of W_eff slots.
Does the finite horizon break PND's init-robustness? Hypothesis: solo-copy gives
fast intra-visit consensus, so PACE stays robust to τ_0 EXCEPT in the tight-window
regime (W_eff/N ≈ 1) where too few feedback rounds elapse before the window drains
— there an aggressive init collapses (early collisions) and a timid init wastes
early slots.

Init distributions (per-STA τ_0), all fed to the identical PND-MIMD engine:
  optimal   τ_0 = 1/N                (population-matched, best guess)
  rand_u01  τ_0 ~ U(0,1)             (fully randomized — PND paper's stress case)
  rand_u2N  τ_0 ~ U(0, 2/N)          (randomized, centered on optimal)
  high      τ_0 = 0.5                (aggressive: over-attempt)
  low       τ_0 = 1/W_eff            (timid: under-attempt)

oracle (τ=1/|viable| each slot) is the upper-bound reference (init-free).

4 panels:
  (a) efficiency vs N               (W_eff=50, uniform PPDU)
  (b) efficiency vs W_eff/N         (N=20) — tight→loose window sweep (KEY)
  (c) mean-τ trajectory, one visit  (N=20, W_eff=50) — inits collapse onto oracle track
  (d) efficiency by init: tight (W_eff/N=1) vs loose (W_eff/N=5) bar

Run:
  .venv/bin/python harq_sim/run_step9_fig22.py
  .venv/bin/python harq_sim/run_step9_fig22.py --fast
  .venv/bin/python harq_sim/run_step9_fig22.py --base-csv results/step9/fig22/data.csv
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

# Init configs are the "methods" swept here; "oracle" is the init-free upper bound.
INIT_CONFIGS = ["optimal", "rand_u01", "rand_u2N", "high", "low"]
METHODS_22   = ["oracle"] + INIT_CONFIGS

PPDU_DIST = "uniform"           # U[3,12] heterogeneous — matches fig17 finite-window setting

N_LIST_22    = [10, 20, 30, 50]
WEFF_LIST_22 = [20, 50, 100, 200]     # at N=20 → W_eff/N ∈ {1, 2.5, 5, 10}
SEEDS_22     = [42, 123, 456, 789, 1234]
FULL_VISITS_22 = 1000

FAST_N_LIST_22    = [10, 20]
FAST_WEFF_LIST_22 = [20, 50]
FAST_SEEDS_22     = [42]
FAST_VISITS_22    = 50

FIELDS_22 = [
    "init", "N", "W_eff", "seed",
    "efficiency", "W_eff_utilization", "collision_rate",
    "oracle_successes",
]

_STYLE_22 = {
    "oracle":   dict(color="#2ca02c", ls="--", lw=1.8, marker="D", ms=6),
    "optimal":  dict(color="#08306b", ls="-",  lw=2.2, marker="o", ms=6),
    "rand_u01": dict(color="#d62728", ls="-",  lw=2.0, marker="s", ms=6),
    "rand_u2N": dict(color="#ff7f0e", ls="-",  lw=1.8, marker="^", ms=6),
    "high":     dict(color="#9467bd", ls="-.", lw=1.8, marker="v", ms=6),
    "low":      dict(color="#17becf", ls=":",  lw=2.0, marker="P", ms=6),
}
_LABEL_22 = {
    "oracle":   "Oracle (τ=1/|V(t)|)",
    "optimal":  "optimal  τ₀=1/N",
    "rand_u01": "randomized  τ₀~U(0,1)",
    "rand_u2N": "randomized  τ₀~U(0,2/N)",
    "high":     "aggressive  τ₀=0.5",
    "low":      "timid  τ₀=1/W_eff",
}


# ─── Init sampler ─────────────────────────────────────────────────────────────

def sample_init_tau(init: str, N: int, W_eff: int, rng: np.random.Generator) -> np.ndarray:
    if init == "optimal":
        return np.full(N, 1.0 / N)
    if init == "rand_u01":
        return rng.uniform(0.0, 1.0, size=N)
    if init == "rand_u2N":
        return rng.uniform(0.0, 2.0 / N, size=N)
    if init == "high":
        return np.full(N, 0.5)
    if init == "low":
        return np.full(N, 1.0 / max(W_eff, 1))
    raise ValueError(init)


# ─── PACE single-visit (PND MIMD, configurable init τ_0) ──────────────────────

def _run_pace_visit(
    W_eff: int, ppdus: np.ndarray, rng: np.random.Generator,
    init_tau: np.ndarray, oracle_successes: int,
) -> dict:
    """PACE = PND MIMD + PPDU-aware self-exclusion, starting from init_tau."""
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff

    tau = np.clip(init_tau.astype(float).copy(), 1e-4, 1.0)
    _solo_sender_tau = 0.0

    successes = 0
    useful_slots = 0
    total_slots = 0
    collision_slots = 0

    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k_viable = int(viable.sum())
        if k_viable == 0:
            break

        tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)
        n_tx = int(tx.sum())
        outcome_idle = (n_tx == 0)
        outcome_coll = (n_tx > 1)
        outcome_solo = (n_tx == 1)

        if outcome_solo:
            i = int(np.where(tx)[0][0])
            _solo_sender_tau = float(tau[i])
            succeeded[i] = True
            W_rem -= int(ppdus[i])
            successes += 1
            useful_slots += int(ppdus[i])
            tau[i] = 0.0
        else:
            W_rem -= 1

        total_slots += 1
        if outcome_coll:
            collision_slots += 1

        # ── PND MIMD update (no CD): DW viable STAs adapt by slot outcome ──────
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

    efficiency = (successes / oracle_successes) if oracle_successes > 0 else 0.0
    col_rate = (collision_slots / total_slots) if total_slots > 0 else 0.0
    weff_util = useful_slots / W_eff if W_eff > 0 else 0.0
    return {
        "efficiency":        efficiency,
        "W_eff_utilization": weff_util,
        "collision_rate":    col_rate,
        "oracle_successes":  oracle_successes,
    }


# ─── Trajectory (panel c): mean-τ over viable STAs vs W_rem ────────────────────

def run_trajectory_visit(
    inits: list[str], N: int, W_eff: int, seed: int,
) -> dict[str, dict]:
    rng_ppdu = np.random.default_rng(seed * 10001 + 7)
    ppdus = _f17.sample_ppdu(PPDU_DIST, N, rng_ppdu)
    results: dict[str, dict] = {}

    # oracle track
    rng = np.random.default_rng(seed)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    w_hist, t_hist = [], []
    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k = int(viable.sum())
        if k == 0:
            break
        w_hist.append(W_rem)
        t_hist.append(1.0 / k)
        tx = rng.random(N) < np.where(viable, 1.0 / k, 0.0)
        if int(tx.sum()) == 1:
            i = int(np.where(tx)[0][0])
            succeeded[i] = True
            W_rem -= int(ppdus[i])
        else:
            W_rem -= 1
    results["oracle"] = {"w_rem": w_hist, "tau": t_hist,
                         "ppdu_thresholds": sorted(ppdus.tolist())}

    for init in inits:
        rng = np.random.default_rng(seed + 1)
        tau = np.clip(sample_init_tau(init, N, W_eff, rng), 1e-4, 1.0)
        succeeded = np.zeros(N, dtype=bool)
        W_rem = W_eff
        _solo_tau = 0.0
        w_hist, t_hist = [], []
        while True:
            viable = (~succeeded) & (ppdus <= W_rem)
            k = int(viable.sum())
            if k == 0:
                break
            w_hist.append(W_rem)
            vt = tau[viable]
            t_hist.append(float(vt.mean()) if len(vt) else 0.0)

            tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)
            n_tx = int(tx.sum())
            if n_tx == 1:
                i = int(np.where(tx)[0][0])
                _solo_tau = float(tau[i])
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                tau[i] = 0.0
            else:
                W_rem -= 1

            if n_tx == 1:
                for kk in range(N):
                    if not tx[kk] and not succeeded[kk] and viable[kk]:
                        tau[kk] = _solo_tau
            elif n_tx > 1:
                for kk in range(N):
                    if not tx[kk] and not succeeded[kk] and viable[kk]:
                        tau[kk] /= _f17.PND_C_COLL
            else:
                for kk in range(N):
                    if not tx[kk] and not succeeded[kk] and viable[kk]:
                        tau[kk] *= _f17.PND_C_IDLE
            for kk in range(N):
                if not succeeded[kk]:
                    tau[kk] = float(np.clip(tau[kk], 1e-4, 1.0))
        results[init] = {"w_rem": w_hist, "tau": t_hist,
                         "ppdu_thresholds": sorted(ppdus.tolist())}
    return results


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(n_visits: int, n_list: list, weff_list: list, seeds: list) -> list[dict]:
    init_idx = {m: i for i, m in enumerate(METHODS_22)}
    rows = []
    total = len(n_list) * len(weff_list) * len(seeds) * len(METHODS_22)
    done = 0

    for N in n_list:
        for W_eff in weff_list:
            for seed in seeds:
                rng_ppdu = np.random.default_rng(seed * 10001 + 7)
                oracle_data = []
                for v in range(n_visits):
                    ppdus = _f17.sample_ppdu(PPDU_DIST, N, rng_ppdu)
                    rng_o = np.random.default_rng(seed * 100003 + v)
                    os_v = _f17._run_oracle_visit(W_eff, ppdus, rng_o)
                    oracle_data.append((ppdus.copy(), os_v))

                for init in METHODS_22:
                    m_idx = init_idx[init]
                    eff_l, util_l, col_l, os_l = [], [], [], []
                    for v, (ppdus, os_v) in enumerate(oracle_data):
                        rng_v = np.random.default_rng(seed * 200003 + v * 17 + m_idx)
                        if init == "oracle":
                            succ = _f17._run_oracle_visit(W_eff, ppdus, rng_v)
                            eff = (succ / os_v) if os_v > 0 else 0.0
                            # oracle util: recompute useful slots quickly
                            util = _oracle_util(W_eff, ppdus, seed * 100003 + v)
                            res = {"efficiency": eff, "W_eff_utilization": util,
                                   "collision_rate": float("nan"), "oracle_successes": os_v}
                        else:
                            it = sample_init_tau(init, len(ppdus), W_eff, rng_v)
                            res = _run_pace_visit(W_eff, ppdus, rng_v, it, os_v)
                        eff_l.append(res["efficiency"])
                        util_l.append(res["W_eff_utilization"])
                        col_l.append(res["collision_rate"])
                        os_l.append(res["oracle_successes"])

                    rows.append({
                        "init":              init,
                        "N":                 N,
                        "W_eff":             W_eff,
                        "seed":              seed,
                        "efficiency":        float(np.mean(eff_l)),
                        "W_eff_utilization": float(np.mean(util_l)),
                        "collision_rate":    float(np.nanmean(col_l)),
                        "oracle_successes":  float(np.mean(os_l)),
                    })
                    done += 1
                    print(f"  [{done:4d}/{total}] N={N:2d}  W_eff={W_eff:3d}  "
                          f"{init:<10} seed={seed}", flush=True)
    return rows


def _oracle_util(W_eff: int, ppdus: np.ndarray, seed: int) -> float:
    """Oracle window utilization Σ(ppdu·succ)/W_eff for one visit."""
    rng = np.random.default_rng(seed)
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    useful = 0
    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k = int(viable.sum())
        if k == 0:
            break
        tx = rng.random(N) < (viable * (1.0 / k))
        if int(tx.sum()) == 1:
            i = int(np.where(tx)[0][0])
            succeeded[i] = True
            W_rem -= int(ppdus[i])
            useful += int(ppdus[i])
        else:
            W_rem -= 1
    return useful / W_eff if W_eff > 0 else 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean22(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _std22(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.std(finite)) if finite else 0.0


# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows) -> None:
    W_ref = 50
    avail_n = sorted({r["N"] for r in rows if r["W_eff"] == W_ref})
    for m in METHODS_22:
        means = [_mean22(rows, "efficiency", init=m, W_eff=W_ref, N=n) for n in avail_n]
        ax.plot(avail_n, means, label=_LABEL_22[m], **_STYLE_22[m])
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("N (contending STAs)", fontsize=11)
    ax.set_ylabel("Efficiency  (successes / oracle)", fontsize=10)
    ax.set_xticks(avail_n)
    ax.legend(fontsize=7.5, frameon=True, loc="lower left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) Efficiency vs N  (W_eff={W_ref}, {PPDU_DIST} PPDU)", fontsize=10)


def _panel_b(ax, rows) -> None:
    N_ref = 20
    avail_w = sorted({r["W_eff"] for r in rows if r["N"] == N_ref})
    for m in METHODS_22:
        x = [w / N_ref for w in avail_w]
        means = [_mean22(rows, "efficiency", init=m, W_eff=w, N=N_ref) for w in avail_w]
        ax.plot(x, means, label=_LABEL_22[m], **_STYLE_22[m])
    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", ls=":", lw=1.0, label="W_eff = N (tight)")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("W_eff / N  (log scale, left = tight window)", fontsize=10)
    ax.set_ylabel("Efficiency  (successes / oracle)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, loc="lower right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(b) Init sensitivity vs window tightness  (N={N_ref})", fontsize=10)


def _panel_c(ax, traj: dict) -> None:
    if not traj:
        ax.text(0.5, 0.5, "no trajectory", ha="center", va="center", transform=ax.transAxes)
        return
    ppdu_th = None
    for m in METHODS_22:
        if m not in traj:
            continue
        d = traj[m]
        if d["w_rem"]:
            st = dict(_STYLE_22[m]); st.pop("marker", None); st.pop("ms", None)
            ax.plot(d["w_rem"], d["tau"], label=_LABEL_22[m], **st)
        if ppdu_th is None:
            ppdu_th = d.get("ppdu_thresholds", [])
    if ppdu_th:
        first = True
        for th in sorted(set(ppdu_th)):
            ax.axvline(th, color="gray", ls=":", lw=0.8, alpha=0.5,
                       label="STA ppduᵢ thresholds" if first else None)
            first = False
    ax.invert_xaxis()
    ax.set_xlabel("W_rem (remaining slots, left = high)", fontsize=10)
    ax.set_ylabel("mean τ over viable STAs", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(c) τ trajectory, single visit — inits collapse onto oracle\n"
                 "(N=20, W_eff=50, seed=42)", fontsize=10)


def _panel_d(ax, rows) -> None:
    N_ref = 20
    regimes = [(20, "tight\nW_eff/N=1"), (100, "loose\nW_eff/N=5")]
    inits = INIT_CONFIGS
    x = np.arange(len(regimes))
    width = 0.15
    offsets = np.linspace(-(len(inits) - 1) / 2, (len(inits) - 1) / 2, len(inits)) * width
    for m, off in zip(inits, offsets):
        means = [_mean22(rows, "efficiency", init=m, W_eff=w, N=N_ref) for w, _ in regimes]
        stds = [_std22(rows, "efficiency", init=m, W_eff=w, N=N_ref) for w, _ in regimes]
        ax.bar(x + off, means, width, label=_LABEL_22[m],
               color=_STYLE_22[m]["color"], alpha=0.85, edgecolor="white")
        ax.errorbar(x + off, means, yerr=stds, fmt="none", ecolor="black",
                    capsize=2, elinewidth=0.8)
    # oracle reference lines per regime
    for xi, (w, _) in zip(x, regimes):
        o = _mean22(rows, "efficiency", init="oracle", W_eff=w, N=N_ref)
        ax.plot([xi - 0.4, xi + 0.4], [o, o], color="#2ca02c", ls="--", lw=1.5,
                label="oracle" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in regimes])
    ax.set_ylabel("Efficiency  (successes / oracle)", fontsize=10)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=7.0, frameon=True, loc="lower left", ncol=2)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7, axis="y")
    ax.set_title(f"(d) Init efficiency: tight vs loose window  (N={N_ref})", fontsize=10)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows) -> None:
    print("\n=== Hypothesis Check ===")
    N = 20

    print("\nH1: at loose window (W_eff=100, N=20) all inits ≈ optimal (init-robust like PND)")
    opt = _mean22(rows, "efficiency", init="optimal", W_eff=100, N=N)
    for m in INIT_CONFIGS:
        v = _mean22(rows, "efficiency", init=m, W_eff=100, N=N)
        tag = "≈" if abs(v - opt) < 0.03 else "≠"
        print(f"  {m:<10}: {v:.4f} {tag} optimal({opt:.4f})")

    print("\nH2: at tight window (W_eff=20, N=20) inits diverge (finite horizon breaks robustness)")
    opt_t = _mean22(rows, "efficiency", init="optimal", W_eff=20, N=N)
    spread = []
    for m in INIT_CONFIGS:
        v = _mean22(rows, "efficiency", init=m, W_eff=20, N=N)
        spread.append(v)
        print(f"  {m:<10}: {v:.4f}  (Δ vs optimal = {v - opt_t:+.4f})")
    print(f"  → tight-window spread = {max(spread) - min(spread):.4f}")

    print("\nH3: spread(tight) > spread(loose)")
    for w in [20, 50, 100, 200]:
        vals = [_mean22(rows, "efficiency", init=m, W_eff=w, N=N) for m in INIT_CONFIGS]
        print(f"  W_eff/N={w/N:>4.1f}:  spread={max(vals) - min(vals):.4f}  "
              f"[{', '.join(f'{v:.3f}' for v in vals)}]")

    print("\n--- Summary (N=20) efficiency by init × W_eff ---")
    print(f"  {'init':<10}" + "".join(f"{'W'+str(w):>9}" for w in WEFF_LIST_22))
    for m in METHODS_22:
        print(f"  {m:<10}" + "".join(
            f"{_mean22(rows, 'efficiency', init=m, W_eff=w, N=N):>9.4f}" for w in WEFF_LIST_22))


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_22)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path) -> list[dict]:
    int_fields = {"N", "W_eff", "seed"}
    str_fields = {"init"}
    float_fields = {f for f in FIELDS_22 if f not in int_fields and f not in str_fields}
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

def plot(rows, traj, out_dir, fig_dir) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.42, wspace=0.30)
    _panel_a(axes[0, 0], rows)
    _panel_b(axes[0, 1], rows)
    _panel_c(axes[1, 0], traj)
    _panel_d(axes[1, 1], rows)

    fig.suptitle(
        "Fig. 22  Initial Transmission Probability Sensitivity of PACE\n"
        f"(PND MIMD cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE}, {PPDU_DIST} PPDU U[3,12]; "
        "PND is init-robust in infinite horizon — does the finite W_eff break it?)",
        fontsize=11,
    )

    fig_name = "fig22_init_probability"
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
        description="Figure 22 — Initial transmission probability sensitivity of PACE")
    parser.add_argument("--fast", action="store_true",
                        help=f"Quick mode: {FAST_VISITS_22} visits, small grid")
    parser.add_argument("--out-dir", default="results/step9/fig22")
    parser.add_argument("--base-csv", default=None, metavar="PATH",
                        help="Load existing CSV and skip re-simulation")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "manuscript", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = load_csv(args.base_csv)
    else:
        nv = FAST_VISITS_22    if args.fast else FULL_VISITS_22
        nl = FAST_N_LIST_22    if args.fast else N_LIST_22
        wl = FAST_WEFF_LIST_22 if args.fast else WEFF_LIST_22
        sl = FAST_SEEDS_22     if args.fast else SEEDS_22
        total = len(nl) * len(wl) * len(sl) * len(METHODS_22)
        print(f"=== Figure 22 [{'FAST' if args.fast else 'FULL'}]  {nv} visits ===")
        print(f"    inits  : {METHODS_22}")
        print(f"    N      : {nl}")
        print(f"    W_eff  : {wl}")
        print(f"    seeds  : {sl}")
        print(f"    configs: {total}")
        rows = run_sweep(nv, nl, wl, sl)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    check_hypotheses(rows)

    print("\nComputing τ trajectory for panel (c) ...")
    traj = run_trajectory_visit(INIT_CONFIGS, N=20, W_eff=50, seed=42)

    print("\nPlotting ...")
    plot(rows, traj, out_dir, fig_dir)

    print("\nFigure 22 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig22_init_probability.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
