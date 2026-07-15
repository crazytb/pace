"""
Figure 24: τ Persistence (Warm-Start) Across Consecutive NPCA Transitions
          — Mixed Native/Visitor NPCA Channel (v2)

RQ24: Every figure so far resets the visitor τ_0 at each NPCA visit ("cold
start"). Fig 21's mixed model even resets to 1/N_total — a genie that knows not
only its own group size but ALSO the native STA count on the non-primary
channel. In a real deployment a visitor BSS knows at best its own N_visitor;
the native load is invisible until experienced.

The same STA population transitions to the NPCA channel repeatedly. PACE's MIMD
state τ is an implicit estimate of the EFFECTIVE contention level (visitors +
natives), so memory — carrying the adapted τ from one visit into the next
("warm start") — can replace both genies at once.

Model (from fig 21): visitor STAs run PACE (PND MIMD, no-CD), native STAs run
standard DCF (CW0=N_total, ppdu=6) and are present in every visit. Natives'
DCF state is re-drawn each visit (stationary background); only visitors carry τ.

Methods (what the visitor knows at each visit start):
  oracle       τ = 1/|viable(t)| over ALL STAs      (upper bound, init-free)
  cold_genie   reset τ_0 = 1/N_total                (fig21 default; knows native count!)
  cold_nv      reset τ_0 = 1/N_visitor              (realistic cold: own group only)
  cold_high    reset τ_0 = 0.5                      (knows nothing, no memory)
  warm_nv      visit 1: τ_0 = 1/N_visitor, then carry
  warm_high    visit 1: τ_0 = 0.5,        then carry

Carry rule (warm): a succeeded visitor carries the τ it held just BEFORE its
solo success; a non-succeeded visitor carries its final adapted τ. Natives never
carry. Clip [1e-4, 1].

Churn (panel d): before each visit every visitor is independently replaced
w.p. ρ; a replaced visitor's carried τ is re-drawn from the init distribution.

Hypotheses:
  H1 warm_nv ≥ cold_nv, gap grows with N_native   (1/N_visitor over-aggressive
                                                    once natives load the channel)
  H2 warm_high ≫ cold_high                        (memory rescues zero knowledge)
  H3 warm τ_0 fixed point decreases with N_native (carry = implicit estimator of
                                                    TOTAL contention, incl. natives)
  H4 warm benefit vs churn is init-quality asymmetric

Efficiency = Σ visitor successes / Σ oracle visitor successes (ratio-of-means
over reps — avoids Jensen inflation of small per-visit ratios).

Collision-cost model: COLLISION_MODE inherited from run_step9_fig17 ("nocd").

Panels (W_eff=50, visitor PPDU U[3,12], native PPDU=6):
  (a) per-visit visitor efficiency vs transition index   (N_native=10)
  (b) steady-state visitor efficiency vs N_native        (KEY: warm vs cold gap)
  (c) carried τ_0 trajectory of warm_nv per N_native     (log y; 1/N_total refs)
  (d) steady-state efficiency vs churn ρ                 (N_native=10)

Run:
  .venv/bin/python harq_sim/run_step9_fig24.py
  .venv/bin/python harq_sim/run_step9_fig24.py --fast
  .venv/bin/python harq_sim/run_step9_fig24.py --base-csv results/step9/fig24/data.csv
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

# NOTE: "dcf_conv" appended last to keep RNG streams of earlier methods unchanged.
METHODS_24 = ["oracle", "cold_genie", "cold_nv", "cold_high", "warm_nv", "warm_high",
              "dcf_conv"]
WARM_METHODS = ["warm_nv", "warm_high"]

# Conventional NPCA visitor: standard CSMA/CA (802.11 DCF, aCWmin=15 → CW0=16)
DCF_CW_MIN_STD = 16

PPDU_DIST_V = "uniform"               # visitor PPDU U[3,12] — fig17 reference
PPDU_NATIVE = 6                       # fig21 native PPDU

N_VISITOR     = 10
N_NATIVE_LIST = [0, 5, 10, 20]
N_NAT_REF     = 10
W_REF         = 50
SEEDS_24      = [42, 123, 456, 789, 1234]
FULL_REPS     = 20                    # independent sequences per seed
FULL_VISITS   = 50                    # consecutive NPCA transitions per sequence
CHURN_LIST    = [0.0, 0.1, 0.3, 0.5, 1.0]

FAST_N_NATIVE = [0, 10]
FAST_SEEDS    = [42]
FAST_REPS     = 5
FAST_VISITS   = 30
FAST_CHURN    = [0.0, 0.5, 1.0]

FIELDS_24 = [
    "method", "churn", "N_visitor", "N_native", "W_eff", "seed", "visit",
    "efficiency", "util_v", "util_n", "collision_rate", "tau0_mean",
]

_STYLE_24 = {
    "oracle":     dict(color="#2ca02c", ls="--", lw=1.8, marker="D", ms=5),
    "cold_genie": dict(color="#08306b", ls="-",  lw=1.8, marker="o", ms=5),
    "cold_nv":    dict(color="#6baed6", ls="-",  lw=1.8, marker="s", ms=5),
    "cold_high":  dict(color="#d62728", ls=":",  lw=2.0, marker="v", ms=5),
    "warm_nv":    dict(color="#ff7f0e", ls="-",  lw=2.2, marker="^", ms=6),
    "warm_high":  dict(color="#9467bd", ls="-",  lw=2.0, marker="P", ms=6),
    "dcf_conv":   dict(color="#525252", ls="-.", lw=1.8, marker="x", ms=6),
}
_LABEL_24 = {
    "oracle":     "Oracle (τ=1/|V(t)|, all STAs)",
    "cold_genie": "cold, τ₀=1/N_total  (knows native count)",
    "cold_nv":    "cold, τ₀=1/N_visitor  (own group only)",
    "cold_high":  "cold, τ₀=0.5  (no knowledge, no memory)",
    "warm_nv":    "warm, τ₀=1/N_visitor then carry",
    "warm_high":  "warm, τ₀=0.5 then carry",
    "dcf_conv":   "conventional NPCA  (CSMA/CA, CW_min=16)",
}

_NNAT_COLORS = {0: "#9ecae1", 5: "#6baed6", 10: "#3182bd", 20: "#08519c"}


# ─── Init sampler ─────────────────────────────────────────────────────────────

def _init_tau(init: str, N_native: int) -> np.ndarray:
    if init == "genie":
        return np.full(N_VISITOR, 1.0 / max(N_VISITOR + N_native, 1))
    if init == "nv":
        return np.full(N_VISITOR, 1.0 / N_VISITOR)
    if init == "high":
        return np.full(N_VISITOR, 0.5)
    raise ValueError(init)


def _parse_method(method: str) -> tuple[str, str]:
    policy, init = method.split("_", 1)
    return policy, init


# ─── Mixed single visit (visitor PACE with τ carry; natives DCF — fig21 model) ─

def _run_mixed_visit(
    N_native: int, W_eff: int, ppdus: np.ndarray, rng: np.random.Generator,
    init_tau: np.ndarray | None, visitor_mode: str = "pace",
) -> dict:
    """
    One visit on the NPCA channel. Visitor behavior by visitor_mode:
      "pace"   PACE MIMD starting from init_tau
      "oracle" τ = 1/|viable| over all STAs (init-free)
      "dcf"    conventional NPCA — standard CSMA/CA (BEB, CW0=DCF_CW_MIN_STD),
               self-exclusion via Min Duration Threshold (viable mask)
    Natives [N_VISITOR:] always run DCF (CW0=N_total, ppdu=PPDU_NATIVE),
    fresh state each visit. Returns visitor metrics + carry-out τ (pace only).
    """
    oracle_mode = visitor_mode == "oracle"
    dcf_mode = visitor_mode == "dcf"
    N_total = N_VISITOR + N_native
    succeeded = np.zeros(N_total, dtype=bool)
    W_rem = W_eff
    useful_v = 0
    useful_n = 0

    if visitor_mode == "pace":
        tau = np.clip(init_tau.astype(float).copy(), 1e-4, 1.0)
        carry = tau.copy()          # succeeded visitors: τ just before success
    else:
        tau = np.zeros(N_VISITOR)
        carry = np.zeros(N_VISITOR)
    _solo_tau_v = 0.0

    # Conventional-NPCA visitor DCF state (dcf_mode only)
    dcf_cw_vis = np.full(N_VISITOR, DCF_CW_MIN_STD, dtype=np.int64)
    dcf_bo_vis = rng.integers(0, DCF_CW_MIN_STD, size=N_VISITOR).astype(np.int64) \
        if dcf_mode else np.zeros(N_VISITOR, dtype=np.int64)

    # Native DCF state — re-drawn each visit (stationary background traffic)
    dcf_cw_nat = np.full(N_native, N_total, dtype=np.int64)
    dcf_bo_nat = rng.integers(0, max(N_total, 1), size=N_native).astype(np.int64)

    succ_v = 0
    total_slots = 0
    collision_slots = 0

    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k_viable = int(viable.sum())
        if k_viable == 0:
            break

        viable_v = viable[:N_VISITOR]
        viable_n = viable[N_VISITOR:]

        if oracle_mode:
            tx_v = rng.random(N_VISITOR) < np.where(viable_v, 1.0 / k_viable, 0.0)
        elif dcf_mode:
            tx_v = (dcf_bo_vis == 0) & viable_v
        else:
            tx_v = rng.random(N_VISITOR) < np.where(viable_v, tau.clip(1e-4, 1.0), 0.0)
        tx_n = (dcf_bo_nat == 0) & viable_n if N_native > 0 else np.empty(0, dtype=bool)

        tx = np.concatenate([tx_v, tx_n])
        n_tx = int(tx.sum())
        outcome_idle = (n_tx == 0)
        outcome_coll = (n_tx > 1)
        outcome_solo = (n_tx == 1)

        if outcome_solo:
            i = int(np.where(tx)[0][0])
            if i < N_VISITOR:
                _solo_tau_v = float(tau[i])
                carry[i] = float(tau[i])
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                useful_v += int(ppdus[i])
                succ_v += 1
                tau[i] = 0.0
                if dcf_mode:
                    dcf_bo_vis[i] = W_eff + 1   # done for this visit
            else:
                j = i - N_VISITOR
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                useful_n += int(ppdus[i])
                dcf_bo_nat[j] = W_eff + 1
        elif outcome_coll:
            W_rem -= _f17.collision_cost(ppdus, tx)
        else:
            W_rem -= 1

        total_slots += 1
        if outcome_coll:
            collision_slots += 1

        # ── Conventional-NPCA visitor BEB update (dcf_mode) ───────────────────
        if dcf_mode:
            if outcome_coll:
                for j in np.where(tx_v)[0]:
                    j = int(j)
                    dcf_cw_vis[j] = min(int(dcf_cw_vis[j]) * 2, _f17.DCF_CW_MAX)
                    dcf_bo_vis[j] = int(rng.integers(0, max(int(dcf_cw_vis[j]), 1)))
            elif outcome_idle:
                mask = (~succeeded[:N_VISITOR]) & viable_v & (dcf_bo_vis > 0)
                dcf_bo_vis[mask] -= 1
            # solo: medium busy → other visitors' bo frozen

        # ── Visitor PND MIMD update (no CD; pace mode only) ───────────────────
        if visitor_mode == "pace":
            if outcome_solo:
                winner = int(np.where(tx)[0][0])
                if winner < N_VISITOR:
                    # visitor won → DW visitors copy sender's pre-success τ
                    for k in range(N_VISITOR):
                        if not tx_v[k] and not succeeded[k] and viable_v[k]:
                            tau[k] = _solo_tau_v
                # native won → visitor τ unchanged (external event; fig21 rule)
            elif outcome_coll:
                for k in range(N_VISITOR):
                    if not succeeded[k] and viable_v[k] and not tx_v[k]:
                        tau[k] /= _f17.PND_C_COLL
            elif outcome_idle:
                for k in range(N_VISITOR):
                    if not tx_v[k] and not succeeded[k] and viable_v[k]:
                        tau[k] *= _f17.PND_C_IDLE
            for k in range(N_VISITOR):
                if not succeeded[k]:
                    tau[k] = float(np.clip(tau[k], 1e-4, 1.0))

        # ── Native DCF update (fig21 rule) ────────────────────────────────────
        if N_native > 0:
            if outcome_coll:
                for j in np.where(tx_n)[0]:
                    j = int(j)
                    dcf_cw_nat[j] = min(int(dcf_cw_nat[j]) * 2, _f17.DCF_CW_MAX)
                    dcf_bo_nat[j] = int(rng.integers(0, max(int(dcf_cw_nat[j]), 1)))
            elif outcome_idle:
                mask = (~succeeded[N_VISITOR:]) & viable_n & (dcf_bo_nat > 0)
                dcf_bo_nat[mask] -= 1
            # solo: medium busy → other natives' bo frozen

    # Non-succeeded visitors carry their final adapted τ
    if visitor_mode == "pace":
        for k in range(N_VISITOR):
            if not succeeded[k]:
                carry[k] = float(np.clip(tau[k], 1e-4, 1.0))

    col_rate = (collision_slots / total_slots) if total_slots > 0 else 0.0
    return {
        "succ_v":         succ_v,
        "util_v":         useful_v / W_eff if W_eff > 0 else 0.0,
        "util_n":         useful_n / W_eff if W_eff > 0 else 0.0,
        "collision_rate": col_rate,
        "tau_end":        carry,
    }


def _sample_visit_ppdus(N_native: int, rng: np.random.Generator) -> np.ndarray:
    """Visitor PPDUs U[3,12] (resampled per visit) + fixed native PPDU."""
    ppdus_v = _f17.sample_ppdu(PPDU_DIST_V, N_VISITOR, rng)
    return np.concatenate([ppdus_v, np.full(N_native, PPDU_NATIVE, dtype=np.int32)])


# ─── One multi-visit sequence ─────────────────────────────────────────────────

def run_sequence(
    method: str, N_native: int, W_eff: int, visits: int,
    ppdus_list: list[np.ndarray], rng: np.random.Generator, churn: float = 0.0,
) -> dict[str, np.ndarray]:
    succs = np.zeros(visits)
    utils_v = np.zeros(visits)
    utils_n = np.zeros(visits)
    cols = np.full(visits, np.nan)
    tau0s = np.full(visits, np.nan)

    if method in ("oracle", "dcf_conv"):
        mode = "oracle" if method == "oracle" else "dcf"
        for v, ppdus in enumerate(ppdus_list):
            res = _run_mixed_visit(N_native, W_eff, ppdus, rng, None, mode)
            succs[v] = res["succ_v"]
            utils_v[v] = res["util_v"]
            utils_n[v] = res["util_n"]
            cols[v] = res["collision_rate"]
        return {"succs": succs, "utils_v": utils_v, "utils_n": utils_n,
                "cols": cols, "tau0s": tau0s}

    policy, init = _parse_method(method)
    tau_carry = _init_tau(init, N_native)

    for v, ppdus in enumerate(ppdus_list):
        if policy == "cold":
            tau0 = _init_tau(init, N_native)
        else:  # warm
            tau0 = tau_carry.copy()
            if v > 0 and churn > 0.0:
                replaced = rng.random(N_VISITOR) < churn
                if replaced.any():
                    tau0[replaced] = _init_tau(init, N_native)[replaced]

        res = _run_mixed_visit(N_native, W_eff, ppdus, rng, tau0)
        tau_carry = res["tau_end"]
        succs[v] = res["succ_v"]
        utils_v[v] = res["util_v"]
        utils_n[v] = res["util_n"]
        cols[v] = res["collision_rate"]
        tau0s[v] = float(np.mean(tau0))

    return {"succs": succs, "utils_v": utils_v, "utils_n": utils_n,
            "cols": cols, "tau0s": tau0s}


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(
    visits: int, reps: int, n_native_list: list, seeds: list, churn_list: list,
) -> list[dict]:
    rows: list[dict] = []
    m_idx = {m: i for i, m in enumerate(METHODS_24)}
    W_eff = W_REF

    total = len(n_native_list) * len(seeds) * len(METHODS_24) \
        + len([c for c in churn_list if c > 0]) * len(WARM_METHODS) * len(seeds)
    done = 0

    for N_native in n_native_list:
        churns_for_n = churn_list if N_native == N_NAT_REF else [0.0]
        for seed in seeds:
            # Shared per-rep visit schedules + oracle normalization sums
            schedules: list[list[np.ndarray]] = []
            os_sum = np.zeros(visits)
            for r in range(reps):
                rng_ppdu = np.random.default_rng(seed * 10001 + r * 71 + 7)
                sched = [_sample_visit_ppdus(N_native, rng_ppdu) for _ in range(visits)]
                schedules.append(sched)
                rng_o = np.random.default_rng(seed * 100003 + r * 977)
                for v, ppdus in enumerate(sched):
                    os_sum[v] += _run_mixed_visit(N_native, W_eff, ppdus, rng_o,
                                                  None, "oracle")["succ_v"]

            for method in METHODS_24:
                churns = churns_for_n if method in WARM_METHODS else [0.0]
                for churn in churns:
                    succ_acc = np.zeros(visits)
                    uv_acc = np.zeros(visits)
                    un_acc = np.zeros(visits)
                    col_acc = np.zeros(visits)
                    col_cnt = np.zeros(visits)
                    tau_acc = np.zeros(visits)
                    tau_cnt = np.zeros(visits)
                    for r in range(reps):
                        rng_m = np.random.default_rng(
                            seed * 200003 + r * 3163 + m_idx[method] * 29
                            + int(churn * 1000) * 13
                        )
                        sq = run_sequence(method, N_native, W_eff, visits,
                                          schedules[r], rng_m, churn)
                        succ_acc += sq["succs"]
                        uv_acc += sq["utils_v"]
                        un_acc += sq["utils_n"]
                        cmask = ~np.isnan(sq["cols"])
                        col_acc[cmask] += sq["cols"][cmask]
                        col_cnt += cmask
                        tmask = ~np.isnan(sq["tau0s"])
                        tau_acc[tmask] += sq["tau0s"][tmask]
                        tau_cnt += tmask

                    for v in range(visits):
                        # ratio-of-means: Σ visitor succ / Σ oracle visitor succ
                        rows.append({
                            "method":         method,
                            "churn":          churn,
                            "N_visitor":      N_VISITOR,
                            "N_native":       N_native,
                            "W_eff":          W_eff,
                            "seed":           seed,
                            "visit":          v,
                            "efficiency":     (succ_acc[v] / os_sum[v])
                                              if os_sum[v] > 0 else float("nan"),
                            "util_v":         uv_acc[v] / reps,
                            "util_n":         un_acc[v] / reps,
                            "collision_rate": (col_acc[v] / col_cnt[v])
                                              if col_cnt[v] > 0 else float("nan"),
                            "tau0_mean":      (tau_acc[v] / tau_cnt[v])
                                              if tau_cnt[v] > 0 else float("nan"),
                        })
                    done += 1
                    print(f"  [{done:4d}/{total}] N_nat={N_native:2d}  seed={seed:<5d} "
                          f"{method:<10} ρ={churn:.1f}", flush=True)
    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _mean24(rows, metric, **kw) -> float:
    vals = [r[metric] for r in _sel(rows, **kw)]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _steady(rows, metric, visits: int, **kw) -> float:
    vals = [r[metric] for r in _sel(rows, **kw) if r["visit"] >= visits // 2]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _visit_curve(rows, metric, visits: int, **kw) -> list[float]:
    return [_mean24(rows, metric, visit=v, **kw) for v in range(visits)]


# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows, visits: int) -> None:
    x = np.arange(1, visits + 1)
    for m in METHODS_24:
        curve = _visit_curve(rows, "efficiency", visits,
                             method=m, churn=0.0, N_native=N_NAT_REF)
        st = dict(_STYLE_24[m])
        st["ms"] = 3
        st["markevery"] = max(visits // 10, 1)
        ax.plot(x, curve, label=_LABEL_24[m], **st)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("NPCA transition index  (each = one full visit of W_eff slots)",
                  fontsize=10)
    ax.set_ylabel("Visitor efficiency  (succ / oracle succ)", fontsize=10)
    ax.legend(fontsize=7, frameon=True, loc="lower right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) Visitor efficiency across consecutive NPCA transitions\n"
                 f"(N_v={N_VISITOR}, N_nat={N_NAT_REF}, W_eff={W_REF}; "
                 "τ carried BETWEEN visits)", fontsize=10)


def _panel_b(ax, rows, visits: int, n_native_list: list) -> None:
    for m in METHODS_24:
        means = [_steady(rows, "efficiency", visits, method=m, churn=0.0, N_native=n)
                 for n in n_native_list]
        ax.plot(n_native_list, means, label=_LABEL_24[m], **_STYLE_24[m])
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(n_native_list)
    ax.set_xlabel("N_native  (native DCF STAs on the NPCA channel)", fontsize=10)
    ax.set_ylabel("Steady-state visitor efficiency  (visits ≥ V/2)", fontsize=10)
    ax.legend(fontsize=7, frameon=True, loc="lower left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(b) Native load sweep — what must the cold init know?\n"
                 f"(N_v={N_VISITOR}, W_eff={W_REF})", fontsize=10)


def _panel_c(ax, rows, visits: int, n_native_list: list) -> None:
    x = np.arange(1, visits + 1)
    for n in n_native_list:
        curve = _visit_curve(rows, "tau0_mean", visits,
                             method="warm_nv", churn=0.0, N_native=n)
        c = _NNAT_COLORS.get(n, "#333333")
        ax.plot(x, curve, color=c, lw=2.0, label=f"warm_nv, N_nat={n}")
        ax.axhline(1.0 / (N_VISITOR + n), color=c, ls=":", lw=1.2)
    ax.axhline(1.0 / N_VISITOR, color="black", ls="--", lw=1.2,
               label=f"τ=1/N_visitor={1.0 / N_VISITOR:.2f} (cold_nv)")
    ax.set_yscale("log")
    ax.set_xlabel("NPCA transition index  (each = one full visit of W_eff slots)",
                  fontsize=10)
    ax.set_ylabel("mean carried τ₀ at visit start  (log)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(c) Carried τ₀ tracks TOTAL contention incl. natives\n"
                 "(dotted = 1/N_total per N_nat)", fontsize=10)


def _panel_d(ax, rows, visits: int, churn_list: list) -> None:
    for m in WARM_METHODS:
        means = [_steady(rows, "efficiency", visits, method=m, churn=c,
                         N_native=N_NAT_REF) for c in churn_list]
        ax.plot(churn_list, means, label=_LABEL_24[m], **_STYLE_24[m])
    for m, ls in [("cold_genie", "--"), ("cold_nv", "-."), ("cold_high", ":"),
                  ("dcf_conv", "-.")]:
        ref = _steady(rows, "efficiency", visits, method=m, churn=0.0,
                      N_native=N_NAT_REF)
        ax.axhline(ref, color=_STYLE_24[m]["color"], ls=ls, lw=1.5,
                   label=_LABEL_24[m] + " (ref)")
    o = _steady(rows, "efficiency", visits, method="oracle", churn=0.0,
                N_native=N_NAT_REF)
    ax.axhline(o, color="#2ca02c", ls="--", lw=1.2, label="oracle (ref)")
    ax.set_xlabel("churn ρ  (fraction of visitors replaced per visit)", fontsize=10)
    ax.set_ylabel("Steady-state visitor efficiency  (visits ≥ V/2)", fontsize=10)
    ax.legend(fontsize=7, frameon=True, loc="center left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(d) Memory value under visitor churn  "
                 f"(N_v={N_VISITOR}, N_nat={N_NAT_REF}, W_eff={W_REF})", fontsize=10)


def _panel_e(ax, rows, visits: int, n_native_list: list) -> None:
    """Channel efficiency: total W_eff utilization (visitor + native airtime)."""
    for m in METHODS_24:
        tot = [
            _steady(rows, "util_v", visits, method=m, churn=0.0, N_native=n)
            + _steady(rows, "util_n", visits, method=m, churn=0.0, N_native=n)
            for n in n_native_list
        ]
        ax.plot(n_native_list, tot, label=_LABEL_24[m], **_STYLE_24[m])
    ax.set_xticks(n_native_list)
    ax.set_xlabel("N_native  (native DCF STAs on the NPCA channel)", fontsize=10)
    ax.set_ylabel("Channel efficiency  (util_v + util_n)", fontsize=10)
    ax.legend(fontsize=7, frameon=True, loc="lower left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(e) Total channel utilization vs native load\n"
                 f"(N_v={N_VISITOR}, W_eff={W_REF}; steady-state)", fontsize=10)


def _airtime_prop(rows, visits: int, method: str, n: int) -> float:
    """fig21-style proportionality on airtime: visitor share / population share."""
    uv = _steady(rows, "util_v", visits, method=method, churn=0.0, N_native=n)
    un = _steady(rows, "util_n", visits, method=method, churn=0.0, N_native=n)
    if not (uv == uv and un == un) or (uv + un) <= 0:
        return float("nan")
    share = uv / (uv + un)
    ideal = N_VISITOR / (N_VISITOR + n)
    return share / ideal


def _panel_f(ax, rows, visits: int, n_native_list: list) -> None:
    """Visitor/native airtime fairness: proportionality index (1 = fair)."""
    nats = [n for n in n_native_list if n > 0]
    for m in METHODS_24:
        props = [_airtime_prop(rows, visits, m, n) for n in nats]
        ax.plot(nats, props, label=_LABEL_24[m], **_STYLE_24[m])
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="proportional fair (=1)")
    ax.set_xticks(nats)
    ax.set_xlabel("N_native  (native DCF STAs on the NPCA channel)", fontsize=10)
    ax.set_ylabel("Airtime proportionality\n(visitor share / population share)",
                  fontsize=10)
    ax.legend(fontsize=7, frameon=True, loc="upper left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(f) Visitor↔native fairness — >1 = visitors over-grab\n"
                 f"(N_v={N_VISITOR}, W_eff={W_REF}; steady-state)", fontsize=10)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows, visits: int, n_native_list: list, churn_list: list) -> None:
    print("\n=== Hypothesis Check ===")

    print("\nH1: warm_nv vs cold_nv gap grows with N_native "
          "(1/N_visitor over-aggressive under native load)")
    for n in n_native_list:
        cn = _steady(rows, "efficiency", visits, method="cold_nv", churn=0.0, N_native=n)
        wn = _steady(rows, "efficiency", visits, method="warm_nv", churn=0.0, N_native=n)
        cg = _steady(rows, "efficiency", visits, method="cold_genie", churn=0.0,
                     N_native=n)
        print(f"  N_nat={n:2d}: cold_nv={cn:.4f}  warm_nv={wn:.4f}  Δ={wn - cn:+.4f}  "
              f"(cold_genie={cg:.4f})")

    print(f"\nH2: warm_high rescues zero knowledge — N_nat={N_NAT_REF}")
    ch = _steady(rows, "efficiency", visits, method="cold_high", churn=0.0,
                 N_native=N_NAT_REF)
    wh = _steady(rows, "efficiency", visits, method="warm_high", churn=0.0,
                 N_native=N_NAT_REF)
    print(f"  cold_high={ch:.4f}  warm_high={wh:.4f}  "
          f"gain ×{wh / ch if ch > 0 else float('inf'):.1f}")

    print("\nH3: warm_nv τ₀ fixed point vs native load (last-visit mean)")
    for n in n_native_list:
        t_last = _mean24(rows, "tau0_mean", method="warm_nv", churn=0.0,
                         N_native=n, visit=visits - 1)
        print(f"  N_nat={n:2d}: τ₀[V-1]={t_last:.4f}   "
              f"(1/N_v={1.0 / N_VISITOR:.3f}, 1/N_tot={1.0 / (N_VISITOR + n):.4f})")

    print(f"\nH4: warm benefit vs churn ρ (steady efficiency, N_nat={N_NAT_REF})")
    for m in WARM_METHODS:
        line = "  " + m + ": " + "  ".join(
            f"ρ={c:.1f}:{_steady(rows, 'efficiency', visits, method=m, churn=c, N_native=N_NAT_REF):.4f}"
            for c in churn_list)
        print(line)

    print("\n--- Steady-state visitor efficiency by method × N_native (churn=0) ---")
    print(f"  {'method':<11}" + "".join(f"{'nat' + str(n):>9}" for n in n_native_list))
    for m in METHODS_24:
        print(f"  {m:<11}" + "".join(
            f"{_steady(rows, 'efficiency', visits, method=m, churn=0.0, N_native=n):>9.4f}"
            for n in n_native_list))

    print("\n--- Steady-state channel efficiency & airtime fairness (churn=0) ---")
    for n in [x for x in n_native_list if x > 0]:
        print(f"  N_native={n}:")
        print(f"    {'method':<11} {'util_v':>7} {'util_n':>7} {'total':>7} "
              f"{'v_share':>8} {'prop':>6}")
        ideal = N_VISITOR / (N_VISITOR + n)
        for m in METHODS_24:
            uv = _steady(rows, "util_v", visits, method=m, churn=0.0, N_native=n)
            un = _steady(rows, "util_n", visits, method=m, churn=0.0, N_native=n)
            share = uv / (uv + un) if (uv + un) > 0 else float("nan")
            prop = share / ideal if share == share else float("nan")
            print(f"    {m:<11} {uv:>7.4f} {un:>7.4f} {uv + un:>7.4f} "
                  f"{share:>8.4f} {prop:>6.3f}")


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_24)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path) -> list[dict]:
    int_fields = {"N_visitor", "N_native", "W_eff", "seed", "visit"}
    str_fields = {"method"}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k in str_fields:
                    continue
                elif k in int_fields:
                    row[k] = int(v)
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = float("nan")
            rows.append(row)
    return rows


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot(rows, visits: int, n_native_list: list, churn_list: list,
         out_dir: str, fig_dir: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 16.5))
    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    _panel_a(axes[0, 0], rows, visits)
    _panel_b(axes[0, 1], rows, visits, n_native_list)
    _panel_c(axes[1, 0], rows, visits, n_native_list)
    _panel_d(axes[1, 1], rows, visits, churn_list)
    _panel_e(axes[2, 0], rows, visits, n_native_list)
    _panel_f(axes[2, 1], rows, visits, n_native_list)

    fig.suptitle(
        "Fig. 24  τ Persistence (Warm-Start) Across Consecutive NPCA Transitions — "
        "Mixed Native/Visitor Channel\n"
        f"(sequence = {FULL_VISITS} back-to-back visits, visitor τ carried between "
        f"visits; visitors: PACE cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE} "
        f"PPDU U[3,12]; natives: DCF ppdu={PPDU_NATIVE}; "
        f"collision={_f17.COLLISION_MODE})",
        fontsize=10.5,
    )

    fig_name = "fig24_warm_start"
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
        description="Figure 24 — warm-start τ across NPCA transitions, mixed channel")
    parser.add_argument("--fast", action="store_true",
                        help=f"Quick mode: {FAST_REPS} reps × {FAST_VISITS} visits")
    parser.add_argument("--out-dir", default="results/step9/fig24")
    parser.add_argument("--base-csv", default=None, metavar="PATH",
                        help="Load existing CSV and skip re-simulation")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "manuscript", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    visits = FAST_VISITS if args.fast else FULL_VISITS
    n_native_list = FAST_N_NATIVE if args.fast else N_NATIVE_LIST
    churn_list = FAST_CHURN if args.fast else CHURN_LIST

    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = load_csv(args.base_csv)
        visits = max(int(r["visit"]) for r in rows) + 1
        n_native_list = sorted({int(r["N_native"]) for r in rows})
        churn_list = sorted({float(r["churn"]) for r in rows})
    else:
        reps = FAST_REPS if args.fast else FULL_REPS
        seeds = FAST_SEEDS if args.fast else SEEDS_24
        print(f"=== Figure 24 v2 [{'FAST' if args.fast else 'FULL'}] ===")
        print(f"    methods  : {METHODS_24}")
        print(f"    N_visitor: {N_VISITOR}  N_native: {n_native_list}  W_eff: {W_REF}")
        print(f"    visits   : {visits}  reps/seed: {reps}  seeds: {seeds}")
        print(f"    churn    : {churn_list} (warm methods, N_nat={N_NAT_REF} only)")
        print(f"    collision-cost mode: {_f17.COLLISION_MODE}")
        rows = run_sweep(visits, reps, n_native_list, seeds, churn_list)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    check_hypotheses(rows, visits, n_native_list, churn_list)

    print("\nPlotting ...")
    plot(rows, visits, n_native_list, churn_list, out_dir, fig_dir)

    print("\nFigure 24 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig24_warm_start.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
