"""
Figure 21: Native vs. Visitor Fairness — Mixed NPCA Channel Access Competition

RQ21: When NPCA visitor STAs (opportunistic, adaptive protocol) and native STAs
(permanent, standard DCF) compete in the same W_eff window, does PND protect
visitor access while remaining fair to native STAs?

3 panels (W_eff=50):
  (a) Visitor W_eff_utilization vs N_native
  (b) Proportionality index vs N_native  — y=1.0 reference
  (c) Visitor TP vs Native TP scatter    — N_native=10

Visitor methods: oracle, pnd, ema_ad_low, consec_L2, dcf_self_excl, and
Native protocol: always standard DCF (CW0=N_total, ppdu=6)

Run:
  .venv/bin/python harq_sim/run_step9_fig21.py
  .venv/bin/python harq_sim/run_step9_fig21.py --fast
  .venv/bin/python harq_sim/run_step9_fig21.py --base-csv results/step9/fig21/data.csv
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

METHODS_21     = ["oracle", "pnd", "dcf_self_excl"]   # plotted set (CSV retains all methods)

N_VISITOR      = 10
N_NATIVE_LIST  = [0, 5, 10, 20]
WEFF_LIST_21   = [50, 100]
PPDU_NATIVE    = 6

SEEDS_21       = [42, 123, 456, 789, 1234]
FULL_VISITS_21 = 1000

FAST_N_NATIVE_LIST = [0, 5, 10]
FAST_WEFF_LIST_21  = [50]
FAST_SEEDS_21      = [42]
FAST_VISITS_21     = 50

FIELDS_21 = [
    "method", "N_visitor", "N_native", "W_eff", "seed",
    "weff_util_visitor", "weff_util_native", "weff_util_total",
    "proportionality", "visitor_share", "ideal_share",
    "native_preservation",
]

# Reuse fig17 styles for the same methods
_STYLE_21 = {m: _f17._METHOD_STYLE[m] for m in METHODS_21}
_LABEL_21 = {m: _f17._METHOD_LABEL[m] for m in METHODS_21}

_NATIVE_ONLY_IDX = 50   # synthetic RNG index for native-only baseline


# ─── Core simulation ──────────────────────────────────────────────────────────

def _run_visit_mixed(
    visitor_method: str,
    N_visitor: int,
    N_native: int,
    W_eff: int,
    ppdus: np.ndarray,          # shape (N_visitor + N_native,)
    rng: np.random.Generator,
    oracle_successes: int,
) -> dict:
    """
    Visitor STAs [0:N_visitor] : visitor_method (adaptive protocol)
    Native STAs  [N_visitor:]  : always DCF (CW0=N_total, ppdu=PPDU_NATIVE)

    Returns per-visit metrics: weff_util_visitor/native/total, proportionality,
    visitor_share, ideal_share.
    """
    N_total = N_visitor + N_native
    succeeded = np.zeros(N_total, dtype=bool)
    W_rem     = W_eff
    useful_visitor = 0
    useful_native  = 0

    # ── Visitor state ──────────────────────────────────────────────────────────
    tau          = np.full(N_visitor, 1.0 / max(N_total, 1))
    ema_idle_v   = np.full(N_visitor, _f17.IDLE_TARGET)
    consec_idle_v = np.zeros(N_visitor, dtype=np.int32)
    dcf_cw_vis   = np.full(N_visitor, N_total, dtype=np.int64)
    dcf_bo_vis   = rng.integers(0, max(N_total, 1), size=N_visitor).astype(np.int64)
    _solo_tau_v  = 0.0
    and_phase    = 1
    and_slots    = 0
    and_dur      = int(math.ceil(2**and_phase * math.e * math.log(2**and_phase)))

    # ── Native DCF state ───────────────────────────────────────────────────────
    dcf_cw_nat = np.full(N_native, N_total, dtype=np.int64)
    dcf_bo_nat = rng.integers(0, max(N_total, 1), size=N_native).astype(np.int64)

    _ema_methods = frozenset((
        "ema_fixed_low", "ema_fixed_high", "ema_adaptive",
        "ema_ad_low", "ema_ad_med", "ema_no_coll",
    ))

    while True:
        viable   = (~succeeded) & (ppdus <= W_rem)
        k_viable = int(viable.sum())
        if k_viable == 0:
            break

        viable_v = viable[:N_visitor]
        viable_n = viable[N_visitor:]
        tau_oracle = 1.0 / k_viable

        # ── TX decisions ──────────────────────────────────────────────────────
        if N_visitor > 0:
            if visitor_method == "oracle":
                tx_v = rng.random(N_visitor) < np.where(viable_v, tau_oracle, 0.0)
            elif visitor_method == "dcf_self_excl":
                tx_v = (dcf_bo_vis == 0) & viable_v
            elif visitor_method == "and":
                and_p = max(1.0 / (2 ** and_phase), 1e-4)
                tx_v  = rng.random(N_visitor) < np.where(viable_v, and_p, 0.0)
            else:   # pnd / ema_ad_low / consec_L2
                tx_v = rng.random(N_visitor) < np.where(viable_v, tau.clip(1e-4, 1.0), 0.0)
        else:
            tx_v = np.empty(0, dtype=bool)

        if N_native > 0:
            tx_n = (dcf_bo_nat == 0) & viable_n
        else:
            tx_n = np.empty(0, dtype=bool)

        tx   = np.concatenate([tx_v, tx_n])
        n_tx = int(tx.sum())
        outcome_idle = (n_tx == 0)
        outcome_coll = (n_tx > 1)
        outcome_solo = (n_tx == 1)

        # ── Outcome ───────────────────────────────────────────────────────────
        if outcome_solo:
            i = int(np.where(tx)[0][0])
            if ppdus[i] <= W_rem:
                if i < N_visitor:
                    _solo_tau_v = float(tau[i])
                    succeeded[i] = True
                    W_rem -= int(ppdus[i])
                    useful_visitor += int(ppdus[i])
                    tau[i] = 0.0
                    if visitor_method == "dcf_self_excl":
                        dcf_bo_vis[i] = W_eff + 1
                else:
                    j_nat = i - N_visitor
                    succeeded[i] = True
                    W_rem -= int(ppdus[i])
                    useful_native += int(ppdus[i])
                    dcf_bo_nat[j_nat] = W_eff + 1
            else:
                W_rem -= 1
                outcome_solo = False
                outcome_coll = True
        else:
            W_rem -= 1

        # ── Visitor state update ──────────────────────────────────────────────
        if N_visitor > 0 and visitor_method not in ("oracle", "and"):
            if visitor_method == "dcf_self_excl":
                if outcome_coll:
                    for j in np.where(tx_v)[0]:
                        j = int(j)
                        dcf_cw_vis[j] = min(int(dcf_cw_vis[j]) * 2, _f17.DCF_CW_MAX)
                        dcf_bo_vis[j] = int(rng.integers(0, max(int(dcf_cw_vis[j]), 1)))
                elif outcome_idle:
                    mask = (~succeeded[:N_visitor]) & viable_v & (dcf_bo_vis > 0)
                    dcf_bo_vis[mask] -= 1

            elif visitor_method in _ema_methods:
                alpha_down = _f17._ALPHA_DOWN_MAP.get(visitor_method, _f17.ALPHA_DOWN)
                for i in range(N_visitor):
                    if succeeded[i] or not viable_v[i]:
                        continue
                    beta = float(np.clip(_f17.BETA_BASE * N_total / max(W_rem, 1), 0.05, 0.50))
                    ema_idle_v[i] = (1 - beta) * ema_idle_v[i] + beta * float(outcome_idle)
                    gap = ema_idle_v[i] - _f17.IDLE_TARGET
                    if gap > _f17.BAND and viable_v[i]:
                        tau[i] = min(tau[i] * (1 + _f17.ALPHA_UP * gap), 1.0)
                    elif outcome_coll and viable_v[i] and alpha_down > 0.0:
                        tau[i] *= (1.0 - alpha_down)
                    tau[i] = float(np.clip(tau[i], 1e-4, 1.0))

            elif visitor_method in ("consec_L2", "consec_L4"):
                L = 2 if visitor_method == "consec_L2" else 4
                for i in range(N_visitor):
                    if succeeded[i] or not viable_v[i]:
                        continue
                    if outcome_idle:
                        consec_idle_v[i] += 1
                        if consec_idle_v[i] >= L:
                            tau[i] = min(tau[i] * (1 + _f17.ALPHA_UP), 1.0)
                    else:
                        consec_idle_v[i] = 0
                        if outcome_coll:
                            tau[i] *= (1 - _f17.ALPHA_DOWN)
                    tau[i] = float(np.clip(tau[i], 1e-4, 1.0))

            elif visitor_method in ("pnd", "pnd_cd"):
                if outcome_solo:
                    winner = int(np.where(tx)[0][0])
                    if winner < N_visitor:      # visitor won → DW visitors copy τ
                        for k in range(N_visitor):
                            if not tx_v[k] and not succeeded[k] and viable_v[k]:
                                tau[k] = _solo_tau_v
                    # native won → visitor τ unchanged (external event)
                elif outcome_coll:
                    for k in range(N_visitor):
                        if succeeded[k]:
                            continue
                        if viable_v[k]:
                            if not tx_v[k]:
                                tau[k] /= _f17.PND_C_COLL
                            elif visitor_method == "pnd_cd":
                                tau[k] /= _f17.PND_C_COLL
                elif outcome_idle:
                    for k in range(N_visitor):
                        if not tx_v[k] and not succeeded[k] and viable_v[k]:
                            tau[k] *= _f17.PND_C_IDLE
                for k in range(N_visitor):
                    if not succeeded[k]:
                        tau[k] = float(np.clip(tau[k], 1e-4, 1.0))

        if visitor_method == "and" and N_visitor > 0:
            and_slots += 1
            if and_slots >= and_dur and and_phase < 60:
                and_phase += 1
                and_slots  = 0
                and_dur    = int(math.ceil(2**and_phase * math.e * math.log(2**and_phase)))

        # ── Native DCF update ─────────────────────────────────────────────────
        if N_native > 0:
            if outcome_coll:
                for j in np.where(tx_n)[0]:
                    j = int(j)
                    dcf_cw_nat[j] = min(int(dcf_cw_nat[j]) * 2, _f17.DCF_CW_MAX)
                    dcf_bo_nat[j] = int(rng.integers(0, max(int(dcf_cw_nat[j]), 1)))
            elif outcome_idle:
                mask = (~succeeded[N_visitor:]) & viable_n & (dcf_bo_nat > 0)
                dcf_bo_nat[mask] -= 1
            # solo (any winner): medium busy → bo frozen for other native STAs

    # ── Compute metrics ───────────────────────────────────────────────────────
    succ_v = int(succeeded[:N_visitor].sum()) if N_visitor > 0 else 0
    succ_n = int(succeeded[N_visitor:].sum()) if N_native > 0 else 0
    succ_t = succ_v + succ_n

    weff_util_v = useful_visitor / W_eff if W_eff > 0 else 0.0
    weff_util_n = useful_native  / W_eff if W_eff > 0 else 0.0
    weff_util_t = (useful_visitor + useful_native) / W_eff if W_eff > 0 else 0.0

    if N_visitor == 0:
        visitor_share   = 0.0
        ideal_share     = 0.0
        proportionality = float("nan")  # undefined: no visitor STAs
    elif N_native == 0:
        visitor_share   = 1.0
        ideal_share     = 1.0
        proportionality = 1.0
    elif succ_t == 0:
        visitor_share   = 0.0
        ideal_share     = N_visitor / N_total
        proportionality = 1.0      # trivially fair (no one succeeded)
    else:
        visitor_share   = succ_v / succ_t
        ideal_share     = N_visitor / N_total
        proportionality = visitor_share / ideal_share

    return {
        "weff_util_visitor": weff_util_v,
        "weff_util_native":  weff_util_n,
        "weff_util_total":   weff_util_t,
        "visitor_share":     visitor_share,
        "ideal_share":       ideal_share,
        "proportionality":   proportionality,
    }


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep_mixed(
    n_visits: int, n_native_list: list, weff_list: list, seeds: list,
) -> list[dict]:
    """Runs mixed visitor+native simulation for all (method, N_native, W_eff, seed)."""
    method_idx = {m: i for i, m in enumerate(_f17.METHODS)}

    # Step 1: native-only baseline (N_visitor=0, pure DCF)
    native_only_map: dict[tuple, float] = {}
    print("  Computing native-only baselines ...")
    for N_native in n_native_list:
        for W_eff in weff_list:
            for seed in seeds:
                if N_native == 0:
                    native_only_map[(N_native, W_eff, seed)] = 0.0
                    continue
                ppdus_nat = np.full(N_native, PPDU_NATIVE, dtype=np.int32)
                util_list = []
                for v in range(n_visits):
                    rng_o = np.random.default_rng(seed * 100003 + v)
                    os_v  = _f17._run_oracle_visit(W_eff, ppdus_nat, rng_o)
                    rng_v = np.random.default_rng(
                        seed * 200003 + v * 17 + _NATIVE_ONLY_IDX)
                    res = _run_visit_mixed(
                        "dcf_self_excl", 0, N_native, W_eff, ppdus_nat, rng_v, os_v)
                    util_list.append(res["weff_util_native"])
                native_only_map[(N_native, W_eff, seed)] = float(np.mean(util_list))

    # Step 2: mixed simulations
    rows = []
    total = len(n_native_list) * len(weff_list) * len(seeds) * len(METHODS_21)
    done  = 0

    for N_native in n_native_list:
        for W_eff in weff_list:
            for seed in seeds:
                rng_ppdu = np.random.default_rng(seed * 10001 + 7)

                # Pre-generate oracle data (same RNG scheme as fig17/20)
                oracle_data: list[tuple[np.ndarray, int]] = []
                for v in range(n_visits):
                    ppdus_vis = _f17.sample_ppdu("bimodal", N_VISITOR, rng_ppdu)
                    ppdus_nat = np.full(N_native, PPDU_NATIVE, dtype=np.int32)
                    ppdus_all = np.concatenate([ppdus_vis, ppdus_nat]).astype(np.int32)
                    rng_o = np.random.default_rng(seed * 100003 + v)
                    os_v  = _f17._run_oracle_visit(W_eff, ppdus_all, rng_o)
                    oracle_data.append((ppdus_all.copy(), os_v))

                nat_only_base = native_only_map.get((N_native, W_eff, seed), 0.0)

                for method in METHODS_21:
                    m_idx = method_idx.get(method, 0)
                    uv_l, un_l, ut_l = [], [], []
                    pr_l, vs_l, is_l = [], [], []

                    for v, (ppdus_all, os_v) in enumerate(oracle_data):
                        rng_v = np.random.default_rng(
                            seed * 200003 + v * 17 + m_idx)
                        res = _run_visit_mixed(
                            method, N_VISITOR, N_native, W_eff, ppdus_all, rng_v, os_v)
                        uv_l.append(res["weff_util_visitor"])
                        un_l.append(res["weff_util_native"])
                        ut_l.append(res["weff_util_total"])
                        pr_l.append(res["proportionality"])
                        vs_l.append(res["visitor_share"])
                        is_l.append(res["ideal_share"])

                    mean_un = float(np.mean(un_l))
                    if N_native == 0:
                        nat_pres = float("nan")
                    elif nat_only_base > 0:
                        nat_pres = mean_un / nat_only_base
                    else:
                        nat_pres = 0.0

                    rows.append({
                        "method":              method,
                        "N_visitor":           N_VISITOR,
                        "N_native":            N_native,
                        "W_eff":               W_eff,
                        "seed":                seed,
                        "weff_util_visitor":   float(np.mean(uv_l)),
                        "weff_util_native":    mean_un,
                        "weff_util_total":     float(np.mean(ut_l)),
                        "proportionality":     float(np.mean(pr_l)),
                        "visitor_share":       float(np.mean(vs_l)),
                        "ideal_share":         float(np.mean(is_l)),
                        "native_preservation": nat_pres,
                    })
                    done += 1
                    print(f"  [{done:4d}/{total}] N_nat={N_native:2d}  W_eff={W_eff:3d}  "
                          f"{method:<18} seed={seed}", flush=True)

    # Step 3: emit native-only baseline (no NPCA visitor) as explicit rows.
    # weff_util_native here is the native window utilization with N_visitor=0,
    # i.e. the reference against which native_preservation is measured.
    for (N_native, W_eff, seed), base in native_only_map.items():
        rows.append({
            "method":              "native_only",
            "N_visitor":           0,
            "N_native":            N_native,
            "W_eff":               W_eff,
            "seed":                seed,
            "weff_util_visitor":   0.0,
            "weff_util_native":    float(base),
            "weff_util_total":     float(base),
            "proportionality":     float("nan"),
            "visitor_share":       0.0,
            "ideal_share":         float("nan"),
            "native_preservation": 1.0,
        })

    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean_m21(rows: list, metric: str, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]   # drop NaN/inf
    return float(np.mean(finite)) if finite else float("nan")


def _std_m21(rows: list, metric: str, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.std(finite)) if finite else 0.0


# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows: list) -> None:
    W_ref    = 50
    avail_nn = sorted({r["N_native"] for r in rows if r["W_eff"] == W_ref})

    for method in METHODS_21:
        means = [_mean_m21(rows, "weff_util_visitor",
                           method=method, W_eff=W_ref, N_native=n)
                 for n in avail_nn]
        ax.plot(avail_nn, means, label=_LABEL_21[method], **_STYLE_21[method])

    ax.set_xlabel("N_native (permanent NPCA STAs)", fontsize=11)
    ax.set_ylabel("Visitor Window Utilization  Σ(ppdu·succ) / W_eff", fontsize=10)
    ax.set_xticks(avail_nn)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) Visitor TP vs N_native\n"
                 f"(W_eff={W_ref}, N_visitor={N_VISITOR}, bimodal {{4,12}})",
                 fontsize=10)


def _panel_b(ax, rows: list) -> None:
    W_ref    = 50
    avail_nn = sorted({r["N_native"] for r in rows
                       if r["W_eff"] == W_ref and r["N_native"] > 0})

    for method in METHODS_21:
        means = [_mean_m21(rows, "proportionality",
                           method=method, W_eff=W_ref, N_native=n)
                 for n in avail_nn]
        ax.plot(avail_nn, means, label=_LABEL_21[method], **_STYLE_21[method])

    ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Proportional (=1.0)", zorder=1)
    ax.set_xlabel("N_native", fontsize=11)
    ax.set_ylabel("Proportionality  (visitor share / ideal share)", fontsize=10)
    ax.set_xticks(avail_nn)
    ymin = 0.0
    ax.set_ylim(bottom=ymin)
    ax.legend(fontsize=7.5, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(b) Proportionality vs N_native\n"
                 f"(W_eff={W_ref}, y=1.0 = population-proportional fair)",
                 fontsize=10)


def _panel_c(ax, rows: list) -> None:
    W_ref    = 50
    avail_nn = sorted({r["N_native"] for r in rows if r["W_eff"] == W_ref and r["N_native"] > 0})
    N_nat_ref = 10 if 10 in avail_nn else (avail_nn[-1] if avail_nn else 5)

    # Collect valid points first
    points = {}
    for method in METHODS_21:
        x = _mean_m21(rows, "weff_util_visitor",
                      method=method, N_native=N_nat_ref, W_eff=W_ref)
        y = _mean_m21(rows, "weff_util_native",
                      method=method, N_native=N_nat_ref, W_eff=W_ref)
        if x == x and y == y:   # NaN guard
            points[method] = (x, y)

    if not points:
        return

    # native-only baseline (no visitor): native utilization reference
    nat_base = _mean_m21(rows, "weff_util_native",
                         method="native_only", N_native=N_nat_ref, W_eff=W_ref)

    xs, ys = [p[0] for p in points.values()], [p[1] for p in points.values()]
    x_max = max(xs) * 1.30
    y_top = max(ys)
    if nat_base == nat_base:            # include baseline in y-range
        y_top = max(y_top, nat_base)
    y_max = y_top * 1.60   # 위쪽 여유 — annotation 두 줄 공간

    # 기울기 -1 iso-total 직선 (먼저 그려 마커 뒤에 위치)
    for method, (x, y) in points.items():
        C = x + y
        st = _STYLE_21[method]
        # y = C − x 직선을 [0, x_max] × [0, y_max] 범위로 클리핑
        x1 = max(0.0, C - y_max)
        x2 = min(C,   x_max)
        ax.plot([x1, x2], [C - x1, C - x2],
                color=st["color"], ls=":", lw=1.0, alpha=0.45, zorder=1)

    # 마커 + annotation (합산값 포함)
    # per-method annotation 위치 오버라이드: (dx_pt, dy_pt, ha)
    _ANN = {
        "dcf_self_excl": (-6, 4, "right"),
    }
    _DEFAULT_ANN = (6, 4, "left")

    _edge_only = {"x", "+", "|", "_"}
    for method, (x, y) in points.items():
        st   = _STYLE_21[method]
        _m   = st["marker"]
        _mec = st["color"] if _m in _edge_only else "white"
        _mew = st.get("markeredgewidth", 1.8) if _m in _edge_only else 0.6
        ax.plot(x, y, linestyle="none",
                color=st["color"], marker=_m,
                ms=st.get("ms", 8) + 2,
                markeredgecolor=_mec, markeredgewidth=_mew,
                label=_LABEL_21[method], zorder=5)
        dx, dy, ha = _ANN.get(method, _DEFAULT_ANN)
        short = _LABEL_21[method].split(" (")[0]
        ax.annotate(f"{short}\nΣ={x + y:.3f}", xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.0,
                    ha=ha, color=st["color"])

    # native-only reference: horizontal line + star marker at x=0
    if nat_base == nat_base:
        ax.axhline(nat_base, color="black", ls="--", lw=1.3, alpha=0.75, zorder=2,
                   label=f"Native-only (no visitor) = {nat_base:.3f}")
        ax.plot(0, nat_base, marker="*", color="black", ms=14,
                markeredgecolor="white", markeredgewidth=0.6, zorder=6)
        ax.annotate("native-only\n(no NPCA)", xy=(0, nat_base),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=7.0, ha="left", va="top", color="black")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Visitor W_eff_utilization", fontsize=10)
    ax.set_ylabel("Native W_eff_utilization", fontsize=10)
    ax.legend(fontsize=7.0, frameon=True, loc="lower left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(c) Visitor TP vs Native TP Scatter\n"
                 f"(N_native={N_nat_ref}, W_eff={W_ref}, dashed = iso-total)",
                 fontsize=10)

    # ── Zoom inset: crowded adaptive-method cluster ──────────────────────────
    zx0, zx1, zy0, zy1 = 0.45, 0.62, 0.10, 0.32
    in_zoom = {m: (x, y) for m, (x, y) in points.items()
               if zx0 <= x <= zx1 and zy0 <= y <= zy1}
    if in_zoom:
        axins = ax.inset_axes([0.50, 0.52, 0.46, 0.44])
        for method, (x, y) in in_zoom.items():     # iso-total guides (behind)
            C = x + y
            axins.plot([zx0, zx1], [C - zx0, C - zx1],
                       color=_STYLE_21[method]["color"], ls=":", lw=0.9,
                       alpha=0.45, zorder=1)
        for method, (x, y) in in_zoom.items():
            st   = _STYLE_21[method]
            _m   = st["marker"]
            _mec = st["color"] if _m in _edge_only else "white"
            _mew = st.get("markeredgewidth", 1.8) if _m in _edge_only else 0.6
            axins.plot(x, y, linestyle="none", color=st["color"], marker=_m,
                       ms=st.get("ms", 8) + 2, markeredgecolor=_mec,
                       markeredgewidth=_mew, zorder=5)
            short = _LABEL_21[method].split(" (")[0]
            axins.annotate(f"{short}\nΣ={x + y:.3f}", xy=(x, y), xytext=(4, 3),
                           textcoords="offset points", fontsize=6.3,
                           ha="left", color=st["color"])
        axins.set_xlim(zx0, zx1)
        axins.set_ylim(zy0, zy1)
        axins.tick_params(labelsize=6.3)
        axins.grid(True, ls=":", lw=0.5, alpha=0.6)
        axins.set_title("zoom", fontsize=7)
        ax.indicate_inset_zoom(axins, edgecolor="gray", lw=1.0, alpha=0.7)


def _panel_d(ax, rows: list) -> None:
    W_ref    = 50
    avail_nn = sorted({r["N_native"] for r in rows
                       if r["W_eff"] == W_ref and r["N_native"] > 0})

    for method in METHODS_21:
        means = [_mean_m21(rows, "weff_util_total",
                           method=method, W_eff=W_ref, N_native=n)
                 for n in avail_nn]
        ax.plot(avail_nn, means, label=_LABEL_21[method], **_STYLE_21[method])

    # native-only baseline (no visitor): total = native DCF utilization alone
    base_means = [_mean_m21(rows, "weff_util_total",
                            method="native_only", W_eff=W_ref, N_native=n)
                  for n in avail_nn]
    ax.plot(avail_nn, base_means, color="black", ls="--", lw=1.4,
            marker="*", ms=9, markeredgecolor="white", markeredgewidth=0.5,
            label="native-only (no visitor)", zorder=2)

    ax.set_xlabel("N_native (permanent NPCA STAs)", fontsize=11)
    ax.set_ylabel("Total Window Utilization  Σ(ppdu·succ) / W_eff", fontsize=10)
    ax.set_xticks(avail_nn)
    ax.margins(y=0.12)
    ax.legend(fontsize=7.5, frameon=True, loc="lower right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(d) Total TP vs N_native\n"
                 f"(W_eff={W_ref}, visitor+native combined)",
                 fontsize=10)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows: list) -> None:
    print("\n=== Hypothesis Check (W_eff=50) ===")
    W = 50
    avail_nn       = sorted({r["N_native"] for r in rows if r["W_eff"] == W})
    avail_nn_nonzero = [n for n in avail_nn if n > 0]
    print(f"  N_native tested: {avail_nn}\n")

    # H1: weff_util_visitor decreases monotonically with N_native
    print("H1: weff_util_visitor decreases monotonically with N_native")
    all_h1 = True
    for method in METHODS_21:
        vals = [_mean_m21(rows, "weff_util_visitor",
                          method=method, W_eff=W, N_native=n)
                for n in avail_nn]
        mono = all(vals[i] >= vals[i+1] - 0.001 for i in range(len(vals)-1))
        all_h1 = all_h1 and mono
        print(f"  {method:<20}: {[f'{v:.3f}' for v in vals]}  {'✅' if mono else '❌'}")
    print(f"  → H1 overall {'✅ PASS' if all_h1 else '❌ FAIL'}")

    # H2: PND proportionality >= DCF proportionality
    print("\nH2: PND proportionality >= DCF proportionality")
    h2_pass = True
    for n in avail_nn_nonzero:
        pnd_p = _mean_m21(rows, "proportionality", method="pnd",          W_eff=W, N_native=n)
        dcf_p = _mean_m21(rows, "proportionality", method="dcf_self_excl",W_eff=W, N_native=n)
        ok = pnd_p >= dcf_p
        h2_pass = h2_pass and ok
        print(f"  N_native={n:2d}: PND={pnd_p:.4f}  DCF={dcf_p:.4f}  {'✅' if ok else '❌'}")
    print(f"  → H2 overall {'✅ PASS' if h2_pass else '❌ FAIL'}")

    # H3: AND proportionality <= DCF proportionality
    print("\nH3: AND proportionality <= DCF proportionality")
    h3_pass = True
    for n in avail_nn_nonzero:
        and_p = _mean_m21(rows, "proportionality", method="and",          W_eff=W, N_native=n)
        dcf_p = _mean_m21(rows, "proportionality", method="dcf_self_excl",W_eff=W, N_native=n)
        ok = and_p <= dcf_p
        h3_pass = h3_pass and ok
        print(f"  N_native={n:2d}: AND={and_p:.4f}  DCF={dcf_p:.4f}  {'✅' if ok else '❌'}")
    print(f"  → H3 overall {'✅ PASS' if h3_pass else '❌ FAIL'}")

    # H4: PND native TP ≈ DCF native TP (within 10%)
    if avail_nn_nonzero:
        n_ref = 10 if 10 in avail_nn_nonzero else avail_nn_nonzero[-1]
        print(f"\nH4: PND native TP ≈ DCF native TP  (N_native={n_ref}, W_eff={W})")
        pnd_n = _mean_m21(rows, "weff_util_native", method="pnd",          W_eff=W, N_native=n_ref)
        dcf_n = _mean_m21(rows, "weff_util_native", method="dcf_self_excl",W_eff=W, N_native=n_ref)
        diff  = abs(pnd_n - dcf_n) / max(dcf_n, 1e-9)
        print(f"  PND={pnd_n:.4f}  DCF={dcf_n:.4f}  |diff|/DCF={diff:.3f}  "
              + ("✅ (within 10%)" if diff < 0.10 else "❌ (>10% diff)"))

    # H5: DCF visitor proportionality ≈ 1.0
    print("\nH5: DCF visitor proportionality ≈ 1.0 (symmetric → fair)")
    h5_pass = True
    for n in avail_nn_nonzero:
        dcf_p = _mean_m21(rows, "proportionality", method="dcf_self_excl", W_eff=W, N_native=n)
        ok = abs(dcf_p - 1.0) < 0.25
        h5_pass = h5_pass and ok
        print(f"  N_native={n:2d}: DCF prop={dcf_p:.4f}  {'✅' if ok else '❌'}")
    print(f"  → H5 overall {'✅ PASS' if h5_pass else '❌ FAIL'}")

    # Summary table
    n_sum = 10 if 10 in avail_nn_nonzero else (avail_nn_nonzero[-1] if avail_nn_nonzero else 0)
    if n_sum > 0:
        print(f"\n--- Summary (N_native={n_sum}, W_eff={W}) ---")
        print(f"  {'method':<20} {'util_v':>7} {'util_n':>7} {'util_t':>7} "
              f"{'prop':>7} {'nat_pres':>9}")
        for m in METHODS_21:
            uv = _mean_m21(rows, "weff_util_visitor",   method=m, W_eff=W, N_native=n_sum)
            un = _mean_m21(rows, "weff_util_native",    method=m, W_eff=W, N_native=n_sum)
            ut = _mean_m21(rows, "weff_util_total",     method=m, W_eff=W, N_native=n_sum)
            pr = _mean_m21(rows, "proportionality",     method=m, W_eff=W, N_native=n_sum)
            np_ = _mean_m21(rows, "native_preservation",method=m, W_eff=W, N_native=n_sum)
            np_str = f"{np_:9.4f}" if np_ == np_ else "      NaN"
            print(f"  {m:<20} {uv:7.4f} {un:7.4f} {ut:7.4f} {pr:7.4f} {np_str}")


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_21)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path: str) -> list[dict]:
    int_fields   = {"N_visitor", "N_native", "W_eff", "seed"}
    str_fields   = {"method"}
    float_fields = {f for f in FIELDS_21 if f not in int_fields and f not in str_fields}
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

def plot(rows: list, out_dir: str, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.48, wspace=0.40)

    _panel_a(axes[0, 0], rows)
    _panel_b(axes[0, 1], rows)
    _panel_c(axes[1, 0], rows)
    _panel_d(axes[1, 1], rows)

    n_nat_set = sorted({r["N_native"] for r in rows})
    fig.suptitle(
        f"Fig. 21  Native vs. Visitor Fairness — Mixed NPCA Contention\n"
        f"(N_visitor={N_VISITOR}, N_native∈{n_nat_set}, bimodal visitor {{4,12}}, "
        f"native ppdu={PPDU_NATIVE}, PND cc={_f17.PND_C_COLL}/ci={_f17.PND_C_IDLE})",
        fontsize=11,
    )

    fig_name = "fig21_native_visitor_fairness"
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
        description="Figure 21 — Native vs. Visitor fairness in mixed NPCA contention")
    parser.add_argument("--fast",     action="store_true",
                        help=f"Quick mode: {FAST_VISITS_21} visits, small grid")
    parser.add_argument("--out-dir",  default="results/step9/fig21")
    parser.add_argument("--base-csv", default=None, metavar="PATH",
                        help="Load existing CSV and skip re-simulation")
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
        nv = FAST_VISITS_21     if args.fast else FULL_VISITS_21
        nl = FAST_N_NATIVE_LIST if args.fast else N_NATIVE_LIST
        wl = FAST_WEFF_LIST_21  if args.fast else WEFF_LIST_21
        sl = FAST_SEEDS_21      if args.fast else SEEDS_21

        total = len(nl) * len(wl) * len(sl) * len(METHODS_21)
        print(f"=== Figure 21 [{'FAST' if args.fast else 'FULL'}]  {nv} visits ===")
        print(f"    visitor methods : {METHODS_21}")
        print(f"    N_visitor       : {N_VISITOR}")
        print(f"    N_native        : {nl}")
        print(f"    W_eff           : {wl}")
        print(f"    seeds           : {sl}")
        print(f"    total configs   : {total}")
        rows = run_sweep_mixed(nv, nl, wl, sl)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    check_hypotheses(rows)

    print("\nPlotting ...")
    plot(rows, out_dir, fig_dir)

    print("\nFigure 21 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig21_native_visitor_fairness.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
