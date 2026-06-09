"""
Figure 17: PPDU-Aware Self-Exclusion + Adaptive τ — Heterogeneous PPDU Validation

Ablation study: silent self-exclusion × EMA-based τ adaptation in finite NPCA window
with heterogeneous PPDU lengths.

v3 additions:
  - dcf_self_excl : IEEE 802.11 DCF with BEB + self-exclusion
  - ema_ad_low    : EMA adaptive β, ALPHA_DOWN=0.10
  - ema_ad_med    : EMA adaptive β, ALPHA_DOWN=0.25
  - ema_no_coll   : EMA adaptive β, no collision penalty (gap-only τ control)

v4 additions:
  - pnd           : Probabilistic Neighbor Discovery (MIMD, no CD)
                    Song et al., WCNC 2014
  - pnd_cd        : PND with Collision Detection (TX STA also reduces τ on collision)

Output:
  manuscript/figure/fig17_ppdu_aware_tau.{eps,png,pdf}
  results/step9/fig17_v3/data.csv

Run:
  python harq_sim/run_step9_fig17.py [--fast] [--out-dir results/step9/fig17_v3]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Parameters ───────────────────────────────────────────────────────────────

N_LIST    = [10, 20, 30, 50]
WEFF_LIST = [20, 50, 100, 200]
SEEDS     = [42, 123, 456, 789, 1234]

FULL_VISITS = 1000
FAST_VISITS = 50

FAST_N_LIST    = [10, 20]
FAST_WEFF_LIST = [20, 50]
FAST_SEEDS     = [42]

# EMA hyperparameters
ALPHA_UP   = 0.3
ALPHA_DOWN = 0.5          # default; variants use different values
BAND       = 0.05
BETA_BASE  = 0.1
IDLE_TARGET = 1.0 / math.e   # ≈ 0.3679

# DCF hyperparameters
DCF_CW_MAX = 1023

# PND hyperparameters — tuned via fig18 parameter study
# Optimal across N=[20,30,50] W_eff=[50,100,200]: cc=1.2, ci=1.2
# (was 1.5/1.5 in original PND paper; 1.2/1.2 +1-2% in finite NPCA window)
PND_C_COLL = 1.2
PND_C_IDLE = 1.2

# Per-method ALPHA_DOWN values (0.0 = no collision penalty)
_ALPHA_DOWN_MAP = {
    "ema_fixed_low":  ALPHA_DOWN,
    "ema_fixed_high": ALPHA_DOWN,
    "ema_adaptive":   ALPHA_DOWN,
    "ema_ad_low":     0.10,
    "ema_ad_med":     0.25,
    "ema_no_coll":    0.00,
}

METHODS = [
    "oracle",
    "mfg_no_excl",
    "self_excl_only",
    "ema_fixed_low",
    "ema_fixed_high",
    "ema_adaptive",
    "ema_ad_low",
    "ema_ad_med",
    "ema_no_coll",
    "consec_L2",
    "consec_L4",
    "dcf_self_excl",
    "pnd",
    "pnd_cd",
    "and",
]

PPDU_DIST_NAMES = ["homo", "uniform", "bimodal"]

FIELDS = [
    "method", "ppdu_dist", "N", "W_eff", "seed", "visit",
    "successes", "oracle_successes", "efficiency",
    "useful_slots", "W_eff_utilization",
    "viable_slots_wasted", "collision_rate", "tau_rmse", "false_trigger_rate",
]


# ─── PPDU samplers ────────────────────────────────────────────────────────────

def sample_ppdu(dist_name: str, N: int, rng: np.random.Generator) -> np.ndarray:
    if dist_name == "homo":
        return np.full(N, 6, dtype=np.int32)        # was 15; 6 → ~8 tx/visit at W_eff=50
    elif dist_name == "uniform":
        return rng.integers(3, 13, size=N).astype(np.int32)   # U[3,12] mean≈7.5; was U[5,40]
    elif dist_name == "bimodal":
        return rng.choice([4, 12], size=N).astype(np.int32)   # mean=8; was [8,35]
    raise ValueError(dist_name)


# ─── Single-visit simulation ──────────────────────────────────────────────────

def _run_oracle_visit(W_eff: int, ppdus: np.ndarray, rng: np.random.Generator) -> int:
    """Oracle: τ = 1/viable_remaining each round. Returns successes."""
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


def _run_visit(
    method: str, W_eff: int, ppdus: np.ndarray, rng: np.random.Generator,
    oracle_successes: int,
) -> dict:
    N = len(ppdus)
    succeeded = np.zeros(N, dtype=bool)
    W_rem = W_eff
    remaining = N

    # Per-STA τ / EMA / consec state
    tau = np.full(N, 1.0 / N)
    ema_idle = np.full(N, IDLE_TARGET)
    consec_idle = np.zeros(N, dtype=np.int32)
    prev_viable_count = N

    # DCF state — CW0 = N (MFG precommit style)
    dcf_cw = np.full(N, N, dtype=np.int64)
    dcf_bo = rng.integers(0, N, size=N).astype(np.int64)

    # PND: save sender τ before zeroing (used by DW update on solo success)
    _solo_sender_tau = 0.0

    # AND state — phase-based open-loop schedule (Vasudevan et al., MobiCom 2009)
    # Phase i: p = 1/2^i, duration = ceil(2^i * e * ln(2^i)) slots
    and_phase = 1
    and_phase_slots = 0
    and_phase_dur = int(math.ceil(2**and_phase * math.e * math.log(2**and_phase)))

    successes = 0
    useful_slots = 0        # Σ(ppdu_i × success_i) — productive NPCA window usage
    total_slots = 0
    collision_slots = 0
    viable_idle_slots = 0
    tau_sq_err_sum = 0.0
    tau_sq_err_cnt = 0
    false_triggers = 0

    _ema_methods = frozenset(
        ("ema_fixed_low", "ema_fixed_high", "ema_adaptive",
         "ema_ad_low", "ema_ad_med", "ema_no_coll")
    )

    while True:
        viable = (~succeeded) & (ppdus <= W_rem)
        k_viable = int(viable.sum())
        if k_viable == 0:
            break

        tau_oracle_val = 1.0 / k_viable

        # ── Determine TX ──────────────────────────────────────────────────────
        if method == "oracle":
            tx = rng.random(N) < np.where(viable, tau_oracle_val, 0.0)
        elif method == "mfg_no_excl":
            tx = rng.random(N) < np.where(~succeeded, 1.0 / max(remaining, 1), 0.0)
        elif method == "self_excl_only":
            tx = rng.random(N) < np.where(viable, 1.0 / max(remaining, 1), 0.0)
        elif method == "dcf_self_excl":
            tx = (dcf_bo == 0) & viable   # deterministic; non-viable bo frozen
        elif method == "and":
            and_p = max(1.0 / (2 ** and_phase), 1e-4)
            tx = rng.random(N) < np.where(viable, and_p, 0.0)
        else:
            # EMA / consec: per-STA τ, self-exclusion via viable mask
            tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)

        n_tx = int(tx.sum())
        outcome_idle = (n_tx == 0)
        outcome_coll = (n_tx > 1)
        outcome_solo = (n_tx == 1)

        # ── Outcome accounting ────────────────────────────────────────────────
        if outcome_solo:
            i = int(np.where(tx)[0][0])
            if ppdus[i] <= W_rem:
                # Successful TX
                _solo_sender_tau = float(tau[i])   # save before zeroing (PND DW update)
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                successes += 1
                useful_slots += int(ppdus[i])
                remaining -= 1
                tau[i] = 0.0
                if method == "dcf_self_excl":
                    dcf_bo[i] = W_eff + 1   # permanently done
            else:
                # PPDU > W_rem: NPCA_TIMER would expire mid-frame → failure
                # (only possible for mfg_no_excl; dcf/excl methods can't reach here)
                W_rem -= 1
                outcome_solo = False
                outcome_coll = True
                if k_viable > 0:
                    viable_idle_slots += 1
        else:
            W_rem -= 1
            if outcome_idle and k_viable > 0:
                viable_idle_slots += 1

        total_slots += 1
        if outcome_coll:
            collision_slots += 1

        # ── RMSE tracking (EMA / consec methods only) ─────────────────────────
        if method in _ema_methods or method in ("consec_L2", "consec_L4"):
            for i in range(N):
                if viable[i]:
                    tau_sq_err_sum += (tau[i] - tau_oracle_val) ** 2
                    tau_sq_err_cnt += 1

        cur_viable_count = k_viable
        viable_count_stable = (cur_viable_count == prev_viable_count)

        # ── DCF state update ──────────────────────────────────────────────────
        if method == "dcf_self_excl":
            if outcome_coll:
                for j in np.where(tx)[0]:
                    j = int(j)
                    dcf_cw[j] = min(int(dcf_cw[j]) * 2, DCF_CW_MAX)
                    dcf_bo[j] = int(rng.integers(0, max(int(dcf_cw[j]), 1)))
            elif outcome_solo and not succeeded[int(np.where(tx)[0][0])]:
                # solo but failed (ppdu > W_rem) — treated as collision above
                pass
            elif outcome_idle:
                # Decrement backoff for viable non-succeeded STAs (self-excl: non-viable frozen)
                mask = (~succeeded) & viable & (dcf_bo > 0)
                dcf_bo[mask] -= 1
            # solo success: dcf_bo[i] already set to W_eff+1 above

        # ── EMA / consec τ update ─────────────────────────────────────────────
        elif method in _ema_methods:
            alpha_down_val = _ALPHA_DOWN_MAP[method]
            for i in range(N):
                if succeeded[i] or (not viable[i] and not succeeded[i]):
                    continue

                if method == "ema_fixed_low":
                    beta = 0.05
                elif method == "ema_fixed_high":
                    beta = 0.30
                else:
                    beta = float(np.clip(BETA_BASE * N / max(W_rem, 1), 0.05, 0.50))

                ema_idle[i] = (1 - beta) * ema_idle[i] + beta * float(outcome_idle)
                gap = ema_idle[i] - IDLE_TARGET

                if gap > BAND and viable[i]:
                    new_tau = tau[i] * (1 + ALPHA_UP * gap)
                    if viable_count_stable and new_tau > tau[i]:
                        false_triggers += 1
                    tau[i] = min(new_tau, 1.0)
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
                        new_tau = min(tau[i] * (1 + ALPHA_UP), 1.0)
                        if viable_count_stable and new_tau > tau[i]:
                            false_triggers += 1
                        tau[i] = new_tau
                else:
                    consec_idle[i] = 0
                    if outcome_coll and viable[i]:
                        tau[i] *= (1 - ALPHA_DOWN)
                tau[i] = float(np.clip(tau[i], 1e-4, 1.0))

        elif method in ("pnd", "pnd_cd"):
            # PND MIMD update (Song et al., WCNC 2014)
            # DT STAs (tx[i]=True): half-duplex → no update unless pnd_cd + collision
            # DW STAs (tx[i]=False, not succeeded, viable): update by outcome
            if outcome_solo:
                # DW viable STAs copy sender's pre-success τ
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] = _solo_sender_tau
            elif outcome_coll:
                for k in range(N):
                    if succeeded[k]:
                        continue
                    if viable[k]:
                        if not tx[k]:
                            # DW: always reduce
                            tau[k] /= PND_C_COLL
                        elif method == "pnd_cd":
                            # DT with CD: also reduces (knows it collided)
                            tau[k] /= PND_C_COLL
                        # DT without CD: no update (half-duplex, unaware)
            elif outcome_idle:
                # All DW viable STAs increase
                for k in range(N):
                    if not tx[k] and not succeeded[k] and viable[k]:
                        tau[k] *= PND_C_IDLE
            for k in range(N):
                if not succeeded[k]:
                    tau[k] = float(np.clip(tau[k], 1e-4, 1.0))

        # AND phase advance (open-loop, per-slot counter)
        if method == "and":
            and_phase_slots += 1
            if and_phase_slots >= and_phase_dur and and_phase < 60:
                and_phase += 1
                and_phase_slots = 0
                and_phase_dur = int(math.ceil(2**and_phase * math.e * math.log(2**and_phase)))

        prev_viable_count = cur_viable_count

    efficiency = (successes / oracle_successes) if oracle_successes > 0 else 0.0
    col_rate = (collision_slots / total_slots) if total_slots > 0 else 0.0
    tau_rmse = math.sqrt(tau_sq_err_sum / tau_sq_err_cnt) if tau_sq_err_cnt > 0 else 0.0
    ftr = (false_triggers / total_slots) if total_slots > 0 else 0.0

    weff_util = useful_slots / W_eff if W_eff > 0 else 0.0

    return {
        "successes":           successes,
        "oracle_successes":    oracle_successes,
        "efficiency":          efficiency,
        "useful_slots":        useful_slots,
        "W_eff_utilization":   weff_util,
        "viable_slots_wasted": viable_idle_slots,
        "collision_rate":      col_rate,
        "tau_rmse":            tau_rmse,
        "false_trigger_rate":  ftr,
    }


# ─── Trajectory capture (panel c) ────────────────────────────────────────────

def run_trajectory_visit(
    methods_traj: list[str], W_eff: int, ppdus: np.ndarray, rng_seed: int,
) -> dict[str, dict]:
    results = {}
    for method in methods_traj:
        rng = np.random.default_rng(rng_seed)
        N = len(ppdus)
        succeeded = np.zeros(N, dtype=bool)
        W_rem = W_eff
        remaining = N
        tau = np.full(N, 1.0 / N)
        ema_idle = np.full(N, IDLE_TARGET)
        consec_idle = np.zeros(N, dtype=np.int32)

        # DCF state
        dcf_cw = np.full(N, N, dtype=np.int64)
        dcf_bo = rng.integers(0, N, size=N).astype(np.int64)

        _ema_methods = frozenset(
            ("ema_adaptive", "ema_ad_low", "ema_no_coll")
        )
        alpha_down_map = {
            "ema_adaptive": ALPHA_DOWN,
            "ema_ad_low":   0.10,
            "ema_no_coll":  0.00,
        }

        _solo_sender_tau_t = 0.0

        # AND state
        and_phase_t = 1
        and_phase_slots_t = 0
        and_phase_dur_t = int(math.ceil(2**and_phase_t * math.e * math.log(2**and_phase_t)))

        w_rem_history = []
        tau_history = []

        while True:
            viable = (~succeeded) & (ppdus <= W_rem)
            k_viable = int(viable.sum())
            if k_viable == 0:
                break

            w_rem_history.append(W_rem)

            if method == "oracle":
                tau_val = 1.0 / k_viable
                tx = rng.random(N) < np.where(viable, tau_val, 0.0)
                tau_history.append(tau_val)
            elif method == "self_excl_only":
                r = max(remaining, 1)
                tx = rng.random(N) < np.where(viable, 1.0 / r, 0.0)
                tau_history.append(1.0 / r)
            elif method == "dcf_self_excl":
                tx = (dcf_bo == 0) & viable
                viable_cws = dcf_cw[(~succeeded) & viable]
                eff_tau = 1.0 / float(np.mean(viable_cws)) if len(viable_cws) > 0 else 0.0
                tau_history.append(eff_tau)
            elif method == "and":
                and_p_t = max(1.0 / (2 ** and_phase_t), 1e-4)
                tx = rng.random(N) < np.where(viable, and_p_t, 0.0)
                tau_history.append(and_p_t)
            else:
                # EMA / PND methods: per-STA τ
                tx = rng.random(N) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)
                viable_taus = tau[viable]
                tau_history.append(float(viable_taus.mean()) if len(viable_taus) > 0 else 0.0)

            n_tx = int(tx.sum())
            outcome_idle = (n_tx == 0)
            outcome_coll = (n_tx > 1)

            if n_tx == 1:
                i = int(np.where(tx)[0][0])
                _solo_sender_tau_t = float(tau[i])
                succeeded[i] = True
                W_rem -= int(ppdus[i])
                remaining -= 1
                tau[i] = 0.0
                if method == "dcf_self_excl":
                    dcf_bo[i] = W_eff + 1
            else:
                W_rem -= 1

            # DCF update
            if method == "dcf_self_excl":
                if outcome_coll:
                    for j in np.where(tx)[0]:
                        j = int(j)
                        dcf_cw[j] = min(int(dcf_cw[j]) * 2, DCF_CW_MAX)
                        dcf_bo[j] = int(rng.integers(0, max(int(dcf_cw[j]), 1)))
                elif outcome_idle:
                    mask = (~succeeded) & viable & (dcf_bo > 0)
                    dcf_bo[mask] -= 1

            # EMA update
            elif method in _ema_methods:
                alpha_down_val = alpha_down_map[method]
                for i in range(N):
                    if succeeded[i]:
                        continue
                    beta = float(np.clip(BETA_BASE * N / max(W_rem, 1), 0.05, 0.50))
                    ema_idle[i] = (1 - beta) * ema_idle[i] + beta * float(outcome_idle)
                    gap = ema_idle[i] - IDLE_TARGET
                    if gap > BAND and viable[i]:
                        tau[i] = min(tau[i] * (1 + ALPHA_UP * gap), 1.0)
                    elif outcome_coll and viable[i] and alpha_down_val > 0.0:
                        tau[i] *= (1.0 - alpha_down_val)
                    tau[i] = float(np.clip(tau[i], 1e-4, 1.0))

            # PND update
            elif method in ("pnd", "pnd_cd"):
                if n_tx == 1:   # solo success
                    for k in range(N):
                        if not tx[k] and not succeeded[k] and viable[k]:
                            tau[k] = _solo_sender_tau_t
                elif outcome_coll:
                    for k in range(N):
                        if succeeded[k]:
                            continue
                        if viable[k]:
                            if not tx[k]:
                                tau[k] /= PND_C_COLL
                            elif method == "pnd_cd":
                                tau[k] /= PND_C_COLL
                elif outcome_idle:
                    for k in range(N):
                        if not tx[k] and not succeeded[k] and viable[k]:
                            tau[k] *= PND_C_IDLE
                for k in range(N):
                    if not succeeded[k]:
                        tau[k] = float(np.clip(tau[k], 1e-4, 1.0))

            # AND phase advance
            if method == "and":
                and_phase_slots_t += 1
                if and_phase_slots_t >= and_phase_dur_t and and_phase_t < 60:
                    and_phase_t += 1
                    and_phase_slots_t = 0
                    and_phase_dur_t = int(math.ceil(
                        2**and_phase_t * math.e * math.log(2**and_phase_t)))

        results[method] = {
            "w_rem":             w_rem_history,
            "tau":               tau_history,
            "ppdu_thresholds":   sorted(ppdus.tolist()),
        }

    return results


# ─── Main sweep ───────────────────────────────────────────────────────────────

def run_sweep(
    n_visits: int, n_list: list, weff_list: list, seeds: list,
    methods_filter: list | None = None,
) -> list[dict]:
    active_methods = methods_filter if methods_filter else METHODS
    rows = []
    total = len(PPDU_DIST_NAMES) * len(n_list) * len(weff_list) * len(seeds) * len(active_methods)
    done = 0

    method_idx = {m: i for i, m in enumerate(METHODS)}

    for ppdu_dist in PPDU_DIST_NAMES:
        for N in n_list:
            for W_eff in weff_list:
                for seed in seeds:
                    rng_ppdu = np.random.default_rng(seed * 10001 + 7)

                    oracle_results = []
                    for v in range(n_visits):
                        rng_v = np.random.default_rng(seed * 100003 + v)
                        ppdus = sample_ppdu(ppdu_dist, N, rng_ppdu)
                        os_v = _run_oracle_visit(W_eff, ppdus, rng_v)
                        oracle_results.append((ppdus.copy(), os_v))

                    for method in active_methods:
                        agg = {
                            "successes":           0.0,
                            "oracle_successes":    0.0,
                            "efficiency":          0.0,
                            "useful_slots":        0.0,
                            "W_eff_utilization":   0.0,
                            "viable_slots_wasted": 0.0,
                            "collision_rate":      0.0,
                            "tau_rmse":            0.0,
                            "false_trigger_rate":  0.0,
                        }
                        for v, (ppdus, os_v) in enumerate(oracle_results):
                            rng_v = np.random.default_rng(
                                seed * 200003 + v * 17 + method_idx[method]
                            )
                            m = _run_visit(method, W_eff, ppdus, rng_v, os_v)
                            for k in agg:
                                agg[k] += m[k]

                        row = {
                            "method":    method,
                            "ppdu_dist": ppdu_dist,
                            "N":         N,
                            "W_eff":     W_eff,
                            "seed":      seed,
                            "visit":     n_visits,
                        }
                        for k in agg:
                            row[k] = agg[k] / n_visits
                        rows.append(row)

                        done += 1
                        print(f"  [{done:4d}/{total}] {ppdu_dist:<8} N={N:2d}  W_eff={W_eff:3d}  "
                              f"{method:<18} seed={seed}", flush=True)

    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean_metric(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    return float(np.mean(vals)) if vals else 0.0


def _std_metric(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    return float(np.std(vals)) if vals else 0.0


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path: str) -> list[dict]:
    int_fields   = {"N", "W_eff", "seed", "visit"}
    float_fields = {"successes", "oracle_successes", "efficiency",
                    "useful_slots", "W_eff_utilization",
                    "viable_slots_wasted", "collision_rate", "tau_rmse", "false_trigger_rate"}
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


def merge_rows(base_rows: list[dict], new_rows: list[dict], replace_methods: list[str]) -> list[dict]:
    """Drop base rows for replace_methods; append new rows."""
    kept = [r for r in base_rows if r["method"] not in replace_methods]
    return kept + new_rows


# ─── Plot ─────────────────────────────────────────────────────────────────────

_METHOD_STYLE = {
    "oracle":         dict(color="#2ca02c", ls="--",  lw=1.5, marker="D", ms=5),
    "mfg_no_excl":    dict(color="#888888", ls=":",   lw=1.8, marker="s", ms=5),
    "self_excl_only": dict(color="#9467bd", ls="-.",  lw=1.8, marker="^", ms=5),
    "ema_fixed_low":  dict(color="#1f77b4", ls="-.",  lw=1.8, marker="v", ms=5),
    "ema_fixed_high": dict(color="#ff7f0e", ls=":",   lw=1.8, marker="<", ms=5),
    "ema_adaptive":   dict(color="#d62728", ls="-",   lw=2.2, marker="o", ms=6),
    "ema_ad_low":     dict(color="#fdae61", ls="-",   lw=2.0, marker="p", ms=6),
    "ema_ad_med":     dict(color="#f46d43", ls="-",   lw=1.8, marker="h", ms=5),
    "ema_no_coll":    dict(color="#74c476", ls="-",   lw=2.2, marker="*", ms=7),
    "consec_L2":      dict(color="#8c564b", ls="--",  lw=1.5, marker="x", ms=6, markeredgewidth=1.8),
    "consec_L4":      dict(color="#e377c2", ls="--",  lw=1.5, marker="+", ms=7, markeredgewidth=1.8),
    "dcf_self_excl":  dict(color="#08306b", ls="-",   lw=2.2, marker="8", ms=6),
    "pnd":            dict(color="#17becf", ls="-",   lw=2.0, marker="P", ms=6),
    "pnd_cd":         dict(color="#bcbd22", ls="-",   lw=2.2, marker="X", ms=6),
    "and":            dict(color="#7f2704", ls="--",  lw=1.8, marker="d", ms=5),
}
_METHOD_LABEL = {
    "oracle":          "Oracle (τ=1/viable)",
    "mfg_no_excl":     "MFG no-excl (τ=1/rem)",
    "self_excl_only":  "Self-excl only (τ=1/rem)",
    "ema_fixed_low":   "EMA β=0.05 (fixed-low)",
    "ema_fixed_high":  "EMA β=0.30 (fixed-high)",
    "ema_adaptive":    "EMA adaptive β_t  (α↓=0.50)",
    "ema_ad_low":      "EMA adaptive β_t  (α↓=0.10)",
    "ema_ad_med":      "EMA adaptive β_t  (α↓=0.25)",
    "ema_no_coll":     "EMA adaptive β_t  (α↓=0, no-coll)",
    "consec_L2":       "Consec L=2",
    "consec_L4":       "Consec L=4",
    "dcf_self_excl":   "DCF + self-excl (BEB, CW₀=N)",
    "pnd":             f"PND MIMD (no CD, cc={PND_C_COLL}/ci={PND_C_IDLE})",
    "pnd_cd":          f"PND MIMD + CD (cc={PND_C_COLL}/ci={PND_C_IDLE})",
    "and":             "AND (phase-based, open-loop)",
}


def plot(rows: list, traj_data: dict, fig_dir: str) -> None:
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, rows)
    _panel_b(ax_b, rows)
    _panel_c(ax_c, traj_data)
    _panel_d(ax_d, rows)

    fig.suptitle(
        "Fig. 17  PPDU-Aware Self-Exclusion + Adaptive τ — Heterogeneous PPDU Ablation (v7)\n"
        f"(Bernoulli Aloha/DCF, N∈[10,50], PPDU U[3,12], PND cc={PND_C_COLL}/ci={PND_C_IDLE}, +AND baseline)",
        fontsize=12,
    )
    _save_figure(fig, fig_dir, "fig17_ppdu_aware_tau")
    plt.close(fig)


# Panel (a): efficiency vs N — W_eff=50, uniform, key 10 methods
_PANEL_A_SEL = [
    "oracle", "mfg_no_excl", "self_excl_only",
    "dcf_self_excl",
    "ema_adaptive", "ema_no_coll",
    "consec_L2",
    "pnd", "pnd_cd",
    "and",
]

def _panel_a(ax, rows) -> None:
    W_eff_ref = 50
    ppdu_ref  = "uniform"

    for method in _PANEL_A_SEL:
        means = [_mean_metric(rows, "efficiency",
                              method=method, ppdu_dist=ppdu_ref,
                              N=N, W_eff=W_eff_ref)
                 for N in N_LIST]
        ax.plot(N_LIST, means, label=_METHOD_LABEL[method], **_METHOD_STYLE[method])

    ax.set_xlabel("N (contending STAs)", fontsize=10)
    ax.set_ylabel("Efficiency  (successes / oracle)", fontsize=10)
    ax.set_xticks(N_LIST)
    ax.set_ylim(0.70, 1.15)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=7.5, frameon=True, loc="lower left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) Efficiency vs N  (W_eff={W_eff_ref}, {ppdu_ref})", fontsize=10)


# Panel (b): β design + α_down sweep — N=20, uniform, select EMA methods
_PANEL_B_SEL = [
    "oracle", "self_excl_only",
    "ema_fixed_low", "ema_fixed_high",
    "ema_adaptive", "ema_ad_low", "ema_no_coll",
]

def _panel_b(ax, rows) -> None:
    N_ref    = 20
    ppdu_ref = "uniform"

    for method in _PANEL_B_SEL:
        x_vals = [W / N_ref for W in WEFF_LIST]
        means  = [_mean_metric(rows, "efficiency",
                               method=method, ppdu_dist=ppdu_ref,
                               N=N_ref, W_eff=W)
                  for W in WEFF_LIST]
        ax.plot(x_vals, means, label=_METHOD_LABEL[method], **_METHOD_STYLE[method])

    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", ls=":", lw=1.0, label="W_eff=N")
    ax.set_xlabel("W_eff / N  (log scale)", fontsize=10)
    ax.set_ylabel("Efficiency  (successes / oracle)", fontsize=10)
    ax.set_ylim(0.70, 1.15)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=7.5, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(b) β design + α↓ sweep\n(N={N_ref}, {ppdu_ref})", fontsize=10)


def _panel_c(ax, traj_data: dict) -> None:
    """τ trajectory over one visit — W_rem on x-axis (descending)."""
    if not traj_data:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes)
        return

    style_traj = {
        "oracle":         dict(color="#2ca02c", ls="--", lw=1.5),
        "self_excl_only": dict(color="#9467bd", ls="-.", lw=1.8),
        "ema_adaptive":   dict(color="#d62728", ls="-",  lw=2.0),
        "ema_no_coll":    dict(color="#74c476", ls="-",  lw=2.0),
        "dcf_self_excl":  dict(color="#08306b", ls="-",  lw=2.0),
        "pnd":            dict(color="#17becf", ls="-",  lw=2.0),
        "pnd_cd":         dict(color="#bcbd22", ls="-",  lw=2.0),
        "and":            dict(color="#7f2704", ls="--", lw=1.8),
    }
    labels_traj = {
        "oracle":         "Oracle τ",
        "self_excl_only": "Self-excl only τ",
        "ema_adaptive":   "EMA adaptive (α↓=0.50)",
        "ema_no_coll":    "EMA no-coll (α↓=0)",
        "dcf_self_excl":  "DCF self-excl  (1/CW_mean)",
        "pnd":            f"PND MIMD (no CD, cc={PND_C_COLL}/ci={PND_C_IDLE})",
        "pnd_cd":         f"PND MIMD + CD (cc={PND_C_COLL}/ci={PND_C_IDLE})",
        "and":            "AND (phase-based)",
    }

    ppdu_thresholds = None
    for method, data in traj_data.items():
        w_rem = data["w_rem"]
        tau   = data["tau"]
        if w_rem:
            ax.plot(w_rem, tau, label=labels_traj.get(method, method),
                    **style_traj.get(method, {}))
        if ppdu_thresholds is None:
            ppdu_thresholds = data.get("ppdu_thresholds", [])

    if ppdu_thresholds:
        first = True
        for thresh in sorted(set(ppdu_thresholds)):
            lbl = "STA ppdu_i thresholds" if first else None
            ax.axvline(thresh, color="gray", ls=":", lw=0.8, alpha=0.6, label=lbl)
            first = False

    ax.invert_xaxis()
    ax.set_xlabel("W_rem (remaining slots, left=high)", fontsize=10)
    ax.set_ylabel("τ (TX probability)", fontsize=10)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(c) τ trajectory — single visit\n(N=20, W_eff=50, seed=42, uniform)",
                 fontsize=10)


_PANEL_D_SEL = [
    "oracle", "mfg_no_excl", "self_excl_only",
    "ema_no_coll", "pnd", "pnd_cd", "and",
]

def _panel_d(ax, rows) -> None:
    W_eff_ref = 50
    N_ref     = 20
    dists     = PPDU_DIST_NAMES

    x = np.arange(len(dists))
    width = 0.13
    offsets = np.linspace(
        -(len(_PANEL_D_SEL) - 1) / 2, (len(_PANEL_D_SEL) - 1) / 2, len(_PANEL_D_SEL)
    ) * width

    for method, offset in zip(_PANEL_D_SEL, offsets):
        means = [_mean_metric(rows, "efficiency",
                              method=method, ppdu_dist=d, N=N_ref, W_eff=W_eff_ref)
                 for d in dists]
        stds  = [_std_metric(rows, "efficiency",
                             method=method, ppdu_dist=d, N=N_ref, W_eff=W_eff_ref)
                 for d in dists]
        ax.bar(x + offset, means, width,
               label=_METHOD_LABEL[method],
               color=_METHOD_STYLE[method]["color"],
               alpha=0.85, edgecolor="white")
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor="black", capsize=2, elinewidth=0.8)

    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dists)
    ax.set_ylabel("Efficiency  (successes / oracle)", fontsize=10)
    ax.set_ylim(0.70, 1.20)
    ax.legend(fontsize=7.5, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7, axis="y")
    ax.set_title(f"(d) PPDU distribution effect\n(W_eff={W_eff_ref}, N={N_ref})", fontsize=10)


def _save_figure(fig, fig_dir: str, name: str) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ms_fig    = os.path.join(repo_root, "manuscript", "figure")
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


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows: list) -> None:
    print("\n=== Hypothesis Check ===")

    # H1: self_excl_only > mfg_no_excl (after v2 bug fix, reversed)
    print("\nH1: self_excl_only > mfg_no_excl (N>20, uniform, W_eff=50)")
    for N in [30, 50]:
        se = _mean_metric(rows, "efficiency", method="self_excl_only",
                          ppdu_dist="uniform", N=N, W_eff=50)
        mn = _mean_metric(rows, "efficiency", method="mfg_no_excl",
                          ppdu_dist="uniform", N=N, W_eff=50)
        status = "✅ PASS" if se > mn else "❌ FAIL"
        print(f"  N={N}: self_excl_only={se:.4f}  mfg_no_excl={mn:.4f}  {status}")

    # H2: consec_L2 best adaptive
    print("\nH2: consec_L2 > self_excl_only (N=20, uniform, W_eff=50)")
    cl2 = _mean_metric(rows, "efficiency", method="consec_L2",
                       ppdu_dist="uniform", N=20, W_eff=50)
    se  = _mean_metric(rows, "efficiency", method="self_excl_only",
                       ppdu_dist="uniform", N=20, W_eff=50)
    print(f"  consec_L2={cl2:.4f}  self_excl_only={se:.4f}  "
          + ("✅ PASS" if cl2 > se else "❌ FAIL"))

    # H5: homo — all methods ≈ oracle
    print("\nH5: homo — all methods ≈ oracle (N=20, W_eff=50)")
    o_homo = _mean_metric(rows, "efficiency", method="oracle",
                          ppdu_dist="homo", N=20, W_eff=50)
    for m in ["mfg_no_excl", "self_excl_only", "ema_adaptive", "dcf_self_excl", "ema_no_coll"]:
        v = _mean_metric(rows, "efficiency", method=m, ppdu_dist="homo", N=20, W_eff=50)
        status = "≈" if abs(v - o_homo) < 0.05 else "≠"
        print(f"  {m:<20}: {v:.4f} {status} oracle({o_homo:.4f})")

    # H6: bimodal gap > uniform gap
    print("\nH6: bimodal gap > uniform gap (oracle vs mfg_no_excl, N=20, W_eff=50)")
    for dist in ["homo", "uniform", "bimodal"]:
        o  = _mean_metric(rows, "efficiency", method="oracle",
                          ppdu_dist=dist, N=20, W_eff=50)
        mn = _mean_metric(rows, "efficiency", method="mfg_no_excl",
                          ppdu_dist=dist, N=20, W_eff=50)
        print(f"  {dist:<8}: oracle={o:.4f}  mfg_no_excl={mn:.4f}  gap={o - mn:.4f}")

    # H7: dcf_self_excl > self_excl_only (BEB τ-adaptation helps)
    print("\nH7: dcf_self_excl > self_excl_only (N=20, uniform, W_eff=50)")
    dcf = _mean_metric(rows, "efficiency", method="dcf_self_excl",
                       ppdu_dist="uniform", N=20, W_eff=50)
    se  = _mean_metric(rows, "efficiency", method="self_excl_only",
                       ppdu_dist="uniform", N=20, W_eff=50)
    print(f"  dcf_self_excl={dcf:.4f}  self_excl_only={se:.4f}  "
          + ("✅ PASS" if dcf > se else "❌ FAIL"))

    # H8: ema_no_coll > ema_adaptive (no collision penalty reduces τ oscillation)
    print("\nH8: ema_no_coll > ema_adaptive (N=20, uniform, W_eff=50)")
    enc = _mean_metric(rows, "efficiency", method="ema_no_coll",
                       ppdu_dist="uniform", N=20, W_eff=50)
    ea  = _mean_metric(rows, "efficiency", method="ema_adaptive",
                       ppdu_dist="uniform", N=20, W_eff=50)
    print(f"  ema_no_coll={enc:.4f}  ema_adaptive={ea:.4f}  "
          + ("✅ PASS" if enc > ea else "❌ FAIL"))

    # H9: ema_ad_low/med between ema_no_coll and ema_adaptive
    print("\nH9: ema_no_coll ≥ ema_ad_low ≥ ema_ad_med ≥ ema_adaptive (N=20, uniform, W_eff=50)")
    eal = _mean_metric(rows, "efficiency", method="ema_ad_low",
                       ppdu_dist="uniform", N=20, W_eff=50)
    eam = _mean_metric(rows, "efficiency", method="ema_ad_med",
                       ppdu_dist="uniform", N=20, W_eff=50)
    print(f"  ema_no_coll={enc:.4f}  ema_ad_low={eal:.4f}  "
          f"ema_ad_med={eam:.4f}  ema_adaptive={ea:.4f}")
    ordered = (enc >= eal - 0.001) and (eal >= eam - 0.001) and (eam >= ea - 0.001)
    print(f"  → {'✅ PASS (monotone)' if ordered else '❌ FAIL (not monotone)'}")

    # H10: pnd_cd > pnd (CD helps in finite window too)
    print("\nH10: pnd_cd > pnd (N=20, uniform, W_eff=50)")
    pc  = _mean_metric(rows, "efficiency", method="pnd_cd",
                       ppdu_dist="uniform", N=20, W_eff=50)
    pn  = _mean_metric(rows, "efficiency", method="pnd",
                       ppdu_dist="uniform", N=20, W_eff=50)
    print(f"  pnd_cd={pc:.4f}  pnd={pn:.4f}  "
          + ("✅ PASS" if pc > pn else "❌ FAIL"))

    # H11: pnd_cd ≈ ema_no_coll (both eliminate collision backoff in receiving set)
    print("\nH11: pnd_cd vs ema_no_coll (N=20, uniform, W_eff=50)")
    enc2 = _mean_metric(rows, "efficiency", method="ema_no_coll",
                        ppdu_dist="uniform", N=20, W_eff=50)
    diff = abs(pc - enc2)
    print(f"  pnd_cd={pc:.4f}  ema_no_coll={enc2:.4f}  diff={diff:.4f}  "
          + ("≈ similar" if diff < 0.02 else "≠ different"))

    # H12: AND (open-loop) < self_excl_only (adaptive methods all beat open-loop)
    print("\nH12: AND < self_excl_only (N=20, uniform, W_eff=50)")
    and_v = _mean_metric(rows, "efficiency", method="and",
                         ppdu_dist="uniform", N=20, W_eff=50)
    se2   = _mean_metric(rows, "efficiency", method="self_excl_only",
                         ppdu_dist="uniform", N=20, W_eff=50)
    print(f"  and={and_v:.4f}  self_excl_only={se2:.4f}  "
          + ("✅ PASS" if and_v < se2 else "❌ FAIL"))

    # Summary table — uniform, W_eff=50, N=20
    print("\n--- Efficiency summary (uniform, W_eff=50, N=20) ---")
    for m in METHODS:
        v = _mean_metric(rows, "efficiency", method=m, ppdu_dist="uniform", N=20, W_eff=50)
        print(f"  {m:<22} {v:.4f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 17 v7 — PPDU-aware self-exclusion + adaptive τ + DCF + PND + AND ablation")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick mode: {FAST_VISITS} visits, small N/W grid")
    parser.add_argument("--out-dir", default="results/step9/fig17_v7")
    parser.add_argument("--methods", nargs="+", default=None, metavar="METHOD",
                        help="Only simulate these methods (e.g. pnd pnd_cd); "
                             "combine with --base-csv to skip re-running others")
    parser.add_argument("--base-csv", default=None, metavar="PATH",
                        help="Existing data CSV to merge with; rows for --methods are replaced")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.fast:
        n_visits  = FAST_VISITS
        n_list    = FAST_N_LIST
        weff_list = FAST_WEFF_LIST
        seeds     = FAST_SEEDS
    else:
        n_visits  = FULL_VISITS
        n_list    = N_LIST
        weff_list = WEFF_LIST
        seeds     = SEEDS

    methods_filter = args.methods  # None = all methods
    active_methods = methods_filter if methods_filter else METHODS
    total = len(PPDU_DIST_NAMES) * len(n_list) * len(weff_list) * len(seeds) * len(active_methods)

    print(f"=== Figure 17 v6 [{'FAST' if args.fast else 'FULL'}]  {n_visits} visits each ===")
    print(f"    N         ∈ {n_list}")
    print(f"    W_eff     ∈ {weff_list}")
    print(f"    PPDU_DIST   {PPDU_DIST_NAMES}")
    print(f"    seeds       {seeds}")
    print(f"    methods     {active_methods}")
    print(f"    total       {total} configurations")
    if args.base_csv:
        print(f"    base CSV    {args.base_csv}")

    new_rows = run_sweep(n_visits, n_list, weff_list, seeds, methods_filter=methods_filter)

    if args.base_csv and methods_filter:
        print(f"  Merging with {args.base_csv} (replacing: {methods_filter}) ...")
        base_rows = load_csv(args.base_csv)
        rows = merge_rows(base_rows, new_rows, methods_filter)
    else:
        rows = new_rows

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Computing τ trajectory for panel (c) ...")
    rng_ppdu = np.random.default_rng(42 * 10001 + 7)
    ppdus_traj = sample_ppdu("uniform", 20, rng_ppdu)
    traj_data = run_trajectory_visit(
        ["oracle", "self_excl_only", "ema_adaptive",
         "ema_no_coll", "dcf_self_excl", "pnd", "pnd_cd", "and"],
        W_eff=50, ppdus=ppdus_traj, rng_seed=42,
    )

    check_hypotheses(rows)

    print("\nPlotting ...")
    plot(rows, traj_data, out_dir)

    print("\nFigure 17 v7 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : manuscript/figure/fig17_ppdu_aware_tau.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
