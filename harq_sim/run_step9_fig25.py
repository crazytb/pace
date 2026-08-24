"""
Figure 25: Collision-Cost Sensitivity and Mandatory RTS/CTS
           — Mixed Native/Visitor NPCA Channel

RQ25: Fig 23 showed the 1-slot collision assumption AMPLIFIES PACE's advantage
in a visitor-only channel. Fig 24's breakdown revealed why: PACE's dominant
waste is collision airtime (53% of W_eff under no-CD), while conventional
NPCA's is frozen-backoff idle (15%) plus collisions (42%). This figure makes
the collision price an explicit axis on the MIXED channel and answers two
questions:

  (A) Does PACE's performance degrade as a collision costs more slots — and
      does conventional NPCA ever overtake it?
  (B) What happens if RTS/CTS becomes MANDATORY — modeled honestly: every
      successful exchange pays a handshake overhead OH, and a collision
      costs only the RTS-collision time OH_C (no data frames collide).

Setup (from fig24): N_visitor=10 PACE-warm visitors (τ carried across visits,
τ₀=1/N_v), N_native=10 DCF natives (ppdu=450µs), W_eff=3.78ms, visitor PPDU
U[225,900]µs — all times in IEEE 802.11 OFDM-PHY standard units (σ=9µs slots),
V=50 back-to-back NPCA transitions, steady state = visits ≥ V/2.
Methods: dcf_conv (conventional NPCA, CSMA/CA CW_min=16), pace (warm),
oracle (τ=1/|viable|, all STAs).

All overheads are derived from IEEE 802.11 standard parameters — no ad-hoc
ratios: aSlotTime σ=9µs, SIFS=16µs, DIFS=34µs, aPHY-RX-START-Delay=25µs,
RTS/CTS at 24 Mbps control rate = 28µs each (20µs preamble + 2 OFDM symbols),
at 6 Mbps = 52µs each.
  success OH  = RTS+SIFS+CTS+SIFS = 88µs (24M) / 136µs (6M) → 10σ / 15σ
  RTS collision = RTS + CTS_Timeout(SIFS+σ+RX-START=50µs) = 78µs / 102µs → 9σ / 11σ

Sweep A — constant collision cost C ∈ {9,16,31,63,94,125}σ (81µs–1.13ms), plus
"nocd" (collision = max Lᵢ of colliders, basic access). Success = ppdu.

Sweep B — mandatory RTS/CTS at 24 Mbps and 6 Mbps control rates vs basic
access. Success = ppdu + OH, collision = RTS-collision cost. Min Duration
Threshold includes OH.

Hypotheses:
  H1 PACE visitor airtime decreases monotonically in C (collision-heavy protocol)
  H2 PACE/dcf visitor ratio shrinks with C but stays > 1 (both pay per collision)
  H3 CHANNEL-efficiency ranking flips at C ≈ E[L]=62.5σ=562µs (dearer-than-a-
     frame collisions turn PACE's aggression into a net channel loss)
  H4 mandatory RTS/CTS helps PACE far more than conventional NPCA — it removes
     PACE's dominant waste (collisions) but cannot fix dcf's frozen-backoff idle
     → channel-efficiency ranking flips in PACE's favor

Run:
  .venv/bin/python harq_sim/run_step9_fig25.py
  .venv/bin/python harq_sim/run_step9_fig25.py --fast
  .venv/bin/python harq_sim/run_step9_fig25.py --base-csv results/step9/fig25/data.csv
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
import run_step9_fig24 as _f24

# ─── Parameters ───────────────────────────────────────────────────────────────

METHODS_25 = ["dcf_conv", "pace", "oracle"]

N_VISITOR = _f24.N_VISITOR          # 10
N_NATIVE  = 10

# ── IEEE 802.11 standard timing (OFDM PHY, 5 GHz) — sim slot ≡ aSlotTime σ ────
SLOT_US   = 9                        # aSlotTime
SIFS_US   = 16                       # aSIFSTime
DIFS_US   = SIFS_US + 2 * SLOT_US    # 34 µs
PHY_RX_START_US = 25                 # aPHY-RX-START-Delay (OFDM)
RTS_US_24M = 28                      # RTS @ 24 Mbps ctrl rate: 20µs preamble + 2 sym
CTS_US_24M = 28
ACK_US_24M = 28
RTS_US_6M  = 52                      # RTS @ 6 Mbps ctrl rate: 20µs preamble + 8 sym
CTS_US_6M  = 52
ACK_US_6M  = 44
# EIFS = SIFS + ACKTxTime + DIFS (recovery defer after an undecodable frame)
EIFS_US_24M = SIFS_US + ACK_US_24M + DIFS_US           # 78 µs
EIFS_US_6M  = SIFS_US + ACK_US_6M + DIFS_US            # 94 µs


def _us2slot(us: float) -> int:
    return max(int(round(us / SLOT_US)), 1)


# RTS/CTS handshake overhead on SUCCESS = RTS+SIFS+CTS+SIFS
OH_SUCC_24M = _us2slot(RTS_US_24M + SIFS_US + CTS_US_24M + SIFS_US)   # 88µs → 10σ
OH_SUCC_6M  = _us2slot(RTS_US_6M + SIFS_US + CTS_US_6M + SIFS_US)    # 136µs → 15σ
# RTS collision cost = RTS + EIFS (channel busy for the RTS, then EIFS recovery)
COLL_RTS_24M = _us2slot(RTS_US_24M + EIFS_US_24M)                     # 106µs → 12σ
COLL_RTS_6M  = _us2slot(RTS_US_6M + EIFS_US_6M)                       # 146µs → 16σ

# Frame/window scale in σ units (realistic durations):
#   visitor PPDU U[25,100]σ = 225–900 µs  (≈1500B @65Mbps … small A-MPDU / low MCS)
#   native PPDU 50σ = 450 µs;  W_eff 420σ = 3.78 ms (TXOP-scale NPCA window)
PPDU_V_LO, PPDU_V_HI = 25, 100
PPDU_NATIVE_SLOTS    = 50
W_REF     = 420
PPDU_MEAN = (PPDU_V_LO + PPDU_V_HI) / 2      # E[L] = 62.5σ = 562 µs

COST_LIST = [9, 16, 31, 63, 94, 125]         # sweep A collision cost C (σ slots)
# sweep B configs: (label, coll_cost, succ_oh) — all standard-derived
RTS_CONFIGS = [
    ("basic",   "nocd", 0),                       # basic access, coll = max Lᵢ
    ("rts_24m", COLL_RTS_24M, OH_SUCC_24M),       # mandatory RTS/CTS @ 24 Mbps
    ("rts_6m",  COLL_RTS_6M,  OH_SUCC_6M),        # mandatory RTS/CTS @ 6 Mbps
]

SEEDS_25    = [42, 123, 456, 789, 1234]
FULL_REPS   = 20
FULL_VISITS = 50

FAST_SEEDS  = [42]
FAST_REPS   = 5
FAST_VISITS = 30
FAST_COSTS  = [9, 31, 63, 125]

FIELDS_25 = [
    "sweep", "config", "method", "seed",
    "succ_v", "succ_n", "coll_air", "idle_air", "oh_air", "useful",
]

_STYLE_25 = {
    "dcf_conv": dict(color="#525252", ls="-.", lw=1.9, marker="x", ms=6),
    "pace":     dict(color="#ff7f0e", ls="-",  lw=2.2, marker="^", ms=6),
    "oracle":   dict(color="#2ca02c", ls="--", lw=1.8, marker="D", ms=5),
}
_LABEL_25 = {
    "dcf_conv": "conventional NPCA  (CSMA/CA, CW_min=16)",
    "pace":     "PACE warm  (τ carried across visits)",
    "oracle":   "Oracle (τ=1/|V(t)|, all STAs)",
}

# stacked-bar component colors (panel c)
_COMP = [
    ("succ_v",   "#ff7f0e", "visitor data"),
    ("succ_n",   "#6baed6", "native data"),
    ("oh_air",   "#969696", "RTS/CTS overhead"),
    ("coll_air", "#d62728", "collision"),
    ("idle_air", "#ffffff", "idle"),
]


def _sample_ppdus25(rng: np.random.Generator) -> np.ndarray:
    """Visitor PPDU U[25,100]σ (225–900 µs), native PPDU 50σ (450 µs)."""
    ppdus_v = rng.integers(PPDU_V_LO, PPDU_V_HI + 1, size=N_VISITOR)
    return np.concatenate([ppdus_v,
                           np.full(N_NATIVE, PPDU_NATIVE_SLOTS)]).astype(np.int32)


# ─── Single visit (generalized collision cost + success overhead) ─────────────

def _run_visit25(
    ppdus: np.ndarray, rng: np.random.Generator, mode: str,
    tau_init: np.ndarray | None, coll_cost, succ_oh: int,
    native_init: tuple | None = None,
    stats: dict | None = None,
) -> tuple[np.ndarray, int, int, int, np.ndarray | None]:
    """One mixed visit under SATURATED traffic: every STA always holds a
    pending PPDU. A visitor that completes an exchange draws a fresh frame
    (U{PPDU_V_LO..PPDU_V_HI}) and keeps contending; PACE winners keep their
    τ, DCF winners reset CW to CW_min per the standard. Natives likewise.
    coll_cost: int slots or "nocd" (max Lᵢ of colliders).
    succ_oh: handshake slots added to every successful exchange.
    stats: optional counter dict; accumulates per-contention-epoch tallies for
    the drift analysis (epochs, nat_tx, vis_viable, outcome counts, Στ). Purely
    passive — consumes no rng, so results are bit-identical with or without it.
    Returns (per-STA useful airtime, coll_air, idle_slots, oh_air, carry τ)."""
    N_total = N_VISITOR + N_NATIVE
    W_rem = W_REF
    airtime = np.zeros(N_total)
    coll_air = 0
    idle = 0
    oh_air = 0

    tau = np.clip(tau_init.copy(), 1e-4, 1.0) if tau_init is not None else None
    carry = tau.copy() if tau is not None else None
    _solo = 0.0
    cw_v = np.full(N_VISITOR, _f24.DCF_CW_MIN_STD, dtype=np.int64)
    bo_v = rng.integers(0, _f24.DCF_CW_MIN_STD, size=N_VISITOR).astype(np.int64)
    if native_init is None:
        cw_n = np.full(N_NATIVE, _f24.DCF_CW_MIN_STD, dtype=np.int64)
        bo_n = rng.integers(0, _f24.DCF_CW_MIN_STD, size=N_NATIVE).astype(np.int64)
    else:
        # steady-state native contention state (e.g., from a burn-in phase)
        cw_n = native_init[0].copy()
        bo_n = native_init[1].copy()

    while W_rem > 0:
        # Visitor eligibility:
        #  - pace / oracle (FS): PPDU-aware self-exclusion incl. handshake
        #    (Phase b of Algorithm 1 / definition of V(t)).
        #  - dcf_conv: standard-faithful — 802.11bn D1.2 has NO per-frame fit
        #    check (Min Duration Threshold gates only the switch, 37.18.3);
        #    the STA contends until NPCA_TIMER expiry (37.18.4/5).
        if mode in ("dcf_conv", "pace_noexcl"):
            # pace_noexcl: ablation — PACE MIMD without Phase (b); an
            # unfittable frame keeps contending and truncates like dcf_conv.
            vv = np.ones(N_VISITOR, dtype=bool)
        else:
            # dcf_excl: best-case compliant CSMA/CA — the standard leaves the
            # unfittable-frame case unspecified, so a deferring implementation
            # (per-frame fit check) is equally standard-conformant.
            vv = ppdus[:N_VISITOR] + succ_oh <= W_rem
        # Natives: standard DCF on their own primary channel — they neither
        # know nor care about the visitors' window; no fit check.
        vn = np.ones(N_NATIVE, dtype=bool)
        k = int(vv.sum() + vn.sum())
        if k == 0:
            break

        if mode == "oracle":
            tx_v = rng.random(N_VISITOR) < np.where(vv, 1.0 / k, 0.0)
        elif mode in ("dcf_conv", "dcf_excl"):
            tx_v = (bo_v == 0) & vv
        else:  # pace
            tx_v = rng.random(N_VISITOR) < np.where(vv, tau.clip(1e-4, 1.0), 0.0)
        tx_n = (bo_n == 0) & vn
        tx = np.concatenate([tx_v, tx_n])
        n_tx = int(tx.sum())
        solo, coll, idle_o = n_tx == 1, n_tx > 1, n_tx == 0

        if stats is not None:
            # One loop pass = one contention epoch = the unit the drift
            # equation is written in (a slot's *cost* varies, its decision
            # does not).
            stats["epochs"] = stats.get("epochs", 0) + 1
            stats["nat_tx"] = stats.get("nat_tx", 0) + int(tx_n.sum())
            stats["nat_slots"] = stats.get("nat_slots", 0) + int(vn.sum())
            stats["vis_viable"] = stats.get("vis_viable", 0) + int(vv.sum())
            key = "idle" if idle_o else "coll" if coll else \
                ("solo_vis" if int(np.where(tx)[0][0]) < N_VISITOR
                 else "solo_nat")
            stats[key] = stats.get(key, 0) + 1
            if tau is not None and vv.any():
                stats["tau_sum"] = stats.get("tau_sum", 0.0) \
                    + float(tau[vv].sum())
                stats["tau_cnt"] = stats.get("tau_cnt", 0) + int(vv.sum())
                # Within-epoch spread of τ across viable visitors. The analysis
                # assumes a homogeneous population (solo-copy re-synchronises
                # them); this measures how far that holds instead of taking it
                # on faith.
                if int(vv.sum()) > 1:
                    _m = float(tau[vv].mean())
                    if _m > 0.0:
                        stats["tau_cv_sum"] = stats.get("tau_cv_sum", 0.0) \
                            + float(tau[vv].std()) / _m
                        stats["tau_cv_cnt"] = stats.get("tau_cv_cnt", 0) + 1
            # Opt-in per-epoch trace: pass stats={"trace": []} to collect the
            # within-visit trajectory as (W_rem, n_viable, k, rate). W_rem is
            # the state the analysis indexes on: viability and the FS target
            # are functions of the remaining window, not of the epoch number.
            # For DCF the rate proxy is BEB's 2/(CW+1).
            tr = stats.get("trace")
            if tr is not None:
                nvv = int(vv.sum())
                if nvv == 0:
                    rate = 0.0
                elif tau is not None:
                    rate = float(tau[vv].mean())
                elif mode.startswith("dcf"):
                    rate = float(np.mean(2.0 / (cw_v[vv] + 1.0)))
                else:
                    rate = 1.0 / k
                tr.append((int(W_rem), nvv, k, rate))

        if solo:
            i = int(np.where(tx)[0][0])
            need = int(ppdus[i]) + succ_oh
            if i < N_VISITOR:
                if need <= W_rem:
                    # completed visitor exchange — saturated: draw the next
                    # frame and keep contending (PACE keeps its τ; DCF
                    # resets CW to CW_min per the standard)
                    if mode.startswith("pace"):
                        _solo = float(tau[i])
                        carry[i] = float(tau[i])
                    W_rem -= need
                    airtime[i] += int(ppdus[i])
                    oh_air += succ_oh
                    ppdus[i] = int(rng.integers(PPDU_V_LO, PPDU_V_HI + 1))
                    if mode.startswith("dcf"):
                        cw_v[i] = _f24.DCF_CW_MIN_STD
                        bo_v[i] = int(rng.integers(0, _f24.DCF_CW_MIN_STD))
                else:
                    # dcf_conv / pace_noexcl: frame does not fit — transmission
                    # starts anyway and is cut off at NPCA_TIMER expiry.
                    if mode.startswith("pace"):
                        _solo = float(tau[i])
                    coll_air += W_rem
                    W_rem = 0
            else:
                # native frame may straddle the window end: it completes on
                # the channel, but only the in-window portion is accounted.
                occupy = min(need, W_rem)
                oh_part = min(succ_oh, occupy)
                airtime[i] += occupy - oh_part
                oh_air += oh_part
                cw_n[i - N_VISITOR] = _f24.DCF_CW_MIN_STD
                bo_n[i - N_VISITOR] = int(rng.integers(0, _f24.DCF_CW_MIN_STD))
                W_rem -= occupy
        elif coll:
            c = int(ppdus[np.where(tx)[0]].max()) if coll_cost == "nocd" \
                else int(coll_cost)
            c = min(c, W_rem)
            W_rem -= c
            coll_air += c
        else:
            W_rem -= 1
            idle += 1

        # visitor state update
        if mode.startswith("dcf"):
            if coll:
                for j in np.where(tx_v)[0]:
                    j = int(j)
                    cw_v[j] = min(int(cw_v[j]) * 2, _f17.DCF_CW_MAX)
                    bo_v[j] = int(rng.integers(0, max(int(cw_v[j]), 1)))
            elif idle_o:
                m = vv & (bo_v > 0)
                bo_v[m] -= 1
        elif mode.startswith("pace"):
            if solo:
                w_i = int(np.where(tx)[0][0])
                if w_i < N_VISITOR:
                    for kk in range(N_VISITOR):
                        if not tx_v[kk] and vv[kk]:
                            tau[kk] = _solo
            elif coll:
                for kk in range(N_VISITOR):
                    if vv[kk] and not tx_v[kk]:
                        tau[kk] /= _f17.PND_C_COLL
            elif idle_o:
                for kk in range(N_VISITOR):
                    if not tx_v[kk] and vv[kk]:
                        tau[kk] *= _f17.PND_C_IDLE
            for kk in range(N_VISITOR):
                tau[kk] = float(np.clip(tau[kk], 1e-4, 1.0))

        # native DCF update
        if coll:
            for j in np.where(tx_n)[0]:
                j = int(j)
                cw_n[j] = min(int(cw_n[j]) * 2, _f17.DCF_CW_MAX)
                bo_n[j] = int(rng.integers(0, max(int(cw_n[j]), 1)))
        elif idle_o:
            m = vn & (bo_n > 0)
            bo_n[m] -= 1

    if mode.startswith("pace"):
        for kk in range(N_VISITOR):
            carry[kk] = float(np.clip(tau[kk], 1e-4, 1.0))
    return airtime, coll_air, idle, oh_air, carry


# ─── One config: sequences with warm carry, steady-state airtime fractions ────

def run_config(
    method: str, coll_cost, succ_oh: int, seed: int, reps: int, visits: int,
) -> dict:
    m_idx = METHODS_25.index(method)
    sv = sn = ca = idl = oh = 0.0
    for r in range(reps):
        rng_p = np.random.default_rng(seed * 10001 + r * 71 + 7)
        rng = np.random.default_rng(seed * 200003 + r * 3163 + m_idx * 29)
        tau_c = np.full(N_VISITOR, 1.0 / N_VISITOR) if method == "pace" else None
        for v in range(visits):
            ppdus = _sample_ppdus25(rng_p)
            air, c, i, o, carry = _run_visit25(ppdus, rng, method, tau_c,
                                               coll_cost, succ_oh)
            if method == "pace":
                tau_c = carry
            if v >= visits // 2:
                sv += air[:N_VISITOR].sum()
                sn += air[N_VISITOR:].sum()
                ca += c
                idl += i
                oh += o
    norm = reps * (visits - visits // 2) * W_REF
    return {
        "succ_v":   sv / norm,
        "succ_n":   sn / norm,
        "coll_air": ca / norm,
        "idle_air": idl / norm,
        "oh_air":   oh / norm,
        "useful":   (sv + sn) / norm,
    }


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(cost_list: list, seeds: list, reps: int, visits: int) -> list[dict]:
    rows = []
    # Sweep A configs + sweep B configs ("basic" doubles as A's nocd point)
    work: list[tuple[str, str, object, int]] = []
    for C in cost_list:
        work.append(("A", str(C), C, 0))
    for label, cc, oh in RTS_CONFIGS:
        work.append(("B", label, cc, oh))

    total = len(work) * len(seeds) * len(METHODS_25)
    done = 0
    for sweep, config, cc, oh in work:
        for seed in seeds:
            for method in METHODS_25:
                res = run_config(method, cc, oh, seed, reps, visits)
                rows.append({"sweep": sweep, "config": config,
                             "method": method, "seed": seed, **res})
                done += 1
                print(f"  [{done:4d}/{total}] {sweep}:{config:<8} "
                      f"{method:<9} seed={seed}", flush=True)
    return rows


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean25(rows, metric, **kw) -> float:
    vals = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    finite = [v for v in vals if v == v and not math.isinf(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ─── Panels ───────────────────────────────────────────────────────────────────

def _panel_a(ax, rows, cost_list: list) -> None:
    for m in METHODS_25:
        ys = [_mean25(rows, "succ_v", sweep="A", config=str(C), method=m)
              for C in cost_list]
        ax.plot(cost_list, ys, label=_LABEL_25[m], **_STYLE_25[m])
        y_nocd = _mean25(rows, "succ_v", sweep="B", config="basic", method=m)
        ax.plot([80], [y_nocd], marker="*", ms=13, color=_STYLE_25[m]["color"],
                ls="none",
                label="basic access (coll = max Lᵢ)" if m == "dcf_conv" else None)
    ax.axvline(PPDU_MEAN, color="gray", ls=":", lw=1.0)
    ax.set_xlabel(f"collision cost C  (σ={SLOT_US}µs slots;  "
                  f"C=63 ↔ {63 * SLOT_US}µs)", fontsize=10)
    ax.set_ylabel("Visitor useful airtime  (fraction of W_eff)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) Visitor airtime vs collision cost\n"
                 f"(N_v={N_VISITOR}, N_nat={N_NATIVE}, "
                 f"W_eff={W_REF}σ={W_REF * SLOT_US / 1000:.2f}ms)", fontsize=10)


def _panel_b(ax, rows, cost_list: list) -> None:
    for m in METHODS_25:
        ys = [_mean25(rows, "useful", sweep="A", config=str(C), method=m)
              for C in cost_list]
        ax.plot(cost_list, ys, label=_LABEL_25[m], **_STYLE_25[m])
        y_nocd = _mean25(rows, "useful", sweep="B", config="basic", method=m)
        ax.plot([80], [y_nocd], marker="*", ms=13, color=_STYLE_25[m]["color"],
                ls="none")
    ax.axvline(PPDU_MEAN, color="gray", ls=":", lw=1.2,
               label=f"C = E[L] = {PPDU_MEAN:.0f}σ = "
                     f"{PPDU_MEAN * SLOT_US / 1000:.2f}ms")
    ax.set_xlabel(f"collision cost C  (σ={SLOT_US}µs slots)", fontsize=10)
    ax.set_ylabel("Channel efficiency  (total useful airtime)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title("(b) Channel efficiency vs collision cost —\n"
                 "conventional NPCA holds a small channel-side lead under basic access",
                 fontsize=10)


def _panel_c(ax, rows) -> None:
    labels = [c[0] for c in RTS_CONFIGS]
    xlabels = {
        "basic":   "basic access\n(coll = max Lᵢ)",
        "rts_24m": f"RTS/CTS @24Mbps\n(OH={OH_SUCC_24M}σ="
                   f"{OH_SUCC_24M * SLOT_US}µs)",
        "rts_6m":  f"RTS/CTS @6Mbps\n(OH={OH_SUCC_6M}σ="
                   f"{OH_SUCC_6M * SLOT_US}µs)",
    }
    n_grp = len(labels)
    width = 0.26
    xs = np.arange(n_grp)
    offs = {m: (i - 1) * width for i, m in enumerate(METHODS_25)}
    for m in METHODS_25:
        bottoms = np.zeros(n_grp)
        for key, color, comp_label in _COMP:
            vals = np.array([_mean25(rows, key, sweep="B", config=c, method=m)
                             for c in labels])
            ax.bar(xs + offs[m], vals, width, bottom=bottoms, color=color,
                   edgecolor="#333333", linewidth=0.4,
                   label=comp_label if m == "dcf_conv" else None)
            bottoms += vals
    for xi, c in zip(xs, labels):
        for m in METHODS_25:
            ax.text(xi + offs[m], -0.045, {"dcf_conv": "dcf", "pace": "pace",
                                           "oracle": "orc"}[m],
                    ha="center", va="top", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([xlabels[c] for c in labels], fontsize=9)
    ax.tick_params(axis="x", pad=16)
    ax.set_ylabel("Airtime composition  (fraction of W_eff)", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, frameon=True, loc="upper left", ncol=2)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7, axis="y")
    ax.set_title("(c) Where the window goes — basic access vs mandatory RTS/CTS",
                 fontsize=10)


def _panel_d(ax, rows) -> None:
    base_v = {m: _mean25(rows, "succ_v", sweep="B", config="basic", method=m)
              for m in METHODS_25}
    base_u = {m: _mean25(rows, "useful", sweep="B", config="basic", method=m)
              for m in METHODS_25}
    groups = ["rts_24m", "rts_6m"]
    glabels = ["RTS/CTS @24Mbps", "RTS/CTS @6Mbps"]
    metrics = [("succ_v", base_v, "visitor airtime"),
               ("useful", base_u, "channel efficiency")]
    xs = np.arange(len(groups) * len(metrics))
    width = 0.34
    plot_methods = ["dcf_conv", "pace"]
    for k, m in enumerate(plot_methods):
        vals, ticklabels = [], []
        for gi, g in enumerate(groups):
            for metric, base, mlabel in metrics:
                v = _mean25(rows, metric, sweep="B", config=g, method=m)
                vals.append((v / base[m] - 1) * 100 if base[m] > 0 else np.nan)
                ticklabels.append(f"{glabels[gi]}\n{mlabel}")
        ax.bar(xs + (k - 0.5) * width, vals, width,
               color=_STYLE_25[m]["color"], alpha=0.9, edgecolor="white",
               label=_LABEL_25[m])
    ax.set_xticks(xs)
    ax.set_xticklabels(ticklabels, fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Gain over basic access  (%)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=True, loc="upper right")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7, axis="y")
    ax.set_title("(d) Who profits from mandatory RTS/CTS —\n"
                 "it removes PACE's waste, not conventional NPCA's frozen backoff",
                 fontsize=10)


# ─── Hypothesis check ─────────────────────────────────────────────────────────

def check_hypotheses(rows, cost_list: list) -> None:
    print("\n=== Hypothesis Check ===")

    print("\nH1/H2: visitor airtime vs collision cost C")
    print(f"  {'C':>5} {'dcf':>7} {'pace':>7} {'ratio':>7}   "
          f"{'dcf_tot':>8} {'pace_tot':>9}")
    for C in cost_list + ["nocd"]:
        kw = dict(sweep="A", config=str(C)) if C != "nocd" \
            else dict(sweep="B", config="basic")
        d = _mean25(rows, "succ_v", method="dcf_conv", **kw)
        p = _mean25(rows, "succ_v", method="pace", **kw)
        dt = _mean25(rows, "useful", method="dcf_conv", **kw)
        pt = _mean25(rows, "useful", method="pace", **kw)
        print(f"  {str(C):>5} {d:>7.3f} {p:>7.3f} {p / d:>7.2f}   "
              f"{dt:>8.3f} {pt:>9.3f}")

    print("\nH3: channel-efficiency crossover (pace_tot - dcf_tot by C)")
    for C in cost_list:
        dt = _mean25(rows, "useful", sweep="A", config=str(C), method="dcf_conv")
        pt = _mean25(rows, "useful", sweep="A", config=str(C), method="pace")
        print(f"  C={C:>2}: Δ={pt - dt:+.4f}  "
              f"({'pace ahead' if pt > dt else 'dcf ahead'})")

    print("\nH4: mandatory RTS/CTS gains vs basic access")
    for m in ["dcf_conv", "pace"]:
        bv = _mean25(rows, "succ_v", sweep="B", config="basic", method=m)
        bu = _mean25(rows, "useful", sweep="B", config="basic", method=m)
        for g in ["rts_24m", "rts_6m"]:
            gv = _mean25(rows, "succ_v", sweep="B", config=g, method=m)
            gu = _mean25(rows, "useful", sweep="B", config=g, method=m)
            print(f"  {m:<9} {g}: visitor {bv:.3f}→{gv:.3f} "
                  f"({(gv / bv - 1) * 100:+.1f}%)   "
                  f"channel {bu:.3f}→{gu:.3f} ({(gu / bu - 1) * 100:+.1f}%)")


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_25)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def load_csv(path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            for k in FIELDS_25:
                if k in ("sweep", "config", "method", "seed"):
                    continue
                row[k] = float(row[k])
            rows.append(row)
    return rows


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot(rows, cost_list: list, out_dir: str, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.42, wspace=0.30)
    _panel_a(axes[0, 0], rows, cost_list)
    _panel_b(axes[0, 1], rows, cost_list)
    _panel_c(axes[1, 0], rows)
    _panel_d(axes[1, 1], rows)

    fig.suptitle(
        "Fig. 25  Collision-Cost Sensitivity and Mandatory RTS/CTS — "
        "Mixed Native/Visitor NPCA Channel\n"
        f"(N_v={N_VISITOR} PACE-warm visitors + N_nat={N_NATIVE} DCF natives, "
        f"W_eff={W_REF}σ={W_REF * SLOT_US / 1000:.2f}ms, visitor PPDU "
        f"U[{PPDU_V_LO},{PPDU_V_HI}]σ={PPDU_V_LO * SLOT_US}–"
        f"{PPDU_V_HI * SLOT_US}µs, σ=aSlotTime={SLOT_US}µs, "
        f"steady state over {FULL_VISITS} transitions)",
        fontsize=11,
    )

    fig_name = "fig25_collision_cost"
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
        description="Figure 25 — collision-cost sensitivity + mandatory RTS/CTS")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-dir", default="results/step9/fig25")
    parser.add_argument("--base-csv", default=None, metavar="PATH")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "results", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    cost_list = FAST_COSTS if args.fast else COST_LIST

    if args.base_csv:
        print(f"Loading data from {args.base_csv} ...")
        rows = load_csv(args.base_csv)
        cost_list = sorted({int(r["config"]) for r in rows if r["sweep"] == "A"})
    else:
        seeds = FAST_SEEDS if args.fast else SEEDS_25
        reps = FAST_REPS if args.fast else FULL_REPS
        visits = FAST_VISITS if args.fast else FULL_VISITS
        print(f"=== Figure 25 [{'FAST' if args.fast else 'FULL'}] ===")
        print(f"    methods : {METHODS_25}")
        print(f"    costs   : {cost_list} + nocd")
        print(f"    RTS cfgs: {[c[0] for c in RTS_CONFIGS]}")
        print(f"    visits  : {visits}  reps/seed: {reps}  seeds: {seeds}")
        rows = run_sweep(cost_list, seeds, reps, visits)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    check_hypotheses(rows, cost_list)

    print("\nPlotting ...")
    plot(rows, cost_list, out_dir, fig_dir)

    print("\nFigure 25 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : {fig_dir}/fig25_collision_cost.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
