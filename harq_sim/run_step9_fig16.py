"""
Figure 16: Topology-Aware MFG — Adjacency Graph Based NPCA Spatial Reuse

Extends Fig 15 from complete-graph (K_N) to arbitrary adjacency topology.
Non-adjacent STAs succeed simultaneously (spatial reuse).

Output:
  manuscript/figure/fig16_topology_mfg.{eps,png,pdf}
  results/step9/fig16/data.csv

Run:
  python harq_sim/run_step9_fig16.py [--fast] [--out-dir results/step9/fig16]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Parameters ───────────────────────────────────────────────────────────────

N_LIST    = [9, 16, 25, 36, 49]   # all perfect squares (grid-compatible)
WEFF_LIST = [20, 50, 100, 200, 500]
SEEDS     = [42, 123, 456, 789, 1234]

FULL_VISITS        = 200
FAST_VISITS        = 30
ORACLE_VISITS_FULL = 80
ORACLE_VISITS_FAST = 15

TOPOLOGY_TYPES = ["complete", "chain", "grid", "rgg"]
PROTOCOLS      = ["dcf_ieee", "dcf_topology", "mfg_full_N", "mfg_perfect_local",
                  "mfg_carrier_sense", "oracle_topology"]

IEEE_CW0 = 16  # IEEE 802.11 CW_min=15 → backoff [0,15] → SIZE=16


# ─── Topology generation ──────────────────────────────────────────────────────

def make_topology(topo_type: str, N: int, seed: int = 42) -> np.ndarray:
    """Returns (N, N) boolean symmetric adjacency matrix, no self-loops."""
    if topo_type == "complete":
        adj = np.ones((N, N), dtype=bool)
        np.fill_diagonal(adj, False)
        return adj

    if topo_type == "chain":
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N - 1):
            adj[i, i + 1] = adj[i + 1, i] = True
        return adj

    if topo_type == "grid":
        side = int(round(np.sqrt(N)))
        assert side * side == N, f"N={N} not a perfect square"
        adj = np.zeros((N, N), dtype=bool)
        for idx in range(N):
            r, c = divmod(idx, side)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    adj[idx, nr * side + nc] = True
        return adj

    if topo_type == "rgg":
        rng_t = np.random.default_rng(seed * 10007 + 3)
        r_thresh = np.sqrt(np.log(max(N, 2)) / (np.pi * N))
        r = r_thresh * 2.0
        pos = rng_t.random((N, 2))
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        adj = (dist < r) & ~np.eye(N, dtype=bool)
        return adj

    raise ValueError(f"Unknown topology: {topo_type}")


def _greedy_mis(adj: np.ndarray) -> int:
    """Greedy maximum independent set (approximation)."""
    N = adj.shape[0]
    degrees = adj.sum(axis=1).astype(int)
    excluded = np.zeros(N, dtype=bool)
    count = 0
    for _ in range(N):
        available = ~excluded
        if not available.any():
            break
        deg_avail = np.where(available, degrees, N + 1)
        i = int(np.argmin(deg_avail))
        count += 1
        excluded[i] = True
        excluded |= adj[i]
    return count


def topology_stats(adj: np.ndarray) -> dict:
    return {
        "mean_degree": float(adj.sum(axis=1).mean()),
        "alpha_G":     _greedy_mis(adj),
    }


# ─── DCF with topology-aware collisions ──────────────────────────────────────

def sim_dcf_topology(
    adj: np.ndarray, W_eff: int, CW0: int,
    n_visits: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DCF: backoff freezes when any neighbor is in ready state (carrier sense).
    Collision = neighbor also at backoff=0; BEB applied.
    Returns (success/visit, spatial_reuse_rate/visit).
    """
    N     = adj.shape[0]
    V     = n_visits
    CW0   = max(CW0, 1)
    adj_f = adj.astype(np.float32)

    backoffs = rng.integers(0, CW0, size=(V, N))
    cws      = np.full((V, N), CW0, dtype=np.int32)
    active   = np.ones((V, N), dtype=bool)
    success  = np.zeros(V, dtype=np.int32)
    sr_slots = np.zeros(V, dtype=np.int32)

    for _ in range(W_eff):
        if not active.any():
            break

        ready   = active & (backoffs == 0)
        # nbr_rdy[v,i] = True iff some neighbor of i is in ready state
        nbr_rdy = (ready.astype(np.float32) @ adj_f) > 0.5

        solo      = ready & ~nbr_rdy
        collision = ready & nbr_rdy

        sc = solo.sum(axis=1)
        sr_slots += (sc >= 2).astype(np.int32)
        success  += sc
        active   &= ~solo

        if collision.any():
            new_cws  = np.minimum(2 * (cws + 1) - 1, 1023)
            cws      = np.where(collision, new_cws, cws)
            r_rand   = rng.integers(0, 1024, size=(V, N), dtype=np.int32)
            backoffs = np.where(collision, r_rand % (cws + 1), backoffs)

        # Freeze when neighbor is transmitting; decrement otherwise
        dec_mask = active & (backoffs > 0) & ~nbr_rdy & ~ready
        backoffs -= dec_mask.astype(np.int32)

    return success, sr_slots.astype(float) / W_eff


# ─── MFG full-N (global τ = 1/remaining, topology collisions) ────────────────

def sim_mfg_full_N(
    adj: np.ndarray, W_eff: int,
    n_visits: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MFG τ(t) = 1/global_remaining applied to topology model.
    Topology-unaware: uses global count, not local n_i(t).
    """
    N     = adj.shape[0]
    V     = n_visits
    adj_f = adj.astype(np.float32)

    active   = np.ones((V, N), dtype=bool)
    success  = np.zeros(V, dtype=np.int32)
    sr_slots = np.zeros(V, dtype=np.int32)

    for _ in range(W_eff):
        if not active.any():
            break

        remaining = active.sum(axis=1).astype(float)          # (V,)
        tau = np.where(remaining > 0, 1.0 / np.maximum(remaining, 1.0), 0.0)

        rand = rng.random((V, N))
        tx   = active & (rand < tau[:, np.newaxis])

        nbr_tx = (tx.astype(np.float32) @ adj_f) > 0.5
        solo   = tx & ~nbr_tx & active

        sc = solo.sum(axis=1)
        sr_slots += (sc >= 2).astype(np.int32)
        success  += sc
        active   &= ~solo

    return success, sr_slots.astype(float) / W_eff


# ─── MFG perfect local (τ_i = 1/n_i(t)) ─────────────────────────────────────

def sim_mfg_perfect_local(
    adj: np.ndarray, W_eff: int,
    n_visits: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Topology-aware MFG: τ_i*(t) = 1/n_i(t), n_i(t) = |N_i ∩ active| + 1.
    Requires perfect knowledge of active neighbor set each slot.
    """
    N     = adj.shape[0]
    V     = n_visits
    adj_f = adj.astype(np.float32)

    active   = np.ones((V, N), dtype=bool)
    success  = np.zeros(V, dtype=np.int32)
    sr_slots = np.zeros(V, dtype=np.int32)

    for _ in range(W_eff):
        if not active.any():
            break

        # n_i(t) = count of active neighbors + 1 (self)
        n_i   = (active.astype(np.float32) @ adj_f) + 1.0    # (V, N)
        tau_i = np.where(active, 1.0 / np.maximum(n_i, 1.0), 0.0)

        rand  = rng.random((V, N))
        tx    = active & (rand < tau_i)

        nbr_tx = (tx.astype(np.float32) @ adj_f) > 0.5
        solo   = tx & ~nbr_tx & active

        sc = solo.sum(axis=1)
        sr_slots += (sc >= 2).astype(np.int32)
        success  += sc
        active   &= ~solo

    return success, sr_slots.astype(float) / W_eff


# ─── MFG carrier sense (static τ_i = 1/(degree_i + 1)) ──────────────────────

def sim_mfg_carrier_sense(
    adj: np.ndarray, W_eff: int,
    n_visits: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Practical distributed MFG: n_hat_i = degree_i + 1 (initial, no adaptation).
    STA uses static TX probability based on initial neighborhood size.
    """
    N     = adj.shape[0]
    V     = n_visits
    adj_f = adj.astype(np.float32)

    degrees  = adj.sum(axis=1).astype(float)           # (N,)
    tau_stat = 1.0 / np.maximum(degrees + 1.0, 1.0)   # (N,) constant

    active   = np.ones((V, N), dtype=bool)
    success  = np.zeros(V, dtype=np.int32)
    sr_slots = np.zeros(V, dtype=np.int32)

    for _ in range(W_eff):
        if not active.any():
            break

        tau_i = np.where(active, tau_stat[np.newaxis, :], 0.0)
        rand  = rng.random((V, N))
        tx    = active & (rand < tau_i)

        nbr_tx = (tx.astype(np.float32) @ adj_f) > 0.5
        solo   = tx & ~nbr_tx & active

        sc = solo.sum(axis=1)
        sr_slots += (sc >= 2).astype(np.int32)
        success  += sc
        active   &= ~solo

    return success, sr_slots.astype(float) / W_eff


# ─── Oracle: best DCF CW0 ────────────────────────────────────────────────────

def _oracle_candidates(N: int, W_eff: int) -> list[int]:
    cands = set()
    v = max(1, N // 4)
    while v <= max(4 * N, W_eff):
        cands.add(v)
        v = int(v * 1.5) + 1
    cands.update([N // 2, N, 2 * N, 3 * N, W_eff])
    return sorted(c for c in cands if 1 <= c <= max(4 * N, W_eff))


def find_oracle_topology(
    adj: np.ndarray, W_eff: int,
    rng: np.random.Generator, n_visits: int = 80,
) -> tuple[int, float]:
    N = adj.shape[0]
    best_cw, best_mean = 1, 0.0
    for cw in _oracle_candidates(N, W_eff):
        s, _ = sim_dcf_topology(adj, W_eff, cw, n_visits, rng)
        m = float(s.mean())
        if m > best_mean:
            best_mean, best_cw = m, cw
    return best_cw, best_mean


# ─── Main sweep ───────────────────────────────────────────────────────────────

FIELDS = [
    "topology", "N", "W_eff", "protocol", "seed",
    "CW0", "mean_success", "std_success",
    "mean_spatial_reuse", "mean_degree", "alpha_G",
]


def run_sweep(n_visits: int, oracle_visits: int) -> list[dict]:
    rows = []
    total = len(TOPOLOGY_TYPES) * len(N_LIST) * len(SEEDS) * len(WEFF_LIST) * len(PROTOCOLS)
    done  = 0

    for topo in TOPOLOGY_TYPES:
        for N in N_LIST:
            for seed in SEEDS:
                adj   = make_topology(topo, N, seed=seed)
                stats = topology_stats(adj)
                mdeg  = stats["mean_degree"]
                alpha = stats["alpha_G"]

                for W_eff in WEFF_LIST:

                    # 0. DCF IEEE (CW0=16, i.e., IEEE CW_min=15, fixed regardless of N)
                    rng = np.random.default_rng(seed)
                    s, sr = sim_dcf_topology(adj, W_eff, IEEE_CW0, n_visits, rng)
                    rows.append(_mkrow(topo, N, W_eff, "dcf_ieee", seed,
                                       IEEE_CW0, s, sr, mdeg, alpha))
                    done += 1; _log(done, total, topo, N, W_eff, "dcf_ieee", seed)

                    # 1. DCF topology (CW0 = 2N)
                    rng = np.random.default_rng(seed)
                    s, sr = sim_dcf_topology(adj, W_eff, 2 * N, n_visits, rng)
                    rows.append(_mkrow(topo, N, W_eff, "dcf_topology", seed,
                                       2*N, s, sr, mdeg, alpha))
                    done += 1; _log(done, total, topo, N, W_eff, "dcf_topology", seed)

                    # 2. MFG full N
                    rng = np.random.default_rng(seed)
                    s, sr = sim_mfg_full_N(adj, W_eff, n_visits, rng)
                    rows.append(_mkrow(topo, N, W_eff, "mfg_full_N", seed,
                                       None, s, sr, mdeg, alpha))
                    done += 1; _log(done, total, topo, N, W_eff, "mfg_full_N", seed)

                    # 3. MFG perfect local
                    rng = np.random.default_rng(seed)
                    s, sr = sim_mfg_perfect_local(adj, W_eff, n_visits, rng)
                    rows.append(_mkrow(topo, N, W_eff, "mfg_perfect_local", seed,
                                       None, s, sr, mdeg, alpha))
                    done += 1; _log(done, total, topo, N, W_eff, "mfg_perfect_local", seed)

                    # 4. MFG carrier sense
                    rng = np.random.default_rng(seed)
                    s, sr = sim_mfg_carrier_sense(adj, W_eff, n_visits, rng)
                    rows.append(_mkrow(topo, N, W_eff, "mfg_carrier_sense", seed,
                                       None, s, sr, mdeg, alpha))
                    done += 1; _log(done, total, topo, N, W_eff, "mfg_carrier_sense", seed)

                    # 5. Oracle
                    rng_o = np.random.default_rng(seed + 99999)
                    best_cw, best_mean = find_oracle_topology(adj, W_eff, rng_o, oracle_visits)
                    rows.append({
                        "topology": topo, "N": N, "W_eff": W_eff,
                        "protocol": "oracle_topology", "seed": seed,
                        "CW0": best_cw, "mean_success": best_mean, "std_success": 0.0,
                        "mean_spatial_reuse": 0.0,
                        "mean_degree": mdeg, "alpha_G": alpha,
                    })
                    done += 1; _log(done, total, topo, N, W_eff, "oracle_topology", seed)

    return rows


def _mkrow(topo, N, W_eff, proto, seed, cw0, s_arr, sr_arr, mdeg, alpha):
    return {
        "topology": topo, "N": N, "W_eff": W_eff,
        "protocol": proto, "seed": seed,
        "CW0": cw0,
        "mean_success":      float(s_arr.mean()),
        "std_success":       float(s_arr.std()),
        "mean_spatial_reuse": float(sr_arr.mean()),
        "mean_degree": mdeg, "alpha_G": alpha,
    }


def _log(done, total, topo, N, W_eff, proto, seed):
    print(f"  [{done:4d}/{total}] {topo:<8}  N={N:2d}  W_eff={W_eff:3d}  "
          f"{proto:<22}  seed={seed}", flush=True)


# ─── Statistics helpers ───────────────────────────────────────────────────────

def _mean_seeds(rows, **kw) -> float:
    vals = [r["mean_success"] for r in rows
            if all(r[k] == v for k, v in kw.items())]
    return float(np.mean(vals)) if vals else 0.0


def _sr_mean_seeds(rows, **kw) -> float:
    vals = [r["mean_spatial_reuse"] for r in rows
            if all(r[k] == v for k, v in kw.items())]
    return float(np.mean(vals)) if vals else 0.0


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


# ─── Plot ─────────────────────────────────────────────────────────────────────

_PROTO_STYLE = {
    "dcf_ieee":          dict(color="#9467bd", ls="--", lw=1.8, marker="x", ms=6, markeredgewidth=1.8),
    "dcf_topology":      dict(color="#888888", ls="--", lw=1.8, marker="s", ms=5),
    "mfg_full_N":        dict(color="#1f77b4", ls="-.", lw=2.5, marker="^", ms=7, zorder=4),
    "mfg_perfect_local": dict(color="#d62728", ls="-",  lw=2.2, marker="o", ms=6),
    "mfg_carrier_sense": dict(color="#ff7f0e", ls=":",  lw=1.8, marker="v", ms=5),
    "oracle_topology":   dict(color="#2ca02c", ls="--", lw=1.5, marker="D", ms=5),
}
_PROTO_LABEL = {
    "dcf_ieee":          "DCF IEEE (CW_min=15, fixed)",
    "dcf_topology":      "DCF (CW₀=2N, local coll.)",
    "mfg_full_N":        "MFG τ=1/N (global)",
    "mfg_perfect_local": "MFG τᵢ=1/nᵢ(t) (perfect local)",
    "mfg_carrier_sense": "MFG carrier sense (static nᵢ)",
    "oracle_topology":   "Oracle DCF (best CW₀)",
}
_TOPO_COLOR = {
    "complete": "#888888",
    "chain":    "#1f77b4",
    "grid":     "#2ca02c",
    "rgg":      "#d62728",
}
_TOPO_LABEL = {
    "complete": "Complete K_N",
    "chain":    "Chain P_N",
    "grid":     "2D Grid",
    "rgg":      "RGG (r=2·r_thresh)",
}


def plot(rows: list, fig_dir: str) -> None:
    fig = plt.figure(figsize=(22, 16))
    gs  = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.42)

    # Panel (a): success vs N, W_eff=20 (tight-window regime maximises visible differences)
    W_eff_a = 20
    for col, topo in enumerate(TOPOLOGY_TYPES):
        ax = fig.add_subplot(gs[0, col])
        _panel_a(ax, rows, topo, W_eff_a, col == 0)

    # Panel (b): topology gain heatmap — averaged over chain/grid/rgg
    ax_b = fig.add_subplot(gs[1, :2])
    _panel_b_gain_heatmap(ax_b, rows)

    # Panel (c): spatial reuse rate vs W_eff/N
    ax_c = fig.add_subplot(gs[1, 2:])
    _panel_c_spatial_reuse(ax_c, rows)

    # Panel (d): information model gap — perfect vs carrier sense
    ax_d = fig.add_subplot(gs[2, :])
    _panel_d_info_model(ax_d, rows)

    fig.suptitle(
        "Fig. 16  Topology-Aware MFG — Adjacency Graph Based NPCA Spatial Reuse\n"
        "(non-adjacent STAs succeed simultaneously; collision domain = carrier-sense neighborhood)",
        fontsize=12,
    )
    _save_figure(fig, fig_dir, "fig16_topology_mfg")
    plt.close(fig)


def _panel_a(ax, rows, topo, W_eff, show_ylabel):
    protos = ["dcf_ieee", "dcf_topology", "mfg_full_N", "mfg_perfect_local", "oracle_topology"]
    for proto in protos:
        means = [_mean_seeds(rows, topology=topo, N=N, W_eff=W_eff, protocol=proto)
                 for N in N_LIST]
        if not any(m > 0 for m in means):
            continue
        # For complete graph, mfg_full_N ≡ mfg_perfect_local (theory) — mark it explicitly
        if topo == "complete" and proto == "mfg_full_N":
            ax.plot(N_LIST, means, label="MFG full_N (≡ perfect, K_N)",
                    **{**_PROTO_STYLE[proto], "lw": 1.2, "ls": "--", "alpha": 0.5})
        else:
            ax.plot(N_LIST, means, label=_PROTO_LABEL[proto], **_PROTO_STYLE[proto])

    ax.set_xlabel("N (STAs)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Mean success / visit", fontsize=9)
    ax.set_xticks(N_LIST)
    ax.legend(fontsize=6.0, frameon=True, loc="upper left")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(a) {_TOPO_LABEL[topo]}\nW_eff={W_eff}", fontsize=9)


def _panel_b_gain_heatmap(ax, rows):
    """Topology gain = (mfg_perfect_local − mfg_full_N) / mfg_full_N × 100%.
    Averaged over chain, grid, rgg."""
    topos_sparse = ["chain", "grid", "rgg"]
    gain_mat = np.zeros((len(N_LIST), len(WEFF_LIST)))

    for i, N in enumerate(N_LIST):
        for j, W_eff in enumerate(WEFF_LIST):
            gains = []
            for topo in topos_sparse:
                pfx  = _mean_seeds(rows, topology=topo, N=N, W_eff=W_eff,
                                   protocol="mfg_perfect_local")
                full = _mean_seeds(rows, topology=topo, N=N, W_eff=W_eff,
                                   protocol="mfg_full_N")
                if full > 0:
                    gains.append((pfx - full) / full * 100.0)
            gain_mat[i, j] = float(np.mean(gains)) if gains else 0.0

    vmax = max(float(gain_mat.max()), 1.0)
    vmin = min(float(gain_mat.min()), 0.0)
    im = ax.imshow(gain_mat, aspect="auto", cmap="YlOrRd",
                   vmin=vmin, vmax=vmax, origin="lower")

    ax.set_xticks(range(len(WEFF_LIST)))
    ax.set_xticklabels(WEFF_LIST)
    ax.set_yticks(range(len(N_LIST)))
    ax.set_yticklabels(N_LIST)
    ax.set_xlabel("W_eff (available NPCA slots)", fontsize=9)
    ax.set_ylabel("N (STAs)", fontsize=9)

    for i in range(len(N_LIST)):
        for j in range(len(WEFF_LIST)):
            ax.text(j, i, f"{gain_mat[i, j]:+.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="black" if gain_mat[i, j] < 0.65 * vmax else "white")

    # W_eff = N and W_eff = 2N diagonals
    N_arr = np.array(N_LIST, dtype=float)
    W_arr = np.array(WEFF_LIST, dtype=float)
    for mult, ls, lbl in [(1, ":", "W_eff=N"), (2, "--", "W_eff=2N")]:
        xs, ys = [], []
        for j, W in enumerate(WEFF_LIST):
            for i, N in enumerate(N_LIST):
                if abs(W - mult * N) <= (W_arr[1] - W_arr[0]) * 0.6:
                    xs.append(j); ys.append(i)
        if xs:
            ax.plot(xs, ys, "k" + ls, lw=1.2, label=lbl)

    plt.colorbar(im, ax=ax, label="Gain % (perfect_local vs full_N)", fraction=0.04)
    ax.set_title("(b) Topology gain: MFG perfect_local vs mfg_full_N\n"
                 "(avg over chain / grid / rgg)", fontsize=9)
    ax.legend(fontsize=8, frameon=True, loc="lower right")


def _panel_c_spatial_reuse(ax, rows):
    """Spatial reuse fraction vs W_eff/N for N=25."""
    N_ref = 25
    for topo in ["chain", "grid", "rgg"]:
        x_vals = [W / N_ref for W in WEFF_LIST]
        y_vals = [_sr_mean_seeds(rows, topology=topo, N=N_ref, W_eff=W,
                                  protocol="mfg_perfect_local")
                  for W in WEFF_LIST]
        ax.plot(x_vals, y_vals, color=_TOPO_COLOR[topo], lw=1.8,
                marker="o", ms=5, label=_TOPO_LABEL[topo])

    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", ls=":", lw=1.0, label="W_eff = N")
    ax.set_xlabel("W_eff / N  (log scale)", fontsize=9)
    ax.set_ylabel("Spatial reuse fraction\n(slots with ≥2 simultaneous successes)", fontsize=9)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(c) Spatial reuse vs W_eff/N  (N={N_ref}, mfg_perfect_local)", fontsize=9)


def _panel_d_info_model(ax, rows):
    """Information gap: (perfect_local − carrier_sense) / perfect_local × 100%."""
    W_eff_d = 50
    for topo in ["chain", "grid", "rgg"]:
        gaps = []
        for N in N_LIST:
            pfx = _mean_seeds(rows, topology=topo, N=N, W_eff=W_eff_d,
                               protocol="mfg_perfect_local")
            cs  = _mean_seeds(rows, topology=topo, N=N, W_eff=W_eff_d,
                               protocol="mfg_carrier_sense")
            gap = (pfx - cs) / max(pfx, 1e-9) * 100.0 if pfx > 0 else 0.0
            gaps.append(gap)
        ax.plot(N_LIST, gaps, color=_TOPO_COLOR[topo], lw=1.8,
                marker="o", ms=5, label=_TOPO_LABEL[topo])

    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("N (STAs)", fontsize=9)
    ax.set_ylabel("Info cost: (perfect − carrier_sense) / perfect  [%]", fontsize=9)
    ax.set_xticks(N_LIST)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.set_title(f"(d) Information model gap — perfect local vs carrier sense  (W_eff={W_eff_d})",
                 fontsize=9)


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


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 16 — Topology-aware MFG NPCA spatial reuse")
    parser.add_argument("--fast",    action="store_true",
                        help=f"Quick validation: {FAST_VISITS} visits/config")
    parser.add_argument("--out-dir", default="results/step9/fig16")
    args = parser.parse_args()

    n_visits      = FAST_VISITS        if args.fast else FULL_VISITS
    oracle_visits = ORACLE_VISITS_FAST if args.fast else ORACLE_VISITS_FULL
    out_dir       = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    total_configs = len(TOPOLOGY_TYPES) * len(N_LIST) * len(SEEDS) * len(WEFF_LIST) * len(PROTOCOLS)
    print(f"=== Figure 16 [{'FAST' if args.fast else 'FULL'}]  {n_visits} visits each ===")
    print(f"    Topologies : {TOPOLOGY_TYPES}")
    print(f"    N          ∈ {N_LIST}")
    print(f"    W_eff      ∈ {WEFF_LIST}")
    print(f"    Seeds        {SEEDS}")
    print(f"    Protocols    {PROTOCOLS}")
    print(f"    Total configs: {total_configs}")

    rows = run_sweep(n_visits, oracle_visits)

    csv_path = os.path.join(out_dir, "data.csv")
    save_csv(rows, csv_path)

    print("Plotting ...")
    plot(rows, out_dir)

    print("\nFigure 16 complete.")
    print(f"  Data    : {csv_path}")
    print(f"  Figures : manuscript/figure/fig16_topology_mfg.{{eps,png,pdf}}")


if __name__ == "__main__":
    main()
