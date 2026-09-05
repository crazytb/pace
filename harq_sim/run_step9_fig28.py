"""
Figure 28 (paper Fig. eval-4, fig4-1/fig4-2): Tracking the Time-Varying Target

Paper Subsection D. Within-visit τ dynamics in the reference setting of
Subsections A/B (N=20 visitors, M=10 natives, W_eff=420σ=3.78ms):

  FS target        τ*(t) = 1/|V(t)|   — rises as the viable set drains
  PACE             mean τ_i over viable visitors (Algorithm 1, τ0=1/W_eff)
  Standard NPCA    measured per-slot transmission frequency of viable
                   visitors (deferring implementation, per-frame fit check)

Curves are averaged over many visits, binned by elapsed time; log-y.
Two single-panel figures (basic / RTS) for the LaTeX subfigure pair.

Run:
  .venv/bin/python harq_sim/run_step9_fig28.py
  .venv/bin/python harq_sim/run_step9_fig28.py --fast
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import math

import run_step9_fig17 as _f17
import run_step9_fig25 as _f25
import run_step9_fig26 as _f26

# ─── Parameters ───────────────────────────────────────────────────────────────

N_V, M, W = 20, 10, 420
N_BINS = 42
FULL_VISITS = 500
FAST_VISITS = 30

ACCESS_CONFIGS = [
    ("basic", "nocd", 0),
    ("rts",   _f25.COLL_RTS_24M, _f25.OH_SUCC_24M),
]

_STYLE = {
    "dcf_excl": dict(_f26._STYLE_26["dcf_excl"]),
    "pace":     dict(_f26._STYLE_26["pace"]),
    "oracle":   dict(_f26._STYLE_26["oracle"]),
    "pace_dyn": dict(color="#d62728", ls=":", lw=2.0),
}
_LABEL = {
    "dcf_excl": "Standard NPCA (measured)",
    "pace":     "PACE-static (measured)",
    "pace_dyn": r"PACE-dynamic ($c=\exp(C/\sqrt{W_\mathrm{eff}})$)",
    "oracle":   "FS target ($\\tau^*{=}1/|\\mathcal{V}(t)|$)",
}


# ─── Instrumented single visit ────────────────────────────────────────────────

def visit_trace(mode: str, ppdus: np.ndarray, rng: np.random.Generator,
                coll_cost, succ_oh: int) -> list[tuple[int, float]]:
    """Returns [(elapsed_slots, tracked_value)] at every decision epoch.
    mode: 'pace' | 'oracle' | 'dcf_excl'. Natives always DCF."""
    N_total = N_V + M
    W_rem = W
    tau = np.full(N_V, 1.0 / W)
    _solo = 0.0
    cw_v = np.full(N_V, 16, dtype=np.int64)
    bo_v = rng.integers(0, 16, size=N_V).astype(np.int64)
    cw_n = np.full(M, 16, dtype=np.int64)
    bo_n = rng.integers(0, 16, size=M).astype(np.int64)
    out = []

    while W_rem > 0:
        # dcf_excl: deferring standard implementation — per-frame fit check
        # (the draft leaves the unfittable case unspecified; see fig25 engine)
        vv = ppdus[:N_V] + succ_oh <= W_rem
        vn = np.ones(M, dtype=bool)
        k = int(vv.sum() + vn.sum())
        if k == 0:
            break
        t = W - W_rem

        if mode == "oracle":
            out.append((t, 1.0 / k))
            tx_v = rng.random(N_V) < np.where(vv, 1.0 / k, 0.0)
        elif mode == "pace":
            tx_v = rng.random(N_V) < np.where(vv, tau.clip(1e-4, 1.0), 0.0)
            if vv.any():
                # measured per-slot transmission frequency of viable visitors
                out.append((t, float(tx_v[vv].mean())))
        else:  # dcf_excl
            tx_v = (bo_v == 0) & vv
            if vv.any():
                out.append((t, float(tx_v[vv].mean())))
        tx_n = (bo_n == 0) & vn
        tx = np.concatenate([tx_v, tx_n])
        n_tx = int(tx.sum())

        if n_tx == 1:
            i = int(np.where(tx)[0][0])
            need = int(ppdus[i]) + succ_oh
            if i < N_V:
                if need <= W_rem:
                    # saturated: winner draws next frame, keeps contending
                    _solo = float(tau[i])
                    if mode == "dcf_excl":
                        cw_v[i] = 16
                        bo_v[i] = int(rng.integers(0, 16))
                    W_rem -= need
                    ppdus[i] = int(rng.integers(_f25.PPDU_V_LO,
                                                _f25.PPDU_V_HI + 1))
                else:
                    W_rem = 0          # unreachable under fit-checked modes
            else:
                cw_n[i - N_V] = 16
                bo_n[i - N_V] = int(rng.integers(0, 16))
                W_rem -= min(need, W_rem)
        elif n_tx > 1:
            c = int(ppdus[np.where(tx)[0]].max()) if coll_cost == "nocd" \
                else int(coll_cost)
            W_rem -= min(c, W_rem)
        else:
            W_rem -= 1

        if mode == "pace":
            if n_tx == 1 and int(np.where(tx)[0][0]) < N_V:
                for kk in range(N_V):
                    if not tx_v[kk] and vv[kk]:
                        tau[kk] = _solo
            elif n_tx > 1:
                for kk in range(N_V):
                    if vv[kk] and not tx_v[kk]:
                        tau[kk] /= _f17.PND_C_COLL
            elif n_tx == 0:
                for kk in range(N_V):
                    if not tx_v[kk] and vv[kk]:
                        tau[kk] *= _f17.PND_C_IDLE
            for kk in range(N_V):
                tau[kk] = float(np.clip(tau[kk], 1e-4, 1.0))
        elif mode == "dcf_excl":
            if n_tx > 1:
                for j in np.where(tx_v)[0]:
                    j = int(j)
                    cw_v[j] = min(int(cw_v[j]) * 2, _f17.DCF_CW_MAX)
                    bo_v[j] = int(rng.integers(0, max(int(cw_v[j]), 1)))
            elif n_tx == 0:
                mask = vv & (bo_v > 0)
                bo_v[mask] -= 1
        # natives
        if n_tx > 1:
            for j in np.where(tx_n)[0]:
                j = int(j)
                cw_n[j] = min(int(cw_n[j]) * 2, _f17.DCF_CW_MAX)
                bo_n[j] = int(rng.integers(0, max(int(cw_n[j]), 1)))
        elif n_tx == 0:
            mask = vn & (bo_n > 0)
            bo_n[mask] -= 1
    return out


# ─── Binned average across visits ─────────────────────────────────────────────

C_WRULE = 10.16          # section 4.5.40, calibrated at alpha = 0.5


def binned(mode: str, coll_cost, succ_oh: int, visits: int) -> tuple:
    sums = np.zeros(N_BINS)
    cnts = np.zeros(N_BINS)
    # pace_dyn is PACE with the coefficient taken from the window rather than
    # fixed; at this figure's single W_eff it differs only by 1.64 against 1.50
    saved = (_f17.PND_C_COLL, _f17.PND_C_IDLE)
    if mode == "pace_dyn":
        _f17.PND_C_COLL = _f17.PND_C_IDLE = math.exp(C_WRULE / math.sqrt(W))
    inner = "pace" if mode == "pace_dyn" else mode
    for r in range(visits):
        rng_p = np.random.default_rng(9000 + r)
        rng = np.random.default_rng(31337 + r * 13)
        ppdus = np.concatenate([
            rng_p.integers(_f25.PPDU_V_LO, _f25.PPDU_V_HI + 1, size=N_V),
            np.full(M, _f25.PPDU_NATIVE_SLOTS)]).astype(np.int32)
        for t, val in visit_trace(inner, ppdus, rng, coll_cost, succ_oh):
            b = min(int(N_BINS * t / W), N_BINS - 1)
            sums[b] += val
            cnts[b] += 1
    _f17.PND_C_COLL, _f17.PND_C_IDLE = saved
    xs = (np.arange(N_BINS) + 0.5) * W / N_BINS * 9 / 1000   # ms
    ys = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return xs, ys


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_one(access: str, coll_cost, succ_oh: int, visits: int,
             fig_dir: str, out_dir: str, fig_name: str,
             leg_loc: str = "best") -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for mode in ["dcf_excl", "pace", "pace_dyn", "oracle"]:
        xs, ys = binned(mode, coll_cost, succ_oh, visits)
        st = dict(_STYLE[mode])
        st["marker"] = None
        ax.plot(xs, ys, label=_LABEL[mode], **{k: v for k, v in st.items()
                                               if k not in ("marker", "ms")})
    ax.set_yscale("log")
    ax.set_ylim(top=0.4)   # empty top band reserved for the legend
    ax.set_xlabel("Elapsed time in the visit (ms)")
    ax.set_ylabel("Per-slot transmission rate")
    ax.legend(fontsize=7.5, frameon=True, loc=leg_loc,
              handlelength=1.5, borderpad=0.3, labelspacing=0.3)
    ax.grid(color="0.9", lw=0.4, which="both")
    ax.set_axisbelow(True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-dir", default="results/step9/fig28")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(repo_root, "results", "figure")
    os.makedirs(fig_dir, exist_ok=True)

    visits = FAST_VISITS if args.fast else FULL_VISITS
    locs = {"fig4-1": "upper left", "fig4-2": "upper left"}
    for (access, cc, oh), name in zip(ACCESS_CONFIGS, ["fig4-1", "fig4-2"]):
        print(f"--- {access} ({visits} visits) ---")
        plot_one(access, cc, oh, visits, fig_dir, args.out_dir, name,
                 leg_loc=locs[name])
    print("\nDone → results/figure/fig4-{1,2}.*")


if __name__ == "__main__":
    main()
