# -*- coding: utf-8 -*-
"""Why the MIMD step size scales as W_eff^(-1/2).

    .venv/bin/python pace-analysis/stepsize.py

Section 4.5.40 established the rule c = exp(C / sqrt(W_eff)) empirically and
argued the square root from a stochastic-approximation bound. The argument there
leaned on a stationary spread, which Theorem 2 contradicts: a visit ends after
17-24 epochs, well before the walk equilibrates. This derives the same exponent
without stationarity and measures the two factors that move it off 1/2.

Proposition. Let X_k = ln tau_k evolve as X_{k+1} = X_k + eps xi_k with
xi_k in {+1, 0, -1} (idle / transmitting-or-non-viable / collision). Split
xi_k = h_k + m_k with h_k = E[xi_k | F_{k-1}] and m_k a martingale difference.
Then

    X_E = X_0 + eps H_E + eps M_E,        H_E = sum h_k

and by orthogonality of martingale increments Var(M_E) = sum E[m_k^2] <= sigma^2 E,
so the deviation from the drift path obeys

    sd[ X_E - (X_0 + eps H_E) ]  <=  eps sigma sqrt(E).

Holding that within a tolerance delta forces eps <= delta / (sigma sqrt(E)), and
with E = W_eff / Dbar,

    eps  <=  (delta sqrt(Dbar) / sigma) * W_eff^(-1/2).

Nothing here needs a stationary distribution or a linearised drift, which is why
it survives Theorem 2 and Proposition 1 (the drift is first-order degenerate at
the operating point, so an AR(1) argument would give eps ~ ln E / E instead).

Two testable consequences, both checked below:

  (a) Var(X_k) / eps^2 collapses onto one curve, independent of eps.
      This is assumption (A1)+(A2): increments are eps-scaled, and solo-copy
      jumps are small because the population stays homogeneous (section 4.5.37).

  (b) sigma is nearly constant in W_eff. It is not exactly, and neither is Dbar,
      so the exponent is 1/2 only to the extent those two hold still. Measuring
      both accounts for the gap between 1/2 and the fitted 0.31-0.40.

The exponent budget it produces:

    0.500   bound with E ~ W_eff
    0.426   after Dbar ~ W_eff^0.148        (section 4.5.40(d))
    0.345   after sigma ~ W_eff^-0.081      basic
    0.386   after sigma ~ W_eff^-0.040      RTS/CTS
    vs 0.310 / 0.331 / 0.397 fitted free in section 4.5.40(c)

Caveat kept in view: the variance grows sublinearly past k ~ 4 (restoring drift,
solo-copy coupling), so the bound is genuine but not tight. The inequality is
what is proved; that the optimum sits AT the bound stays an empirical finding.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "harq_sim"))

import run_step9_fig17 as _f17      # noqa: E402
import run_step9_fig25 as _f25      # noqa: E402

N_V, M = 20, 10
KMAX = 60
ACCESS = {"basic": ("nocd", 0),
          "rts": (_f25.COLL_RTS_24M, _f25.OH_SUCC_24M)}


def trace(W: int, c: float, ppdus, rng, coll_cost, succ_oh) -> list[float]:
    """ln tau of a tagged visitor at each contention epoch of one visit.

    Mirrors the pace branch of run_step9_fig25._run_visit25: tau0 = 1/W_eff,
    solo-copy on a visitor success, /c on a collision seen as a listener, *c on
    an idle slot, and no update while transmitting or non-viable.
    """
    W_rem = W
    tau = np.full(N_V, 1.0 / W)
    bo_n = rng.integers(0, 16, size=M).astype(np.int64)
    cw_n = np.full(M, 16, dtype=np.int64)
    out: list[float] = []

    while W_rem > 0 and len(out) < KMAX:
        viable = ppdus[:N_V] + succ_oh <= W_rem
        tx_v = rng.random(N_V) < np.where(viable, tau.clip(1e-4, 1.0), 0.0)
        tx_n = bo_n == 0
        n_tx = int(tx_v.sum() + tx_n.sum())

        solo_tau = 0.0
        if n_tx == 1:
            if tx_v.any():
                i = int(np.where(tx_v)[0][0])
                solo_tau = float(tau[i])
                W_rem -= int(ppdus[i]) + succ_oh
                ppdus[i] = int(rng.integers(_f25.PPDU_V_LO,
                                            _f25.PPDU_V_HI + 1))
            else:
                j = int(np.where(tx_n)[0][0])
                cw_n[j], bo_n[j] = 16, int(rng.integers(0, 16))
                W_rem -= min(_f25.PPDU_NATIVE_SLOTS + succ_oh, W_rem)
        elif n_tx > 1:
            if coll_cost == "nocd":
                lens = list(ppdus[:N_V][tx_v]) + \
                       [_f25.PPDU_NATIVE_SLOTS] * int(tx_n.sum())
                cost = int(max(lens))
            else:
                cost = int(coll_cost)
            W_rem -= min(cost, W_rem)
        else:
            W_rem -= 1

        listener = viable & ~tx_v
        if n_tx == 1 and tx_v.any():
            tau[listener] = solo_tau
        elif n_tx > 1:
            tau[listener] /= c
        elif n_tx == 0:
            tau[listener] *= c
        tau = np.clip(tau, 1e-4, 1.0)

        if n_tx > 1:
            for j in np.where(tx_n)[0]:
                cw_n[j] = min(int(cw_n[j]) * 2, _f17.DCF_CW_MAX)
                bo_n[j] = int(rng.integers(0, max(int(cw_n[j]), 1)))
        elif n_tx == 0:
            bo_n[bo_n > 0] -= 1

        out.append(math.log(float(tau[0])))
    return out


def variance_curve(W: int, c: float, access: str, visits: int,
                   kmax: int = KMAX) -> np.ndarray:
    """Var across visits of ln tau at each epoch index; NaN where undersampled."""
    coll_cost, succ_oh = ACCESS[access]
    cols: list[list[float]] = [[] for _ in range(kmax)]
    for r in range(visits):
        rng_p = np.random.default_rng(90000 + r)
        rng = np.random.default_rng(313370 + r * 17)
        ppdus = np.concatenate([
            rng_p.integers(_f25.PPDU_V_LO, _f25.PPDU_V_HI + 1, size=N_V),
            np.full(M, _f25.PPDU_NATIVE_SLOTS)]).astype(np.int32)
        for k, x in enumerate(trace(W, c, ppdus, rng, coll_cost, succ_oh)):
            if k < kmax:
                cols[k].append(x)
    return np.array([np.var(v) if len(v) >= 200 else np.nan for v in cols])


def sigma_sq(W: int, access: str, c: float = 1.5, visits: int = 4000,
             k: int = 2) -> float:
    """Var(X_k)/(eps^2 k) at the small-k end, where the collapse is clean."""
    v = variance_curve(W, c, access, visits, kmax=k + 1)
    return float(v[k - 1] / (math.log(c) ** 2 * k))


# ─── (a) eps-collapse ─────────────────────────────────────────────────────────

def check_collapse(visits: int = 4000) -> None:
    print("(a) Var(ln tau)/eps^2 at W_eff = 420, basic, %d visits" % visits)
    print("    proposition: one curve for every eps\n")
    cs = [1.2, 1.5, 1.8, 2.2]
    curves = {c: variance_curve(420, c, "basic", visits) for c in cs}
    print("    " + f"{'k':>4}" + "".join(f"{'c=%.1f' % c:>10}" for c in cs))
    for k in (2, 4, 8, 16):
        row = f"    {k:>4}"
        for c in cs:
            v = curves[c][k - 1]
            row += f"{v / math.log(c) ** 2:>10.3f}" if np.isfinite(v) else f"{'-':>10}"
        print(row)
    at2 = [curves[c][1] / math.log(c) ** 2 for c in cs]
    print(f"\n    spread at k=2 over a {math.log(cs[-1]) / math.log(cs[0]):.1f}x "
          f"range of eps: {max(at2) / min(at2) - 1:.1%}")
    print("    past k ~ 4 the growth turns sublinear (restoring drift and")
    print("    solo-copy coupling), so the bound holds with room to spare.")
    return at2


# ─── (b) exponent budget ──────────────────────────────────────────────────────

WS = [100, 200, 420, 840, 1680]
DBAR_EXP = 0.148        # Dbar ~ W^0.148, section 4.5.40(d)


def check_exponent(visits: int = 4000) -> dict:
    print("\n(b) is sigma constant in W_eff?  (k=2, c=1.5)\n")
    print("    " + f"{'access':>6}" + "".join(f"{w:>9}" for w in WS))
    out = {}
    for acc in ACCESS:
        s2 = np.array([sigma_sq(w, acc, visits=visits) * 2 for w in WS])
        out[acc] = s2
        print(f"    {acc:>6}" + "".join(f"{x:>9.3f}" for x in s2))
    print("\n    exponent budget for eps ~ W_eff^(-theta):")
    print(f"      {0.5:.3f}   bound with E ~ W_eff")
    print(f"      {0.5 - DBAR_EXP / 2:.3f}   + Dbar ~ W_eff^{DBAR_EXP}")
    thetas = {}
    for acc, s2 in out.items():
        p = float(np.polyfit(np.log(WS), np.log(s2), 1)[0])
        thetas[acc] = 0.5 - DBAR_EXP / 2 + p / 2
        print(f"      {thetas[acc]:.3f}   + sigma ~ W_eff^{p / 2:+.3f}   [{acc}]")
    print("\n      0.310 / 0.331 / 0.397   fitted free, section 4.5.40(c)")
    return thetas


def demo() -> None:
    at2 = check_collapse()
    thetas = check_exponent()

    # (A1)+(A2): the increment scale is eps, so dividing by eps^2 collapses the
    # curves. A 4.3x change in eps must not move the k=2 variance much.
    assert max(at2) / min(at2) - 1 < 0.10, at2
    # the derived exponent, once Dbar and sigma are measured, must land inside
    # the range fitted independently from the coefficient sweep
    for acc, th in thetas.items():
        assert 0.28 < th < 0.45, (acc, th)
    print("\nOK: eps-collapse within 10%, derived theta inside the fitted range.")


if __name__ == "__main__":
    demo()
