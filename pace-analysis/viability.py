"""Section 5.1-5.2 — the viable population and the structurally dead tail.

    .venv/bin/python pace-analysis/viability.py

A visitor may start an exchange only while its frame still fits:

    i in V(t)  <=>  L_i + L_hs <= W_rem(t)          (run_step9_fig25.py:213)

With L ~ U{PPDU_V_LO..PPDU_V_HI} the marginal expectation is closed form,

    E|V(t)| = N_vis * F_L(W_rem(t) - L_hs)

and it gives the FS target line 1/|V(t)| analytically instead of by simulation.

Two consequences the manuscript uses:

  * the window has a dead tail. Below W_rem = PPDU_V_LO + L_hs = 35 slots no
    visitor can start at all, so the last 8.3% of W_eff is structurally lost.
    (The plan's first draft put this at 27% by reading L_hs as 88 slots rather
     than 88 us; see PACE_TWC_ANALYSIS.md section 1.)
  * viability is never regained. A STA's frame is fixed until it succeeds and
    W_rem only shrinks, so |V(t)| is non-increasing and the FS target 1/|V(t)|
    is non-decreasing. That monotonicity is what the section 4.4 lemma leans on.

ponytail: the closed form treats the n viable STAs as independent draws. Under
saturated traffic a STA redraws L after each success, which correlates its
viability with the epoch it last won. The check below measures the resulting
error against the engine rather than assuming it away (plan section 7).
"""
from __future__ import annotations

import math

import numpy as np

import params as P

f25 = P.engine()
N_LEN = P.PPDU_V_HI - P.PPDU_V_LO + 1          # 76 equiprobable frame lengths


def f_len(x: float) -> float:
    """CDF of the visitor PPDU length: P(L <= x) for L ~ U{LO..HI}."""
    return min(max((math.floor(x) - P.PPDU_V_LO + 1) / N_LEN, 0.0), 1.0)


def p_viable(w_rem: int) -> float:
    """Probability that one visitor's current frame still fits."""
    return f_len(w_rem - P.L_HS)


def expected_viable(w_rem: int, n_vis: int) -> float:
    """E|V(t)| given the remaining window."""
    return n_vis * p_viable(w_rem)


def survival(w_from: int, w_to: int) -> float:
    """P(a STA viable at w_from is still viable at w_to), w_to <= w_from.

    The DP needs this rather than the marginal: conditioning on being viable
    now truncates the frame-length distribution.
    """
    num, den = f_len(w_to - P.L_HS), f_len(w_from - P.L_HS)
    return 0.0 if den <= 0.0 else min(num / den, 1.0)


def fs_target(w_rem: int, n_vis: int, n_nat: int = None) -> float:
    """Fair-share transmission probability 1/k over all contenders."""
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    k = expected_viable(w_rem, n_vis) + n_nat
    return 1.0 / k if k > 0 else 0.0


def dead_tail() -> tuple[int, float]:
    """(smallest W_rem at which a visitor can start, its fraction of W_eff)."""
    return P.min_start(), P.dead_fraction()


def measure_viable(n_vis: int, mode: str = "pace",
                   access: str = "rts") -> dict[int, float]:
    """Engine ground truth: mean observed |V| bucketed by W_rem."""
    f25.N_VISITOR, f25.N_NATIVE = n_vis, P.N_NATIVE
    buckets: dict[int, list] = {}
    try:
        for r in range(P.REPS):
            rng_p = np.random.default_rng(10001 + r * 71 + 7)
            rng = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
            for v in range(P.VISITS):
                ppdus = f25._sample_ppdus25(rng_p)
                tau0 = (np.full(n_vis, P.TAU_0)
                        if mode.startswith(("pace", "oracle")) else None)
                st: dict = {"trace": []}
                f25._run_visit25(ppdus, rng, mode, tau0,
                                 *P.ACCESS[access], stats=st)
                if v < P.VISITS // 2:
                    continue
                for w_rem, nvv, _k, _rate in st["trace"]:
                    buckets.setdefault(w_rem // 20 * 20, []).append(nvv)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    return {w: float(np.mean(v)) for w, v in sorted(buckets.items())}


def _main() -> None:
    start, frac = dead_tail()
    print(f"visitor PPDU L ~ U{{{P.PPDU_V_LO}..{P.PPDU_V_HI}}}, "
          f"L_hs = {P.L_HS} slots")
    print(f"min W_rem to start = {start} slots -> dead tail "
          f"{frac * 100:.1f}% of W_eff ({P.W_EFF} slots)")
    print(f"all viable while W_rem >= {P.PPDU_V_HI + P.L_HS} "
          f"(t <= {P.W_EFF - P.PPDU_V_HI - P.L_HS})")

    for n_vis in (20, 50):
        obs = measure_viable(n_vis)
        print(f"\n=== N_vis={n_vis}: closed form vs engine ===")
        print(f"{'W_rem':>7} {'E|V| model':>11} {'engine':>8} {'err':>7} "
              f"{'FS target':>10}")
        for w in sorted(obs, reverse=True):
            pred = expected_viable(w + 10, n_vis)     # bucket midpoint
            print(f"{w:>7} {pred:11.2f} {obs[w]:8.2f} "
                  f"{pred - obs[w]:+7.2f} {fs_target(w + 10, n_vis):10.4f}")


def _self_check() -> None:
    # CDF endpoints
    assert f_len(P.PPDU_V_LO - 1) == 0.0
    assert abs(f_len(P.PPDU_V_LO) - 1 / N_LEN) < 1e-12
    assert f_len(P.PPDU_V_HI) == 1.0 and f_len(P.W_EFF) == 1.0

    # dead tail matches section 5.2 after the unit correction
    start, frac = dead_tail()
    assert start == 35 and abs(frac - 35 / 420) < 1e-12
    assert expected_viable(start - 1, 20) == 0.0
    assert expected_viable(P.PPDU_V_HI + P.L_HS, 20) == 20.0

    # |V| is non-increasing as the window drains, so the FS target rises
    ws = list(range(P.W_EFF, 0, -1))
    vs = [expected_viable(w, 20) for w in ws]
    assert all(a >= b - 1e-12 for a, b in zip(vs, vs[1:])), "|V| not monotone"
    ts = [fs_target(w, 20) for w in ws]
    assert all(a <= b + 1e-12 for a, b in zip(ts, ts[1:])), "target not monotone"

    # conditional survival is a probability and is exact at the endpoints
    assert survival(P.W_EFF, P.W_EFF) == 1.0
    assert survival(P.W_EFF, start - 1) == 0.0
    assert 0.0 < survival(200, 100) < 1.0

    # the closed form must track the engine within a couple of stations
    for n_vis in (20, 50):
        obs = measure_viable(n_vis)
        errs = [abs(expected_viable(w + 10, n_vis) - obs[w]) for w in obs]
        rel = max(errs) / n_vis
        assert rel < 0.12, f"N_vis={n_vis}: worst relative error {rel:.3f}"
    print("\nviability.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
