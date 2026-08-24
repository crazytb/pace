"""Section 5.3 — finite-horizon recursion for channel airtime.

    .venv/bin/python pace-analysis/dp.py

Evaluates the expected useful airtime of one NPCA visit under PACE by backward
recursion over the remaining window. This is an evaluation, not an optimisation:
the transmission probability follows PACE's own MIMD dynamics.

State (w, k). w is the remaining window in slots. k is the lattice index of the
transmission probability, tau = clip(tau_0 * c^k, tau_floor, 1), which is the
whole state a STA carries (section 3.1). The viable population is not carried
as a state: viability.expected_viable(w) tracks the engine to within 3%, so
n = n(w) is folded in as a mean-field function of w.

Per epoch, with n = n(w) visitors at tau and N_nat natives at tau_nat:

    P_idle  = q0v*q0n   cost 1,                k+1, airtime 0
    P_svis  = q1v*q0n   cost E[L|viable]+L_hs, k,   airtime E[L|viable]
    P_snat  = q0v*q1n   cost L_nat+L_hs,       k,   airtime L_nat
    P_coll  = rest      cost L_col,            k-1, airtime 0

The k-transitions mirror the engine exactly: a native solo success advertises no
tau, so listeners leave k alone (run_step9_fig25.py:290), while a collision
pushes every listener down whoever caused it (:294-297).

ponytail: the visitor success cost uses E[L | viable] rather than summing over
the 76 frame lengths. Costs enter linearly here, so the mean is exact for the
airtime term and approximate only in where it lands the next state. Swap in the
full sum if the residual demands it.
"""
from __future__ import annotations

import numpy as np

import params as P
import viability as V

f25 = P.engine()

K_LO = int(np.floor(np.log(P.TAU_FLOOR / P.TAU_0) / np.log(P.C_MIMD)))   # -18
K_HI = int(np.ceil(np.log(1.0 / P.TAU_0) / np.log(P.C_MIMD)))            # +33


# n(w) is a mean-field float, so tau = 1 would evaluate (1-tau)^(n-1) as a
# negative power of zero. The engine's ceiling is 1.0 but k that high is never
# reached (measured tau_end ~ 0.013), so clipping just below is harmless.
TAU_CEIL = 1.0 - 1e-9


def tau_of(k: int) -> float:
    return float(np.clip(P.TAU_0 * P.C_MIMD ** k, P.TAU_FLOOR, TAU_CEIL))


def mean_len_viable(w: int) -> float:
    """E[L | L + L_hs <= w] for L ~ U{LO..HI}; 0 if nothing fits."""
    hi = min(int(w - P.L_HS), P.PPDU_V_HI)
    return 0.0 if hi < P.PPDU_V_LO else (P.PPDU_V_LO + hi) / 2.0


def basic_collision_cost(w: int, pv: np.ndarray, n_vis: int,
                         tau_nat: float, n_nat: int) -> np.ndarray:
    """Expected collision cost under basic access: the longest colliding frame.

    Under "nocd" the engine charges max L_i over the colliders
    (run_step9_fig25.py:268), so the cost is an order statistic, not a mean.
    Conditioned on a collision the expected number of colliders is

        E[M | M >= 2] = (mu - P(M=1)) / P(M >= 2),   mu = n_vis*pv + n_nat*tau_nat

    and for m draws from a continuous U[lo, hi] the expected maximum is
    lo + (hi-lo)*m/(m+1). Natives contribute a fixed 50-slot frame, folded in
    as a floor once a native is likely to be among the colliders.

    ponytail: this uses E[max | E[M]] rather than E[max | M] summed over the
    collider-count distribution, and treats the truncated frame length as
    continuous. Both are second-order next to replacing the plain mean; do the
    exact sum only if the residual still demands it.
    """
    lo = P.PPDU_V_LO
    hi = min(int(w - P.L_HS), P.PPDU_V_HI)
    if hi < lo:                                   # no visitor frame fits
        return np.full_like(pv, float(P.PPDU_NATIVE))
    mu_v, mu_n = n_vis * pv, n_nat * tau_nat
    mu = mu_v + mu_n
    p0 = (1 - pv) ** n_vis * (1 - tau_nat) ** n_nat
    p1 = (n_vis * pv * (1 - pv) ** (n_vis - 1) * (1 - tau_nat) ** n_nat
          + (1 - pv) ** n_vis * n_nat * tau_nat * (1 - tau_nat) ** (n_nat - 1))
    p2 = np.maximum(1.0 - p0 - p1, 1e-12)
    m = np.clip((mu - p1) / p2, 2.0, float(n_vis + n_nat))
    e_max_v = lo + (hi - lo) * m / (m + 1.0)
    # a native is among the colliders with probability ~ mu_n/mu; its frame is
    # a constant PPDU_NATIVE, so the overall max is at least that when likely
    frac_n = np.clip(mu_n / np.maximum(mu, 1e-12), 0.0, 1.0)
    return np.maximum(e_max_v, frac_n * P.PPDU_NATIVE)


def solve(n_vis: int, tau_nat: float = None, n_nat: int = None,
          access: str = "rts") -> np.ndarray:
    """Backward recursion. Returns table[w, k-K_LO] = expected useful slots."""
    tau_nat = P.TAU_NAT if tau_nat is None else tau_nat
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    coll_cost, succ_oh = P.ACCESS[access]
    nocd = coll_cost == "nocd"
    q0n = (1.0 - tau_nat) ** n_nat
    q1n = n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1)

    nk = K_HI - K_LO + 1
    tab = np.zeros((P.W_EFF + 1, nk))
    taus = np.array([tau_of(k) for k in range(K_LO, K_HI + 1)])

    for w in range(1, P.W_EFF + 1):
        # Each visitor is independently viable with probability p_via and, if
        # viable, transmits with probability tau. Keeping n_vis an integer and
        # folding viability into the per-STA rate avoids the fractional-count
        # mean field, whose (1-tau)^(n-1) diverges once n drops below one.
        p_via = V.p_viable(w)
        l_vis = mean_len_viable(w)
        # visitor success: viable by definition, so the frame always fits
        w_svis = max(w - int(round(l_vis)) - succ_oh, 0)
        # native success: no fit check, the engine truncates at the window end
        occupy_n = min(P.PPDU_NATIVE + succ_oh, w)
        air_n = max(occupy_n - min(succ_oh, occupy_n), 0)
        w_snat = w - occupy_n
        pv = p_via * taus
        q0v = (1.0 - pv) ** n_vis
        q1v = n_vis * pv * (1.0 - pv) ** (n_vis - 1)
        p_idle = q0v * q0n
        p_svis = q1v * q0n
        p_snat = q0v * q1n
        p_coll = np.maximum(1.0 - p_idle - p_svis - p_snat, 0.0)

        cost_c = (basic_collision_cost(w, pv, n_vis, tau_nat, n_nat) if nocd
                  else np.full(nk, float(coll_cost)))
        w_coll = np.maximum(w - np.minimum(cost_c, w), 0).astype(int)

        up = np.clip(np.arange(nk) + 1, 0, nk - 1)      # k+1, saturating
        dn = np.clip(np.arange(nk) - 1, 0, nk - 1)      # k-1, saturating
        tab[w] = (p_idle * tab[w - 1][up]
                  + p_svis * (l_vis + tab[w_svis])
                  + p_snat * (air_n + tab[w_snat])
                  + p_coll * tab[w_coll, dn])
    return tab


def total_airtime(n_vis: int, **kw) -> float:
    """Expected useful airtime of a visit, normalised by W_eff."""
    tab = solve(n_vis, **kw)
    return float(tab[P.W_EFF, -K_LO]) / P.W_EFF        # start at k=0, tau=tau_0


def forward(n_vis: int, tau_nat: float = None, n_nat: int = None,
            access: str = "rts") -> tuple[np.ndarray, dict]:
    """Forward pass over the same chain.

    Returns (mass, tally). mass[w, k] is the expected number of epochs spent in
    state (w, k) over one visit; tally counts expected epochs by outcome. The
    backward recursion only yields a total, so this is what lets the model be
    compared against the engine trajectory-by-trajectory rather than at a single
    number, which is what the plan's sections 3.2 and 6 ask for.
    """
    tau_nat = P.TAU_NAT if tau_nat is None else tau_nat
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    coll_cost, succ_oh = P.ACCESS[access]
    nocd = coll_cost == "nocd"
    q0n = (1.0 - tau_nat) ** n_nat
    q1n = n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1)

    nk = K_HI - K_LO + 1
    taus = np.array([tau_of(k) for k in range(K_LO, K_HI + 1)])
    # mass[w, k]: probability of occupying state (w, k) at some epoch
    mass = np.zeros((P.W_EFF + 1, nk))
    mass[P.W_EFF, -K_LO] = 1.0
    tally = {"idle": 0.0, "solo_vis": 0.0, "solo_nat": 0.0, "coll": 0.0}

    for w in range(P.W_EFF, 0, -1):
        m = mass[w]
        if m.sum() <= 1e-15:
            continue
        p_via = V.p_viable(w)
        l_vis = mean_len_viable(w)
        pv = p_via * taus
        q0v = (1.0 - pv) ** n_vis
        q1v = n_vis * pv * (1.0 - pv) ** (n_vis - 1)
        p = {"idle": q0v * q0n, "solo_vis": q1v * q0n, "solo_nat": q0v * q1n}
        p["coll"] = np.maximum(1.0 - sum(p.values()), 0.0)
        for key, pk in p.items():
            tally[key] += float((m * pk).sum())
        cost_c = (basic_collision_cost(w, pv, n_vis, tau_nat, n_nat) if nocd
                  else np.full(nk, float(coll_cost)))
        occupy_n = min(P.PPDU_NATIVE + succ_oh, w)
        up = np.clip(np.arange(nk) + 1, 0, nk - 1)
        dn = np.clip(np.arange(nk) - 1, 0, nk - 1)
        np.add.at(mass[w - 1], up, m * p["idle"])
        nxt = max(w - int(round(l_vis)) - succ_oh, 0)
        if nxt > 0:
            mass[nxt] += m * p["solo_vis"]
        if w - occupy_n > 0:
            mass[w - occupy_n] += m * p["solo_nat"]
        wc = np.maximum(w - np.minimum(cost_c, w), 0).astype(int)
        for j in range(nk):
            if wc[j] > 0:
                mass[wc[j], dn[j]] += m[j] * p["coll"][j]
    return mass, tally


def outcome_mix(n_vis: int, **kw) -> dict:
    """Predicted epoch-outcome fractions plus the expected epoch count."""
    _mass, tally = forward(n_vis, **kw)
    tot = sum(tally.values())
    return {k: v / tot for k, v in tally.items()} | {"epochs": tot}


def tau_trajectory(n_vis: int, bin_w: int = 20, min_mass: float = 0.005,
                   **kw) -> dict:
    """Predicted E[tau | W_rem] and E[|V| | W_rem], bucketed for plotting.

    Each bucket's tau is weighted by the occupancy mass. Buckets holding less
    than min_mass of the total are dropped: with the large jumps a success or a
    collision makes, some W_rem ranges are reachable only by an improbably long
    idle run, so their conditional tau is both huge and meaningless. Under basic
    access one such bucket carried 0.00% of the model's mass and 5 of the
    simulation's 6416 samples, and left alone it dominated the error metric and
    put a spurious spike in the figure.
    """
    mass, _tally = forward(n_vis, **kw)
    taus = np.array([tau_of(k) for k in range(K_LO, K_HI + 1)])
    total = mass.sum()
    w_mid, e_tau, e_nv, e_m = [], [], [], []
    for lo in range(0, P.W_EFF + bin_w, bin_w):
        sel = mass[lo:min(lo + bin_w, P.W_EFF + 1)]
        tot = sel.sum()
        if total <= 0 or tot / total < min_mass:
            continue
        mid = min(lo + bin_w / 2, float(P.W_EFF))
        if V.expected_viable(int(mid), n_vis) <= 0.0:
            continue        # no visitor can transmit here, tau is meaningless
        w_mid.append(mid)
        e_tau.append(float((sel * taus).sum() / tot))
        e_nv.append(V.expected_viable(int(mid), n_vis))
        e_m.append(float(tot / total))
    return {"w_rem": np.array(w_mid), "tau": np.array(e_tau),
            "n_viable": np.array(e_nv), "mass": np.array(e_m)}


def measured(n_vis: int, access: str = "rts", mode: str = "pace") -> dict:
    """Engine ground truth for the same configuration."""
    f25.N_VISITOR, f25.N_NATIVE = n_vis, P.N_NATIVE
    av = an = 0.0
    cnt = 0
    try:
        for r in range(P.REPS):
            rng_p = np.random.default_rng(10001 + r * 71 + 7)
            rng = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
            for v in range(P.VISITS):
                ppdus = f25._sample_ppdus25(rng_p)
                tau0 = (np.full(n_vis, P.TAU_0)
                        if mode.startswith(("pace", "oracle")) else None)
                air, _c, _i, _o, _ = f25._run_visit25(
                    ppdus, rng, mode, tau0, *P.ACCESS[access])
                if v >= P.VISITS // 2:
                    av += air[:n_vis].sum()
                    an += air[n_vis:].sum()
                    cnt += 1
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    norm = cnt * P.W_EFF
    return {"visitor": av / norm, "native": an / norm, "total": (av + an) / norm}


def _main() -> None:
    print(f"state space: w in [0,{P.W_EFF}], k in [{K_LO},{K_HI}] "
          f"-> {(P.W_EFF + 1) * (K_HI - K_LO + 1):,} states")
    for access in ("rts", "basic"):
        print(f"\n=== {access} ===")
        print(f"{'N_vis':>6} {'DP total':>9} {'engine':>8} {'err':>8} "
              f"{'eng vis':>8} {'eng nat':>8}")
        for n_vis in (5, 10, 20, 50):
            dp = total_airtime(n_vis, access=access)
            m = measured(n_vis, access=access)
            print(f"{n_vis:>6} {dp:9.3f} {m['total']:8.3f} "
                  f"{dp - m['total']:+8.3f} {m['visitor']:8.3f} "
                  f"{m['native']:8.3f}")


def _self_check() -> None:
    assert K_LO < 0 < K_HI
    assert abs(tau_of(0) - P.TAU_0) < 1e-15
    assert tau_of(K_LO - 5) == P.TAU_FLOOR and tau_of(K_HI + 5) == TAU_CEIL
    assert mean_len_viable(P.W_EFF) == (P.PPDU_V_LO + P.PPDU_V_HI) / 2
    assert mean_len_viable(P.min_start()) == P.PPDU_V_LO
    assert mean_len_viable(P.min_start() - 1) == 0.0

    tab = solve(20)
    assert np.all(tab >= 0.0) and np.all(tab[0] == 0.0)
    # airtime cannot exceed the window
    assert tab.max() <= P.W_EFF + 1e-9, tab.max()
    # more visitors must not reduce total channel airtime
    tot = [total_airtime(n) for n in (5, 10, 20, 50)]
    assert all(0.0 < t < 1.0 for t in tot), tot
    print("\ndp.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
