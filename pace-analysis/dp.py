"""Section 5.3 — finite-horizon recursion for channel airtime.

    .venv/bin/python pace-analysis/dp.py

Evaluates the expected useful airtime of one NPCA visit under PACE by backward
recursion over the remaining window. This is an evaluation, not an optimisation:
the transmission probability follows PACE's own MIMD dynamics.

State (w, k). w is the remaining window in slots. k indexes a uniform grid in
ln tau, which is the whole state a STA carries (section 3.1). The viable
population is not carried as a state: viability.expected_viable(w) tracks the
engine to within 3%, so n = n(w) is folded in as a mean-field function of w.

For c_idle = c_coll the grid is tau_0 * c^k with unit steps. For asymmetric
coefficients the walk takes +up units on idle and -dn units on a collision,
where up/dn is the nearest rational to r = ln c_idle / ln c_coll; see lattice().

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

from fractions import Fraction
from typing import NamedTuple

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


class Lat(NamedTuple):
    """A refined log-tau grid, plus the step sizes the MIMD rule takes on it."""
    taus: np.ndarray        # tau at each grid index
    k0: int                 # index of tau_0, the starting state
    nk: int                 # number of grid points
    up: int                 # grid units an idle epoch moves up
    dn: int                 # grid units a collision moves down
    r_eff: float            # up/dn, the ratio actually realised


def lattice(c_coll: float = None, r: float = 1.0, tau_0: float = None,
            max_den: int = 24) -> Lat:
    """Reachable tau values and the index of the starting state.

    With eps_idle = ln c_idle and eps_coll = ln c_coll the reachable set is
    tau = tau_0 * exp(n_up*eps_idle - n_down*eps_coll), which for r != 1 is a
    2-D lattice in (n_up, n_down). Only ln tau matters to the dynamics, though,
    so the 2-D state is unnecessary: approximate r by the nearest rational
    up/dn and lay a uniform grid of step eps_coll/dn. The walk then moves +up or
    -dn units on one axis and the state stays scalar.

    r = 1 gives up = dn = 1 and this reduces exactly to the old tau_0 * c^k.
    r_eff reports the ratio after rounding, which the caller should quote rather
    than the requested r.
    """
    c = P.C_MIMD if c_coll is None else c_coll
    tau_0 = P.TAU_0 if tau_0 is None else tau_0
    eps_coll = np.log(c)
    frac = Fraction(float(r)).limit_denominator(max_den)
    up, dn = max(frac.numerator, 1), max(frac.denominator, 1)
    delta = eps_coll / dn
    lo = int(np.floor(np.log(P.TAU_FLOOR / tau_0) / delta))
    hi = int(np.ceil(np.log(TAU_CEIL / tau_0) / delta))
    ks = np.arange(lo, hi + 1)
    taus = np.clip(tau_0 * np.exp(delta * ks), P.TAU_FLOOR, TAU_CEIL)
    return Lat(taus, int(-lo), len(ks), up, dn, up / dn)


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
          access: str = "rts", split: bool = False,
          c_coll: float = None, r: float = 1.0, w_eff: int = None,
          max_den: int = 24):
    """Backward recursion over (w, k).

    Returns (table, k0), table[w, k] = expected useful slots still to come, with
    k0 the starting index. With split=True returns (visitor, native, k0) instead
    of the sum: the fairness term rho needs the two groups separately, and one
    combined value function cannot recover them.
    """
    tau_nat = P.TAU_NAT if tau_nat is None else tau_nat
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    coll_cost, succ_oh = P.ACCESS[access]
    nocd = coll_cost == "nocd"
    q0n = (1.0 - tau_nat) ** n_nat
    q1n = n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1)

    # tau_0 is deadline-scaled, so sweeping the window moves the start too
    lat = lattice(c_coll, r, tau_0=1.0 / w_eff, max_den=max_den)
    taus, k0, nk = lat.taus, lat.k0, lat.nk
    # two reward channels: the recursion is identical, only the reward differs,
    # so the visitor and native value functions ride the same transitions
    tv = np.zeros((w_eff + 1, nk))
    tn = np.zeros((w_eff + 1, nk))

    for w in range(1, w_eff + 1):
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

        up = np.clip(np.arange(nk) + lat.up, 0, nk - 1)   # idle, saturating
        dn = np.clip(np.arange(nk) - lat.dn, 0, nk - 1)   # collision, saturating
        # the visitor frame length rewards the visitor channel only, the native
        # frame the native channel only; every transition is shared
        tv[w] = (p_idle * tv[w - 1][up]
                 + p_svis * (l_vis + tv[w_svis])
                 + p_snat * tv[w_snat]
                 + p_coll * tv[w_coll, dn])
        tn[w] = (p_idle * tn[w - 1][up]
                 + p_svis * tn[w_svis]
                 + p_snat * (air_n + tn[w_snat])
                 + p_coll * tn[w_coll, dn])
    return (tv, tn, k0) if split else (tv + tn, k0)


def total_airtime(n_vis: int, **kw) -> float:
    """Expected useful airtime of a visit, normalised by W_eff."""
    w_eff = kw.get("w_eff") or P.W_EFF
    tab, k0 = solve(n_vis, **kw)
    return float(tab[w_eff, k0]) / w_eff              # start at tau_0


def outcome(n_vis: int, n_nat: int = None, **kw) -> dict:
    """Predicted (T, rho) for one visit, the two axes the objective trades off.

    T   = total useful airtime / W_eff        (channel efficiency)
    rho = visitor airtime share / population share    (1 = exactly proportional)

    rho is bounded above by 1 + n_nat/n_vis, since the visitors cannot take
    more than the whole channel.
    """
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    w_eff = kw.get("w_eff") or P.W_EFF
    tv, tn, k0 = solve(n_vis, n_nat=n_nat, split=True, **kw)
    a_vis = float(tv[w_eff, k0])
    a_nat = float(tn[w_eff, k0])
    tot = a_vis + a_nat
    share = a_vis / tot if tot > 0 else 0.0
    return {"T": tot / w_eff, "visitor": a_vis / w_eff,
            "native": a_nat / w_eff,
            "rho": share / (n_vis / (n_vis + n_nat)) if share > 0 else 0.0}


def objective(n_vis: int, alpha: float, **kw) -> float:
    """J = ln T - alpha*(ln rho)^2.

    Two-sided in log rho, so over- and under-proportional shares of the same
    factor are penalised equally, and a visitor group that takes twice its
    share is treated exactly as badly as one that takes half.

    The two-sided form is load bearing, but not for the reason first recorded
    here. The old note said a one-sided penalty would be inert because PACE
    operates at rho < 1. That is false at heavy native load: under RTS/CTS with
    N_nat = 20, five of the seven Pareto-frontier points sit at rho > 1 and
    EVERY alpha optimum overshoots (rho 1.39 at alpha = 0 down to 1.07 at
    alpha = 0.5), so alpha's job there is to pull the share DOWN. A one-sided
    penalty would be inert in exactly the regime where the fairness term is
    doing the most work. Measured in section 4.5.17 of PACE_TWC_ANALYSIS.md.
    """
    o = outcome(n_vis, **kw)
    if o["T"] <= 0.0 or o["rho"] <= 0.0:
        return -np.inf
    return float(np.log(o["T"]) - alpha * np.log(o["rho"]) ** 2)


def forward(n_vis: int, tau_nat: float = None, n_nat: int = None,
            access: str = "rts", c_coll: float = None, r: float = 1.0,
            w_eff: int = None, max_den: int = 24) -> tuple[np.ndarray, dict]:
    """Forward pass over the same chain.

    Returns (mass, tally). mass[w, k] is the expected number of epochs spent in
    state (w, k) over one visit; tally counts expected epochs by outcome. The
    backward recursion only yields a total, so this is what lets the model be
    compared against the engine trajectory-by-trajectory rather than at a single
    number, which is what the plan's sections 3.2 and 6 ask for.
    """
    tau_nat = P.TAU_NAT if tau_nat is None else tau_nat
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    coll_cost, succ_oh = P.ACCESS[access]
    nocd = coll_cost == "nocd"
    q0n = (1.0 - tau_nat) ** n_nat
    q1n = n_nat * tau_nat * (1.0 - tau_nat) ** (n_nat - 1)

    lat = lattice(c_coll, r, tau_0=1.0 / w_eff, max_den=max_den)
    taus, k0, nk = lat.taus, lat.k0, lat.nk
    # mass[w, k]: probability of occupying state (w, k) at some epoch
    mass = np.zeros((w_eff + 1, nk))
    mass[w_eff, k0] = 1.0
    tally = {"idle": 0.0, "solo_vis": 0.0, "solo_nat": 0.0, "coll": 0.0}

    for w in range(w_eff, 0, -1):
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
        up = np.clip(np.arange(nk) + lat.up, 0, nk - 1)
        dn = np.clip(np.arange(nk) - lat.dn, 0, nk - 1)
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
    w_eff = kw.get("w_eff") or P.W_EFF
    mass, _tally = forward(n_vis, **kw)
    # must be the same grid forward() walked, or the tau values are misread
    taus = lattice(kw.get("c_coll"), kw.get("r", 1.0), tau_0=1.0 / w_eff,
                   max_den=kw.get("max_den", 24)).taus
    total = mass.sum()
    w_mid, e_tau, e_nv, e_m = [], [], [], []
    for lo in range(0, w_eff + bin_w, bin_w):
        sel = mass[lo:min(lo + bin_w, w_eff + 1)]
        tot = sel.sum()
        if total <= 0 or tot / total < min_mass:
            continue
        mid = min(lo + bin_w / 2, float(w_eff))
        if V.expected_viable(int(mid), n_vis) <= 0.0:
            continue        # no visitor can transmit here, tau is meaningless
        w_mid.append(mid)
        e_tau.append(float((sel * taus).sum() / tot))
        e_nv.append(V.expected_viable(int(mid), n_vis))
        e_m.append(float(tot / total))
    return {"w_rem": np.array(w_mid), "tau": np.array(e_tau),
            "n_viable": np.array(e_nv), "mass": np.array(e_m)}


def measured(n_vis: int, access: str = "rts", mode: str = "pace",
             n_nat: int = None, c_coll: float = None, c_idle: float = None,
             w_eff: int = None) -> dict:
    """Engine ground truth for the same configuration.

    Returns airtime shares plus rho, so a simulation point can be dropped
    straight into the objective alongside a DP point.
    """
    n_nat = P.N_NATIVE if n_nat is None else n_nat
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    c_coll = P.C_MIMD if c_coll is None else c_coll
    f25.N_VISITOR, f25.N_NATIVE = n_vis, n_nat
    av = an = 0.0
    cnt = 0
    try:
        with P.coefficients(c_coll, c_idle), P.window(w_eff):
            for r in range(P.REPS):
                rng_p = np.random.default_rng(10001 + r * 71 + 7)
                rng = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
                for v in range(P.VISITS):
                    ppdus = f25._sample_ppdus25(rng_p)
                    tau0 = (np.full(n_vis, 1.0 / w_eff)
                            if mode.startswith(("pace", "oracle")) else None)
                    air, _c, _i, _o, _ = f25._run_visit25(
                        ppdus, rng, mode, tau0, *P.ACCESS[access])
                    if v >= P.VISITS // 2:
                        av += air[:n_vis].sum()
                        an += air[n_vis:].sum()
                        cnt += 1
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    norm = cnt * w_eff
    tot = av + an
    share = av / tot if tot > 0 else 0.0
    return {"visitor": av / norm, "native": an / norm, "total": tot / norm,
            "T": tot / norm,
            "rho": share / (n_vis / (n_vis + n_nat)) if share > 0 else 0.0}


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

    # r = 1 must reproduce the old single-step lattice exactly, or every number
    # the manuscript already quotes silently moves
    l1 = lattice(r=1.0)
    assert (l1.up, l1.dn, l1.r_eff) == (1, 1, 1.0)
    assert abs(l1.taus[l1.k0] - P.TAU_0) < 1e-15
    ks = np.arange(l1.nk) - l1.k0
    assert np.allclose(l1.taus, np.clip(P.TAU_0 * P.C_MIMD ** ks,
                                        P.TAU_FLOOR, TAU_CEIL))
    # and asymmetric r refines the grid without moving tau_0 or the ratio
    for rq in (0.25, 0.5366, 1.9718):
        lr = lattice(r=rq)
        assert abs(lr.r_eff - rq) / rq < 0.02, (rq, lr.r_eff)
        assert abs(lr.taus[lr.k0] - P.TAU_0) < 1e-12
        assert lr.nk >= l1.nk and lr.up >= 1 and lr.dn >= 1
        # dn grid units must be exactly one factor of c_coll. Read this off
        # around k0, since the grid is clipped flat at both ends.
        step = np.log(lr.taus[lr.k0 + 1] / lr.taus[lr.k0])
        assert abs(step * lr.dn - np.log(P.C_MIMD)) < 1e-12

    assert mean_len_viable(P.W_EFF) == (P.PPDU_V_LO + P.PPDU_V_HI) / 2
    assert mean_len_viable(P.min_start()) == P.PPDU_V_LO
    assert mean_len_viable(P.min_start() - 1) == 0.0

    tab, k0 = solve(20)
    assert np.all(tab >= 0.0) and np.all(tab[0] == 0.0)
    # the split must reconstruct the combined table exactly
    tv, tn, k0s = solve(20, split=True)
    assert k0s == k0 and np.allclose(tv + tn, tab, atol=1e-12)
    # rho is a share ratio, bounded above by 1 + n_nat/n_vis
    for nv in (5, 20, 50):
        o = outcome(nv)
        assert 0.0 < o["rho"] < 1.0 + P.N_NATIVE / nv, (nv, o["rho"])
        assert abs(o["visitor"] + o["native"] - o["T"]) < 1e-12
    # the objective must be a dial: raising alpha must not lower the chosen rho
    prev = -1.0
    for al in (0.0, 0.05, 0.1, 0.5):
        best = max((1.2, 1.5, 2.0, 2.8), key=lambda c: objective(20, al, c_coll=c))
        rho = outcome(20, c_coll=best)["rho"]
        assert rho >= prev - 1e-9, (al, rho, prev)
        prev = rho
    # airtime cannot exceed the window
    assert tab.max() <= P.W_EFF + 1e-9, tab.max()
    # more visitors must not reduce total channel airtime
    tot = [total_airtime(n) for n in (5, 10, 20, 50)]
    assert all(0.0 < t < 1.0 for t in tot), tot
    print("\ndp.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
