"""Section 4.5.5 — the finite-horizon layer: the coefficient scale, and how far
the analytically chosen ratio falls short of a two-dimensional search.

    .venv/bin/python pace-analysis/optimise.py

The obvious reduction is the one equilibrium.py supports: fix the ratio r at
its drift value r* and search the remaining scale. It does not work, and the
way it fails is the result.

    r*  = equilibrium.r_star(tau_J)                  closed form
    s*  = argmax_s J_FH(exp(r* s), exp(s))           1-D, over the DP
    G_J = exp(J_analytic) / exp(J_oracle)  in (0, 1]

Measured against a two-dimensional engine oracle, G_J = 0.83-0.96: r* is
systematically too SMALL (0.26-0.65 against an engine optimum near 1.0-1.5).
The reason is Theorem 2. r* places the equilibrium correctly, but a visit ends
long before the walk arrives, and forcing a small r shrinks eps_idle = r*eps_coll,
which is what drives the ramp. The scale search then runs to its bracket trying
to compensate. That runaway and the one in section 4.5.3 are the same event.

What the engine actually says is simpler. On a grid over (eps_idle, eps_coll),
J is sharply peaked in eps_idle and nearly flat in eps_coll: over a 3.6x sweep
of eps_coll the objective moves by 0.03-0.09 under RTS/CTS, while the same
sweep in eps_idle moves it by up to 1.5. The design variable is the UP step,

    eps_idle* = C(N_vis, alpha, mode) / sqrt(W_eff)

with the exponent measured at -0.44 to -0.49 across scenarios. That is the
diffusion balance: reach grows like eps*E and jitter like eps*sqrt(E), so the
scale that trades them off goes as 1/sqrt(E) ~ 1/sqrt(W_eff). Section 4.5.1
derived this correctly and attached it to the wrong coefficient.

eps_coll is not identifiable within one visit, which is exactly why the closed
form for it diverged. A flat objective has no interior optimum to find.

ponytail: the DP oracle is the default because a DP-vs-DP comparison isolates
the question being asked from the DP's own residual against the engine, which
validate.py measures separately. sim_oracle() and eps_idle_star() run the
engine, for the claims that have to hold on the real dynamics.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

import dp
import equilibrium as EQ
import params as P

# c_coll from a hair above 1 to 4. Below S_LO the walk cannot move within the
# window at all; above S_HI a single collision wipes out more than a decade of
# tau and the lattice picture stops meaning anything.
S_LO, S_HI = math.log(1.02), math.log(4.0)
R_LO, R_HI = 0.05, 6.0


def _den_for(s: float, max_den: int = 24, nk_cap: int = 1200) -> int:
    """Rational-approximation denominator that keeps the grid affordable.

    The grid has eps_coll/den spacing over a fixed span of ln tau, so nk grows
    like den/s. Small s already gives a fine grid, so coarsen the approximation
    of r there rather than paying for both.
    """
    span = math.log(dp.TAU_CEIL / P.TAU_FLOOR)
    return max(1, min(max_den, int(nk_cap * s / span)))


def j_fh(n_vis: int, alpha: float, r: float, s: float, **kw) -> float:
    """Finite-horizon objective at coefficients (e^{r s}, e^{s})."""
    if not (S_LO * 0.5 <= s <= S_HI * 1.5):
        return -math.inf
    return dp.objective(n_vis, alpha, c_coll=math.exp(s), r=r,
                        max_den=_den_for(s), **kw)


def s_star(n_vis: int, alpha: float, r: float, **kw) -> float:
    """The scale, given the ratio. One bounded scalar search over the DP."""
    res = minimize_scalar(lambda s: -j_fh(n_vis, alpha, r, s, **kw),
                          bounds=(S_LO, S_HI), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def analytic(n_vis: int, alpha: float, **kw) -> dict:
    """The two-step design: r* in closed form, then s* over the DP."""
    pop = {k: v for k, v in kw.items() if k in ("tau_nat", "n_nat")}
    eq = EQ.design(n_vis, alpha, **{k: v for k, v in kw.items()
                                    if k in ("tau_nat", "n_nat", "access")})
    r = min(max(eq["r_star"], R_LO), R_HI)
    s = s_star(n_vis, alpha, r, **kw)
    return {"r": r, "s": s, "c_coll": math.exp(s), "c_idle": math.exp(r * s),
            "J": j_fh(n_vis, alpha, r, s, **kw),
            "tau_J": eq["tau_J"], "x_J": eq["x_J"], "r_star_raw": eq["r_star"],
            **{f"eq_{k}": v for k, v in eq.items() if k in ("T_eq", "rho_eq")},
            **pop}


def oracle(n_vis: int, alpha: float, n_r: int = 13, **kw) -> dict:
    """Two-dimensional search over (r, s), the benchmark r* is measured against.

    Coarse geometric sweep in r with a nested exact scale search, which is the
    cheap way round: s_star is one bounded scalar solve and r is the axis whose
    shape is in question.
    """
    grid = np.geomspace(R_LO, R_HI, n_r)
    best = None
    for r in grid:
        s = s_star(n_vis, alpha, float(r), **kw)
        j = j_fh(n_vis, alpha, float(r), s, **kw)
        if best is None or j > best["J"]:
            best = {"r": float(r), "s": s, "J": j,
                    "c_coll": math.exp(s), "c_idle": math.exp(float(r) * s)}
    return best


def gap(n_vis: int, alpha: float, **kw) -> dict:
    """G_J and the ratio the finite horizon actually wanted."""
    a, o = analytic(n_vis, alpha, **kw), oracle(n_vis, alpha, **kw)
    return {"G_J": math.exp(a["J"] - o["J"]),
            "r_analytic": a["r"], "r_oracle": o["r"],
            "c_coll_a": a["c_coll"], "c_idle_a": a["c_idle"],
            "c_coll_o": o["c_coll"], "c_idle_o": o["c_idle"],
            "J_a": a["J"], "J_o": o["J"], "x_J": a["x_J"]}


# ─── the design variable that does close: the up step ────────────────────────

EI_GRID = np.exp(np.linspace(math.log(0.12), math.log(1.10), 13))


def sim_J(n_vis: int, alpha: float, c_idle: float, c_coll: float,
          access: str = "rts", w_eff: int = None, n_nat: int = None,
          **kw) -> float:
    """The objective as the engine realises it, not as the DP predicts it."""
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    m = dp.measured(n_vis, access=access, c_coll=c_coll, c_idle=c_idle,
                    w_eff=w_eff, n_nat=n_nat, **kw)
    if m["T"] <= 0.0 or m["rho"] <= 0.0:
        return -math.inf
    return math.log(m["T"]) - alpha * math.log(m["rho"]) ** 2


def dp_J(n_vis: int, alpha: float, c_idle: float, c_coll: float,
         access: str = "rts", w_eff: int = None, n_nat: int = None,
         **kw) -> float:
    """The same objective as sim_J, predicted by the DP instead of measured.

    The DP parameterises the walk by (c_coll, r), so the up step enters as
    r = ln c_idle / ln c_coll. Everything else matches sim_J exactly, which is
    what makes the two comparable curve by curve.
    """
    r = math.log(c_idle) / math.log(c_coll)
    return dp.objective(n_vis, alpha, c_coll=c_coll, r=r, access=access,
                        w_eff=w_eff, n_nat=n_nat, max_den=12, **kw)


def eps_idle_star(n_vis: int, alpha: float, w_eff: int = None,
                  access: str = "rts", c_coll: float = 1.4,
                  grid: np.ndarray = None, band: float = 0.05, **kw) -> float:
    """Engine-optimal up step, located by a least-squares parabola in log eps.

    The peak is broad: at W_eff = 420 between two and five grid points sit
    within 0.02 of the maximum. A three-point parabolic refinement around the
    raw argmax therefore reads mostly noise, and fitting every point within
    `band` of the peak instead uses the shape rather than the tip.

    eps_coll is held fixed. The engine says J barely depends on it inside the
    homogeneous range (see dispersion()), and sweeping both would only put the
    unidentifiable axis back into a measurement of the identifiable one.
    """
    grid = EI_GRID if grid is None else grid
    js = np.array([sim_J(n_vis, alpha, math.exp(float(e)), c_coll,
                         access=access, w_eff=w_eff, **kw) for e in grid])
    lg = np.log(grid)
    sel = js >= js.max() - band
    if sel.sum() >= 3:
        a2, a1, _a0 = np.polyfit(lg[sel], js[sel], 2)
        if a2 < 0:                                  # a genuine interior peak
            v = -a1 / (2 * a2)
            if lg[sel].min() <= v <= lg[sel].max():
                return float(math.exp(v))
    return float(grid[int(js.argmax())])


def scaling_law(n_vis: int, alpha: float, access: str = "rts",
                windows: tuple = (150, 300, 420, 840, 1680), **kw) -> dict:
    """Fit eps_idle* = C * W_eff^b over the engine, and report the fit quality.

    b is the claim: the diffusion balance predicts -1/2. C absorbs the log
    distance to the operating point, which is scenario dependent.
    """
    es = [eps_idle_star(n_vis, alpha, w_eff=w, access=access, **kw)
          for w in windows]
    b, a = np.polyfit(np.log(windows), np.log(es), 1)
    c_half = float(np.mean([e * math.sqrt(w) for e, w in zip(es, windows)]))
    err = [abs(e - c_half / math.sqrt(w)) / e for e, w in zip(es, windows)]
    return {"windows": list(windows), "eps": es, "exponent": float(b),
            "C_fit": float(math.exp(a)), "C_half": c_half,
            "mean_err": float(np.mean(err)), "max_err": float(max(err))}


def collapse(n_vis: int, alpha: float, access: str = "rts",
             windows: tuple = (150, 300, 420, 840, 1680),
             grid: np.ndarray = None, c_coll: float = 1.4,
             band: float = 0.35, thetas: np.ndarray = None,
             jfun=None, **kw) -> dict:
    """Locate the scaling exponent by data collapse rather than by argmax.

    scaling_law() reads one number off each window, the position of a peak that
    is broad enough that the reading is mostly noise. This uses every measured
    point instead. Subtract each window's own maximum, so every curve peaks at
    zero and only its SHAPE is left, then rescale the abscissa

        u = eps_idle * W_eff^theta

    and ask which theta makes the curves lie on top of one another. If the
    diffusion balance holds, theta = 1/2 collapses them; if no theta does, the
    curves differ in width and there is no scaling law to find.

    Collapse quality is the residual of one quadratic in ln u fitted to the
    pooled points. Restricting to within `band` of each peak keeps the fit on
    the region a quadratic can describe: far tails are not parabolic and would
    otherwise decide the answer.

    Returns the best theta, the residual as a function of theta (so the width
    of the minimum can be read as an uncertainty), and the pooled points for
    plotting.
    """
    grid = EI_GRID if grid is None else grid
    thetas = np.linspace(0.0, 1.0, 51) if thetas is None else thetas
    jfun = sim_J if jfun is None else jfun          # swap in dp_J to predict
    curves, dropped = {}, []
    lg = np.log(grid)
    for w in windows:
        js = np.array([jfun(n_vis, alpha, math.exp(float(e)), c_coll,
                            access=access, w_eff=w, **kw) for e in grid])
        js = js - js.max()                      # peak at zero, shape only
        # A window too short to adapt in has no optimal adaptation rate to
        # measure. At W_eff = 150 a visit fits one or two frames and the curve
        # is flat across the whole feasible range: the fitted curvature is
        # -0.02 and the vertex extrapolates to eps = 22, outside any usable
        # grid. Including such a window does not weaken the collapse, it makes
        # the question meaningless, so drop it and say so.
        sel = js >= -band
        ok = False
        if sel.sum() >= 3:
            a2, a1, _ = np.polyfit(lg[sel], js[sel], 2)
            ok = a2 < -0.1 and lg.min() <= -a1 / (2 * a2) <= lg.max()
        # An interior peak also has to fall away on BOTH sides inside the grid.
        # The curvature test alone passes a ragged, near-flat curve: at
        # W_eff = 150 under basic access the objective is multi-modal and its
        # right edge is only 0.04 below the maximum, so a local quadratic finds
        # spurious curvature.
        top = int(np.argmax(js))
        ok = ok and 0 < top < len(js) - 1 \
            and js[:top].min() <= -0.10 and js[top + 1:].min() <= -0.10
        (curves.setdefault(w, js) if ok else dropped.append(w))

    def residual(theta: float) -> float:
        u, y = [], []
        for w, js in curves.items():
            sel = js >= -band
            u.extend(np.log(grid[sel]) + theta * math.log(w))
            y.extend(js[sel])
        if len(u) < 6:
            return math.inf
        u, y = np.array(u), np.array(y)
        fit = np.polyval(np.polyfit(u, y, 2), u)
        return float(np.sqrt(np.mean((y - fit) ** 2)))

    if len(curves) < 3:
        return {"theta": float("nan"), "rms": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "rms_at_0": float("nan"), "gain": 1.0,
                "u_star": float("nan"),
                "curves": curves, "grid": grid, "dropped": dropped,
                "windows": sorted(curves), "residual": None, "thetas": thetas}
    res = np.array([residual(float(t)) for t in thetas])
    best = float(thetas[int(np.argmin(res))])
    # width of the minimum: the thetas whose residual is within 10% of the best
    near = thetas[res <= res.min() * 1.1]
    r0 = residual(0.0)

    # the master curve's peak, at theta = 1/2. This is the constant C in
    # eps_idle* = C / sqrt(W_eff), read off the collapse rather than fitted
    # window by window.
    u, y = [], []
    for w, js in curves.items():
        sel = js >= -band
        u.extend(np.log(grid[sel]) + 0.5 * math.log(w))
        y.extend(js[sel])
    a2, a1, _ = np.polyfit(np.array(u), np.array(y), 2)
    u_star = math.exp(-a1 / (2 * a2)) if a2 < 0 else float("nan")

    return {"theta": best, "residual": res, "thetas": thetas,
            "rms": float(res.min()), "rms_at_0": float(r0),
            "gain": float(r0 / res.min()) if res.min() > 0 else float("inf"),
            "lo": float(near.min()), "hi": float(near.max()),
            "u_star": float(u_star),
            "curves": curves, "grid": grid, "dropped": dropped,
            "windows": sorted(curves)}


# The design constant, calibrated at alpha = 0.2 over nine scenarios (section
# 4.5.8). It moves by 15% across populations and depends only weakly on the
# access mode, so two numbers cover everything measured.
C_DESIGN = {"rts": 10.15, "basic": 7.92}
SHIPPED = (1.2, 1.2)                            # (c_coll, c_idle) in the engine


def design_coefficients(access: str = "rts", w_eff: int = None,
                        c_coll: float = 1.4) -> tuple[float, float]:
    """(c_coll, c_idle) from the design rule eps_idle = C / sqrt(W_eff).

    c_coll is an argument rather than a derived value because the finite window
    does not identify it from J (section 4.5.4): the objective moves by under
    0.07 across a 3.6x sweep. That is NOT the same as irrelevant, though. It
    moves the allocation a lot, rho running 0.98 to 0.69 as c_coll goes 1.2 to
    2.0, and J is flat only because the airtime and fairness effects cancel.

    There is deliberately no alpha argument. C does drift with alpha when it is
    fitted (up to 1.75x over alpha in [0.05, 0.5]), but making the rule
    alpha-aware measured WORSE than leaving it fixed: mean G_J 0.980 against
    0.998, with 5 of 30 cases below 0.95 against none (section 4.5.10). The
    objective is broad enough in eps_idle that one value serves every alpha,
    so alpha selects the outcome, not the coefficient.
    """
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    return c_coll, math.exp(C_DESIGN[access] / math.sqrt(w_eff))


def design_gain(n_vis: int, access: str = "rts", w_eff: int = None,
                n_nat: int = None, alphas: tuple = (0.05, 0.2),
                c_coll: float = 1.4) -> dict:
    """Engine-measured outcome of the design rule against the shipped constants.

    Reports the PRIMITIVE quantities, total useful airtime and the visitor's
    share ratio, and not only J. J was defined in this work, so a comparison
    made solely in J is close to circular: of course coefficients chosen to
    maximise it score well on it. T and rho are checkable independently of
    whether a reader accepts the objective.
    """
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    cc, ci = design_coefficients(access, w_eff, c_coll)
    base = dp.measured(n_vis, access=access, c_coll=SHIPPED[0],
                       c_idle=SHIPPED[1], w_eff=w_eff, n_nat=n_nat)
    new = dp.measured(n_vis, access=access, c_coll=cc, c_idle=ci,
                      w_eff=w_eff, n_nat=n_nat)

    def js(m):
        return {a: math.log(m["T"]) - a * math.log(m["rho"]) ** 2
                for a in alphas}
    return {"c_idle": ci, "c_coll": cc,
            "T_base": base["T"], "T_new": new["T"],
            "dT": new["T"] / base["T"] - 1.0,
            "rho_base": base["rho"], "rho_new": new["rho"],
            "vis_base": base["visitor"], "vis_new": new["visitor"],
            "nat_base": base["native"], "nat_new": new["native"],
            "dJ": {a: js(new)[a] - js(base)[a] for a in alphas}}


def ceiling_hits(n_vis: int, c_idle: float, c_coll: float = 1.4,
                 w_eff: int = None, access: str = "rts", reps: int = 3) -> dict:
    """How often a viable visitor's tau sits at the engine's 1.0 clip.

    The engine clips tau into [1e-4, 1.0] after every MIMD update
    (run_step9_fig25.py:350), so a long idle run saturates rather than pushing
    tau past one. If the optimum sat against that clip, the scaling law would
    be reporting where a boundary is, not where the diffusion balance is, so
    this has to be checked rather than assumed.
    """
    f25 = P.engine()
    w_eff = P.W_EFF if w_eff is None else int(w_eff)
    f25.N_VISITOR, f25.N_NATIVE = n_vis, P.N_NATIVE
    hits = tot = 0
    peak = 0.0
    per_visit = []
    try:
        with P.coefficients(c_coll, c_idle), P.window(w_eff):
            for r in range(reps):
                rp = np.random.default_rng(10001 + r * 71 + 7)
                rg = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
                for _ in range(P.VISITS):
                    st: dict = {"trace": []}
                    f25._run_visit25(f25._sample_ppdus25(rp), rg, "pace",
                                     np.full(n_vis, 1.0 / w_eff),
                                     *P.ACCESS[access], stats=st)
                    seen = [rate for _w, nvv, _k, rate in st["trace"] if nvv > 0]
                    tot += len(seen)
                    hits += sum(1 for v in seen if v >= 0.999)
                    peak = max([peak] + seen)
                    per_visit.append(max(seen) if seen else 0.0)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    return {"frac": hits / max(tot, 1), "peak": peak,
            "mean_visit_peak": float(np.mean(per_visit))}


def dispersion(n_vis: int, c_coll: float, c_idle: float = None,
               access: str = "rts", reps: int = 3) -> float:
    """Within-epoch coefficient of variation of tau across viable visitors.

    The DP carries one tau per state, so its answers are trustworthy only while
    the population stays homogeneous. This measures where that stops being true
    and is what bounds the admissible coefficient range, in place of an assumed
    eps_max.
    """
    f25 = P.engine()
    f25.N_VISITOR, f25.N_NATIVE = n_vis, P.N_NATIVE
    st: dict = {}
    try:
        with P.coefficients(c_coll, c_idle):
            for r in range(reps):
                rp = np.random.default_rng(10001 + r * 71 + 7)
                rg = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
                for _ in range(P.VISITS):
                    f25._run_visit25(f25._sample_ppdus25(rp), rg, "pace",
                                     np.full(n_vis, P.TAU_0),
                                     *P.ACCESS[access], stats=st)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    return float(st["tau_cv_sum"] / st["tau_cv_cnt"])


def sim_oracle(n_vis: int, alpha: float, n_r: int = 5, n_s: int = 5,
               access: str = "rts", **kw) -> dict:
    """The same search run on the engine rather than the DP.

    Coarse by necessity: one engine point is ~0.6 s against the DP's ~20 ms, so
    this confirms a handful of scenarios rather than sweeping them.
    """
    best = None
    for r in np.geomspace(R_LO, R_HI, n_r):
        for s in np.geomspace(S_LO, S_HI, n_s):
            c_c, c_i = math.exp(s), math.exp(float(r) * s)
            m = dp.measured(n_vis, access=access, c_coll=c_c, c_idle=c_i, **kw)
            if m["T"] <= 0.0 or m["rho"] <= 0.0:
                continue
            j = math.log(m["T"]) - alpha * math.log(m["rho"]) ** 2
            if best is None or j > best["J"]:
                best = {"r": float(r), "s": float(s), "J": j,
                        "c_coll": c_c, "c_idle": c_i, "T": m["T"],
                        "rho": m["rho"]}
    return best


def _main() -> None:
    for access in ("rts", "basic"):
        print(f"\n=== analytic vs 2-D oracle ({access}, N_nat=10) ===")
        print(f"{'N_vis':>6} {'alpha':>6} {'x_J':>6} {'r*':>7} {'r_orc':>7} "
              f"{'c_coll*':>8} {'c_idle*':>8} {'c_coll_o':>9} {'c_idle_o':>9} "
              f"{'G_J':>7}")
        for n_vis in (10, 20, 50):
            for al in (0.05, 0.2, 0.5):
                g = gap(n_vis, al, access=access)
                print(f"{n_vis:>6} {al:6.2f} {g['x_J']:6.3f} "
                      f"{g['r_analytic']:7.3f} {g['r_oracle']:7.3f} "
                      f"{g['c_coll_a']:8.3f} {g['c_idle_a']:8.3f} "
                      f"{g['c_coll_o']:9.3f} {g['c_idle_o']:9.3f} "
                      f"{g['G_J']:7.4f}")

    print("\n=== the up step is what closes: eps_idle* = C / sqrt(W_eff) ===")
    print(f"{'access':>6} {'N_vis':>6} {'alpha':>6} {'exponent':>9} {'C':>7} "
          f"{'mean err':>9} {'eps* at 420':>12}")
    for access, n_vis, al in (("rts", 20, 0.2), ("rts", 50, 0.05),
                              ("basic", 20, 0.2), ("basic", 10, 0.05)):
        f = scaling_law(n_vis, al, access=access)
        i = f["windows"].index(420)
        print(f"{access:>6} {n_vis:>6} {al:6.2f} {f['exponent']:9.3f} "
              f"{f['C_half']:7.2f} {f['mean_err'] * 100:8.1f}% "
              f"{f['eps'][i]:12.4f}")
    print("  the diffusion balance predicts -1/2")

    print("\n=== where the DP may be trusted: tau dispersion ===")
    print(f"{'c':>6} {'tau CV':>8}")
    for c in (1.05, 1.2, 1.5, 2.0, 3.0, 4.0):
        print(f"{c:6.2f} {dispersion(20, c):8.3f}")

    print("\n=== the shipped baseline (1.2, 1.2) against the oracle ===")
    print(f"{'access':>6} {'N_vis':>6} {'alpha':>6} {'J(1.2,1.2)':>11} "
          f"{'J_oracle':>9} {'G_J':>7}")
    for access in ("rts", "basic"):
        for n_vis in (10, 20, 50):
            for al in (0.05, 0.2):
                jb = dp.objective(n_vis, al, c_coll=P.C_MIMD, r=1.0,
                                  access=access)
                o = oracle(n_vis, al, access=access)
                print(f"{access:>6} {n_vis:>6} {al:6.2f} {jb:11.4f} "
                      f"{o['J']:9.4f} {math.exp(jb - o['J']):7.4f}")


def _self_check() -> None:
    # the grid helper must stay inside its cap and never degenerate
    span = math.log(dp.TAU_CEIL / P.TAU_FLOOR)
    for s in np.geomspace(S_LO, S_HI, 20):
        den = _den_for(float(s))
        assert 1 <= den <= 24
        assert dp.lattice(c_coll=math.exp(float(s)), r=0.5366,
                          max_den=den).nk <= 1400, s

    # J_FH must reproduce dp.objective at the shipped coefficients
    s0 = math.log(P.C_MIMD)
    assert abs(j_fh(20, 0.05, 1.0, s0) - dp.objective(20, 0.05, r=1.0)) < 1e-9

    # Forcing r = r* drives the scale search into its bracket. This is the
    # documented failure, not an accident, so pin it: if it ever stops
    # happening the section 4.5.4 argument needs revisiting.
    hit = [s_star(20, al, EQ.design(20, al, access=ac)["r_star"], access=ac)
           for ac in ("rts", "basic") for al in (0.2, 0.5)]
    assert any(s > S_HI - 1e-2 for s in hit), hit

    # G_J is a loss ratio: the analytic design can never beat the 2-D oracle it
    # is a restriction of, up to the oracle's grid resolution
    for access in ("rts", "basic"):
        for al in (0.05, 0.2):
            g = gap(20, al, access=access)
            assert 0.0 < g["G_J"] <= 1.02, (access, al, g["G_J"])
            assert R_LO <= g["r_analytic"] <= R_HI
            # and the equilibrium ratio must come out below what the finite
            # horizon wants, which is the whole point of section 4.5.4
            assert g["r_analytic"] < g["r_oracle"], (access, al, g)

    # tau dispersion must grow with the step size, and must be small where the
    # DP is being trusted. This is the measured validity bound.
    cvs = [dispersion(20, c) for c in (1.2, 1.5, 2.0, 3.0)]
    assert all(a < b for a, b in zip(cvs, cvs[1:])), cvs
    assert cvs[0] < 0.05 and cvs[-1] > 0.2, cvs

    # The up step is the identifiable direction. Compare spreads inside the
    # range the dispersion check above certifies (eps <= 0.7, i.e. c <= 2);
    # beyond it the population is no longer homogeneous and both axes bite.
    eps = (0.15, 0.25, 0.40, 0.55, 0.70)
    g = np.array([[sim_J(20, 0.2, math.exp(ei), math.exp(ec)) for ec in eps]
                  for ei in eps])
    i, j = np.unravel_index(g.argmax(), g.shape)
    spread_c = g[i].max() - g[i].min()          # move eps_coll, hold eps_idle
    spread_i = g[:, j].max() - g[:, j].min()    # move eps_idle, hold eps_coll
    assert spread_i > 2.5 * spread_c, (spread_i, spread_c)
    print("\noptimise.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
