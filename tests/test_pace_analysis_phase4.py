"""Phase 4: coefficient design.

Pins the section 4.5.4 result, which reversed the section 4.5.3 plan: the
design variable is the up step eps_idle, not the ratio r, and eps_coll is not
identifiable within one visit. Each claim the manuscript will make gets one
test, so a later "simplification" cannot quietly unmake the argument.

The engine-backed tests are slow by the standards of this suite (seconds, not
milliseconds). They are here rather than in a notebook because these are the
numbers the paper quotes.
"""
import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "pace-analysis"), os.path.join(_ROOT, "harq_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dp                       # noqa: E402
import equilibrium as EQ        # noqa: E402
import optimise as O            # noqa: E402
import params as P              # noqa: E402


# ── the lattice generalisation must not disturb the shipped case ─────────────

def test_r_one_reproduces_the_original_lattice():
    """Every number already in the manuscript rides on this."""
    lat = dp.lattice(r=1.0)
    assert (lat.up, lat.dn, lat.r_eff) == (1, 1, 1.0)
    ks = np.arange(lat.nk) - lat.k0
    assert np.allclose(lat.taus, np.clip(P.TAU_0 * P.C_MIMD ** ks,
                                         P.TAU_FLOOR, dp.TAU_CEIL))
    assert abs(dp.total_airtime(20, access="rts") - 0.750) < 0.005


@pytest.mark.parametrize("r", (0.25, 0.5366, 1.5, 1.9718))
def test_asymmetric_lattice_is_a_faithful_refinement(r):
    lat = dp.lattice(r=r)
    assert abs(lat.r_eff - r) / r < 0.02
    assert abs(lat.taus[lat.k0] - P.TAU_0) < 1e-12
    # dn grid units is exactly one factor of c_coll, read off away from the
    # clipped ends
    step = math.log(lat.taus[lat.k0 + 1] / lat.taus[lat.k0])
    assert abs(step * lat.dn - math.log(P.C_MIMD)) < 1e-12


def test_w_eff_sweep_moves_tau_0_with_the_deadline():
    """tau_0 = 1/W_eff is deadline-scaled, so the grid anchor must follow."""
    for w in (105, 420, 1680):
        lat = dp.lattice(tau_0=1.0 / w)
        assert abs(lat.taus[lat.k0] - 1.0 / w) < 1e-14
        assert 0.0 < dp.total_airtime(20, w_eff=w) < 1.0


# ── the ratio: closed form, and why it is not the design variable ────────────

@pytest.mark.parametrize("n", (10, 20, 50))
@pytest.mark.parametrize("tau_nat", (0.0, P.TAU_NAT))
def test_r_star_reduces_to_the_drift_equation(n, tau_nat):
    """r* generalises section 4.2: at the r = 1 equilibrium it must return 1."""
    import drift as D
    n_nat = P.N_NATIVE if tau_nat else 0
    ts = D.tau_star(n, tau_nat)
    assert abs(EQ.r_star(ts, n, tau_nat=tau_nat, n_nat=n_nat) - 1.0) < 1e-9


def test_r_star_continuum_limit():
    """e^x - 1 - x is the native-FREE limit; section 4.5.3 quoted it without
    that qualifier, which is where its numbers came from."""
    for x in (0.2, 0.5, 1.0):
        assert abs(EQ.r_star(x / 500, 500, tau_nat=0.0, n_nat=0)
                   - EQ.r_star_continuum(x)) / EQ.r_star_continuum(x) < 0.01


def test_native_load_raises_the_required_ratio():
    """The plan hypothesised the opposite. More natives means more observed
    collisions, so holding an equilibrium needs a larger up step."""
    vals = [EQ.r_star(0.378 / 20, 20, tau_nat=P.TAU_NAT if nn else 0.0,
                      n_nat=nn) for nn in (0, 5, 10, 20)]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals
    assert vals[2] / vals[0] > 5.0, vals        # about 7x at N_nat = 10


@pytest.mark.parametrize("access", ("rts", "basic"))
def test_equilibrium_ratio_undershoots_the_finite_horizon(access):
    """Section 4.5.4's central negative result. Theorem 2 in coefficient form:
    r* places the equilibrium correctly but the visit never gets there."""
    g = O.gap(20, 0.2, access=access)
    assert g["r_analytic"] < g["r_oracle"], g
    assert 0.7 < g["G_J"] <= 1.02, g["G_J"]


# ── alpha is not optional ────────────────────────────────────────────────────

@pytest.mark.parametrize("access", ("rts", "basic"))
def test_equilibrium_layer_silences_visitors_at_alpha_zero(access):
    """A property of the SATURATED equilibrium model, not of the system.

    There, every visitor is always viable and adding visitor traffic only adds
    collisions, so alpha = 0 drives the operating point to zero. Do not read
    this as "without a fairness term NPCA is pointless": the engine disagrees,
    which the next test pins.
    """
    assert EQ.tau_J(20, 0.0, access=access) * 20 < 0.05
    assert EQ.tau_J(20, 0.2, access=access) > EQ.tau_J(20, 0.0, access=access)


def test_the_engine_does_not_degenerate_at_alpha_zero():
    """The finite window does not share the equilibrium layer's degeneracy.

    Silent visitors leave the window underfilled, because the natives cannot
    take it up on their own, so pure airtime still wants a substantial up step.
    At alpha = 0 the engine's optimum yields the HIGHEST total airtime of any
    alpha, which is exactly what alpha = 0 should do, and rho is far from zero.
    """
    grid = np.exp(np.linspace(math.log(0.02), math.log(2.5), 21))
    js = np.array([O.sim_J(10, 0.0, math.exp(float(e)), 1.4, w_eff=P.W_EFF)
                   for e in grid])
    eps = float(grid[int(js.argmax())])
    assert eps > 0.15, eps                       # not silence
    m = dp.measured(10, c_coll=1.4, c_idle=math.exp(eps))
    assert m["rho"] > 0.25, m
    # and it really is the airtime-maximising choice
    m2 = dp.measured(10, c_coll=1.4, c_idle=math.exp(9.41 / math.sqrt(P.W_EFF)))
    assert m["T"] > m2["T"], (m["T"], m2["T"])


@pytest.mark.parametrize("access", ("rts", "basic"))
def test_alpha_is_a_monotone_dial(access):
    prev_t = prev_r = -1.0
    for al in (0.0, 0.05, 0.2, 0.5):
        d = EQ.design(20, al, access=access)
        assert d["tau_J"] >= prev_t - 1e-9 and d["rho_eq"] >= prev_r - 1e-9
        prev_t, prev_r = d["tau_J"], d["rho_eq"]


# ── the up step is the identifiable direction, and it scales as W^-1/2 ───────

def test_up_step_is_identifiable_and_down_step_is_not():
    """Inside the range the dispersion check certifies. Outside it the
    asymmetry weakens, which is why the claim must carry its range."""
    eps = (0.15, 0.25, 0.40, 0.55, 0.70)
    g = np.array([[O.sim_J(20, 0.2, math.exp(ei), math.exp(ec)) for ec in eps]
                  for ei in eps])
    i, j = np.unravel_index(g.argmax(), g.shape)
    assert (g[:, j].max() - g[:, j].min()) > 2.5 * (g[i].max() - g[i].min())


@pytest.mark.parametrize("access,n_vis", [("rts", 10), ("rts", 20),
                                          ("basic", 20)])
def test_up_step_follows_the_diffusion_scaling(access, n_vis):
    """eps_idle* ~ W_eff^-1/2 where the ramp is the binding constraint.

    Reach grows like eps*E and jitter like eps*sqrt(E), so the balance sits at
    1/sqrt(E) ~ 1/sqrt(W_eff). Measured -0.47 to -0.61 at alpha = 0.2 with a
    7-11% residual. The scope conditions are the subject of the next test.
    """
    f = O.scaling_law(n_vis, 0.2, access=access)
    assert -0.65 < f["exponent"] < -0.35, f["exponent"]
    assert f["mean_err"] < 0.15, f["mean_err"]


@pytest.mark.parametrize("access", ("rts", "basic"))
def test_data_collapse_supports_the_half_power(access):
    """The robust form of the scaling claim.

    Subtract each window's own maximum so only the SHAPE is left, rescale the
    abscissa by W_eff^theta, and ask which theta lays the curves on top of one
    another. Uses every measured point rather than the position of a broad
    peak, which is what made the argmax fit unreliable.
    """
    c = O.collapse(10, 0.2, access=access)
    assert c["lo"] <= 0.5 <= c["hi"], (c["theta"], c["lo"], c["hi"])
    assert c["gain"] > 2.0, c["gain"]           # rescaling really does help
    assert 6.0 < c["u_star"] < 14.0, c["u_star"]


def test_collapse_reports_unmeasurable_scenarios_instead_of_guessing():
    """A window too short to adapt in has no optimal adaptation rate, and a
    flat objective has no peak to place. The filter must say so rather than
    return a number: the argmax fit gave this scenario an exponent of +0.06."""
    c = O.collapse(50, 0.05, access="rts", windows=(300, 420, 840, 1680))
    assert not c["windows"] and len(c["dropped"]) == 4, c
    assert math.isnan(c["theta"])
    # and the short window is rejected where the rest of the sweep is fine
    ok = O.collapse(10, 0.2, access="basic")
    assert 150 in ok["dropped"] and len(ok["windows"]) >= 3, ok["dropped"]


def test_the_scaling_law_needs_alpha_to_condition_the_objective():
    """At small alpha the law cannot be measured, and the reason is not noise.

    J = ln T - alpha*(ln rho)^2, and T is nearly flat in eps_idle: the r sweep
    moves it by about 2%. The curvature that locates an optimum comes from the
    fairness term, so at alpha = 0.05 the peak is too broad to place and the
    fit degrades to a 38-52% residual with the wrong sign. This is the same
    fact as test_pure_airtime_objective_silences_the_visitors, seen from the
    conditioning side, and it is a scope condition the manuscript must state.
    """
    sharp = O.scaling_law(20, 0.2, access="rts")
    flat = O.scaling_law(50, 0.05, access="rts")
    assert sharp["mean_err"] < 0.15 < flat["mean_err"], (sharp, flat)
    assert sharp["exponent"] < -0.35 < flat["exponent"], (sharp, flat)


# ── the measured validity bound, in place of an assumed eps_max ──────────────

def test_one_up_step_serves_every_alpha():
    """E4, and the answer to "where did alpha go?".

    The design rule takes W_eff but not alpha, which looks like an omission
    since the original goal was to choose coefficients GIVEN alpha. It is a
    measured result: the objective is broad enough in eps_idle that the single
    value 9.41/sqrt(W) lands within a few per cent of the per-case optimum for
    every alpha from 0.05 to 0.5. Alpha picks the operating point, not the
    coefficient.
    """
    grid = np.exp(np.linspace(math.log(0.12), math.log(1.8), 15))
    w = P.W_EFF
    eps_fixed = 9.41 / math.sqrt(w)
    picks = []
    for alpha in (0.05, 0.2, 0.5):
        js = np.array([O.sim_J(10, alpha, math.exp(float(e)), 1.4, w_eff=w)
                       for e in grid])
        picks.append(float(grid[int(js.argmax())]))
        got = O.sim_J(10, alpha, math.exp(eps_fixed), 1.4, w_eff=w)
        assert math.exp(got - js.max()) > 0.95, (alpha, got, js.max())
    # The picks stay within about one grid step. They came out identical on
    # one earlier grid, but that was its resolution, not a fact: a finer grid
    # separates alpha = 0.05 from the rest. The substantive claim is the G_J
    # bound above, not exact equality.
    assert max(picks) / min(picks) < 1.6, picks

    # the plateau has an upper edge, though. Past alpha ~ 1 the fixed rule
    # degrades, so "any nonzero alpha" is not the claim to make.
    wide = np.exp(np.linspace(math.log(0.02), math.log(2.5), 21))
    js = np.array([O.sim_J(10, 5.0, math.exp(float(e)), 1.4, access="basic",
                           w_eff=w) for e in wide])
    got = O.sim_J(10, 5.0, math.exp(eps_fixed), 1.4, access="basic", w_eff=w)
    assert math.exp(got - js.max()) < 0.75, math.exp(got - js.max())


@pytest.mark.parametrize("access", ("rts", "basic"))
def test_the_design_buys_fairness_not_airtime(access):
    """E3, and the claim the manuscript is allowed to make.

    Against the shipped (1.2, 1.2) the derived coefficients do NOT raise total
    airtime, they lower it slightly. What they buy is proportional fairness:
    rho moves a long way towards one. Saying it the other way round would be
    wrong, so pin the direction of both.
    """
    g = O.design_gain(20, access=access)
    assert g["T_new"] < g["T_base"], g          # airtime is spent, not gained
    assert g["T_new"] > 0.90 * g["T_base"], g   # but only a few per cent of it
    assert g["rho_new"] > 1.4 * g["rho_base"], g
    assert g["rho_new"] <= 1.0, g               # corrects under-share, no overshoot
    assert g["vis_new"] > g["vis_base"] and g["nat_new"] < g["nat_base"], g


def test_the_shipped_constant_is_fine_at_long_windows():
    """Why a W_eff rule is needed at all: the shipped 1.2 is close to what the
    rule itself prescribes once the window is long, so the two perform alike.
    The rule earns its keep as the window shrinks."""
    assert abs(O.design_coefficients("rts", 1680)[1] - 1.2) < 0.12
    long_ = O.design_gain(20, access="rts", w_eff=1680)
    short = O.design_gain(20, access="rts", w_eff=420)
    assert abs(long_["dJ"][0.2]) < 0.05, long_["dJ"]
    assert short["dJ"][0.2] > 4 * abs(long_["dJ"][0.2]), (short["dJ"],
                                                          long_["dJ"])


def test_equilibrium_ratio_becomes_right_once_the_window_is_long():
    """The validity boundary of the equilibrium theory, not its refutation.

    Fixing eps_idle at the design value and deriving eps_coll = eps_idle / r*
    closes the chain analytically. It loses at the reference window and wins
    once the window is long enough for the walk to arrive, which is the finite
    horizon of Theorem 2 speaking and nothing else.
    """
    n_vis, alpha, access = 10, 0.2, "rts"
    tau_j = EQ.tau_J(n_vis, alpha, access=access)
    r_star = EQ.r_star(tau_j, n_vis)

    def gap(w):
        eps = 10.15 / math.sqrt(w)
        c_idle = math.exp(eps)
        def j(c_coll):
            m = dp.measured(n_vis, access=access, c_coll=c_coll,
                            c_idle=c_idle, w_eff=w)
            return math.log(m["T"]) - alpha * math.log(m["rho"]) ** 2
        return j(math.exp(eps / r_star)) - j(c_idle)

    short, long_ = gap(P.W_EFF), gap(6720)
    assert short < 0.0 < long_, (short, long_)
    # the derived ratio really is well below one, which is what makes the
    # implied c_coll large and the back-off hard
    assert 0.2 < r_star < 0.9, r_star


def test_the_ratio_is_not_a_well_posed_target_at_any_alpha():
    """The original plan was alpha -> tau_J -> r* -> coefficients. This is where
    the last step finally dies.

    Sweeping alpha with c_coll held fixed makes r move by construction, so the
    two dimensions have to be swept together. Doing that, eps_idle stays put
    while eps_coll slides along a flat direction, so the apparent r* jumps
    around without J changing. The near-optimal band in r is the same wide
    range at every alpha, which is what "not identified" means.
    """
    ei = np.exp(np.linspace(math.log(0.15), math.log(1.3), 7))
    ec = np.exp(np.linspace(math.log(0.10), math.log(0.75), 5))
    picks, bands = [], []
    for alpha in (0.05, 0.2, 0.5):
        g = np.array([[O.sim_J(10, alpha, math.exp(float(a)), math.exp(float(b)))
                       for b in ec] for a in ei])
        i, j = np.unravel_index(g.argmax(), g.shape)
        picks.append(ei[i])
        rs = [ei[a] / ec[b] for a in range(len(ei)) for b in range(len(ec))
              if g[a, b] >= g.max() - 0.02]
        bands.append((min(rs), max(rs)))
    assert max(picks) == min(picks), picks       # the up step does not move
    for lo, hi in bands:
        assert hi / lo > 3.0, bands              # the ratio is wide open
    # and the band is the same one at every alpha, so alpha does not pick an r
    assert max(b[0] for b in bands) / min(b[0] for b in bands) < 1.5, bands


def test_collision_cost_sets_how_identified_the_down_step_is():
    """Why RTS/CTS and basic differ, isolated from the overhead that also
    differs between them.

    eps_coll governs only the response to a collision, so when collisions are
    cheap that whole channel carries little weight in the objective and any
    value does about as well. Make collisions expensive and the recovery policy
    starts to matter. The up step, meanwhile, does not care.
    """
    ei = np.exp(np.linspace(math.log(0.15), math.log(1.3), 7))
    ec = np.exp(np.linspace(math.log(0.10), math.log(0.90), 6))
    spreads, ups = [], []
    for name, cost in (("_t_cheap", 6), ("_t_mid", 25), ("_t_nocd", "nocd")):
        P.ACCESS[name] = (cost, P.L_HS)          # overhead held at the RTS value
        try:
            g = np.array([[O.sim_J(10, 0.2, math.exp(float(a)),
                                   math.exp(float(b)), access=name)
                           for b in ec] for a in ei])
        finally:
            del P.ACCESS[name]
        i, j = np.unravel_index(g.argmax(), g.shape)
        spreads.append(g[i].max() - g[i].min())
        ups.append(ei[i])
    assert spreads[-1] > 2.0 * spreads[0], spreads   # expensive -> identified
    assert all(a < b for a, b in zip(spreads, spreads[1:])), spreads
    assert max(ups) == min(ups), ups                 # the up step is robust


def test_down_step_is_unidentified_by_J_but_not_irrelevant():
    """A correction to the first reading of section 4.5.4.

    eps_coll barely moves J, which is why the closed form for it diverged. That
    is not the same as being irrelevant: it moves the allocation a great deal,
    and J is flat only because the airtime and fairness effects cancel.
    """
    ts, rs = [], []
    for c_coll in (1.2, 1.4, 2.0):
        g = O.design_gain(10, access="rts", c_coll=c_coll)
        ts.append(g["T_new"])
        rs.append(g["rho_new"])
    assert max(ts) / min(ts) < 1.05, ts          # airtime hardly notices
    assert max(rs) / min(rs) > 1.30, rs          # the split very much does
    assert all(a > b for a, b in zip(rs, rs[1:])), rs


def test_the_design_constant_is_nearly_scenario_free():
    """E1's usable result. The diffusion balance predicts the exponent but not
    the constant: fitting C from the equilibrium layer was worse than assuming
    it fixed (3.16x spread against 1.63x). It does not matter, because C barely
    moves. Pinned at two scenarios per access mode to keep the run affordable.
    """
    ws = (420, 1680)
    grid = np.exp(np.linspace(math.log(0.12), math.log(1.6), 13))

    def c_of(n_vis, access):
        es = []
        for w in ws:
            js = np.array([O.sim_J(n_vis, 0.2, math.exp(float(e)), 1.4,
                                   access=access, w_eff=w) for e in grid])
            es.append(grid[int(js.argmax())])
        return float(np.mean([e * math.sqrt(w) for e, w in zip(es, ws)]))

    cs = [c_of(n, a) for a in ("rts", "basic") for n in (10, 20)]
    assert max(cs) / min(cs) < 1.8, cs           # spread stays modest
    assert all(6.0 < c < 13.0 for c in cs), cs   # and lands near 9.4


def test_log_airtime_makes_the_design_independent_of_units():
    """Why J uses ln T rather than T.

    Both forms are monotone scalarisations of the same two objectives, so both
    trace the same Pareto frontier and at moderate alpha they pick the same
    point. The difference is that alpha carries units in the plain-T form: it
    trades an ABSOLUTE airtime against a squared log ratio, so the design moves
    when airtime is reported in slots instead of a fraction of the window.
    ln T only shifts by a constant under rescaling, so the argmax cannot move.
    """
    grid = np.exp(np.linspace(math.log(0.12), math.log(3.0), 19))
    pts = [dp.measured(10, access="basic", c_coll=1.4,
                       c_idle=math.exp(float(e))) for e in grid]
    T = np.array([p["T"] for p in pts])
    pen = 0.2 * np.log(np.array([p["rho"] for p in pts])) ** 2

    log_choice = {int(np.argmax(np.log(T * s) - pen))
                  for s in (1.0, 420.0, 420 * 9.0, 0.42)}
    raw_choice = {int(np.argmax(T * s - pen))
                  for s in (1.0, 420.0, 420 * 9.0, 0.42)}
    assert len(log_choice) == 1, "ln T must be scale invariant"
    assert len(raw_choice) > 1, "plain T should have been unit dependent"


def test_dp_predicts_the_shape_but_overshoots_the_constant():
    """E2, the claim section IV is allowed to make.

    The DP carries one tau per state, so it cannot see the dispersion a large
    up step creates and recommends a bigger one than the engine wants. The bias
    grows with the window, because more epochs accumulate more dispersion. It
    is a bias in the CONSTANT, not in the shape: following the DP still lands
    within a few percent of the engine's own optimum.
    """
    grid = O.EI_GRID
    ratios, losses = [], []
    for w in (420, 1680):
        js = np.array([O.sim_J(10, 0.2, math.exp(float(e)), 1.4, w_eff=w)
                       for e in grid])
        jd = np.array([O.dp_J(10, 0.2, math.exp(float(e)), 1.4, w_eff=w)
                       for e in grid])
        i, k = int(js.argmax()), int(jd.argmax())
        ratios.append(grid[k] / grid[i])
        losses.append(math.exp(js[k] - js[i]))       # cost of trusting the DP
    assert all(r >= 1.0 for r in ratios), ratios     # biased high, never low
    assert ratios[-1] > ratios[0], ratios            # and worse at long windows
    assert all(g > 0.90 for g in losses), losses     # yet cheap to follow


def test_objective_takes_a_coefficient_not_a_log_step():
    """A bug that produced a plausible-looking but garbage DP curve.

    Passing eps_idle where c_idle belongs makes r = ln(eps)/ln(c_coll) negative
    for eps < 1, and the objective jumps around by two nats between adjacent
    grid points. The smooth curve is the correct call.
    """
    eps = np.log(np.array([1.2, 1.4, 1.7, 2.0, 2.4]))
    good = [O.dp_J(10, 0.2, math.exp(float(e)), 1.4, w_eff=840,
                   access="basic") for e in eps]
    assert all(b > a for a, b in zip(good, good[1:])) or \
        max(good) - min(good) < 1.0, good           # smooth, no two-nat jumps
    assert all(abs(b - a) < 0.5 for a, b in zip(good, good[1:])), good
    # the mis-call drives r negative, which is the signature of the bug
    assert math.log(0.364) / math.log(1.4) < 0


def test_the_tau_ceiling_does_not_bind_at_the_optimum():
    """tau is clipped into [1e-4, 1] after every update, so a long idle run
    saturates instead of running past one. The scaling law would be reporting
    that boundary rather than the diffusion balance if the optimum sat against
    it, so pin that it does not: at eps_idle* the clip is never reached, and it
    only starts to bite well above.
    """
    star = O.ceiling_hits(10, math.exp(10.77 / math.sqrt(P.W_EFF)))
    assert star["frac"] == 0.0, star
    assert star["peak"] < 0.95, star           # never even close
    assert star["mean_visit_peak"] < 0.30, star
    # and it does bite once the step is a few times too large, so the check
    # above is a real constraint rather than one that can never fail
    assert O.ceiling_hits(10, math.exp(2.3))["frac"] > 0.20


def test_dispersion_grows_with_the_step_and_bounds_the_dp():
    """The DP carries one tau per state, so it is only usable while the
    population stays homogeneous. This is section 4.5.3's missing constraint,
    measured rather than assumed."""
    cvs = [O.dispersion(20, c) for c in (1.2, 1.5, 2.0, 3.0)]
    assert all(a < b for a, b in zip(cvs, cvs[1:])), cvs
    assert cvs[0] < 0.05, cvs          # the shipped coefficient is safe
    assert cvs[-1] > 0.2, cvs          # c = 3 is not
