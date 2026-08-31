"""Scenario-wise MIMD coefficients: analytical candidate vs simulation oracle.

    .venv/bin/python pace-analysis/coeff_oracle.py --subset
    .venv/bin/python pace-analysis/coeff_oracle.py --full --jobs 8

For each scenario S = (N_vis, N_nat, W_eff, F_L, access, alpha) this produces

    (c_idle,c_coll)_A   argmax of the finite-horizon DP objective  (analytical)
    (c_idle,c_coll)_O   argmax of a 2-D search on the ENGINE       (oracle)

and then evaluates BOTH on an independent Monte Carlo run, on evaluation seeds
that the oracle search never saw, with

    G_J = exp(J_A - J_O),   J = ln max(T,delta) - alpha (ln max(rho,delta))^2.

Three things this file is careful about, because each has bitten this project:

1.  J is computed ONCE from the pooled airtime totals, never averaged over
    per-visit J values. ln and the square are both non-linear.
2.  The analytical candidate's reported performance is a SIMULATION number.
    Mixing dp.objective into G_J would compare a model to an engine.
3.  Tuning and evaluation seeds are disjoint by construction (assert_disjoint),
    so the oracle cannot be scored on the sample that selected it.

The DP is the only analytical layer that carries both coefficient axes. Its
known departures from the engine, recorded in metadata.json rather than assumed
away: native attempts are an exogenous constant tau_nat instead of the engine's
frozen-backoff DCF, the visitor population is treated as homogeneous in tau, and
the (c_idle, c_coll) pair is quantised onto a rational lattice, so the achieved
ratio is reported next to the requested one.

ponytail: no scheduler, no resume. A full run is ~1-2 h on 8 cores; if it dies,
rerun it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict

import numpy as np

import dp
import equilibrium as EQ
import optimise as OPT
import params as P

DELTA = 1e-9                       # floor inside the logs of J
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "coeff_oracle")

# Independent random streams. The candidate coefficients are deliberately NOT
# part of either key, which is what makes the comparison paired: every candidate
# sees the same PPDU draws and the same initial backoffs.
STREAM_PPDU, STREAM_ENGINE = 0xA1, 0xB2
STREAM_INIT = 0xC3          # tau_0 draws, kept off the PPDU stream so the two
#                             initialisation arms see the same workload

# Search box in log coefficients. The defaults are the brief's; CAP_* are the
# physical stops the expansion is allowed to reach (c in [1.01, 4]) before the
# point is declared boundary-limited.
EPS_LO, EPS_HI = 0.05, 0.80
CAP_LO, CAP_HI = math.log(1.01), math.log(4.0)

ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.5)
AXES = {"n_vis": (5, 10, 20, 50), "n_nat": (0, 5, 10, 20),
        "w_eff": (105, 210, 420, 840, 1680), "access": ("rts", "basic")}

# Eight cores that move one axis at a time off the manuscript's operating point
# (N_vis=10, N_nat=10, W_eff=420, RTS/CTS).
SUBSET = [(10, 10, 420, "rts"), (10, 10, 420, "basic"),
          (50, 10, 420, "rts"), (10, 10, 105, "rts"),
          (10, 10, 1680, "rts"), (10, 0, 420, "rts"),
          (20, 20, 840, "basic"), (5, 5, 210, "rts")]

TUNE_SEEDS = tuple(range(1, 13))        # oracle search only
EVAL_SEEDS = tuple(range(101, 161))     # final numbers only
DEV_SEEDS = tuple(range(9001, 9010))    # debugging; used by the tests


@dataclass(frozen=True)
class Scn:
    """A scenario core. alpha is not a member: it changes only the objective
    read off an already-measured (T, rho), never the simulation itself."""
    n_vis: int
    n_nat: int
    w_eff: int
    access: str

    def tag(self) -> str:
        return f"{self.access}_W{self.w_eff}_v{self.n_vis}_n{self.n_nat}"


# ─── engine measurement ──────────────────────────────────────────────────────

def _rngs(scn: Scn, seed: int):
    """Exogenous-workload stream and engine stream for one sequence.

    Both depend on the scenario and the seed only. The engine stream also drives
    the transmission draws, which diverge between candidates as soon as the
    dynamics differ, so the sharing is exact for the initial state and the PPDU
    sequence and statistical thereafter (brief section 8's fallback)."""
    key = (seed, scn.n_vis, scn.n_nat, scn.w_eff, len(scn.access))
    return (np.random.default_rng((STREAM_PPDU,) + key),
            np.random.default_rng((STREAM_ENGINE,) + key))


def batch(scn: Scn, c_idle: float, c_coll: float, seeds, visits: int,
          tau0: str = "one_probe") -> list:
    """Run one candidate over `seeds` sequences of `visits` NPCA transitions.

    Returns one row per seed holding SUMS, not means: the pooled aggregation in
    aggregate() and the paired bootstrap both need additive quantities.

    tau0 selects the initial transmission probability: "one_probe" is the
    shipped 1/W_eff, "uniform" draws U(0,1) per visitor per visit from its own
    stream so the PPDU sequence stays paired with the one_probe arm."""
    # 1.0 disables that update entirely, which is a legitimate control point
    # (no up step / no down step); anything below 1 inverts the rule.
    assert c_idle >= 1.0 and c_coll >= 1.0, (c_idle, c_coll)
    f25 = P.engine()
    old = (f25.N_VISITOR, f25.N_NATIVE)
    f25.N_VISITOR, f25.N_NATIVE = scn.n_vis, scn.n_nat
    rows = []
    try:
        with P.coefficients(c_coll, c_idle), P.window(scn.w_eff):
            for s in seeds:
                rng_p, rng = _rngs(scn, s)
                rng_i = np.random.default_rng(
                    (STREAM_INIT, s, scn.n_vis, scn.n_nat, scn.w_eff,
                     len(scn.access)))
                st: dict = {}
                av = an = 0.0
                coll = idle = oh = 0
                for _ in range(visits):
                    air, c_air, idl, o_air, _ = f25._run_visit25(
                        f25._sample_ppdus25(rng_p), rng, "pace",
                        (rng_i.random(scn.n_vis) if tau0 == "uniform"
                         else np.full(scn.n_vis, 1.0 / scn.w_eff)),
                        *P.ACCESS[scn.access], stats=st)
                    av += float(air[:scn.n_vis].sum())
                    an += float(air[scn.n_vis:].sum())
                    coll += int(c_air)
                    idle += int(idl)
                    oh += int(o_air)
                rows.append({
                    "seed": int(s), "visits": int(visits),
                    "A_vis": av, "A_nat": an,
                    "coll_air": coll, "idle_slots": idle, "oh_air": oh,
                    "epochs": st.get("epochs", 0),
                    "n_idle": st.get("idle", 0), "n_coll": st.get("coll", 0),
                    "n_solo_vis": st.get("solo_vis", 0),
                    "n_solo_nat": st.get("solo_nat", 0),
                    "tau_cv_sum": st.get("tau_cv_sum", 0.0),
                    "tau_cv_cnt": st.get("tau_cv_cnt", 0)})
    finally:
        f25.N_VISITOR, f25.N_NATIVE = old
    return rows


def aggregate(rows: list, scn: Scn, alpha: float) -> dict:
    """Pool first, then form T, rho and J once (brief section 3)."""
    n = sum(r["visits"] for r in rows)
    norm = n * scn.w_eff
    av = sum(r["A_vis"] for r in rows)
    an = sum(r["A_nat"] for r in rows)
    tot = av + an
    share = av / tot if tot > 0 else 0.0
    pop = scn.n_vis / (scn.n_vis + scn.n_nat)
    rho = share / pop if share > 0 else 0.0
    ep = sum(r["epochs"] for r in rows)
    cv_c = sum(r["tau_cv_cnt"] for r in rows)
    return {
        "T": tot / norm, "A_vis": av / norm, "A_nat": an / norm, "rho": rho,
        "J": (math.log(max(tot / norm, DELTA))
              - alpha * math.log(max(rho, DELTA)) ** 2),
        "coll_frac": sum(r["coll_air"] for r in rows) / norm,
        "idle_frac": sum(r["idle_slots"] for r in rows) / norm,
        "oh_frac": sum(r["oh_air"] for r in rows) / norm,
        "solo_vis_frac": sum(r["n_solo_vis"] for r in rows) / max(ep, 1),
        "epochs_per_visit": ep / max(n, 1),
        "tau_cv": (sum(r["tau_cv_sum"] for r in rows) / cv_c) if cv_c else
        float("nan"),
        "visits": n, "seqs": len(rows)}


def objective(T: float, rho: float, alpha: float) -> float:
    return math.log(max(T, DELTA)) - alpha * math.log(max(rho, DELTA)) ** 2


# ─── the two candidates ──────────────────────────────────────────────────────

class SimGrid:
    """Memoised engine evaluations of (eps_idle, eps_coll) on tuning seeds.

    (T, rho) do not depend on alpha, so one grid serves every alpha in the
    scenario. Keys are rounded logs; the same point requested by two alphas'
    refinements costs one run."""

    def __init__(self, scn: Scn, seeds, visits: int):
        self.scn, self.seeds, self.visits = scn, seeds, visits
        self.pts: dict = {}

    def __call__(self, ei: float, ec: float) -> dict:
        key = (round(float(ei), 6), round(float(ec), 6))
        if key not in self.pts:
            a = aggregate(batch(self.scn, math.exp(key[0]), math.exp(key[1]),
                                self.seeds, self.visits), self.scn, 0.0)
            self.pts[key] = {"eps_idle": key[0], "eps_coll": key[1],
                             "c_idle": math.exp(key[0]),
                             "c_coll": math.exp(key[1]),
                             "T": a["T"], "rho": a["rho"]}
        return self.pts[key]


def _argmax(pts, alpha):
    return max(pts, key=lambda p: objective(p["T"], p["rho"], alpha))


def _on_edge(v, axis) -> bool:
    return abs(v - min(axis)) < 1e-9 or abs(v - max(axis)) < 1e-9


def search(evalpt, alphas, lo=EPS_LO, hi=EPS_HI, n_coarse=6, n_fine=5,
           max_expand=3) -> dict:
    """Coarse grid, boundary expansion, then a per-alpha fine grid.

    The expansion is shared across alphas: if any alpha's coarse argmax sits on
    an edge the box grows in that direction, until every alpha is interior or a
    physical cap is reached. Refinement is per alpha because the peaks separate
    once the fairness weight bites."""
    axis = list(np.geomspace(lo, hi, n_coarse))
    for _ in range(max_expand):
        pts = [evalpt(ei, ec) for ei in axis for ec in axis]
        need_lo = need_hi = False
        for al in alphas:
            b = _argmax(pts, al)
            for v in (b["eps_idle"], b["eps_coll"]):
                need_lo |= abs(v - axis[0]) < 1e-9
                need_hi |= abs(v - axis[-1]) < 1e-9
        grew = False
        if need_hi and axis[-1] < CAP_HI - 1e-9:
            hi = min(hi * 1.8, CAP_HI)
            grew = True
        if need_lo and axis[0] > CAP_LO + 1e-9:
            lo = max(lo / 1.8, CAP_LO)
            grew = True
        if not grew:
            break
        axis = list(np.geomspace(lo, hi, n_coarse + 2))

    pts = [evalpt(ei, ec) for ei in axis for ec in axis]
    out = {}
    step = axis[1] / axis[0]
    for al in alphas:
        b = _argmax(pts, al)
        fi = np.geomspace(max(b["eps_idle"] / step, CAP_LO),
                          min(b["eps_idle"] * step, CAP_HI), n_fine)
        fc = np.geomspace(max(b["eps_coll"] / step, CAP_LO),
                          min(b["eps_coll"] * step, CAP_HI), n_fine)
        local = [evalpt(ei, ec) for ei in fi for ec in fc]
        best = _argmax(pts + local, al)
        # near-optimal region: everything within 1% of the best utility
        j_b = objective(best["T"], best["rho"], al)
        near = [p for p in pts + local
                if objective(p["T"], p["rho"], al) >= j_b + math.log(0.99)]
        out[al] = {
            "best": best,
            "boundary": _on_edge(best["eps_idle"], axis)
            or _on_edge(best["eps_coll"], axis),
            "capped": (abs(axis[-1] - CAP_HI) < 1e-9
                       or abs(axis[0] - CAP_LO) < 1e-9),
            "box": (float(axis[0]), float(axis[-1])),
            "spacing": float(math.log(step)),
            "near_c_idle": (min(p["c_idle"] for p in near),
                            max(p["c_idle"] for p in near)),
            "near_c_coll": (min(p["c_coll"] for p in near),
                            max(p["c_coll"] for p in near)),
            "near_n": len(near)}
    return out


def dp_point(scn: Scn, ei: float, ec: float) -> dict:
    """(T, rho) as the finite-horizon DP predicts them, with the lattice's
    ACHIEVED ratio reported: r is quantised to a rational approximation."""
    den = OPT._den_for(ec)
    lat = dp.lattice(c_coll=math.exp(ec), r=ei / ec, tau_0=1.0 / scn.w_eff,
                     max_den=den)
    o = dp.outcome(scn.n_vis, n_nat=scn.n_nat, access=scn.access,
                   w_eff=scn.w_eff, c_coll=math.exp(ec), r=ei / ec,
                   max_den=den)
    return {"eps_idle": lat.r_eff * ec, "eps_coll": ec,
            "c_idle": math.exp(lat.r_eff * ec), "c_coll": math.exp(ec),
            "eps_idle_req": ei, "r_eff": lat.r_eff,
            "T": o["T"], "rho": o["rho"]}


def analytic(scn: Scn, alphas, n_grid=9) -> dict:
    """2-D argmax of the DP objective. Same box and the same boundary
    bookkeeping as the engine search, so the two are read the same way."""
    axis = list(np.geomspace(EPS_LO, EPS_HI, n_grid))
    pts = [dp_point(scn, ei, ec) for ei in axis for ec in axis]
    out = {}
    for al in alphas:
        b = _argmax(pts, al)
        out[al] = {"best": b,
                   "boundary": _on_edge(b["eps_idle_req"], axis)
                   or _on_edge(b["eps_coll"], axis),
                   "J_model": objective(b["T"], b["rho"], al)}
    return out


def design_rule_candidate(scn: Scn, c_coll: float = 1.2) -> tuple[float, float]:
    """The manuscript's shipped one-parameter rule, eps_idle = C / sqrt(W_eff).

    Included as a CANDIDATE, never as the analytical optimum: C is calibrated
    (section 4.5.8 rejected its derivation), so this is a design heuristic being
    measured, not a model prediction being tested. It is the only candidate whose
    input, W_eff, the standard already hands a visitor for free."""
    return math.exp(OPT.C_DESIGN[scn.access] / math.sqrt(scn.w_eff)), c_coll


def equilibrium_candidate(scn: Scn, alpha: float) -> dict:
    """r fixed at the drift ratio r*(tau_J), scale then chosen over the DP.

    Kept as a separate column because it answers a different question from the
    analytical argmax: it is what the EQUILIBRIUM layer would pick, and section
    4.5.14 says the visit ends long before that layer applies."""
    eq = EQ.design(scn.n_vis, alpha, n_nat=scn.n_nat, access=scn.access)
    r = min(max(eq["r_star"], OPT.R_LO), OPT.R_HI)
    s = OPT.s_star(scn.n_vis, alpha, r, access=scn.access, w_eff=scn.w_eff,
                   n_nat=scn.n_nat)
    return {"c_coll": math.exp(s), "c_idle": math.exp(r * s),
            "r_star": eq["r_star"], "tau_J": eq["tau_J"]}


# ─── evaluation and the paired bootstrap ─────────────────────────────────────

def bootstrap(rows_a: list, rows_o: list, scn: Scn, alpha: float,
              n_boot: int = 2000, seed: int = 20260828) -> dict:
    """Paired bootstrap over evaluation sequences.

    The same resampled index set is applied to both candidates, which is the
    whole point of running them on shared seeds: the noise common to a sequence
    cancels in the difference."""
    assert [r["seed"] for r in rows_a] == [r["seed"] for r in rows_o]
    rng = np.random.default_rng(seed)
    n = len(rows_a)
    pop = scn.n_vis / (scn.n_vis + scn.n_nat)

    def cols(rows):
        return (np.array([r["A_vis"] for r in rows]),
                np.array([r["A_nat"] for r in rows]),
                np.array([r["visits"] for r in rows], float))
    av_a, an_a, v_a = cols(rows_a)
    av_o, an_o, v_o = cols(rows_o)
    idx = rng.integers(0, n, size=(n_boot, n))

    def stats(av, an, v):
        # pooled per resample, exactly as aggregate() pools over the full set
        sv, sn, nv = av[idx].sum(1), an[idx].sum(1), v[idx].sum(1)
        tot = sv + sn
        T = tot / (nv * scn.w_eff)
        rho = np.where(tot > 0, sv / np.maximum(tot, DELTA) / pop, 0.0)
        J = (np.log(np.maximum(T, DELTA))
             - alpha * np.log(np.maximum(rho, DELTA)) ** 2)
        return T, rho, J
    T_a, r_a, J_a = stats(av_a, an_a, v_a)
    T_o, r_o, J_o = stats(av_o, an_o, v_o)

    def ci(x):
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {"G_J_ci": ci(np.exp(J_a - J_o)), "dJ_ci": ci(J_a - J_o),
            "dT_ci": ci(T_a - T_o), "drho_ci": ci(r_a - r_o),
            "n_boot": n_boot}


def run_scenario(core: tuple, alphas=ALPHAS, tune_seeds=TUNE_SEEDS,
                 tune_visits: int = 40, eval_seeds=EVAL_SEEDS,
                 eval_visits: int = 30, n_boot: int = 2000,
                 with_eq: bool = True, keep_rows: bool = False) -> dict:
    """One scenario core: search on tuning seeds, score on evaluation seeds."""
    scn = Scn(*core)
    assert not (set(tune_seeds) & set(eval_seeds)), "seed sets overlap"
    grid = SimGrid(scn, tune_seeds, tune_visits)
    orc = search(grid, alphas)
    ana = analytic(scn, alphas)

    cache: dict = {}

    def ev(c_idle, c_coll):
        key = (round(c_idle, 9), round(c_coll, 9))
        if key not in cache:
            cache[key] = batch(scn, c_idle, c_coll, eval_seeds, eval_visits)
        return cache[key]

    rows, raw = [], []
    for al in alphas:
        cands = {
            "current": (1.2, 1.2),
            "analytic": (ana[al]["best"]["c_idle"], ana[al]["best"]["c_coll"]),
            "oracle": (orc[al]["best"]["c_idle"], orc[al]["best"]["c_coll"]),
            "design_rule": design_rule_candidate(scn),
        }
        if with_eq:
            e = equilibrium_candidate(scn, al)
            cands["equilibrium_ratio"] = (e["c_idle"], e["c_coll"])
        meas = {}
        for name, (ci_, cc_) in cands.items():
            r = ev(ci_, cc_)
            meas[name] = aggregate(r, scn, al)
            meas[name].update(c_idle=ci_, c_coll=cc_)
            if keep_rows:
                for x in r:
                    raw.append({**asdict(scn), "alpha": al, "candidate": name,
                                "c_idle": ci_, "c_coll": cc_, **x})
        bs = bootstrap(ev(*cands["analytic"]), ev(*cands["oracle"]), scn, al,
                       n_boot=n_boot)
        row = {**asdict(scn), "alpha": al,
               "c_idle_A": cands["analytic"][0], "c_coll_A": cands["analytic"][1],
               "c_idle_O": cands["oracle"][0], "c_coll_O": cands["oracle"][1],
               "G_J": math.exp(meas["analytic"]["J"] - meas["oracle"]["J"]),
               "boundary_O": orc[al]["boundary"], "capped_O": orc[al]["capped"],
               "boundary_A": ana[al]["boundary"],
               "J_model_A": ana[al]["J_model"],
               "near_c_idle_lo": orc[al]["near_c_idle"][0],
               "near_c_idle_hi": orc[al]["near_c_idle"][1],
               "near_c_coll_lo": orc[al]["near_c_coll"][0],
               "near_c_coll_hi": orc[al]["near_c_coll"][1],
               "near_n": orc[al]["near_n"],
               "box_lo": orc[al]["box"][0], "box_hi": orc[al]["box"][1],
               "grid_spacing": orc[al]["spacing"],
               "tune_visits": len(tune_seeds) * tune_visits,
               "eval_visits": len(eval_seeds) * eval_visits,
               "tune_points": len(grid.pts), **bs}
        for name, m in meas.items():
            for k in ("T", "rho", "J", "A_vis", "A_nat", "coll_frac",
                      "idle_frac", "solo_vis_frac", "epochs_per_visit",
                      "tau_cv"):
                row[f"{k}_{name}"] = m[k]
            row[f"c_idle_{name}"] = m["c_idle"]
            row[f"c_coll_{name}"] = m["c_coll"]
        rows.append(row)

    gridrows = [{**asdict(scn), **p, **{f"J_a{al}": objective(p["T"], p["rho"], al)
                                        for al in alphas}}
                for p in grid.pts.values()]
    return {"rows": rows, "grid": gridrows, "raw": raw}


# ─── reporting ───────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: list) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _tables(rows: list, outdir: str) -> None:
    hdr = ("| Mode | W | Nv | Nn | a | (ci,cc)_A | (ci,cc)_O | T_A/T_O | "
           "rho_A/rho_O | G_J [95% CI] |")
    md = [hdr, "|" + "---|" * 10]
    tex = [r"\begin{tabular}{llrrrccccc}", r"\hline",
           r"Mode & $W$ & $N_v$ & $N_n$ & $\alpha$ & $(c_i,c_c)_A$ & "
           r"$(c_i,c_c)_O$ & $T_A/T_O$ & $\rho_A/\rho_O$ & $G_J$ \\", r"\hline"]
    for r in rows:
        ca = f"({r['c_idle_A']:.2f},{r['c_coll_A']:.2f})"
        co = f"({r['c_idle_O']:.2f},{r['c_coll_O']:.2f})"
        tr = f"{r['T_analytic']:.3f}/{r['T_oracle']:.3f}"
        rr = f"{r['rho_analytic']:.3f}/{r['rho_oracle']:.3f}"
        gj = (f"{r['G_J']:.3f} [{r['G_J_ci'][0]:.3f}, {r['G_J_ci'][1]:.3f}]"
              if isinstance(r["G_J_ci"], tuple) else f"{r['G_J']:.3f}")
        flag = "*" if r["boundary_O"] else ""
        md.append(f"| {r['access']} | {r['w_eff']} | {r['n_vis']} | "
                  f"{r['n_nat']} | {r['alpha']} | {ca} | {co}{flag} | {tr} | "
                  f"{rr} | {gj} |")
        tex.append(f"{r['access']} & {r['w_eff']} & {r['n_vis']} & "
                   f"{r['n_nat']} & {r['alpha']} & {ca} & {co}{flag} & {tr} & "
                   f"{rr} & {r['G_J']:.3f} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    md.append("")
    md.append("`*` = oracle argmax on the search boundary.")
    with open(os.path.join(outdir, "summary_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(outdir, "summary_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")


def _heatmap(scn: Scn, rows: list, alpha: float, outdir: str,
             seeds=TUNE_SEEDS, visits: int = 40, n: int = 13) -> None:
    """J over a DENSE regular (c_idle, c_coll) grid, with the three candidates.

    Deliberately not reused from the search grid: that one is a coarse mesh plus
    per-alpha refinements, so it has holes and would render as a patchwork
    rather than a surface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    g = SimGrid(scn, seeds, visits)
    axis = np.geomspace(EPS_LO, EPS_HI, n)
    z = np.array([[objective(g(ei, ec)["T"], g(ei, ec)["rho"], alpha)
                   for ei in axis] for ec in axis])
    c = np.exp(axis)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    m = ax.pcolormesh(c, c, z, shading="gouraud", cmap="viridis")
    fig.colorbar(m, ax=ax, label=r"$J$ (tuning seeds)")
    ax.contour(c, c, z, levels=12, colors="w", linewidths=0.4, alpha=0.7)
    r = next(x for x in rows if x["alpha"] == alpha)
    ax.plot(1.2, 1.2, "*", color="w", ms=14, mec="k", label="current (1.2, 1.2)")
    ax.plot(r["c_idle_A"], r["c_coll_A"], "o", color="tab:red", mec="k",
            label="analytic (DP)")
    ax.plot(r["c_idle_O"], r["c_coll_O"], "s", color="tab:orange", mec="k",
            label="simulation oracle")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(ScalarFormatter())
        a.set_minor_formatter(NullFormatter())
    ax.set_xticks([1.1, 1.3, 1.5, 1.8, 2.2])
    ax.set_yticks([1.1, 1.3, 1.5, 1.8, 2.2])
    ax.set_xlabel(r"$c_{\mathrm{idle}}$")
    ax.set_ylabel(r"$c_{\mathrm{coll}}$")
    ax.set_title(f"{scn.tag()}, " r"$\alpha$=" f"{alpha}")
    ax.legend(fontsize=7, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"heatmap_{scn.tag()}_a{alpha}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    _write_csv(os.path.join(outdir, f"heatmap_{scn.tag()}_a{alpha}.csv"),
               list(g.pts.values()))


def proportionality(rho: float) -> float:
    """F = exp(-(ln rho)^2), the fairness axis of the Pareto view.

    J = ln T - alpha (ln rho)^2 is exactly ln T + alpha ln F, so exp(J) = T F^a
    and the objective is a weighted geometric mean of the two goals. F folds
    over-share and under-share together: rho = 2 and rho = 1/2 both give F =
    0.62, so a plot in F must distinguish the two branches by colour."""
    return math.exp(-math.log(max(rho, DELTA)) ** 2)


def pareto_front(pts: list) -> list:
    """The non-dominated subset of [{T, F, ...}], sorted by T.

    A point survives if nothing else has T >= and F >= with one strict. O(n^2),
    which is nothing at a few hundred grid points."""
    out = [p for p in pts
           if not any((q["T"] >= p["T"] and q["F"] >= p["F"])
                      and (q["T"] > p["T"] or q["F"] > p["F"]) for q in pts)]
    return sorted(out, key=lambda p: p["T"])


def fig_pareto(scn: Scn, outdir: str, alphas=ALPHAS, seeds=TUNE_SEEDS,
               visits: int = 40, n: int = 13) -> dict:
    """Efficiency against proportionality, with the alpha-selected operating pts.

    Every point on this figure comes from ONE sample (the tuning seeds), grid
    cloud and alpha optima alike. Drawing the cloud from tuning seeds and the
    optima from evaluation seeds would put a selection effect on the picture:
    an optimum could then plot off its own frontier, which is a sampling
    artefact and not a property of the trade-off. G_J in the tables is the
    out-of-sample question and is measured separately.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = SimGrid(scn, seeds, visits)
    axis = np.geomspace(EPS_LO, EPS_HI, n)
    for ei in axis:
        for ec in axis:
            g(ei, ec)
    refs = {"current": (1.2, 1.2), "design rule": design_rule_candidate(scn)}
    for ci, cc in refs.values():
        g(math.log(ci), math.log(cc))
    pts = [dict(p, F=proportionality(p["rho"])) for p in g.pts.values()]
    front = pareto_front(pts)

    best = {}
    for al in alphas:
        b = max(pts, key=lambda p: objective(p["T"], p["rho"], al))
        assert any(q is b for q in front), (
            f"alpha={al} optimum is dominated: J is Pareto-consistent, so this "
            "cannot happen unless the objective and the axes disagree")
        best[al] = b

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    under = [p for p in pts if p["rho"] <= 1.0]
    over = [p for p in pts if p["rho"] > 1.0]
    ax.scatter([p["T"] for p in under], [p["F"] for p in under], s=9,
               c="0.72", label=r"candidates, $\rho<1$ (visitor under-served)")
    ax.scatter([p["T"] for p in over], [p["F"] for p in over], s=9,
               marker="^", c="0.45",
               label=r"candidates, $\rho>1$ (visitor over-served)")
    ax.plot([p["T"] for p in front], [p["F"] for p in front], "-",
            color="tab:blue", lw=1.6, zorder=3, label="Pareto frontier")
    cmap = plt.get_cmap("plasma")
    for i, al in enumerate(sorted(best)):
        b = best[al]
        ax.plot(b["T"], b["F"], "o", ms=9, zorder=5, mec="k", mew=0.7,
                color=cmap(0.1 + 0.75 * i / max(len(best) - 1, 1)),
                label=rf"$\alpha$={al}: $(c_\mathrm{{idle}},c_\mathrm{{coll}})$="
                      rf"({b['c_idle']:.2f}, {b['c_coll']:.2f}), "
                      rf"$\rho$={b['rho']:.2f}")
    for (name, (ci, cc)), mk in zip(refs.items(), ("*", "D")):
        p = next(q for q in pts if abs(q["c_idle"] - ci) < 1e-6
                 and abs(q["c_coll"] - cc) < 1e-6)
        ax.plot(p["T"], p["F"], mk, ms=15 if mk == "*" else 8, mec="k",
                mew=0.9, color="w", zorder=6,
                label=f"{name}: ({ci:.2f}, {cc:.2f}), "
                      r"$\rho$="f"{p['rho']:.2f}")
    ax.set_xlabel(r"total useful airtime  $T$")
    ax.set_ylabel(r"proportionality  $F=\exp[-(\ln\rho)^2]$")
    ax.set_title(f"{scn.tag()} — efficiency vs proportionality")
    ax.grid(color="0.88", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=1, framealpha=1.0, borderaxespad=0.0,
              title=rf"$c_\mathrm{{coll}}$ fixed at 1.20 for the two references; "
                    rf"design rule $c_\mathrm{{idle}}=e^{{C/\sqrt{{W}}}}$, "
                    rf"$C$={OPT.C_DESIGN[scn.access]}",
              title_fontsize=6.5)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(os.path.join(outdir, f"pareto_{scn.tag()}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    _write_csv(os.path.join(outdir, f"pareto_{scn.tag()}.csv"),
               [dict(p, on_front=any(q is p for q in front)) for p in pts])
    return {"n_points": len(pts), "n_front": len(front),
            "best": {a: (b["c_idle"], b["c_coll"], b["T"], b["rho"])
                     for a, b in best.items()}}


def _meta(outdir: str, cores: list, args) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    meta = {
        "commit": commit, "dirty": dirty,
        "objective": "J = ln max(T,delta) - alpha * (ln max(rho,delta))^2",
        "delta": DELTA,
        "T": "(A_vis + A_nat) / (n_visits * W_eff), useful airtime only "
             "(handshake overhead excluded, as in the engine)",
        "rho": "(A_vis/(A_vis+A_nat)) / (N_vis/(N_vis+N_nat))",
        "aggregation": "pooled over all evaluation sequences, then J once",
        "G_J": "exp(J_A - J_O), both from the SAME evaluation simulator",
        "cores": [dict(zip(("n_vis", "n_nat", "w_eff", "access"), c))
                  for c in cores],
        "alphas": list(ALPHAS),
        "seeds": {"tuning": list(TUNE_SEEDS), "evaluation": list(EVAL_SEEDS),
                  "dev": list(DEV_SEEDS), "bootstrap": 20260828},
        "visits": {"tuning_per_seed": args.tune_visits,
                   "evaluation_per_seed": args.eval_visits},
        "search": {"box_eps": [EPS_LO, EPS_HI], "cap_eps": [CAP_LO, CAP_HI],
                   "coarse": 6, "fine": 5, "max_expand": 3,
                   "note": "log-spaced in eps = ln c; expansion is shared "
                           "across alphas, refinement is per alpha"},
        "n_boot": args.n_boot,
        "F_L": {"visitor": f"U[{P.PPDU_V_LO},{P.PPDU_V_HI}] slots",
                "native": f"{P.PPDU_NATIVE} slots", "slot_us": P.SLOT_US},
        "engine": {"module": "harq_sim/run_step9_fig25.py",
                   "entry": "_run_visit25", "mode": "pace",
                   "tau_0": "1/W_eff", "tau_clip": [1e-4, 1.0],
                   "traffic": "saturated"},
        "analytical_model": {
            "module": "pace-analysis/dp.py",
            "native_approximation":
                "exogenous constant tau_nat = %.3f per contention epoch; the "
                "engine instead runs standard DCF natives with frozen backoff, "
                "so the DP does not see native backoff freezing" % P.TAU_NAT,
            "population": "homogeneous in tau (one tau per DP state); the "
                          "engine carries N_vis independent tau values",
            "lattice": "(c_coll, r) with r rationally approximated; the "
                       "ACHIEVED c_idle is reported, not the requested one",
            "viability": "closed-form E|V(t)| from viability.py"},
        "known_gaps": [
            "The DP fixes F_L at the engine's U[25,100]; the F_L axis in the "
            "brief is therefore not swept.",
            "eps_coll is weakly identified within one visit (section 4.5.3/"
            "4.5.4): a flat ridge means the reported oracle c_coll is one point "
            "of a near-optimal set, and the near_c_coll columns give its width.",
        ],
    }
    with open(os.path.join(outdir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _worker(a):
    core, kw = a
    try:
        return core, run_scenario(core, **kw)
    except Exception as e:                                  # keep the sweep up
        return core, {"rows": [], "grid": [], "raw": [], "error": repr(e)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true",
                    help="the whole Cartesian product (160 cores)")
    ap.add_argument("--subset", action="store_true", help="8 cores (default)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tune-visits", type=int, default=40)
    ap.add_argument("--eval-visits", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-eq", action="store_true",
                    help="skip the equilibrium-ratio column")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--raw", action="store_true",
                    help="write per-seed evaluation rows (large)")
    args = ap.parse_args(argv)

    cores = ([tuple(c) for c in itertools.product(
        AXES["n_vis"], AXES["n_nat"], AXES["w_eff"], AXES["access"])]
        if args.full else SUBSET)
    outdir = args.out + ("_full" if args.full else "")
    os.makedirs(outdir, exist_ok=True)

    kw = dict(tune_visits=args.tune_visits, eval_visits=args.eval_visits,
              n_boot=args.n_boot, with_eq=not args.no_eq, keep_rows=args.raw)
    todo = [(c, kw) for c in cores]
    results = []
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, (core, res) in enumerate(ex.map(_worker, todo), 1):
                print(f"[{i}/{len(todo)}] {Scn(*core).tag()}"
                      + (f"  ERROR {res.get('error')}" if "error" in res
                         else ""), flush=True)
                results.append((core, res))
    else:
        for i, t in enumerate(todo, 1):
            core, res = _worker(t)
            print(f"[{i}/{len(todo)}] {Scn(*core).tag()}"
                  + (f"  ERROR {res.get('error')}" if "error" in res else ""),
                  flush=True)
            results.append((core, res))

    rows = [r for _c, res in results for r in res["rows"]]
    grid = [g for _c, res in results for g in res["grid"]]
    raw = [x for _c, res in results for x in res["raw"]]
    for r in rows:                                  # CSV-friendly CI columns
        for k in ("G_J_ci", "dJ_ci", "dT_ci", "drho_ci"):
            r[k + "_lo"], r[k + "_hi"] = r[k]
    _write_csv(os.path.join(outdir, "scenario_results.csv"),
               [{k: v for k, v in r.items() if not isinstance(v, tuple)}
                for r in rows])
    _write_csv(os.path.join(outdir, "oracle_grid.csv"), grid)
    if raw:
        _write_csv(os.path.join(outdir, "evaluation_raw.csv"), raw)
    _tables(rows, outdir)
    _meta(outdir, cores, args)

    for core, res in results[:2]:
        if res["rows"]:
            for al in (0.05, 0.2):
                _heatmap(Scn(*core), res["rows"], al, outdir,
                         visits=args.tune_visits)
            p = fig_pareto(Scn(*core), outdir, visits=args.tune_visits)
            print(f"  pareto {Scn(*core).tag()}: {p['n_front']}/{p['n_points']} "
                  f"points on the frontier")

    print(f"\nwrote {len(rows)} scenario rows to {outdir}")
    bad = [r for r in rows if r["boundary_O"]]
    if bad:
        print(f"WARNING: {len(bad)} oracle argmax on the search boundary "
              f"(capped: {sum(1 for r in bad if r['capped_O'])})")
    print(f"G_J: min {min(r['G_J'] for r in rows):.3f}  "
          f"median {float(np.median([r['G_J'] for r in rows])):.3f}  "
          f"max {max(r['G_J'] for r in rows):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
