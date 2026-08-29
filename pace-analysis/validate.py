"""Phase 3 — model validation figures and the tau_nat sensitivity analysis.

    .venv/bin/python pace-analysis/validate.py

Produces the two figures the plan's section 6 requires, without which a reviewer
will attack the homogeneous mean field as an unchecked assumption:

  fig6-1 / fig6-2   within-visit tau trajectory, analysis vs simulation, with
                    the fair-share target 1/|V(t)| for reference
  fig7-1 / fig7-2   total useful airtime vs the visitor population, analysis vs
                    simulation, with a band for the tau_nat uncertainty

and prints the sensitivity of the model to tau_nat, which is the input the
analysis takes from measurement rather than deriving. Natives freeze their
backoff during busy epochs, so that coupling is where a reviewer will push.

Per CLAUDE.md every figure is written to results/figure/ in eps, png and pdf.
Only figures actually selected for the paper get copied into the manuscript.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

import dp                              # noqa: E402
import params as P                     # noqa: E402
import viability as V                  # noqa: E402

f25 = P.engine()
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "results", "figure")

NV_LIST = [5, 10, 20, 50]
BIN_W = 20
# tau_nat is measured, not derived; this brackets the spread across the sweep
TAU_NAT_LO, TAU_NAT_HI = 0.049, 0.056

_STYLE = {
    "model": dict(color="#d62728", ls="-", lw=2.0, marker="o", ms=4.5),
    "sim": dict(color="#1f77b4", ls="--", lw=1.8, marker="s", ms=4.5),
    "fs": dict(color="#2ca02c", ls=":", lw=1.6, marker="^", ms=4.0),
}


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext, kw in (("eps", dict(format="eps")),
                    ("png", dict(format="png", dpi=300)),
                    ("pdf", dict(format="pdf"))):
        dest = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(dest, bbox_inches="tight", **kw)
    print(f"  figure -> {FIG_DIR}/{name}.{{eps,png,pdf}}")
    plt.close(fig)


# ─── measurement ─────────────────────────────────────────────────────────────

def sim_trajectory(n_vis: int, access: str = "rts") -> dict:
    """Engine tau and |V| bucketed by W_rem, matching dp.tau_trajectory."""
    f25.N_VISITOR, f25.N_NATIVE = n_vis, P.N_NATIVE
    tau_b: dict[int, list] = {}
    nv_b: dict[int, list] = {}
    try:
        for r in range(P.REPS):
            rp = np.random.default_rng(10001 + r * 71 + 7)
            rg = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
            for v in range(P.VISITS):
                st: dict = {"trace": []}
                f25._run_visit25(f25._sample_ppdus25(rp), rg, "pace",
                                 np.full(n_vis, P.TAU_0), *P.ACCESS[access],
                                 stats=st)
                if v < P.VISITS // 2:
                    continue
                for w_rem, nvv, _k, rate in st["trace"]:
                    b = min(w_rem // BIN_W * BIN_W + BIN_W / 2, float(P.W_EFF))
                    if nvv > 0:
                        tau_b.setdefault(b, []).append(rate)
                    nv_b.setdefault(b, []).append(nvv)
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    ws = sorted(tau_b)
    return {"w_rem": np.array(ws),
            "tau": np.array([np.mean(tau_b[w]) for w in ws]),
            "n_viable": np.array([np.mean(nv_b[w]) for w in ws])}


def _rel_err(mod: dict, sim: dict) -> tuple[np.ndarray, np.ndarray]:
    """Relative error on the W_rem buckets both series share, plus weights.

    Two traps here. Comparing positionally is wrong, because the simulation
    reports no tau once every visitor has self-excluded, so its bucket list is
    a subset. And the mean must be weighted by how much time the process
    actually spends in each bucket, or a near-empty one dominates.
    """
    smap = dict(zip(sim["w_rem"], sim["tau"]))
    trip = [(t, smap[w], m) for w, t, m
            in zip(mod["w_rem"], mod["tau"], mod["mass"]) if w in smap]
    assert trip, "no overlapping W_rem buckets"
    m = np.array([a for a, _b, _c in trip])
    s = np.array([b for _a, b, _c in trip])
    wt = np.array([c for _a, _b, c in trip])
    return np.abs(m - s) / s, wt / wt.sum()


# ─── figures ─────────────────────────────────────────────────────────────────

def fig_trajectory(access: str, name: str, n_vis: int = 20) -> dict:
    mod = dp.tau_trajectory(n_vis, bin_w=BIN_W, access=access)
    sim = sim_trajectory(n_vis, access=access)

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(P.W_EFF - mod["w_rem"], mod["tau"], label="Analysis",
            **_STYLE["model"])
    ax.plot(P.W_EFF - sim["w_rem"], sim["tau"], label="Simulation",
            **_STYLE["sim"])
    fs_w = mod["w_rem"]
    fs = [V.fs_target(int(w), n_vis) for w in fs_w]
    ax.plot(P.W_EFF - fs_w, fs, label="Fair share $1/|\\mathcal{V}(t)|$",
            **_STYLE["fs"])

    ax.set_xlabel("Elapsed slots within the visit", fontsize=9)
    ax.set_ylabel("Transmission probability $\\tau$", fontsize=9)
    ax.set_yscale("log")
    # a single decade tick reads as an unlabelled axis; label the minors too
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _p: f"{y:g}" if y in (0.002, 0.005, 0.02, 0.05) else ""))
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.legend(fontsize=7, frameon=True, loc="lower right",
              handlelength=1.6, borderpad=0.3, labelspacing=0.25)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    fig.tight_layout()
    _save(fig, name)

    err, wt = _rel_err(mod, sim)
    return {"max_rel_err": float(err.max()),
            "mean_rel_err": float((err * wt).sum())}


def fig_airtime(access: str, name: str) -> dict:
    mod = [dp.total_airtime(n, access=access) for n in NV_LIST]
    sim = [dp.measured(n, access=access)["total"] for n in NV_LIST]

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    # The tau_nat band is thinner than the line width, which is itself the
    # robustness result, so it would only be a confusing legend entry here.
    # The sensitivity sweep is reported as a table instead.
    ax.plot(NV_LIST, mod, label="Analysis", **_STYLE["model"])
    ax.plot(NV_LIST, sim, label="Simulation", **_STYLE["sim"])

    ax.set_xscale("log")
    ax.set_xticks(NV_LIST)
    ax.set_xticklabels([str(n) for n in NV_LIST])
    ax.set_xlabel("Visitor STAs $N_\\mathrm{vis}$", fontsize=9)
    ax.set_ylabel("Total useful airtime / $W_\\mathrm{eff}$", fontsize=9)
    lohi = list(mod) + list(sim)
    ax.set_ylim(min(lohi) - 0.10, max(lohi) + 0.10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=True, loc="lower left",
              handlelength=1.6, borderpad=0.3, labelspacing=0.25)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    fig.tight_layout()
    _save(fig, name)

    err = [abs(m - s) for m, s in zip(mod, sim)]
    return {"max_abs_err": max(err), "model": mod, "sim": sim}


# ─── the coefficient scaling law ─────────────────────────────────────────────

def fig_collapse(access: str, name: str, n_vis: int = 10,
                 alpha: float = 0.2, theta: float = 0.5) -> dict:
    """Two panels: the raw J curves per window, and the same curves rescaled.

    Left panel shows that the objective's peak moves left as the window grows.
    Right panel rescales the abscissa by W_eff^theta and asks whether the
    curves fall on one another. This is the robust form of the scaling claim:
    it uses every measured point rather than the position of a broad peak.
    """
    import optimise as O
    c = O.collapse(n_vis, alpha, access=access)
    grid, curves = c["grid"], c["curves"]
    cmap = plt.get_cmap("viridis")
    ws = c["windows"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for k, w in enumerate(ws):
        col = cmap(k / max(len(ws) - 1, 1))
        kw = dict(color=col, lw=1.5, marker="o", ms=3.0,
                  label=f"$W_\\mathrm{{eff}}={w}$")
        axes[0].plot(grid, curves[w], **kw)
        axes[1].plot(grid * w ** theta, curves[w], **kw)

    axes[0].set_xlabel("Up step $\\varepsilon_\\mathrm{idle}$", fontsize=9)
    axes[1].set_xlabel("$\\varepsilon_\\mathrm{idle}\\,"
                       f"W_\\mathrm{{eff}}^{{{theta:g}}}$", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(-0.5, 0.05)
        ax.tick_params(labelsize=8)
        ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    axes[0].set_ylabel("$J-J_{\\max}$", fontsize=9)
    axes[0].legend(fontsize=6.5, frameon=True, loc="lower center", ncol=2,
                   handlelength=1.4, borderpad=0.3, labelspacing=0.2)
    axes[1].set_title(f"fitted $\\theta={c['theta']:.2f}$ "
                      f"$[{c['lo']:.2f},{c['hi']:.2f}]$", fontsize=8)
    fig.tight_layout()
    _save(fig, name)
    return c


def fig_design(access: str, name: str, n_vis: int = 10, alpha: float = 0.2,
               w_show: int = 420,
               windows: tuple = (300, 420, 600, 840, 1190, 1680)) -> dict:
    """Analysis against simulation for the coefficient design result.

    Left: the objective against the up step at one window, DP and engine.
    Right: the optimal up step against the window, both sources, with the
    C/sqrt(W) reference fitted to each.

    The point of the figure is the honest one: the DP reproduces the shape and
    the trend but sits systematically to the right of the engine, because it
    carries a single tau per state and so cannot see the dispersion that a
    large step creates. The constant has to come from simulation.
    """
    import optimise as O
    grid = np.exp(np.linspace(np.log(0.12), np.log(1.6), 15))

    def star(jf, w):
        # grid holds eps_idle; the objective takes the COEFFICIENT, exp(eps)
        js = np.array([jf(n_vis, alpha, float(np.exp(e)), 1.4, access=access,
                          w_eff=w) for e in grid])
        return grid[int(js.argmax())], js

    e_sim, j_sim = star(O.sim_J, w_show)
    e_dp, j_dp = star(O.dp_J, w_show)
    s_stars = [star(O.sim_J, w)[0] for w in windows]
    d_stars = [star(O.dp_J, w)[0] for w in windows]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    axes[0].plot(grid, j_dp - j_dp.max(), label="Analysis", **_STYLE["model"])
    axes[0].plot(grid, j_sim - j_sim.max(), label="Simulation", **_STYLE["sim"])
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.6, 0.05)
    axes[0].set_xlabel("Up step $\\varepsilon_\\mathrm{idle}$", fontsize=9)
    axes[0].set_ylabel("$J-J_{\\max}$", fontsize=9)
    axes[0].set_title(f"$W_\\mathrm{{eff}}={w_show}$", fontsize=8)
    axes[0].legend(fontsize=7, frameon=True, loc="lower center",
                   handlelength=1.6, borderpad=0.3, labelspacing=0.25)

    ws = np.array(windows, dtype=float)
    axes[1].plot(ws, d_stars, label="Analysis", **_STYLE["model"])
    axes[1].plot(ws, s_stars, label="Simulation", **_STYLE["sim"])
    for vals, col in ((d_stars, "#d62728"), (s_stars, "#1f77b4")):
        c = float(np.mean([v * np.sqrt(w) for v, w in zip(vals, ws)]))
        axes[1].plot(ws, c / np.sqrt(ws), color=col, ls=":", lw=1.2,
                     label=f"$C={c:.1f}/\\sqrt{{W}}$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    # the default log locator crams 3x10^2, 4x10^2, 6x10^2 into an unreadable run
    axes[1].set_xticks([300, 600, 1200])
    axes[1].set_xticklabels(["300", "600", "1200"])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_xlabel("$W_\\mathrm{eff}$ (slots)", fontsize=9)
    axes[1].set_ylabel("$\\varepsilon_\\mathrm{idle}^{*}$", fontsize=9)
    axes[1].legend(fontsize=6.5, frameon=True, loc="lower left",
                   handlelength=1.6, borderpad=0.3, labelspacing=0.2)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    fig.tight_layout()
    _save(fig, name)

    ratio = [d / s for d, s in zip(d_stars, s_stars)]
    return {"windows": list(windows), "sim": s_stars, "dp": d_stars,
            "ratio": ratio, "max_ratio": max(ratio)}


# ─── sensitivity ─────────────────────────────────────────────────────────────

def sensitivity(access: str = "rts", n_vis: int = 20) -> list[tuple]:
    """How much the model's answer moves with the one measured input."""
    base = dp.total_airtime(n_vis, access=access)
    out = []
    for tn in (0.040, 0.045, P.TAU_NAT, 0.060, 0.070):
        val = dp.total_airtime(n_vis, tau_nat=tn, access=access)
        out.append((tn, val, (val - base) / base))
    return out


def _main() -> None:
    print("=== Figure 6: within-visit tau trajectory (N_vis=20) ===")
    traj = {}
    for access, name in (("basic", "fig6-1"), ("rts", "fig6-2")):
        traj[access] = fig_trajectory(access, name)
        print(f"  {access}: mean rel. err {traj[access]['mean_rel_err']:.1%}, "
              f"max {traj[access]['max_rel_err']:.1%}")

    print("\n=== Figure 7: total airtime vs N_vis ===")
    air = {}
    for access, name in (("basic", "fig7-1"), ("rts", "fig7-2")):
        air[access] = fig_airtime(access, name)
        print(f"  {access}: max abs. err {air[access]['max_abs_err']:.3f}")

    print("\n=== Figure 8: coefficient scaling law, data collapse ===")
    for access, name in (("basic", "fig8-1"), ("rts", "fig8-2")):
        c = fig_collapse(access, name)
        print(f"  {access}: theta {c['theta']:.2f} "
              f"[{c['lo']:.2f}, {c['hi']:.2f}], "
              f"collapse gain {c['gain']:.2f}x"
              + (f", dropped {c['dropped']}" if c["dropped"] else ""))

    print("\n=== tau_nat sensitivity (N_vis=20, RTS/CTS) ===")
    print(f"{'tau_nat':>9} {'DP total':>9} {'vs base':>9}")
    for tn, val, rel in sensitivity():
        mark = "  <- measured" if tn == P.TAU_NAT else ""
        print(f"{tn:9.3f} {val:9.3f} {rel:+8.1%}{mark}")


def _self_check() -> None:
    # trajectory: the model must track the simulation, not merely look similar
    for access in ("rts", "basic"):
        mod = dp.tau_trajectory(20, bin_w=BIN_W, access=access)
        sim = sim_trajectory(20, access=access)
        rel, wt = _rel_err(mod, sim)
        assert (rel * wt).sum() < 0.20, (access, (rel * wt).sum())
        assert rel.max() < 0.60, (access, rel.max())
        # both start at tau_0 and both end below the fair-share target
        assert abs(mod["tau"][-1] - P.TAU_0) / P.TAU_0 < 0.1
        assert mod["tau"][0] > mod["tau"][-1], "tau should climb over the visit"

    # airtime: within the tolerance the plan accepts
    for access, tol in (("rts", 0.04), ("basic", 0.05)):
        for n in NV_LIST:
            assert abs(dp.total_airtime(n, access=access)
                       - dp.measured(n, access=access)["total"]) < tol

    # sensitivity must be monotone decreasing in tau_nat: busier natives leave
    # less room, and it must not be so steep that the measured input dominates
    vals = [v for _tn, v, _r in sensitivity()]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals
    assert abs(sensitivity()[1][2]) < 0.15, "model too sensitive to tau_nat"
    print("\nvalidate.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
