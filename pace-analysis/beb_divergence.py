"""Lemma (section 4.4) — BEB's entry point is population-blind.

    .venv/bin/python pace-analysis/beb_divergence.py

The original claim in PACE_TWC_ANALYSIS.md was "BEB diverges monotonically from
the rising target". Measurement refutes it in that direction: BEB *converges
toward* the target over a visit. Two premises were wrong.

  Refuted P1  "no recovery path inside the window". A DCF solo winner resets CW
              to CW_min (run_step9_fig25.py:268-270) under saturated traffic.
  Refuted P2  "BEB falls below the target". BEB starts far ABOVE it. Its entry
              rate is 2/(CW_min+1) = 0.1176 against a target 1/k of 0.033 at
              k=30, so it is over-aggressive by 3.5x and the CW growth that
              follows moves it toward the target, not away.

What is actually true, and what the manuscript should claim instead:

  (a) BEB enters every visit at 2/(CW_min+1) regardless of the contender count,
      so it is over-aggressive by 2k/(CW_min+1), a factor growing LINEARLY in k.
  (b) Correcting that needs log2(2k/(CW_min+1)) CW doublings per STA, and every
      doubling is purchased with a collision epoch.
  (c) PACE enters under-aggressive at 1/W_eff and corrects upward on idle
      epochs, which cost one slot, and every listener steps up together.

So the two schemes approach the target from opposite sides and the finite window
prices the two errors very differently. In a visit affording ~21 epochs, BEB's
corrective path consumes the airtime it is trying to conserve, while PACE's
costs almost nothing. This is why PACE wins despite ending FURTHER from the
target in probability space, which the trajectory table below makes explicit.
"""
from __future__ import annotations

import math

import numpy as np

import params as P

f25 = P.engine()
VISITS, REPS, MAX_EPOCH = P.VISITS, P.REPS, P.MAX_EPOCH
CW_MIN = P.CW_MIN
BEB_ENTRY = P.BEB_ENTRY                # 2/(CW_min+1), independent of population


def entry_overshoot(k: int) -> float:
    """How many times too aggressive BEB is on entry, against the FS target."""
    return BEB_ENTRY * k


def doublings_needed(k: int) -> float:
    """CW doublings, hence collisions, to bring BEB down to the target."""
    return max(math.log2(entry_overshoot(k)), 0.0)


def run(mode: str, n_vis: int, n_nat: int = 10) -> tuple[np.ndarray, dict]:
    """Per-epoch mean (n_viable, k, rate, target) plus whole-visit tallies."""
    f25.N_VISITOR, f25.N_NATIVE = n_vis, n_nat
    acc = [[] for _ in range(MAX_EPOCH)]
    agg: dict = {}
    try:
        for r in range(REPS):
            rng_p = np.random.default_rng(10001 + r * 71 + 7)
            rng = np.random.default_rng(200003 + r * 3163 + n_vis * 211)
            for v in range(VISITS):
                ppdus = f25._sample_ppdus25(rng_p)
                tau0 = (np.full(n_vis, P.TAU_0)
                        if mode.startswith(("pace", "oracle")) else None)
                st: dict = {"trace": []}
                f25._run_visit25(ppdus, rng, mode, tau0,
                                 *P.ACCESS['rts'], stats=st)
                if v < VISITS // 2:
                    continue
                for key in ("epochs", "idle", "coll", "solo_vis", "solo_nat"):
                    agg[key] = agg.get(key, 0) + st.get(key, 0)
                for e, (_wr, nvv, k, rate) in enumerate(st["trace"][:MAX_EPOCH]):
                    acc[e].append((nvv, k, rate, 1.0 / k if k else 0.0))
    finally:
        f25.N_VISITOR, f25.N_NATIVE = P.N_VISITOR, P.N_NATIVE
    tr = np.array([np.mean(a, axis=0) if a else [np.nan] * 4 for a in acc])
    return tr, agg


def _main() -> None:
    print(f"BEB entry rate 2/(CW_min+1) = {BEB_ENTRY:.4f}   "
          f"(CW_min={CW_MIN}, population-blind)")
    print(f"\n(a) entry over-aggression grows linearly in k")
    print(f"{'k':>5} {'target 1/k':>11} {'overshoot':>10} {'(b) collisions':>15}")
    for k in (20, 30, 40, 60, 110):
        print(f"{k:>5} {1 / k:11.4f} {entry_overshoot(k):9.2f}x "
              f"{doublings_needed(k):15.2f}")

    for n_vis in (20, 50):
        d, da = run("dcf_excl", n_vis)
        p, pa = run("pace", n_vis)
        ok = np.isfinite(d[:, 0]) & np.isfinite(p[:, 0])
        print(f"\n=== N_vis={n_vis}, N_nat=10, RTS/CTS ===")
        print(f"{'epoch':>6} {'|V|':>6} {'target':>8} | {'dcf rate':>9} "
              f"{'dcf/tgt':>8} | {'pace tau':>9} {'pace/tgt':>9}")
        for e in range(0, MAX_EPOCH, 3):
            if not ok[e]:
                continue
            print(f"{e:>6} {d[e, 0]:6.1f} {d[e, 3]:8.4f} | {d[e, 2]:9.4f} "
                  f"{d[e, 2] / d[e, 3]:7.2f}x | {p[e, 2]:9.4f} "
                  f"{p[e, 2] / p[e, 3]:8.2f}x")
        print(f"  dcf  approaches from ABOVE: "
              f"{d[ok, 2][0] / d[ok, 3][0]:.2f}x -> {d[ok, 2][-1] / d[ok, 3][-1]:.2f}x")
        print(f"  pace approaches from BELOW: "
              f"{p[ok, 2][0] / p[ok, 3][0]:.2f}x -> {p[ok, 2][-1] / p[ok, 3][-1]:.2f}x")
        for name, a in (("dcf", da), ("pace", pa)):
            e = a["epochs"]
            print(f"  {name:>4} epochs/visit {e / (REPS * (VISITS - VISITS // 2)):5.1f}"
                  f"  idle {a['idle'] / e:.3f}  coll {a['coll'] / e:.3f}"
                  f"  solo_vis {a['solo_vis'] / e:.3f}")


def _self_check() -> None:
    # (a) the entry point ignores the population, so overshoot scales with k
    assert abs(BEB_ENTRY - 2 / 17) < 1e-12
    assert entry_overshoot(60) / entry_overshoot(30) == 2.0
    assert doublings_needed(30) > 1.5 and doublings_needed(60) > 2.5

    d, da = run("dcf_excl", 20)
    p, pa = run("pace", 20)
    ok = np.isfinite(d[:, 0]) & np.isfinite(p[:, 0])
    tgt, drate, prate = d[ok, 3], d[ok, 2], p[ok, 2]

    # the FS target is non-decreasing: viability is lost and never regained
    assert np.all(np.diff(tgt) >= -1e-12), "target not monotone"

    # BEB sits ABOVE the target throughout and moves toward it (refutes the
    # original "diverges" claim)
    assert np.all(drate > tgt), "BEB was expected above the target"
    assert drate[-1] / tgt[-1] < drate[0] / tgt[0], "BEB should converge"

    # PACE sits BELOW the target for the whole visit and never overshoots it,
    # so it never pays BEB's collision price. Its ratio is non-monotone: tau
    # climbs faster than the target early, then the target outruns it once
    # self-exclusion starts shrinking |V(t)|.
    pr = prate / tgt
    assert np.all(prate < tgt), "PACE was expected below the target"
    assert pr.max() < 0.5, f"PACE got unexpectedly close to the target: {pr.max()}"
    peak = int(np.argmax(pr))
    assert 0 < peak < len(pr) - 1, f"expected an interior peak, got {peak}"

    # the asymmetry that decides the outcome: BEB's correction is bought with
    # collisions, PACE's with idle epochs
    dc, pc = da["coll"] / da["epochs"], pa["coll"] / pa["epochs"]
    di, pi = da["idle"] / da["epochs"], pa["idle"] / pa["epochs"]
    assert dc > pc, (dc, pc)
    assert pi > di, (pi, di)
    print("\nbeb_divergence.py self-check: OK")


if __name__ == "__main__":
    _main()
    _self_check()
