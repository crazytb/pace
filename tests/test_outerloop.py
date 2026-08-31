# -*- coding: utf-8 -*-
"""Section 4.5.29: the per-visit adaptation loop."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pace-analysis"))

import coeff_oracle as CO      # noqa: E402
import outerloop as OL         # noqa: E402


@pytest.mark.parametrize("c0", [1.05, 1.20, 2.50])
def test_loop_converges_to_the_same_place_from_any_start(c0):
    """The point of the outer loop: it has the initialisation-independence the
    inner tau loop does not (section 4.5.24 measured that one collapsing from a
    U(0,1) start). Converged c must not depend on where it began."""
    scn = CO.Scn(20, 10, 420, "rts")
    r = OL.run_sequence(scn, 101, 0.5, c0, n_visits=90)
    assert 1.2 < r["c_final"] < 1.8, (c0, r["c_final"])


def test_the_observable_needs_nothing_the_station_cannot_count():
    """q is built from idle, visitor-solo and visitor-collision counts only.
    If it ever needed N_vis, N_nat or W_eff the rule would not be deployable."""
    src = open(os.path.join(os.path.dirname(OL.__file__),
                            "outerloop.py")).read()
    body = src[src.index("def run_sequence"):src.index("def job")]
    q_line = [l for l in body.splitlines() if "q = idle / den" in l]
    assert q_line, "the observable moved; re-check what it depends on"
    assert "n_vis" not in body.split("den = idle + sv + cv")[1].split("c =")[0]


def test_setpoints_are_ordered_by_alpha():
    """A larger fairness weight pulls the target tau up, which means fewer idle
    epochs, so q* must fall as alpha rises."""
    qs = [OL.Q_STAR[a] for a in (0.25, 0.5, 1.0)]
    assert qs == sorted(qs, reverse=True)


def test_c_stays_inside_the_box():
    scn = CO.Scn(50, 20, 210, "basic")
    r = OL.run_sequence(scn, 102, 1.0, 3.9, n_visits=40)
    lo, hi = OL.C_BOX
    assert all(lo <= c <= hi for c, _ in r["trace"])


def test_a_bigger_step_really_does_lower_q():
    """The sign of the update rests on q decreasing in c. If that inverted, the
    loop would run away from the setpoint instead of toward it."""
    scn = CO.Scn(20, 10, 420, "rts")
    seen = []
    for c in (1.05, 1.5, 2.2):
        m = CO.aggregate(CO.batch(scn, c, c, CO.EVAL_SEEDS[:8], 12), scn, 0.0)
        den = m["idle_ep_frac"] + m["solo_vis_frac"] + m["coll_vis_per_ep"]
        seen.append(m["idle_ep_frac"] / den)
    assert seen[0] > seen[1] > seen[2], seen
