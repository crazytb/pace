"""c_idle* over the FULL scenario set, on a grid fine enough to trust an argmax.

    .venv/bin/python pace-analysis/cidle_scan.py

Written because the coarse 7-point c_idle grid gave an argmax that moved with
the interpolation window (C came out 10.2 or 12.9 depending on the fit), and
reading a few rows off that table produced four separate over-generalisations.
So: every scenario, every alpha, one table, and no claim that is not a
min/median/max over the whole of it.

c_coll is pinned at 1.20 throughout. Section 4.5.22 measured that c_idle* moves
by a median 1.13x over c_coll in [1.05, 1.70], so this fixes the axis rather
than pretending it does not exist; the residual is reported, not hidden.

alpha is post-hoc: T and rho do not depend on it, so one simulation per
(scenario, c_idle) serves every alpha.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import coeff_oracle as CO

C_COLL = 1.20
EPS = np.geomspace(0.08, 1.00, 16)          # ln c_idle, 16 points
CIDLE = [float(math.exp(e)) for e in EPS]
ALPHAS = (0.25, 0.5, 0.75, 1.0)
WS = (105, 210, 420, 840, 1680)
NVS = (5, 10, 20, 50)
SCEN = [(nv, 10, w, ac) for ac in ("rts", "basic") for w in WS for nv in NVS]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "cidle_scan")


def measure(core):
    scn = CO.Scn(*core)
    rows = []
    for ci in CIDLE:
        m = CO.aggregate(CO.batch(scn, ci, C_COLL, CO.EVAL_SEEDS, 30), scn, 0.0)
        rows.append({"access": scn.access, "w_eff": scn.w_eff,
                     "n_vis": scn.n_vis, "n_nat": scn.n_nat,
                     "c_coll": C_COLL, "c_idle": round(ci, 5),
                     "eps_idle": round(math.log(ci), 5),
                     "T": round(m["T"], 5), "rho": round(m["rho"], 5)})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, part in enumerate(ex.map(measure, SCEN), 1):
            rows += part
            r = part[0]
            print(f"[{i}/{len(SCEN)}] {r['access']}_W{r['w_eff']}_v{r['n_vis']}",
                  flush=True)
    path = os.path.join(OUT, "data.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
