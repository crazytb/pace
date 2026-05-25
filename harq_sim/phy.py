"""
PHY layer model for HARQ-NPCA simulation.

Guidelines §15   Channel model — probabilistic PER (Step 2+)
Guidelines §9.3  HARQ combining success probability (Step 3+: accumulated SNR)
Guidelines §9.4  MCS selection — ARQ/fresh TX can select new MCS per attempt

Step 2 (ARQ-only): each attempt independently evaluates PHY success.
Step 3 (HARQ-CC): accumulated_snr_linear = sum(snr_linear over attempts)
                  effective_snr_db = 10*log10(accumulated) → used for p_success
"""

from __future__ import annotations

import math
import random

# Simplified 802.11ax MCS SNR thresholds (dB) — AWGN, 20 MHz, NSS=1
# Adapted from IEEE 802.11ax Table 27-52 approximate sensitivity values
MCS_SNR_THRESHOLDS: dict[int, float] = {
    0:  5.0,   # BPSK  1/2
    1:  8.0,   # QPSK  1/2
    2: 11.0,   # QPSK  3/4
    3: 14.0,   # 16-QAM 1/2
    4: 17.0,   # 16-QAM 3/4
    5: 20.0,   # 64-QAM 2/3
    6: 23.0,   # 64-QAM 3/4
    7: 26.0,   # 64-QAM 5/6
}

SIGMOID_STEEPNESS: float = 1.0   # logistic curve steepness (a in §15.2)


def select_mcs(snr_db: float) -> int:
    """Select highest feasible MCS for the given SNR (guidelines §9.4)."""
    for mcs in range(7, -1, -1):
        if snr_db >= MCS_SNR_THRESHOLDS[mcs]:
            return mcs
    return 0  # fallback: BPSK 1/2


def success_prob(effective_snr_db: float, mcs: int) -> float:
    """Logistic PER success probability (guidelines §15.2).

    p = 1 / (1 + exp(-a × (SNR_eff − threshold[MCS])))
    """
    threshold = MCS_SNR_THRESHOLDS.get(mcs, MCS_SNR_THRESHOLDS[0])
    return 1.0 / (1.0 + math.exp(-SIGMOID_STEEPNESS * (effective_snr_db - threshold)))


def attempt_success(effective_snr_db: float, mcs: int) -> bool:
    """Bernoulli trial for TX success using logistic PER model (guidelines §15.2)."""
    return random.random() < success_prob(effective_snr_db, mcs)


# ─────────────────────────────────────────────────────────────────────────────
# HARQ-CC combining helpers (Step 3 onward)
# ─────────────────────────────────────────────────────────────────────────────
def snr_db_to_linear(snr_db: float) -> float:
    """dB → linear (for Chase Combining accumulation, guidelines §9.3)."""
    return 10.0 ** (snr_db / 10.0)


def snr_linear_to_db(snr_linear: float) -> float:
    """Linear → dB."""
    return 10.0 * math.log10(max(snr_linear, 1e-12))
