"""
Step 6: Reward module — normalize_metrics(), compute_reward(), INTENT_PROFILES.

Guidelines §19  Reward design (raw → normalized → weighted reward)
Guidelines §20  Intent-based reward profiles (5 profiles)

Usage:
    from harq_sim.reward import INTENT_PROFILES, normalize_metrics, compute_reward

    agg = sim.compute_metrics()["aggregate"]
    norm = normalize_metrics(agg)
    reward = compute_reward(norm, INTENT_PROFILES["throughput"]["weights"],
                            INTENT_PROFILES["throughput"]["constraints"], agg)
"""
from __future__ import annotations

from typing import Optional

from harq_sim.configs import (
    REWARD_THROUGHPUT_REF, REWARD_DELAY_REF, REWARD_D95_REF,
    REWARD_ENERGY_REF, REWARD_LEGACY_REF,
    REWARD_PACKET_LOSS_MAX, REWARD_P95_DELAY_MAX, REWARD_LEGACY_DEGRADATION_MAX,
    REWARD_LAMBDA_LOSS, REWARD_LAMBDA_DELAY, REWARD_LAMBDA_LEGACY,
)

__all__ = [
    "INTENT_PROFILES",
    "DEFAULT_REFS",
    "normalize_metrics",
    "compute_reward",
]

# ─────────────────────────────────────────────────────────────────────────────
# Intent profiles  (guidelines §20)
# Each profile: weights (must sum to 1.0) + constraints dict
# ─────────────────────────────────────────────────────────────────────────────
INTENT_PROFILES: dict = {
    "throughput": {
        "weights": {
            "throughput":        0.45,
            "delay":             0.10,
            "tail_delay":        0.05,
            "packet_loss":       0.10,
            "collision":         0.10,
            "fairness":          0.10,
            "energy":            0.05,
            "legacy_protection": 0.05,
        },
        "constraints": {
            "packet_loss_max":        0.10,
            "p95_delay_max":          500.0,
            "legacy_degradation_max": 0.30,
        },
    },
    "delay_sensitive": {
        "weights": {
            "throughput":        0.10,
            "delay":             0.35,
            "tail_delay":        0.25,
            "packet_loss":       0.10,
            "collision":         0.05,
            "fairness":          0.05,
            "energy":            0.05,
            "legacy_protection": 0.05,
        },
        "constraints": {
            "packet_loss_max":        0.05,
            "p95_delay_max":          300.0,
            "legacy_degradation_max": 0.20,
        },
    },
    "qos_aware": {
        "weights": {
            "throughput":        0.25,
            "delay":             0.20,
            "tail_delay":        0.15,
            "packet_loss":       0.15,
            "collision":         0.05,
            "fairness":          0.10,
            "energy":            0.05,
            "legacy_protection": 0.05,
        },
        "constraints": {
            "packet_loss_max":        0.05,
            "p95_delay_max":          400.0,
            "legacy_degradation_max": 0.25,
        },
    },
    "fair_coexistence": {
        "weights": {
            "throughput":        0.15,
            "delay":             0.10,
            "tail_delay":        0.10,
            "packet_loss":       0.10,
            "collision":         0.10,
            "fairness":          0.25,
            "energy":            0.05,
            "legacy_protection": 0.15,
        },
        "constraints": {
            "packet_loss_max":        0.10,
            "p95_delay_max":          500.0,
            "legacy_degradation_max": 0.15,
        },
    },
    "energy_aware": {
        "weights": {
            "throughput":        0.15,
            "delay":             0.15,
            "tail_delay":        0.10,
            "packet_loss":       0.10,
            "collision":         0.05,
            "fairness":          0.10,
            "energy":            0.25,
            "legacy_protection": 0.10,
        },
        "constraints": {
            "packet_loss_max":        0.10,
            "p95_delay_max":          500.0,
            "legacy_degradation_max": 0.30,
        },
    },
}

# Default reference values for normalization (overridable per-call)
DEFAULT_REFS: dict = {
    "throughput_ref":         REWARD_THROUGHPUT_REF,
    "delay_ref":              REWARD_DELAY_REF,
    "p95_delay_ref":          REWARD_D95_REF,
    "energy_ref":             REWARD_ENERGY_REF,
    "legacy_degradation_ref": REWARD_LEGACY_REF,
}


# ─────────────────────────────────────────────────────────────────────────────
# normalize_metrics  (guidelines §19.2)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_metrics(metrics: dict, refs: Optional[dict] = None) -> dict:
    """Normalize raw aggregate metrics to [0, 1].

    Parameters
    ----------
    metrics : dict
        Aggregate metrics from ``Simulator.compute_metrics()["aggregate"]``.
    refs : dict, optional
        Override reference values. Merged with DEFAULT_REFS.

    Returns
    -------
    dict with keys: T_hat, D_hat, D95_hat, loss_hat, collision_hat,
                    fairness_hat, energy_hat, legacy_hat
    """
    r = {**DEFAULT_REFS, **(refs or {})}

    T_hat = min(
        metrics.get("aggregate_throughput", 0) / max(r["throughput_ref"], 1),
        1.0,
    )

    mean_delay = metrics.get("mean_access_delay", 0.0)
    D_hat      = max(0.0, 1.0 - min(mean_delay / max(r["delay_ref"], 1.0), 1.0))

    p95_delay = metrics.get("p95_access_delay", 0.0)
    D95_hat   = max(0.0, 1.0 - min(p95_delay / max(r["p95_delay_ref"], 1.0), 1.0))

    loss_hat = max(0.0, 1.0 - metrics.get("packet_loss_probability", 0.0))

    collision_hat = max(0.0, 1.0 - min(metrics.get("collision_probability", 0.0), 1.0))

    fairness_hat = max(0.0, min(metrics.get("jain_fairness_index", 1.0), 1.0))

    energy     = metrics.get("total_energy_uj", 0.0)
    energy_hat = max(0.0, 1.0 - min(energy / max(r["energy_ref"], 1.0), 1.0))

    legacy     = metrics.get("legacy_throughput_degradation", 0.0)
    legacy_hat = max(0.0, 1.0 - min(legacy / max(r["legacy_degradation_ref"], 1.0), 1.0))

    return {
        "T_hat":        T_hat,
        "D_hat":        D_hat,
        "D95_hat":      D95_hat,
        "loss_hat":     loss_hat,
        "collision_hat": collision_hat,
        "fairness_hat": fairness_hat,
        "energy_hat":   energy_hat,
        "legacy_hat":   legacy_hat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_reward  (guidelines §19.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    normalized:  dict,
    weights:     dict,
    constraints: Optional[dict] = None,
    raw_metrics: Optional[dict] = None,
    refs:        Optional[dict] = None,
) -> float:
    """Apply reward template and optional constraint penalties.

    Parameters
    ----------
    normalized : dict
        Output of ``normalize_metrics()``.
    weights : dict
        Intent weight profile (values should sum to 1.0).
    constraints : dict, optional
        Constraint thresholds. Expected keys:
        ``packet_loss_max``, ``p95_delay_max``, ``legacy_degradation_max``.
    raw_metrics : dict, optional
        Raw aggregate metrics; required for constraint penalty computation.
    refs : dict, optional
        Reference values used to normalize delay violation (merged with DEFAULT_REFS).

    Returns
    -------
    float
        Scalar reward.
    """
    w = weights
    n = normalized

    reward = (
        w.get("throughput",        0.0) * n.get("T_hat",        0.0)
        + w.get("delay",           0.0) * n.get("D_hat",        0.0)
        + w.get("tail_delay",      0.0) * n.get("D95_hat",      0.0)
        + w.get("packet_loss",     0.0) * n.get("loss_hat",     0.0)
        + w.get("collision",       0.0) * n.get("collision_hat", 0.0)
        + w.get("fairness",        0.0) * n.get("fairness_hat", 0.0)
        + w.get("energy",          0.0) * n.get("energy_hat",   0.0)
        + w.get("legacy_protection", 0.0) * n.get("legacy_hat", 0.0)
    )

    if constraints and raw_metrics:
        r = {**DEFAULT_REFS, **(refs or {})}

        loss   = raw_metrics.get("packet_loss_probability", 0.0)
        p95    = raw_metrics.get("p95_access_delay", 0.0)
        legacy = raw_metrics.get("legacy_throughput_degradation", 0.0)

        lmax  = constraints.get("packet_loss_max",        REWARD_PACKET_LOSS_MAX)
        dmax  = constraints.get("p95_delay_max",          REWARD_P95_DELAY_MAX)
        lgmax = constraints.get("legacy_degradation_max", REWARD_LEGACY_DEGRADATION_MAX)

        if loss > lmax:
            reward -= REWARD_LAMBDA_LOSS * (loss - lmax)

        if p95 > dmax:
            # Normalize delay violation to unit scale before applying lambda
            reward -= REWARD_LAMBDA_DELAY * (p95 - dmax) / max(r["p95_delay_ref"], 1.0)

        if legacy > lgmax:
            reward -= REWARD_LAMBDA_LEGACY * (legacy - lgmax)

    return reward
