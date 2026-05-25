"""
Step 7: LLM Reward Designer for harq_sim.

Translates operator intent (natural language) into a validated reward weight profile.
The LLM output is constrained to a fixed JSON schema — it never generates executable code.

Guidelines §21  LLM as reward designer
Guidelines §24  Grid-best reward baseline (validate_reward_profile used as gate)

Usage:
    from harq_sim.llm_reward_designer import LLMRewardDesigner, validate_reward_profile

    designer = LLMRewardDesigner(use_mock=True)
    profile  = designer.design_reward("delay sensitive XR traffic")
    # → {"intent_name": "delay_sensitive", "weights": {...}, "constraints": {...}}

    from harq_sim.reward import normalize_metrics, compute_reward
    norm   = normalize_metrics(agg)
    reward = compute_reward(norm, profile["weights"], profile["constraints"], agg)
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from harq_sim.configs import SLOT_DURATION_US, REWARD_P95_DELAY_MAX, REWARD_PACKET_LOSS_MAX, REWARD_LEGACY_DEGRADATION_MAX
from harq_sim.reward import INTENT_PROFILES

__all__ = [
    "LLMRewardDesigner",
    "validate_reward_profile",
]

# ─────────────────────────────────────────────────────────────────────────────
# Mock intent-keyword mapping (ordered by specificity)
# ─────────────────────────────────────────────────────────────────────────────
_MOCK_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["delay", "latency", "jitter", "xr", "rtc", "voice", "video", "real-time", "realtime", "interactive"],
     "delay_sensitive"),
    (["throughput", "speed", "bandwidth", "download", "upload", "fast", "high-speed"],
     "throughput"),
    (["energy", "battery", "power", "saving", "efficient", "low-power"],
     "energy_aware"),
    (["fair", "coexist", "legacy", "coexistence", "equal", "neighbor"],
     "fair_coexistence"),
    (["qos", "quality", "service", "mixed", "balance", "general"],
     "qos_aware"),
]
_MOCK_DEFAULT = "qos_aware"

# ─────────────────────────────────────────────────────────────────────────────
# LLM system prompt (cached across calls with cache_control: ephemeral)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a wireless-network reward designer for IEEE 802.11bn NPCA-HARQ systems.

Given an operator intent description, output a reward weight profile as JSON.

Rules:
1. "weights" values must be >= 0 and sum to exactly 1.0.
2. All 8 weight keys must be present.
3. Output JSON only — no explanation, no markdown code fences.

JSON schema:
{
  "intent_name": "<throughput | delay_sensitive | qos_aware | fair_coexistence | energy_aware>",
  "weights": {
    "throughput":        <float>,
    "delay":             <float>,
    "tail_delay":        <float>,
    "packet_loss":       <float>,
    "collision":         <float>,
    "fairness":          <float>,
    "energy":            <float>,
    "legacy_protection": <float>
  },
  "constraints": {
    "packet_loss_max":        <float, 0.0–1.0>,
    "p95_delay_max_ms":       <int, milliseconds>,
    "legacy_degradation_max": <float, 0.0–1.0>
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_reward_profile(profile: dict) -> bool:
    """Validate a reward profile dict.

    Checks:
    - 'weights' key present
    - all weight values >= 0
    - weights sum == 1.0 (tolerance 1e-6)
    - 'constraints' key present

    Returns True on success; raises ValueError on any violation.
    """
    if "weights" not in profile:
        raise ValueError("Profile missing 'weights' key")

    weights = profile["weights"]
    for k, v in weights.items():
        if v < 0:
            raise ValueError(f"Weight '{k}' = {v:.6f} is negative")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Weights sum = {total:.8f}, expected 1.0 (delta={abs(total-1.0):.2e})"
        )

    if "constraints" not in profile:
        raise ValueError("Profile missing 'constraints' key")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# LLMRewardDesigner
# ─────────────────────────────────────────────────────────────────────────────

class LLMRewardDesigner:
    """Translate operator intent (natural language) into a validated reward weight profile.

    Parameters
    ----------
    use_mock : bool
        If True, select a predefined profile by keyword matching — no API call.
    model : str
        Anthropic model ID used when use_mock=False.
    """

    def __init__(
        self,
        use_mock: bool = True,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.use_mock = use_mock
        self.model    = model
        self._client  = None

        if not use_mock:
            try:
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                ) from exc

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

    # ── public API ────────────────────────────────────────────────────────────

    def design_reward(self, intent: str) -> dict:
        """Convert natural-language intent to a validated reward profile.

        Parameters
        ----------
        intent : str
            Operator intent description, e.g. "delay sensitive XR traffic".

        Returns
        -------
        dict
            Validated profile with keys 'intent_name', 'weights', 'constraints'.
            'constraints' uses slot-based delay (p95_delay_max in slots).
        """
        if self.use_mock:
            raw = self._mock_profile(intent)
        else:
            raw = self._llm_profile(intent)

        profile = self._normalize_constraints(raw)
        validate_reward_profile(profile)
        return profile

    # ── private helpers ───────────────────────────────────────────────────────

    def _mock_profile(self, intent: str) -> dict:
        """Select best-matching predefined profile by keyword matching."""
        intent_lower = intent.lower()
        for keywords, profile_name in _MOCK_KEYWORD_MAP:
            if any(kw in intent_lower for kw in keywords):
                base = INTENT_PROFILES[profile_name]
                return {
                    "intent_name":  profile_name,
                    "weights":      dict(base["weights"]),
                    "constraints":  dict(base["constraints"]),
                }
        base = INTENT_PROFILES[_MOCK_DEFAULT]
        return {
            "intent_name":  _MOCK_DEFAULT,
            "weights":      dict(base["weights"]),
            "constraints":  dict(base["constraints"]),
        }

    def _llm_profile(self, intent: str) -> dict:
        """Call Anthropic API to generate reward profile."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type":          "text",
                    "text":          _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": f"Operator intent: {intent}"}
            ],
        )
        text = response.content[0].text
        return self._parse_response(text)

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from LLM response text."""
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse LLM response as JSON: {stripped[:200]!r}"
        )

    def _normalize_constraints(self, raw: dict) -> dict:
        """Convert ms-based delay constraint to slot-based; fill defaults."""
        c_in  = raw.get("constraints", {})
        c_out: dict = {}

        c_out["packet_loss_max"] = float(
            c_in.get("packet_loss_max", REWARD_PACKET_LOSS_MAX)
        )

        if "p95_delay_max_ms" in c_in:
            ms = float(c_in["p95_delay_max_ms"])
            c_out["p95_delay_max"] = ms * 1_000.0 / SLOT_DURATION_US
        else:
            c_out["p95_delay_max"] = float(
                c_in.get("p95_delay_max", REWARD_P95_DELAY_MAX)
            )

        c_out["legacy_degradation_max"] = float(
            c_in.get("legacy_degradation_max", REWARD_LEGACY_DEGRADATION_MAX)
        )

        return {
            "intent_name": raw.get("intent_name", _MOCK_DEFAULT),
            "weights":     raw["weights"],
            "constraints": c_out,
        }
