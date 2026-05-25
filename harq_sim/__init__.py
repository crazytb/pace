"""HARQ-NPCA simulation module — Step 8: Baseline Comparison."""
from harq_sim.channel import Channel
from harq_sim.enums import (
    Action, ChannelType, FailureReason, NPCA_ACTIONS,
    PacketStatus, STAMode, TrafficClass, TxType,
)
from harq_sim.llm_reward_designer import LLMRewardDesigner, validate_reward_profile
from harq_sim.packet import Packet, TransmissionAttempt
from harq_sim.policy import NPCAHARQPolicy
from harq_sim.reward import INTENT_PROFILES, normalize_metrics, compute_reward
from harq_sim.sta import STA
from harq_sim.simulator import Simulator

__all__ = [
    "Channel",
    "Action", "ChannelType", "FailureReason", "NPCA_ACTIONS",
    "PacketStatus", "STAMode", "TrafficClass", "TxType",
    "LLMRewardDesigner", "validate_reward_profile",
    "Packet", "TransmissionAttempt",
    "NPCAHARQPolicy",
    "INTENT_PROFILES", "normalize_metrics", "compute_reward",
    "STA", "Simulator",
]
