"""
Packet and TransmissionAttempt data classes for HARQ-NPCA simulation.

Guidelines §8 (TransmissionAttempt), §3.1 (Packet fields),
§14 (QoS deadline / HARQ validity horizon).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

from harq_sim.enums import (
    ChannelType, FailureReason, PacketStatus, TrafficClass, TxType,
)

# QoS latency deadline in slots (9 μs/slot)
_SLOT_US = 9.0
MAX_DELAY_SLOTS: dict[TrafficClass, Optional[int]] = {
    TrafficClass.XR:          int(10e3  / _SLOT_US),   # 10 ms
    TrafficClass.VOICE:       int(20e3  / _SLOT_US),   # 20 ms
    TrafficClass.VIDEO:       int(50e3  / _SLOT_US),   # 50 ms
    TrafficClass.BEST_EFFORT: None,                     # no deadline
}

_id_counter = itertools.count(1)


# ─────────────────────────────────────────────────────────────────────────────
# TransmissionAttempt  (guidelines §8)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TransmissionAttempt:
    packet_id:            int
    sta_id:               int
    channel_type:         ChannelType
    tx_type:              TxType
    mcs:                  int
    start_time:           int
    duration:             int
    success:              bool           = False
    failure_reason:       FailureReason  = FailureReason.NONE
    collision:            bool           = False
    snr_db:               float          = 0.0
    effective_snr_db:     float          = 0.0
    harq_combining_count: int            = 0


# ─────────────────────────────────────────────────────────────────────────────
# Packet  (guidelines §3.1)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Packet:
    arrival_time:  int
    size_bits:     int          = 8000
    traffic_class: TrafficClass = TrafficClass.BEST_EFFORT
    current_mcs:   int          = 0

    # Auto-assigned unique ID
    packet_id: int = field(default_factory=lambda: next(_id_counter))

    # Retry / HARQ counters
    retry_count: int = 0
    harq_count:  int = 0

    status: PacketStatus = PacketStatus.PENDING

    transmission_history: List[TransmissionAttempt] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Deadline helpers  (guidelines §14.1 QoS deadline)
    # ------------------------------------------------------------------
    @property
    def latency_deadline(self) -> Optional[int]:
        max_delay = MAX_DELAY_SLOTS.get(self.traffic_class)
        return None if max_delay is None else self.arrival_time + max_delay

    def is_deadline_expired(self, current_slot: int) -> bool:
        dl = self.latency_deadline
        return False if dl is None else current_slot > dl

    def deadline_remaining(self, current_slot: int) -> Optional[int]:
        dl = self.latency_deadline
        return None if dl is None else max(0, dl - current_slot)

    def __repr__(self) -> str:
        return (
            f"Packet(id={self.packet_id}, retry={self.retry_count}, "
            f"harq={self.harq_count}, status={self.status.value})"
        )
