"""
HARQ soft combining buffer — Chase Combining (HARQ-CC).

Guidelines §9.1  HARQBuffer class definition
Guidelines §9.3  accumulated_snr_linear = Σ snr_linear over failed attempts
Guidelines §9.4  MCS constraint: HARQ_RETX must use the same MCS as original TX
Guidelines §14.2 HARQ validity horizon = first_tx_slot + channel_coherence_time

HARQ-CC operation:
  1. PHY failure (not collision): store(packet, snr_linear, slot) → buffer initialized
  2. Retransmit as HARQ_RETX with harq_buffer.original_mcs (MCS constraint §9.4)
  3. At TX end: effective_snr_db = linear_to_db(accumulated + current_snr_linear)
     4a. PHY success → flush()
     4b. PHY fail   → store() again → accumulates into existing buffer

Collision does NOT store soft information (guidelines §9.2, §15.3).
"""

from __future__ import annotations

from typing import Optional

from harq_sim import phy


class HARQBuffer:
    """Per-STA HARQ-CC soft combining buffer.

    Stores accumulated SNR (linear) across failed PHY attempts for a single
    in-flight packet.  One buffer per STA; only one packet can be in the
    HARQ pipeline at a time.
    """

    def __init__(self, validity_horizon: int = 200) -> None:
        """
        Args:
            validity_horizon: Buffer lifetime in slots after the first failed
                              attempt.  Approximates channel coherence time
                              (200 slots × 9 μs/slot = 1.8 ms).
        """
        self.validity_horizon: int = validity_horizon

        # State — all reset by flush()
        self.active:                  bool          = False
        self.packet_id:               Optional[int] = None
        self.original_mcs:            int           = 0
        self.combining_count:         int           = 0     # number of stored failures
        self.accumulated_snr_linear:  float         = 0.0   # Σ snr_linear
        self.first_tx_slot:           Optional[int] = None
        self.last_tx_slot:            Optional[int] = None
        self.validity_deadline:       Optional[int] = None  # first_tx_slot + validity_horizon

    # ──────────────────────────────────────────────────────────────────────────
    # Core operations
    # ──────────────────────────────────────────────────────────────────────────

    def store(self, packet, snr_linear: float, current_slot: int) -> None:
        """Store soft information from a PHY-failure attempt.

        May be called multiple times for the same packet (HARQ_RETX also fails).
        Initializes the buffer on the first call; accumulates on subsequent calls.
        If a stale entry for a different packet is found, it is flushed first.

        Guidelines §9.2: only PHY_ERROR triggers storage (not collision).
        Guidelines §9.3: accumulated_snr_linear += snr_linear (Chase Combining).
        """
        if self.active and self.packet_id != packet.packet_id:
            # Stale buffer for a different packet — should not normally happen
            # because flush() is called on delivery/drop, but guard defensively.
            self.flush()

        if not self.active:
            self.active                  = True
            self.packet_id               = packet.packet_id
            self.original_mcs            = packet.current_mcs
            self.combining_count         = 0
            self.accumulated_snr_linear  = 0.0
            self.first_tx_slot           = current_slot
            self.validity_deadline       = current_slot + self.validity_horizon

        self.accumulated_snr_linear += snr_linear
        self.combining_count        += 1
        self.last_tx_slot            = current_slot

    def effective_snr_db(self, new_snr_linear: float = 0.0) -> float:
        """Effective SNR after Chase Combining (guidelines §9.3).

        Computes 10·log10(accumulated_snr_linear + new_snr_linear).

        Args:
            new_snr_linear: SNR (linear) of the current in-progress attempt.
                            Pass 0.0 to query the accumulated-only result.
        """
        return phy.snr_linear_to_db(self.accumulated_snr_linear + new_snr_linear)

    def is_valid(self, current_slot: int) -> bool:
        """True when the buffer is active and within the validity horizon.

        Guidelines §14.2 — validity_deadline = first_tx_slot + coherence_time.
        An expired buffer should be flushed and the STA should fall back to ARQ.
        """
        if not self.active:
            return False
        if self.validity_deadline is None:
            return True
        return current_slot <= self.validity_deadline

    def flush(self) -> None:
        """Clear all stored soft information (guidelines §9.1 flush())."""
        self.active                  = False
        self.packet_id               = None
        self.original_mcs            = 0
        self.combining_count         = 0
        self.accumulated_snr_linear  = 0.0
        self.first_tx_slot           = None
        self.last_tx_slot            = None
        self.validity_deadline       = None

    # ──────────────────────────────────────────────────────────────────────────
    # Debug
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.active:
            return "HARQBuffer(inactive)"
        return (
            f"HARQBuffer(pkt={self.packet_id}, mcs={self.original_mcs}, "
            f"count={self.combining_count}, "
            f"eff_snr={self.effective_snr_db():.1f}dB, "
            f"deadline_slot={self.validity_deadline})"
        )
