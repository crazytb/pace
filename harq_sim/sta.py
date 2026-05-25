"""
STA state machine for HARQ-NPCA simulation — Step 1 (NPCA without HARQ).

D1.2 §37.18.3  Switching to the NPCA channel
  → can_transition_to_npca(): condition 1 implemented
  → _start_npca_transition(): save primary state, init NPCA state, start switching delay

D1.2 §37.18.4  NPCA transmission rules
  → pt 3:  use same EDCA param set on NPCA channel  (modelled via npca_initial_qsrc)
  → pt 4a: NPCA_TIMER = NPCA_PPDU_REM_DUR − switch_back_delay
  → pt 4b: return to primary within aSlotTime of NPCA_TIMER expiry

Guidelines §5  Primary/NPCA EDCA state separation
  → _save_primary_state() / _restore_primary_state()
  → _init_npca_state() with fresh CW from npca_initial_qsrc

Guidelines §18  Backoff update after success/failure
  → reset_backoff_after_success() / increase_backoff_after_failure()
  → NPCA CW increase does NOT propagate to primary CW

Guidelines §7  AP absence failure
  → handle_tx_result() uses FailureReason.AP_ABSENCE_DUE_TO_NPCA
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from harq_sim.channel import Channel
from harq_sim.enums import (
    ChannelType, FailureReason, NPCA_MODES,
    PacketStatus, STAMode, TxType,
)
from harq_sim.packet import Packet, TransmissionAttempt

# ──────────────────────────────────────────────────────────────────────────────
# 802.11 AC_BE EDCA defaults (used for both primary and NPCA channels)
# D1.2 §37.18.4 pt 3: same EDCA parameter set on NPCA as on BSS primary
# ──────────────────────────────────────────────────────────────────────────────
CW_MIN: int = 15
CW_MAX: int = 1023


@dataclass
class TxRequest:
    """Pending transmission request — consumed by Simulator each slot."""
    sta_id:       int
    channel_type: ChannelType
    duration:     int
    packet:       Optional[Packet]
    tx_type:      TxType
    mcs:          int = 0


class STA:
    # ──────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        sta_id:                    int,
        primary_channel:           Channel,
        npca_channel:              Optional[Channel] = None,
        npca_enabled:              bool = False,
        ppdu_duration:             int = 33,       # slots
        switching_delay:           int = 1,        # radio transition (slots) — D1.2 NPCA switching delay
        switch_back_delay:         int = 1,        # D1.2 NPCA switch back delay (slots)
        npca_min_duration_threshold: int = 0,      # D1.2 §37.18.3.1.c.i minimum duration threshold
        npca_initial_qsrc:         int = 0,        # initial NPCA CW exponent (research variable)
        retry_limit:               int = 7,
        ap_on_primary:             bool = True,    # False when AP is on NPCA (single-radio model)
        infinite_queue:            bool = True,    # auto-generate new packets when queue empties
    ):
        self.sta_id                     = sta_id
        self.primary_channel            = primary_channel
        self.npca_channel               = npca_channel
        self.npca_enabled               = npca_enabled
        self.ppdu_duration              = ppdu_duration
        self.switching_delay            = switching_delay
        self.switch_back_delay          = switch_back_delay
        self.npca_min_duration_threshold = npca_min_duration_threshold
        self.npca_initial_qsrc          = npca_initial_qsrc
        self.retry_limit                = retry_limit
        self.ap_on_primary              = ap_on_primary
        self.infinite_queue             = infinite_queue

        # ── Primary EDCA state (guidelines §5.1) ──────────────────────────────
        self.primary_cw:              int = CW_MIN
        self.primary_backoff_counter: int = random.randint(0, CW_MIN)
        self.primary_backoff_stage:   int = 0
        self.primary_retry_counter:   int = 0

        # ── NPCA EDCA state (guidelines §5.2) ─────────────────────────────────
        self.npca_cw:              int = CW_MIN
        self.npca_backoff_counter: int = 0
        self.npca_backoff_stage:   int = 0
        self.npca_retry_counter:   int = 0

        # ── Saved primary state for restore after NPCA (guidelines §5.3/5.4) ──
        self.saved_primary_state: Optional[dict] = None

        # ── intra-BSS NAV (D1.2 §37.18.3.1.e) — simplified: always 0 ────────
        self.intra_bss_nav: int = 0

        # ── Mode state machine ────────────────────────────────────────────────
        self.mode:      STAMode = STAMode.PRIMARY_BACKOFF
        self.next_mode: STAMode = self.mode

        # ── Radio switching countdown ─────────────────────────────────────────
        self.switching_remain: int = 0

        # ── NPCA_TIMER (D1.2 §37.18.4 pt 4a) ────────────────────────────────
        # Set at switch time to: obss_remain − switch_back_delay
        # Decremented each slot while in any NPCA mode.
        self.npca_timer: int = 0

        # ── TX state ──────────────────────────────────────────────────────────
        self.tx_remaining:   int             = 0
        self.tx_request:     Optional[TxRequest] = None
        self.current_packet: Optional[Packet]    = None

        # ── Packet queue ──────────────────────────────────────────────────────
        self.packet_queue: Deque[Packet] = deque()
        self._pkt_arrival_counter: int   = 0

        # ── Episode statistics ────────────────────────────────────────────────
        self.stats: dict = {
            "primary_tx_success": 0,
            "primary_tx_fail":    0,
            "npca_tx_success":    0,
            "npca_tx_fail":       0,
            "npca_transitions":   0,
            "switch_backs":       0,
            "packets_delivered":  0,
            "packets_dropped":    0,
            "ap_absence_failures": 0,
        }

        # ── Trace log (optional, filled by simulator) ─────────────────────────
        self.trace: list = []

        # ── TX completion event (read by simulator each slot, then cleared) ──
        # Set inside handle_tx_result(success=True) so the logger can capture it.
        self._completed_tx: Optional[dict] = None

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA initial CW   (guidelines §5.3)
    # npca_cw = 2^qsrc × (CW_MIN + 1) − 1
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_npca_cw_init(self, qsrc: Optional[int] = None) -> int:
        q = self.npca_initial_qsrc if qsrc is None else qsrc
        return 2 ** q * (CW_MIN + 1) - 1

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA transition condition  (D1.2 §37.18.3 Condition 1)
    # ──────────────────────────────────────────────────────────────────────────
    def can_transition_to_npca(self, slot: int) -> bool:
        if not self.npca_enabled or self.npca_channel is None:
            return False
        # b. Primary channel busy due to inter-BSS PPDU
        if not self.primary_channel.is_busy_by_obss(slot):
            return False
        # c.i. NPCA_PPDU_REM_DUR ≥ NPCA Minimum Duration Threshold
        if self.primary_channel.obss_remain < self.npca_min_duration_threshold:
            return False
        # d. NPCA channel does not overlap with OBSS PPDU channel (always true)
        if self.npca_channel.overlaps_with_obss_ppdu:
            return False
        # e. Intra-BSS NAV = 0
        if self.intra_bss_nav > 0:
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Primary state save / restore  (guidelines §5.3 / §5.4)
    # ──────────────────────────────────────────────────────────────────────────
    def _save_primary_state(self) -> None:
        self.saved_primary_state = {
            "cw":              self.primary_cw,
            "backoff_counter": self.primary_backoff_counter,
            "backoff_stage":   self.primary_backoff_stage,
            "retry_counter":   self.primary_retry_counter,
        }

    def _restore_primary_state(self) -> None:
        if self.saved_primary_state is not None:
            self.primary_cw              = self.saved_primary_state["cw"]
            self.primary_backoff_counter = self.saved_primary_state["backoff_counter"]
            self.primary_backoff_stage   = self.saved_primary_state["backoff_stage"]
            self.primary_retry_counter   = self.saved_primary_state["retry_counter"]
            self.saved_primary_state     = None

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA EDCA state initialization  (guidelines §5.3)
    # D1.2 §37.18.4 pt 3: same EDCA param set — here modelled as fresh CW from qsrc
    # ──────────────────────────────────────────────────────────────────────────
    def _init_npca_state(self, qsrc: Optional[int] = None) -> None:
        self.npca_cw              = self._compute_npca_cw_init(qsrc)
        self.npca_backoff_counter = random.randint(0, self.npca_cw)
        self.npca_backoff_stage   = 0
        self.npca_retry_counter   = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Backoff management after success / failure  (guidelines §18)
    # Key invariant: NPCA CW changes do NOT affect primary CW.
    # ──────────────────────────────────────────────────────────────────────────
    def reset_backoff_after_success(self, channel_type: ChannelType) -> None:
        if channel_type == ChannelType.PRIMARY:
            self.primary_backoff_stage   = 0
            self.primary_cw              = CW_MIN
            self.primary_backoff_counter = random.randint(0, self.primary_cw)
        else:  # NPCA
            self.npca_backoff_stage   = 0
            self.npca_cw              = self._compute_npca_cw_init()
            self.npca_backoff_counter = random.randint(0, self.npca_cw)

    def increase_backoff_after_failure(self, channel_type: ChannelType) -> None:
        if channel_type == ChannelType.PRIMARY:
            self.primary_backoff_stage   += 1
            self.primary_cw              = min(2 * (self.primary_cw + 1) - 1, CW_MAX)
            self.primary_backoff_counter = random.randint(0, self.primary_cw)
            self.primary_retry_counter   += 1
        else:  # NPCA — increase NPCA CW only, primary CW unchanged
            self.npca_backoff_stage   += 1
            self.npca_cw              = min(2 * (self.npca_cw + 1) - 1, CW_MAX)
            self.npca_backoff_counter = random.randint(0, self.npca_cw)
            self.npca_retry_counter   += 1

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA switch-back decision
    # D1.2 §37.18.4 pt 4b: return within aSlotTime of NPCA_TIMER expiry
    # ──────────────────────────────────────────────────────────────────────────
    def _should_switch_back(self) -> bool:
        return (
            self.npca_timer <= 0
            or self.primary_channel.obss_remain <= self.switch_back_delay
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NPCA transition helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _start_npca_transition(self, slot: int) -> None:
        """Save primary state and start NPCA switching delay."""
        self._save_primary_state()
        self._init_npca_state()
        # D1.2 §37.18.4 pt 4a: NPCA_TIMER = NPCA_PPDU_REM_DUR − switch_back_delay
        self.npca_timer       = max(0, self.primary_channel.obss_remain - self.switch_back_delay)
        self.switching_remain = self.switching_delay
        self.next_mode        = STAMode.NPCA_SWITCHING
        self.stats["npca_transitions"] += 1

    def _start_switch_back(self) -> None:
        """Begin radio switching back to BSS primary channel."""
        self.switching_remain = self.switching_delay
        self.next_mode        = STAMode.SWITCH_BACK
        self.stats["switch_backs"] += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Packet queue helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _peek_head(self) -> Optional[Packet]:
        if self.infinite_queue and not self.packet_queue:
            self._pkt_arrival_counter += 1
            self.packet_queue.append(Packet(
                arrival_time=self._pkt_arrival_counter,
            ))
        return self.packet_queue[0] if self.packet_queue else None

    def _dequeue_current(self) -> None:
        if self.current_packet and self.packet_queue:
            if self.packet_queue[0] is self.current_packet:
                self.packet_queue.popleft()
        self.current_packet = None

    # ──────────────────────────────────────────────────────────────────────────
    # Main step  (called once per slot by Simulator)
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, slot: int) -> None:
        self.tx_request    = None
        self._completed_tx = None   # clear last slot's completion event

        # Decrement NPCA_TIMER every slot while in any NPCA mode
        if self.mode in NPCA_MODES and self.npca_timer > 0:
            self.npca_timer -= 1

        if   self.mode == STAMode.PRIMARY_BACKOFF: self._handle_primary_backoff(slot)
        elif self.mode == STAMode.PRIMARY_FROZEN:  self._handle_primary_frozen(slot)
        elif self.mode == STAMode.PRIMARY_TX:      self._handle_primary_tx(slot)
        elif self.mode == STAMode.NPCA_SWITCHING:  self._handle_npca_switching(slot)
        elif self.mode == STAMode.NPCA_BACKOFF:    self._handle_npca_backoff(slot)
        elif self.mode == STAMode.NPCA_FROZEN:     self._handle_npca_frozen(slot)
        elif self.mode == STAMode.NPCA_TX:         self._handle_npca_tx(slot)
        elif self.mode == STAMode.SWITCH_BACK:     self._handle_switch_back(slot)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-mode handlers
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_primary_backoff(self, slot: int) -> None:
        # 1. Intra-BSS busy → freeze (no NPCA opportunity — NAV set implicitly)
        if self.primary_channel.is_busy_by_intra_bss(slot):
            self.next_mode = STAMode.PRIMARY_FROZEN
            return

        # 2. OBSS detected → consider NPCA transition (D1.2 §37.18.3 Condition 1)
        if self.primary_channel.is_busy_by_obss(slot):
            if self.can_transition_to_npca(slot):
                self._start_npca_transition(slot)
            else:
                self.next_mode = STAMode.PRIMARY_FROZEN
            return

        # 3. Channel idle: count down backoff or transmit
        if self.primary_backoff_counter == 0:
            pkt = self._peek_head()
            if pkt is not None:
                self.current_packet = pkt
                self.tx_request = TxRequest(
                    sta_id=self.sta_id,
                    channel_type=ChannelType.PRIMARY,
                    duration=self.ppdu_duration,
                    packet=pkt,
                    tx_type=TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX,
                )
                self.tx_remaining = self.ppdu_duration
                self.next_mode    = STAMode.PRIMARY_TX
        else:
            self.primary_backoff_counter -= 1

    def _handle_primary_frozen(self, slot: int) -> None:
        # OBSS appeared while frozen → check NPCA opportunity
        if self.primary_channel.is_busy_by_obss(slot):
            if self.can_transition_to_npca(slot):
                self._start_npca_transition(slot)
                return

        if not self.primary_channel.is_busy(slot):
            self.next_mode = STAMode.PRIMARY_BACKOFF

    def _handle_primary_tx(self, slot: int) -> None:
        self.tx_remaining -= 1
        if self.tx_remaining == 0:
            # TX complete — self-report success (collision was already reported by simulator)
            self.handle_tx_result(True, FailureReason.NONE, ChannelType.PRIMARY, slot)

    def _handle_npca_switching(self, slot: int) -> None:
        """Count down radio switching delay before entering NPCA backoff."""
        self.switching_remain -= 1
        if self.switching_remain == 0:
            # D1.2 §37.18.4 pt 1: STA shall be ready to TX no later than switching delay
            if self.npca_channel.is_busy(slot):
                self.next_mode = STAMode.NPCA_FROZEN
            else:
                self.next_mode = STAMode.NPCA_BACKOFF

    def _handle_npca_backoff(self, slot: int) -> None:
        # Check NPCA_TIMER / primary OBSS expiry → switch back
        if self._should_switch_back():
            self._start_switch_back()
            return

        if self.npca_channel.is_busy(slot):
            self.next_mode = STAMode.NPCA_FROZEN
            return

        if self.npca_backoff_counter == 0:
            pkt = self.current_packet or self._peek_head()
            if pkt is not None:
                self.current_packet = pkt
                # TX duration bounded by remaining OBSS time (D1.2 §37.18.4 pt 4a)
                tx_dur = min(self.ppdu_duration, self.primary_channel.obss_remain)
                self.tx_request = TxRequest(
                    sta_id=self.sta_id,
                    channel_type=ChannelType.NPCA,
                    duration=tx_dur,
                    packet=pkt,
                    tx_type=TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX,
                )
                self.tx_remaining = tx_dur
                self.next_mode    = STAMode.NPCA_TX
        else:
            self.npca_backoff_counter -= 1

    def _handle_npca_frozen(self, slot: int) -> None:
        # Primary OBSS ended or NPCA_TIMER expired → switch back
        if self._should_switch_back():
            self._start_switch_back()
            return

        if not self.npca_channel.is_busy(slot):
            self.next_mode = STAMode.NPCA_BACKOFF

    def _handle_npca_tx(self, slot: int) -> None:
        self.tx_remaining -= 1
        if self.tx_remaining == 0:
            # TX complete — self-report success
            self.handle_tx_result(True, FailureReason.NONE, ChannelType.NPCA, slot)

    def _handle_switch_back(self, slot: int) -> None:
        """Count down radio switching delay back to primary channel."""
        self.switching_remain -= 1
        if self.switching_remain == 0:
            # D1.2 §37.18.4: restore primary EDCA state (guidelines §5.4)
            self._restore_primary_state()
            self.next_mode = STAMode.PRIMARY_BACKOFF

    # ──────────────────────────────────────────────────────────────────────────
    # TX result handler  (called by Simulator after collision resolution)
    # Guidelines §17
    # ──────────────────────────────────────────────────────────────────────────
    def handle_tx_result(
        self,
        success: bool,
        failure_reason: FailureReason,
        channel_type: ChannelType,
        slot: int,
    ) -> None:
        pkt = self.current_packet
        if pkt is None:
            return

        attempt = TransmissionAttempt(
            packet_id=pkt.packet_id,
            sta_id=self.sta_id,
            channel_type=channel_type,
            tx_type=TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX,
            mcs=pkt.current_mcs,
            start_time=slot - self.ppdu_duration + 1,
            duration=self.ppdu_duration,
            success=success,
            failure_reason=failure_reason,
            collision=(failure_reason == FailureReason.COLLISION),
        )
        pkt.transmission_history.append(attempt)

        if success:
            # Record completion event for CSV logger before dequeue clears packet
            self._completed_tx = {
                "channel_type": channel_type,
                "tx_type": TxType.NEW if pkt.retry_count == 0 else TxType.ARQ_RETX,
                "packet_id": pkt.packet_id,
                "retry_count": pkt.retry_count,
            }
            pkt.status = PacketStatus.DELIVERED
            self._dequeue_current()
            self.reset_backoff_after_success(channel_type)
            if channel_type == ChannelType.PRIMARY:
                self.stats["primary_tx_success"] += 1
                self.next_mode = STAMode.PRIMARY_BACKOFF
            else:
                self.stats["npca_tx_success"] += 1
                if self._should_switch_back():
                    self._start_switch_back()
                else:
                    self.next_mode = STAMode.NPCA_BACKOFF
            self.stats["packets_delivered"] += 1
            return

        # ── Failure path ──────────────────────────────────────────────────────
        pkt.retry_count += 1
        if channel_type == ChannelType.PRIMARY:
            self.stats["primary_tx_fail"] += 1
        else:
            self.stats["npca_tx_fail"] += 1

        if failure_reason == FailureReason.AP_ABSENCE_DUE_TO_NPCA:
            self.stats["ap_absence_failures"] += 1

        # Drop if retry limit exceeded or deadline passed
        if pkt.retry_count > self.retry_limit:
            pkt.status = PacketStatus.DROPPED
            self._dequeue_current()
            self.stats["packets_dropped"] += 1
            self._post_drop_backoff(channel_type)
            return

        if pkt.is_deadline_expired(slot):
            pkt.status = PacketStatus.DROPPED
            self._dequeue_current()
            self.stats["packets_dropped"] += 1
            self._post_drop_backoff(channel_type)
            return

        # Retry: increase CW (NPCA CW ↑ does NOT affect primary CW)
        self.increase_backoff_after_failure(channel_type)

        # After failed NPCA TX: check switch-back
        if channel_type == ChannelType.NPCA:
            if self._should_switch_back():
                self._start_switch_back()
            else:
                self.next_mode = STAMode.NPCA_BACKOFF
        else:
            self.next_mode = STAMode.PRIMARY_BACKOFF

    def _post_drop_backoff(self, channel_type: ChannelType) -> None:
        """Reset backoff after packet drop so STA can contend again."""
        self.reset_backoff_after_success(channel_type)
        if channel_type == ChannelType.NPCA:
            if self._should_switch_back():
                self._start_switch_back()
            else:
                self.next_mode = STAMode.NPCA_BACKOFF
        else:
            self.next_mode = STAMode.PRIMARY_BACKOFF

    # ──────────────────────────────────────────────────────────────────────────
    # Commit next_mode at end of each slot (called by Simulator)
    # ──────────────────────────────────────────────────────────────────────────
    def commit_mode(self) -> None:
        self.mode = self.next_mode

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def reset_episode_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0
        self.trace.clear()

    def __repr__(self) -> str:
        return (
            f"STA(id={self.sta_id}, mode={self.mode.name}, "
            f"p_cw={self.primary_cw}, n_cw={self.npca_cw}, "
            f"npca_timer={self.npca_timer})"
        )
