"""
Channel model for HARQ-NPCA simulation.

Key D1.2 mapping:
  obss_remain  ↔  NPCA_PPDU_REM_DUR  (§37.18.3.1.c.i)
  busy_source  ↔  INTRA / OBSS distinction needed for NPCA transition condition (§37.18.3.1.b)
"""

import random
from typing import List, Tuple, Optional, Callable


class Channel:
    def __init__(
        self,
        channel_id: int,
        obss_generation_rate: float = 0.0,
        obss_duration_range: Tuple[int, int] = (20, 200),
        obss_duration_sampler: Optional[Callable[[], int]] = None,
    ):
        self.channel_id = channel_id
        self.obss_generation_rate = obss_generation_rate
        self.obss_duration_range = obss_duration_range
        self.obss_duration_sampler = obss_duration_sampler

        # Intra-BSS occupation (STA/AP transmissions within this BSS)
        self._intra_end_slot: int = 0
        self.occupied_remain: int = 0

        # Inter-BSS (OBSS) traffic list: (label, start_slot, duration, src_bss)
        self.obss_traffic: List[Tuple[str, int, int, int]] = []

        # Derived each slot by update()
        self.obss_remain: int = 0           # = NPCA_PPDU_REM_DUR (D1.2 §37.18.3)
        self.busy_source: Optional[str] = None  # "INTRA" | "OBSS" | None

    # ------------------------------------------------------------------
    # Slot update — call once per slot before any STA step()
    # ------------------------------------------------------------------
    def update(self, slot: int) -> None:
        # Expire finished OBSS traffic
        self.obss_traffic = [
            t for t in self.obss_traffic if t[1] + t[2] > slot
        ]

        # Intra-BSS remaining duration
        self.occupied_remain = max(0, self._intra_end_slot - slot)

        # OBSS remaining duration (max over all active OBSS PPDUs)
        active_rems = [
            start + dur - slot
            for _, start, dur, _ in self.obss_traffic
            if start <= slot < start + dur
        ]
        self.obss_remain = max(active_rems) if active_rems else 0

        if self.occupied_remain > 0:
            self.busy_source = "INTRA"
        elif self.obss_remain > 0:
            self.busy_source = "OBSS"
        else:
            self.busy_source = None

    # ------------------------------------------------------------------
    # OBSS generation — call after update() each slot
    # ------------------------------------------------------------------
    def generate_obss(self, slot: int) -> None:
        if self.obss_generation_rate == 0:
            return
        if self.is_busy(slot):
            return
        if random.random() < self.obss_generation_rate:
            duration = (
                self.obss_duration_sampler()
                if self.obss_duration_sampler is not None
                else random.randint(*self.obss_duration_range)
            )
            self.obss_traffic.append((
                f"obss_ch{self.channel_id}_s{slot}",
                slot, duration, -1,
            ))

    # ------------------------------------------------------------------
    # Intra-BSS occupation (called by simulator on TX grant)
    # ------------------------------------------------------------------
    def occupy_intra(self, slot: int, duration: int) -> None:
        self._intra_end_slot = slot + duration
        self.occupied_remain = duration

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    def is_busy(self, slot: int) -> bool:
        return self.occupied_remain > 0 or self.obss_remain > 0

    def is_busy_by_obss(self, slot: int) -> bool:
        return self.obss_remain > 0

    def is_busy_by_intra_bss(self, slot: int) -> bool:
        return self.occupied_remain > 0

    # D1.2 §37.18.3.1.d: NPCA channel must not overlap with the OBSS PPDU channel.
    # In our model, primary and NPCA are always distinct Channel objects → always True.
    @property
    def overlaps_with_obss_ppdu(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"Channel(id={self.channel_id}, "
            f"obss_remain={self.obss_remain}, "
            f"intra_remain={self.occupied_remain})"
        )
