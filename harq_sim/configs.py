"""
Default configuration constants for HARQ-NPCA simulation.
"""

# ── 802.11 EDCA (AC_BE default) ───────────────────────────────────────────────
CW_MIN: int = 15
CW_MAX: int = 1023

# ── Time ──────────────────────────────────────────────────────────────────────
SLOT_DURATION_US: float = 9.0      # μs per slot (802.11ax)
SIFS_SLOTS: int         = 2        # aSIFSTime ≈ 16 μs → ~2 slots
DIFS_SLOTS: int         = 4        # aDIFSTime = SIFS + 2×aSlotTime ≈ 34 μs → ~4 slots

# ── NPCA parameters ───────────────────────────────────────────────────────────
DEFAULT_SWITCHING_DELAY:   int = 1   # radio transition slots (NPCA switching delay)
DEFAULT_SWITCH_BACK_DELAY: int = 1   # D1.2 NPCA switch back delay (slots)
DEFAULT_NPCA_QSRC:         int = 0   # initial NPCA CW exponent (qsrc=0 → CW=15)
DEFAULT_NPCA_MIN_DURATION_THRESHOLD: int = 0   # D1.2 §37.18.3.1.c.i (0 = always allowed)

# ── Frame / transmission ──────────────────────────────────────────────────────
DEFAULT_PPDU_DURATION: int  = 33    # slots (~297 μs for a typical A-MPDU)
DEFAULT_RETRY_LIMIT:   int  = 7     # dot11ShortRetryLimit

# ── OBSS traffic ─────────────────────────────────────────────────────────────
DEFAULT_OBSS_RATE:          float      = 0.05
DEFAULT_OBSS_DURATION_RANGE: tuple     = (20, 200)   # slots

# ── Energy model (IEEE 802.11ax TG 11-14-0980-16-00ax) ───────────────────────
ENERGY_TX_PER_SLOT_UJ:        float = 2.772   # 280 mA × 1.1 V × 9 μs
ENERGY_LISTEN_PER_SLOT_UJ:    float = 0.495   # 50 mA × 1.1 V × 9 μs
ENERGY_NPCA_TRANSITION_UJ:    float = 0.75    # radio switching event

# ── Step 5: Adaptive CW_npca_init (guidelines §13) ───────────────────────────
NPCA_QSRC_MIN: int = 0              # minimum qsrc clamp
NPCA_QSRC_MAX: int = 5              # maximum qsrc clamp
NPCA_FAILURE_WINDOW: int = 10       # sliding window size for NPCA failure rate
NPCA_TRANSITION_WINDOW: int = 50    # recent-slot window for global transition count
NPCA_TRANSITION_THRESHOLD: int = 3  # "too many transitions" threshold
URGENT_DEADLINE_THRESHOLD: int = 50 # remaining slots < this → aggressive (qsrc--)

# ── Step 6: Reward module reference values ────────────────────────────────────
REWARD_THROUGHPUT_REF:         int   = 50       # reference: packets delivered per episode
REWARD_DELAY_REF:              float = 150.0    # mean access delay reference (slots)
REWARD_D95_REF:                float = 400.0    # p95 delay reference (slots)
REWARD_ENERGY_REF:             float = 3000.0   # total energy reference per episode (μJ)
REWARD_LEGACY_REF:             float = 0.30     # legacy degradation reference (fraction)

# Constraint thresholds (defaults)
REWARD_PACKET_LOSS_MAX:        float = 0.10   # 10% max packet loss
REWARD_P95_DELAY_MAX:          float = 500.0  # max p95 delay (slots)
REWARD_LEGACY_DEGRADATION_MAX: float = 0.30   # 30% max legacy degradation
REWARD_LAMBDA_LOSS:            float = 1.0    # packet-loss constraint penalty scale
REWARD_LAMBDA_DELAY:           float = 1.0    # delay constraint penalty scale (normalized)
REWARD_LAMBDA_LEGACY:          float = 1.0    # legacy-degradation constraint penalty scale

# ── QoS deadlines (slots) — computed from MAX_DELAY_SLOTS in packet.py ───────
# XR: 10 ms, VOICE: 20 ms, VIDEO: 50 ms, BEST_EFFORT: None
