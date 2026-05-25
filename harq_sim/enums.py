from enum import Enum, auto


class STAMode(Enum):
    """STA operating mode state machine.

    D1.2 §37.18.3 switching conditions → NPCA_SWITCHING
    D1.2 §37.18.4 transmission rules  → NPCA_BACKOFF / NPCA_TX
    D1.2 §37.18.4 b return rule       → SWITCH_BACK
    """
    PRIMARY_BACKOFF  = auto()   # EDCA backoff on BSS primary channel
    PRIMARY_FROZEN   = auto()   # Backoff frozen (primary channel busy)
    PRIMARY_TX       = auto()   # Transmitting on BSS primary channel
    NPCA_SWITCHING   = auto()   # Radio switching to NPCA primary channel
    NPCA_BACKOFF     = auto()   # EDCA backoff on NPCA primary channel
    NPCA_FROZEN      = auto()   # NPCA backoff frozen (NPCA channel busy)
    NPCA_TX          = auto()   # Transmitting on NPCA primary channel
    SWITCH_BACK      = auto()   # Radio switching back to BSS primary channel


# Convenience frozensets for bulk mode checks
NPCA_MODES = frozenset({
    STAMode.NPCA_SWITCHING,
    STAMode.NPCA_BACKOFF,
    STAMode.NPCA_FROZEN,
    STAMode.NPCA_TX,
    STAMode.SWITCH_BACK,
})

LISTEN_MODES = frozenset({
    STAMode.PRIMARY_BACKOFF,
    STAMode.PRIMARY_FROZEN,
    STAMode.NPCA_BACKOFF,
    STAMode.NPCA_FROZEN,
})

TX_MODES = frozenset({
    STAMode.PRIMARY_TX,
    STAMode.NPCA_TX,
})


class ChannelType(Enum):
    PRIMARY = "PRIMARY"
    NPCA    = "NPCA"


class TxType(Enum):
    NEW        = "NEW"
    ARQ_RETX   = "ARQ_RETX"
    HARQ_RETX  = "HARQ_RETX"   # used from Step 3 onward


class FailureReason(Enum):
    NONE                    = "NONE"
    COLLISION               = "COLLISION"
    PHY_ERROR               = "PHY_ERROR"
    AP_ABSENCE_DUE_TO_NPCA = "AP_ABSENCE_DUE_TO_NPCA"
    DEADLINE_EXPIRED        = "DEADLINE_EXPIRED"
    RETRY_LIMIT_EXCEEDED    = "RETRY_LIMIT_EXCEEDED"
    NPCA_TIMER_EXPIRED      = "NPCA_TIMER_EXPIRED"


class TrafficClass(Enum):
    XR          = "XR"
    VOICE       = "VOICE"
    VIDEO       = "VIDEO"
    BEST_EFFORT = "BEST_EFFORT"


class PacketStatus(Enum):
    PENDING   = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    DELIVERED = "DELIVERED"
    DROPPED   = "DROPPED"
