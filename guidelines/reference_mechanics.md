# 참조: STA 동작 메커니즘 상세 (§4–12, §15–18)

로드 시점: Step 1~4 구현 내용을 참조해야 할 때만 로드. 평소에는 불필요.

---

## 4. STA mode 정의

```python
class STAMode(Enum):
    PRIMARY_BACKOFF = auto()
    PRIMARY_FROZEN = auto()
    PRIMARY_TX = auto()
    NPCA_SWITCHING = auto()
    NPCA_BACKOFF = auto()
    NPCA_FROZEN = auto()
    NPCA_TX = auto()
    SWITCH_BACK = auto()
```

| Mode | 의미 |
|---|---|
| `PRIMARY_BACKOFF` | BSS primary channel에서 EDCA backoff 진행 |
| `PRIMARY_FROZEN` | primary channel busy로 backoff freeze |
| `PRIMARY_TX` | primary channel에서 전송 중 |
| `NPCA_SWITCHING` | NPCA primary channel로 전환 중 |
| `NPCA_BACKOFF` | NPCA primary channel에서 별도 backoff 수행 |
| `NPCA_FROZEN` | NPCA primary channel busy로 NPCA backoff freeze |
| `NPCA_TX` | NPCA primary channel에서 전송 중 |
| `SWITCH_BACK` | BSS primary channel로 복귀 중 |

---

## 5. Primary backoff와 NPCA backoff 관리

### 5.1 Primary EDCA state

```python
primary_backoff_counter
primary_cw
primary_backoff_stage
primary_retry_counter
```

### 5.2 NPCA EDCA state

```python
npca_backoff_counter
npca_cw
npca_backoff_stage
npca_retry_counter
npca_initial_qsrc
```

### 5.3 NPCA transition 시 primary state 저장

```python
saved_primary_state = {
    "cw": primary_cw,
    "backoff_counter": primary_backoff_counter,
    "backoff_stage": primary_backoff_stage,
    "retry_counter": primary_retry_counter,
}

# NPCA EDCA state 초기화
npca_qsrc = initial_npca_qsrc
npca_cw = 2 ** npca_qsrc * (cw_min + 1) - 1
npca_backoff_counter = random.randint(0, npca_cw)
npca_backoff_stage = 0
```

**중요**: NPCA primary channel이 idle이어도 backoff procedure를 수행해야 함. 전환 직후 즉시 전송 금지.

### 5.4 Switch-back 시 primary state 복원

```python
primary_cw = saved_primary_state["cw"]
primary_backoff_counter = saved_primary_state["backoff_counter"]
primary_backoff_stage = saved_primary_state["backoff_stage"]
primary_retry_counter = saved_primary_state["retry_counter"]
# NPCA에서 증가한 npca_cw는 primary CW에 반영하지 않음
```

### 5.5 다시 NPCA로 transition하는 경우

```python
npca_qsrc = initial_npca_qsrc
npca_cw = 2 ** npca_qsrc * (cw_min + 1) - 1
```

adaptive NPCA CW 제어 사용 시 `initial_npca_qsrc`는 policy에 의해 변경될 수 있음.

---

## 6. NPCA transition 조건

### 6.1 기본 조건

```text
NPCA mode enabled
AND BSS primary channel is busy due to inter-BSS PPDU
AND detected PPDU does not overlap with NPCA primary channel
AND expected remaining duration is long enough
AND intra-BSS NAV is zero
```

### 6.2 코드상 단순화된 조건

```python
def can_transition_to_npca(sta, env):
    return (
        sta.npca_enabled
        and env.primary_channel.is_busy
        and env.primary_channel.busy_source == "OBSS"
        and env.primary_channel.remaining_busy_time >= env.npca_min_duration_threshold
        and not env.npca_channel.overlaps_with_obss_ppdu
        and sta.intra_bss_nav == 0
    )
```

### 6.3 핵심 포인트

primary channel busy period가 **NPCA transition opportunity**가 됨:

```text
primary channel busy detected
→ primary backoff freeze
→ if NPCA condition satisfied:
       save primary state
       switch to NPCA
  else:
       remain frozen on primary
```

---

## 7. AP가 NPCA로 이동했을 때 primary STA uplink 처리

AP가 single-radio이고 NPCA primary channel로 switch했다고 가정.

```python
if sta.transmits_on_primary:
    if ap.mode == "PRIMARY":
        evaluate_success_or_collision()
    elif ap.mode == "NPCA":
        transmission_failed_due_to_ap_absence()
```

```python
# ACK timeout으로 처리
sta.retry_counter += 1
sta.primary_backoff_stage += 1
sta.primary_cw = min(2 * (sta.primary_cw + 1) - 1, cw_max)
sta.primary_backoff_counter = random.randint(0, sta.primary_cw)
failure_reason = "AP_ABSENCE_DUE_TO_NPCA"
```

---

## 8. Transmission attempt 구조

```text
TransmissionAttempt:
    packet_id
    sta_id
    channel_type  # PRIMARY or NPCA
    tx_type       # NEW, ARQ_RETX, HARQ_RETX
    mcs
    start_time
    duration
    success
    failure_reason
    collision
    sinr
    effective_snr
    harq_combining_count
```

---

## 9. HARQ buffer 설계

### 9.1 HARQBuffer class

```python
class HARQBuffer:
    def __init__(self):
        self.active = False
        self.packet_id = None
        self.receiver_id = None
        self.original_mcs = None
        self.combining_count = 0
        self.accumulated_snr_linear = 0.0
        self.first_tx_time = None
        self.last_tx_time = None
        self.validity_deadline = None

    def store_failed_attempt(self, packet, snr_linear, current_time): ...
    def combine(self, snr_linear): ...
    def is_valid(self, current_time): ...
    def flush(self): ...
```

### 9.2 HARQ buffer 생성 조건

```python
if failure_reason == "PHY_ERROR":
    harq_buffer.store_failed_attempt(...)
elif failure_reason == "COLLISION":
    do_not_store_soft_info()  # collision 시 soft info 무효
```

### 9.3 HARQ combining success probability

```python
effective_snr_linear = sum(snr_linear over HARQ attempts)
effective_snr_db = 10 * log10(effective_snr_linear)
p_success = 1 / (1 + exp(-a * (effective_snr_db - threshold_db[mcs])))
success = random.random() < p_success
```

### 9.4 HARQ-CC MCS 제약

```python
if tx_type == "HARQ_RETX":
    mcs = harq_buffer.original_mcs  # 이전 전송과 동일 MCS
if tx_type in ["NEW", "ARQ_RETX"]:
    mcs = select_mcs(current_snr)   # 현재 channel에 맞는 MCS 선택
```

---

## 10. Decision action 정의

```python
class Action(Enum):
    STAY_PRIMARY = auto()
    SWITCH_NPCA = auto()
    TX_NEW_PRIMARY = auto()
    TX_NEW_NPCA = auto()
    ARQ_RETX_PRIMARY = auto()
    ARQ_RETX_NPCA = auto()
    HARQ_RETX_PRIMARY = auto()
    HARQ_RETX_NPCA = auto()
    FLUSH_HARQ = auto()
```

Adaptive CW action:

```python
action = {
    "tx_decision": "HARQ_RETX_NPCA",
    "npca_initial_qsrc": q  # q ∈ {0, 1, 2, 3, 4, 5}
}
```

---

## 11. Rule-based baseline 정책

| 정책 | 설명 |
|---|---|
| Legacy ARQ-only | NPCA 없음, HARQ 없음, primary ARQ 재전송 |
| Fixed-NPCA | NPCA 전환 조건 만족 시 이동, CW_npca_init 고정, HARQ 없음 |
| HARQ-only | NPCA 없음, primary에서 HARQ 재전송 |
| NPCA-HARQ fixed CW | NPCA + HARQ, CW_npca_init 고정 |
| Adaptive NPCA-HARQ | NPCA + HARQ, CW_npca_init 상황에 따라 조절 |

---

## 12. NPCA-HARQ decision logic 예시

```python
def select_action(sta, env):
    packet = sta.peek_head_of_line_packet()
    if packet is None:
        return None

    harq_valid = sta.harq_buffer.active and sta.harq_buffer.is_valid(env.time)
    primary_delay_est = estimate_primary_access_delay(sta, env)
    npca_delay_est = estimate_npca_access_delay(sta, env)

    if harq_valid:
        if env.can_use_npca(sta) and npca_delay_est < primary_delay_est:
            return {"tx_decision": "HARQ_RETX_NPCA",
                    "npca_initial_qsrc": select_npca_qsrc(sta, env)}
        else:
            return {"tx_decision": "HARQ_RETX_PRIMARY"}
    else:
        if env.can_use_npca(sta) and npca_delay_est < primary_delay_est:
            return {"tx_decision": "TX_NEW_NPCA",
                    "npca_initial_qsrc": select_npca_qsrc(sta, env)}
        else:
            return {"tx_decision": "TX_NEW_PRIMARY"}
```

---

## 15. Channel model

### 15.1 Probabilistic PER model (Step 2+)

```python
p_success = sigmoid(a * (effective_snr_db - threshold_db[mcs]))
success = random.random() < p_success
```

MCS SNR thresholds (dB):

| MCS | threshold | 변조 |
|---|---|---|
| 0 | 5.0 | BPSK 1/2 |
| 1 | 8.0 | QPSK 1/2 |
| 2 | 11.0 | QPSK 3/4 |
| 3 | 14.0 | 16-QAM 1/2 |
| 4 | 17.0 | 16-QAM 3/4 |
| 5 | 20.0 | 64-QAM 2/3 |
| 6 | 23.0 | 64-QAM 3/4 |
| 7 | 26.0 | 64-QAM 5/6 |

### 15.2 Collision 처리

```python
if num_transmitters_on_same_channel >= 2:
    collision = True
    success = False
    # HARQ soft information 저장 안 함
```

---

## 16. Event loop 설계

```python
for t in range(num_slots):
    env.update_obss_activity()
    env.update_channel_states()
    for sta in sta_list:
        sta.update_mode(env)
    for sta in sta_list:
        sta.perform_backoff(env)
    tx_attempts = collect_transmissions(sta_list)
    results = env.resolve_transmissions(tx_attempts)
    for result in results:
        result.sta.handle_tx_result(result, env)
    logger.record(env, sta_list, results)
```

TX 흐름:
```text
[Slot T] backoff==0 → SNR 샘플링 → MCS 선택 → tx_request 생성
         충돌/AP-absence → handle_tx_result(False) 즉시
         충돌 없음 → channel.occupy_intra(T, ppdu_duration) → STA = PRIMARY_TX

[Slot T+D] tx_remaining==0 → PHY 판정 → handle_tx_result()
```

---

## 17. Transmission result handling

```python
def handle_tx_result(sta, result, env):
    packet = result.packet
    if result.success:
        packet.status = "DELIVERED"
        sta.queue.remove(packet)
        sta.harq_buffer.flush_if_packet(packet.packet_id)
        sta.reset_backoff_after_success(result.channel_type)
    else:
        packet.retry_count += 1
        if result.failure_reason == "PHY_ERROR":
            sta.harq_buffer.store_failed_attempt(...)
        sta.increase_backoff_after_failure(result.channel_type)
```

---

## 18. Backoff update after success/failure

```python
def reset_backoff_after_success(sta, channel_type):
    if channel_type == "PRIMARY":
        sta.primary_backoff_stage = 0
        sta.primary_cw = sta.cw_min
        sta.primary_backoff_counter = random.randint(0, sta.primary_cw)
    elif channel_type == "NPCA":
        sta.npca_backoff_stage = 0
        sta.npca_cw = sta.current_npca_cw_init
        sta.npca_backoff_counter = random.randint(0, sta.npca_cw)

def increase_backoff_after_failure(sta, channel_type):
    if channel_type == "PRIMARY":
        sta.primary_backoff_stage += 1
        sta.primary_cw = min(2 * (sta.primary_cw + 1) - 1, sta.cw_max)
        sta.primary_backoff_counter = random.randint(0, sta.primary_cw)
    elif channel_type == "NPCA":
        sta.npca_backoff_stage += 1
        sta.npca_cw = min(2 * (sta.npca_cw + 1) - 1, sta.cw_max)
        sta.npca_backoff_counter = random.randint(0, sta.npca_cw)
```

**주의**: NPCA 체류 중 실패 → `npca_cw` 증가. switch-back 후 다시 NPCA 전환 시 `npca_cw`는 `initial_npca_qsrc` 기반으로 재초기화.
