# Hybrid ARQ 기반 NPCA 시뮬레이션 코드 작성 지침

## 1. 시뮬레이션의 목적

본 시뮬레이션은 IEEE 802.11bn NPCA 환경에서 **Hybrid ARQ 기반 재전송 제어**가 channel access delay, throughput, packet delivery ratio, collision probability, fairness에 미치는 영향을 분석하기 위한 것이다.

기존 NPCA 시뮬레이터는 다음 동작을 포함한다고 가정한다.

- BSS primary channel에서 EDCA/CSMA/CA 기반 backoff 수행
- BSS primary channel이 inter-BSS PPDU 등에 의해 busy할 때 NPCA transition 수행
- NPCA primary channel에서 별도의 EDCA backoff 수행
- NPCA TXOP 획득 후 data transmission 수행
- NPCA 종료 후 BSS primary channel로 switch-back
- primary channel의 기존 EDCA state 저장 및 복원

본 확장에서는 여기에 다음 기능을 추가한다.

- MPDU/MSDU transmission failure 발생 시 ARQ 또는 HARQ 재전송 선택
- HARQ soft combining gain 모델링
- HARQ buffer 유지/flush
- primary channel 또는 NPCA primary channel 중 재전송 위치 선택
- adaptive `CW_npca_init` 또는 `Initial_NPCA_QSRC` 제어
- LLM-designed reward 또는 intent-based reward profile을 사용할 수 있는 RL interface 제공

---

## 2. 핵심 개념

### 2.1 기존 ARQ

ARQ에서는 전송 실패 시 동일 packet을 재전송하지만, receiver는 이전 실패 전송에서 얻은 soft information을 사용하지 않는다고 가정한다.

```text
Transmission attempt 1: fail
Transmission attempt 2: independent retry
```

성공 확률은 매 attempt마다 현재 channel condition, MCS, collision 여부에 의해 결정된다.

### 2.2 Hybrid ARQ

HARQ에서는 이전 실패 전송에서 얻은 soft information을 receiver buffer에 저장하고, 다음 재전송과 combining하여 decoding success probability를 높인다.

본 시뮬레이션에서는 우선 **HARQ-CC Chase Combining**을 기본으로 한다.

```text
Transmission attempt 1: fail, soft information stored
Transmission attempt 2: same packet retransmitted
Receiver combines attempt 1 + attempt 2
Decoding success probability increases
```

초기 구현에서는 다음과 같이 단순화한다.

- HARQ-CC 사용
- 동일 packet에 대한 retransmission count가 증가할수록 effective SNR 또는 decoding success probability 증가
- HARQ combining은 같은 receiver가 soft buffer를 유지하고 있을 때만 가능
- HARQ buffer lifetime 또는 validity horizon이 지나면 buffer flush

---

## 3. 전체 시스템 구조

### 3.1 주요 객체

기존 코드에 다음 객체 또는 속성을 추가한다.

```text
Environment
 ├── Channel primary_channel
 ├── Channel npca_channel
 ├── STA list
 ├── AP
 ├── event scheduler or slot loop
 ├── logger
 └── reward/evaluation module

STA
 ├── primary EDCA state
 ├── NPCA EDCA state
 ├── queue
 ├── HARQ buffer
 ├── current mode
 ├── transition timers
 ├── retry counters
 └── decision policy

HARQBuffer
 ├── packet_id
 ├── receiver_id
 ├── original_mcs
 ├── combining_count
 ├── accumulated_snr or accumulated_reliability
 ├── first_tx_time
 ├── last_tx_time
 ├── validity_deadline
 └── active flag

Packet
 ├── packet_id
 ├── arrival_time
 ├── size_bits
 ├── traffic_class
 ├── latency_deadline
 ├── retry_count
 ├── harq_count
 ├── current_mcs
 ├── status
 └── transmission_history
```

---

## 4. STA mode 정의

STA는 최소한 다음 mode를 가진다.

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

STA는 primary channel에서 다음 값을 가진다.

```python
primary_backoff_counter
primary_cw
primary_backoff_stage
primary_retry_counter
```

### 5.2 NPCA EDCA state

NPCA channel에서는 별도 값을 가진다.

```python
npca_backoff_counter
npca_cw
npca_backoff_stage
npca_retry_counter
npca_initial_qsrc
```

### 5.3 NPCA transition 시 primary state 저장

NPCA transition이 발생하면 primary EDCA state를 저장한다.

```python
saved_primary_state = {
    "cw": primary_cw,
    "backoff_counter": primary_backoff_counter,
    "backoff_stage": primary_backoff_stage,
    "retry_counter": primary_retry_counter,
}
```

그 후 NPCA EDCA state를 초기화한다.

```python
npca_qsrc = initial_npca_qsrc
npca_cw = 2 ** npca_qsrc * (cw_min + 1) - 1
npca_backoff_counter = random.randint(0, npca_cw)
npca_backoff_stage = 0
```

중요한 점은 **NPCA primary channel이 idle이어도 backoff procedure를 수행해야 한다**는 것이다. 즉 NPCA transition 직후 즉시 전송하지 않는다.

### 5.4 Switch-back 시 primary state 복원

NPCA operation이 끝나고 BSS primary channel로 돌아오면 저장한 primary state를 복원한다.

```python
primary_cw = saved_primary_state["cw"]
primary_backoff_counter = saved_primary_state["backoff_counter"]
primary_backoff_stage = saved_primary_state["backoff_stage"]
primary_retry_counter = saved_primary_state["retry_counter"]
```

NPCA에서 증가한 `npca_cw`는 primary CW에 반영하지 않는다.

### 5.5 다시 NPCA로 transition하는 경우

새로운 NPCA transition이 발생하면 `npca_cw`는 다시 `initial_npca_qsrc` 기반으로 초기화한다.

```python
npca_qsrc = initial_npca_qsrc
npca_cw = 2 ** npca_qsrc * (cw_min + 1) - 1
```

단, adaptive NPCA CW 제어를 사용하는 경우 `initial_npca_qsrc` 또는 `cw_npca_init`은 policy에 의해 변경될 수 있다.

---

## 6. NPCA transition 조건

STA 또는 AP는 BSS primary channel에서 다음과 같은 상황을 감지하면 NPCA transition을 고려한다.

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

### 6.3 중요한 모델링 포인트

기존 Bianchi model에서는 primary channel busy period가 단순히 backoff freezing time으로 처리된다. 하지만 NPCA에서는 이 busy period가 **NPCA transition opportunity**가 된다.

따라서 시뮬레이션에서는 다음 동작을 명확히 구현해야 한다.

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

AP가 single-radio이고 NPCA primary channel로 switch했다고 가정하면, AP는 BSS primary channel을 동시에 수신할 수 없다.

따라서 primary channel에 남아 있는 STA가 AP에게 uplink 전송을 시도하면 다음과 같이 처리한다.

```python
if sta.transmits_on_primary:
    if ap.mode == "PRIMARY":
        evaluate_success_or_collision()
    elif ap.mode == "NPCA":
        transmission_failed_due_to_ap_absence()
```

이 실패는 ACK timeout으로 관측된다.

```python
sta.retry_counter += 1
sta.primary_backoff_stage += 1
sta.primary_cw = min(2 * (sta.primary_cw + 1) - 1, cw_max)
sta.primary_backoff_counter = random.randint(0, sta.primary_cw)
```

로그에는 별도 실패 원인으로 기록한다.

```python
failure_reason = "AP_ABSENCE_DUE_TO_NPCA"
```

---

## 8. Transmission attempt 구조

전송 attempt는 다음 정보를 포함한다.

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
        self.packet_size_bits = None

    def store_failed_attempt(self, packet, snr_linear, current_time):
        ...

    def combine(self, snr_linear):
        ...

    def is_valid(self, current_time):
        ...

    def flush(self):
        ...
```

### 9.2 HARQ buffer 생성 조건

전송 실패 시 다음 조건을 만족하면 HARQ buffer에 저장한다.

```text
failure reason is PHY_DECODING_FAILURE
OR packet collision model allows partial soft information
AND packet retry limit not exceeded
AND receiver can maintain soft buffer
AND traffic class allows HARQ
```

초기 구현에서는 collision이 발생한 경우 soft information이 유효하지 않다고 단순화한다.

```python
if failure_reason == "PHY_ERROR":
    harq_buffer.store_failed_attempt(...)
elif failure_reason == "COLLISION":
    do_not_store_soft_info()
```

추후 확장에서는 collision 중 capture 또는 partial decoding을 고려할 수 있다.

### 9.3 HARQ combining success probability

가장 단순한 방식은 accumulated SNR을 사용한다.

```python
effective_snr_linear = sum(snr_linear over HARQ attempts)
effective_snr_db = 10 * log10(effective_snr_linear)
success = effective_snr_db >= mcs_threshold_db[mcs]
```

또는 logistic success probability를 사용한다.

```python
p_success = 1 / (1 + exp(-a * (effective_snr_db - threshold_db[mcs])))
success = random.random() < p_success
```

### 9.4 HARQ-CC MCS 제약

HARQ-CC를 사용하는 경우 동일 packet의 retransmission은 이전 전송과 동일 MCS를 사용한다고 가정한다.

```python
if tx_type == "HARQ_RETX":
    mcs = harq_buffer.original_mcs
```

반면 ARQ retransmission 또는 fresh transmission은 현재 channel condition에 따라 새 MCS를 선택할 수 있다.

```python
if tx_type in ["NEW", "ARQ_RETX"]:
    mcs = select_mcs(current_snr)
```

---

## 10. Decision action 정의

초기 구현에서는 rule-based policy와 RL policy 모두 지원할 수 있도록 action space를 분리한다.

### 10.1 기본 action

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

### 10.2 Adaptive CW action

NPCA initial CW 제어를 포함하려면 action에 `npca_qsrc_level`을 추가한다.

```python
action = {
    "tx_decision": "HARQ_RETX_NPCA",
    "npca_initial_qsrc": q
}
```

예:

```text
q ∈ {0, 1, 2, 3, 4, 5}
```

또는 직접 CW 값을 선택한다.

```text
cw_npca_init ∈ {7, 15, 31, 63, 127, 255}
```

---

## 11. Rule-based baseline 정책

### 11.1 Legacy ARQ-only

```text
NPCA 없음
HARQ 없음
전송 실패 시 primary channel에서 ARQ 재전송
```

### 11.2 Fixed-NPCA

```text
NPCA transition 조건 만족 시 NPCA로 이동
CW_npca_init은 고정
HARQ 없음
```

### 11.3 HARQ-only

```text
NPCA 없음
전송 실패 시 primary channel에서 HARQ 재전송
HARQ buffer validity 안에서만 combining
```

### 11.4 NPCA-HARQ fixed CW

```text
NPCA 가능
HARQ 가능
CW_npca_init은 고정
실패 packet은 NPCA channel에서 HARQ retransmission 가능
```

### 11.5 Adaptive NPCA-HARQ

```text
NPCA 가능
HARQ 가능
CW_npca_init을 상황에 따라 조절
```

---

## 12. NPCA-HARQ decision logic 예시

초기 rule-based policy는 다음과 같이 구현한다.

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
            return {
                "tx_decision": "HARQ_RETX_NPCA",
                "npca_initial_qsrc": select_npca_qsrc(sta, env)
            }
        else:
            return {
                "tx_decision": "HARQ_RETX_PRIMARY"
            }

    else:
        if env.can_use_npca(sta) and npca_delay_est < primary_delay_est:
            return {
                "tx_decision": "TX_NEW_NPCA",
                "npca_initial_qsrc": select_npca_qsrc(sta, env)
            }
        else:
            return {
                "tx_decision": "TX_NEW_PRIMARY"
            }
```

---

## 13. Adaptive `CW_npca_init` 설계

### 13.1 기본 아이디어

Primary channel에서 CW가 크다는 것은 contention이 심하거나 실패가 많았다는 의미일 수 있다. 이때 여러 STA가 동시에 NPCA transition을 시도하면 NPCA channel에서도 collision burst가 발생할 수 있다.

따라서 `CW_npca_init`은 고정값보다 adaptive하게 설정하는 것이 좋다.

### 13.2 입력 변수

`select_npca_qsrc()`는 다음 정보를 사용한다.

```python
features = {
    "primary_cw": sta.primary_cw,
    "primary_backoff_counter": sta.primary_backoff_counter,
    "primary_busy_ratio": env.primary_channel.busy_ratio,
    "npca_busy_ratio": env.npca_channel.busy_ratio,
    "npca_recent_failure_rate": sta.npca_failure_rate,
    "num_recent_npca_transitions": env.num_recent_npca_transitions,
    "harq_buffer_active": sta.harq_buffer.active,
    "harq_combining_count": sta.harq_buffer.combining_count,
    "packet_deadline_remaining": packet.deadline_remaining,
    "traffic_class": packet.traffic_class,
}
```

### 13.3 Rule-based adaptive QSRC 예시

```python
def select_npca_qsrc(sta, env):
    q = env.default_npca_qsrc

    if sta.primary_cw >= 4 * sta.cw_min:
        q += 1

    if env.num_recent_npca_transitions > env.npca_transition_threshold:
        q += 1

    if sta.npca_failure_rate > 0.3:
        q += 1

    if sta.current_packet.deadline_remaining < env.urgent_deadline_threshold:
        q -= 1

    q = max(env.npca_qsrc_min, min(q, env.npca_qsrc_max))
    return q
```

해석:

- primary CW가 크면 NPCA로 몰릴 가능성이 있으므로 NPCA 초기 CW를 키움
- 최근 NPCA transition 수가 많으면 NPCA contention이 심하다고 보고 CW를 키움
- NPCA 실패율이 높으면 CW를 키움
- deadline이 임박한 packet은 더 aggressive하게 접근하도록 CW를 줄임

---

## 14. Packet deadline 정의

HARQ packet deadline은 표준상 명시된 HARQ deadline이 아니라 시뮬레이션 모델링 변수로 정의한다.

### 14.1 QoS deadline

```python
packet.latency_deadline = packet.arrival_time + max_delay_by_traffic_class
```

예:

```python
max_delay_by_traffic_class = {
    "XR": 10e-3,
    "VOICE": 20e-3,
    "VIDEO": 50e-3,
    "BEST_EFFORT": None,
}
```

### 14.2 HARQ validity horizon

HARQ soft information이 유효한 시간이다.

```python
harq_validity_deadline = first_tx_time + channel_coherence_time
```

### 14.3 Packet discard

다음 조건이면 packet을 drop한다.

```python
if current_time > packet.latency_deadline:
    drop_packet(reason="DEADLINE_EXPIRED")
```

또는 retry limit 초과 시 drop한다.

```python
if packet.retry_count > retry_limit:
    drop_packet(reason="RETRY_LIMIT_EXCEEDED")
```

---

## 15. Channel model

초기 구현에서는 다음 두 가지 중 하나를 사용한다.

### 15.1 Threshold-based model

```python
success = effective_snr_db >= mcs_threshold_db[mcs]
```

### 15.2 Probabilistic PER model

```python
p_success = sigmoid(a * (effective_snr_db - threshold_db[mcs]))
success = random.random() < p_success
```

### 15.3 Collision 처리

```python
if num_transmitters_on_same_channel >= 2:
    collision = True
    success = False
```

초기 구현에서는 collision이 발생하면 HARQ soft information은 저장하지 않는다.

추후 capture effect를 추가할 수 있다.

---

## 16. Event loop 설계

### 16.1 Slot-based loop

가장 단순한 구조는 slot 단위 simulation이다.

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

### 16.2 Backoff update rule

```python
def perform_backoff(sta, env):
    if sta.mode == PRIMARY_BACKOFF:
        if env.primary_channel.is_idle_for_backoff(sta):
            sta.primary_backoff_counter -= 1
        else:
            sta.mode = PRIMARY_FROZEN

    elif sta.mode == PRIMARY_FROZEN:
        if can_transition_to_npca(sta, env):
            sta.start_npca_transition(env)
        elif env.primary_channel.is_idle_for_difs():
            sta.mode = PRIMARY_BACKOFF

    elif sta.mode == NPCA_BACKOFF:
        if env.npca_channel.is_idle_for_backoff(sta):
            sta.npca_backoff_counter -= 1
        else:
            sta.mode = NPCA_FROZEN

    elif sta.mode == NPCA_FROZEN:
        if env.npca_channel.is_idle_for_difs():
            sta.mode = NPCA_BACKOFF
```

### 16.3 Transmission trigger

```python
if sta.mode == PRIMARY_BACKOFF and sta.primary_backoff_counter <= 0:
    sta.start_primary_tx(env)

if sta.mode == NPCA_BACKOFF and sta.npca_backoff_counter <= 0:
    sta.start_npca_tx(env)
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

        if result.tx_type == "HARQ_RETX":
            packet.harq_count += 1

        if packet.retry_count > sta.retry_limit:
            sta.drop_packet(packet, reason="RETRY_LIMIT_EXCEEDED")
            sta.harq_buffer.flush_if_packet(packet.packet_id)
            sta.reset_backoff_after_drop(result.channel_type)
            return

        if packet.is_deadline_expired(env.time):
            sta.drop_packet(packet, reason="DEADLINE_EXPIRED")
            sta.harq_buffer.flush_if_packet(packet.packet_id)
            sta.reset_backoff_after_drop(result.channel_type)
            return

        if result.failure_reason == "PHY_ERROR":
            sta.harq_buffer.store_failed_attempt(
                packet=packet,
                snr_linear=result.snr_linear,
                current_time=env.time
            )

        sta.increase_backoff_after_failure(result.channel_type)
```

---

## 18. Backoff update after success/failure

### 18.1 Success

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
```

### 18.2 Failure

```python
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

주의:

- NPCA 체류 중 실패하면 `npca_cw`는 증가한다.
- 하지만 primary로 switch-back한 뒤 다시 NPCA로 갈 때는 `npca_cw`를 `initial_npca_qsrc` 기반으로 다시 초기화한다.
- packet retry counter는 channel과 무관하게 packet 단위로 유지한다.

---

## 19. Reward 설계

### 19.1 Raw metrics

시뮬레이션은 매 episode마다 다음 raw metric을 계산해야 한다.

```python
metrics = {
    "aggregate_throughput": ...,
    "per_sta_throughput": ...,
    "mean_access_delay": ...,
    "p95_access_delay": ...,
    "p99_access_delay": ...,
    "packet_delivery_ratio": ...,
    "packet_loss_probability": ...,
    "collision_probability_primary": ...,
    "collision_probability_npca": ...,
    "npca_transition_count": ...,
    "npca_transition_rate": ...,
    "average_npca_cw_init": ...,
    "jain_fairness_index": ...,
    "legacy_throughput_degradation": ...,
    "harq_success_gain": ...,
    "harq_buffer_flush_count": ...,
    "ap_absence_failure_count": ...,
}
```

### 19.2 Normalized metric basis

LLM-designed reward를 사용할 경우 모든 metric은 `[0, 1]` 범위로 normalize한다.

```python
T_hat = throughput / throughput_ref
D_hat = 1 - min(delay / delay_ref, 1)
D95_hat = 1 - min(p95_delay / p95_delay_ref, 1)
loss_hat = 1 - packet_loss_probability
collision_hat = 1 - collision_probability
fairness_hat = jain_fairness_index
energy_hat = 1 - min(energy / energy_ref, 1)
legacy_hat = 1 - min(legacy_degradation / legacy_degradation_ref, 1)
```

### 19.3 Reward template

Reward는 항상 동일한 template을 사용한다.

```python
reward = (
    w_T * T_hat
    + w_D * D_hat
    + w_D95 * D95_hat
    + w_loss * loss_hat
    + w_col * collision_hat
    + w_fair * fairness_hat
    + w_energy * energy_hat
    + w_legacy * legacy_hat
)
```

Constraint violation이 있으면 penalty를 추가한다.

```python
if legacy_degradation > legacy_degradation_max:
    reward -= lambda_legacy * (legacy_degradation - legacy_degradation_max)

if packet_loss_probability > packet_loss_max:
    reward -= lambda_loss * (packet_loss_probability - packet_loss_max)

if p95_delay > p95_delay_max:
    reward -= lambda_delay * (p95_delay - p95_delay_max)
```

---

## 20. Intent별 reward profile

### 20.1 Throughput-oriented

```json
{
  "throughput": 0.45,
  "delay": 0.10,
  "tail_delay": 0.05,
  "packet_loss": 0.10,
  "collision": 0.10,
  "fairness": 0.10,
  "energy": 0.05,
  "legacy_protection": 0.05
}
```

### 20.2 Delay-sensitive

```json
{
  "throughput": 0.10,
  "delay": 0.35,
  "tail_delay": 0.25,
  "packet_loss": 0.10,
  "collision": 0.05,
  "fairness": 0.05,
  "energy": 0.05,
  "legacy_protection": 0.05
}
```

### 20.3 QoS-aware

```json
{
  "throughput": 0.25,
  "delay": 0.20,
  "tail_delay": 0.15,
  "packet_loss": 0.15,
  "collision": 0.05,
  "fairness": 0.10,
  "energy": 0.05,
  "legacy_protection": 0.05
}
```

### 20.4 Fair coexistence

```json
{
  "throughput": 0.15,
  "delay": 0.10,
  "tail_delay": 0.10,
  "packet_loss": 0.10,
  "collision": 0.10,
  "fairness": 0.25,
  "energy": 0.05,
  "legacy_protection": 0.15
}
```

### 20.5 Energy-aware

```json
{
  "throughput": 0.15,
  "delay": 0.15,
  "tail_delay": 0.10,
  "packet_loss": 0.10,
  "collision": 0.05,
  "fairness": 0.10,
  "energy": 0.25,
  "legacy_protection": 0.10
}
```

---

## 21. LLM as reward designer를 위한 구현 원칙

LLM이 reward function code를 직접 생성하지 않도록 한다.

LLM의 역할은 다음으로 제한한다.

```text
operator intent
→ normalized reward weight vector
→ constraint threshold
```

### LLM output schema

```json
{
  "intent_name": "delay_sensitive",
  "weights": {
    "throughput": 0.10,
    "delay": 0.35,
    "tail_delay": 0.25,
    "packet_loss": 0.10,
    "collision": 0.05,
    "fairness": 0.05,
    "energy": 0.05,
    "legacy_protection": 0.05
  },
  "constraints": {
    "packet_loss_max": 0.01,
    "p95_delay_max_ms": 10,
    "legacy_degradation_max": 0.10
  }
}
```

### Validator

```python
def validate_reward_profile(profile):
    weights = profile["weights"]
    assert all(v >= 0 for v in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert "constraints" in profile
    return True
```

---

## 22. 실험 시나리오

### 22.1 Scenario variables

다음 변수들을 sweep한다.

```python
num_stas = [5, 10, 20, 30, 50]
obss_load = [0.1, 0.3, 0.5, 0.7]
npca_channel_busy_ratio = [0.0, 0.2, 0.4]
frame_duration = [0.5e-3, 1e-3, 2e-3, 5e-3]
mcs_distribution = ["static", "snr_based"]
cw_npca_init = [7, 15, 31, 63, 127]
harq_enabled = [False, True]
npca_enabled = [False, True]
```

---

## 23. Baseline schemes

반드시 다음 baseline을 구현한다.

```text
1. Legacy EDCA
2. ARQ-only NPCA
3. HARQ-only without NPCA
4. Fixed-CW NPCA-HARQ
5. Adaptive-CW NPCA-HARQ
6. LLM-reward NPCA-HARQ
7. Grid-best reward NPCA-HARQ
```

---

## 24. Grid-best reward baseline

LLM reward profile의 성능을 비교하기 위해 grid search baseline을 둔다.

```python
candidate_weights = generate_weight_grid(step=0.1)

for w in candidate_weights:
    train_rl_agent(reward_weights=w)
    eval_score = evaluate_policy(raw_metrics)
    keep_best()
```

이 baseline은 `oracle`이라고 부르기보다 다음 이름을 사용한다.

```text
grid-best
exhaustive-search baseline
practical upper bound
```

---

## 25. Logging format

각 transmission attempt마다 다음을 기록한다.

```python
log_entry = {
    "time": env.time,
    "sta_id": sta.id,
    "packet_id": packet.id,
    "mode": sta.mode.name,
    "channel": "PRIMARY" or "NPCA",
    "tx_type": "NEW" or "ARQ_RETX" or "HARQ_RETX",
    "mcs": mcs,
    "snr_db": snr_db,
    "effective_snr_db": effective_snr_db,
    "success": success,
    "failure_reason": failure_reason,
    "collision": collision,
    "primary_cw": sta.primary_cw,
    "npca_cw": sta.npca_cw,
    "primary_backoff": sta.primary_backoff_counter,
    "npca_backoff": sta.npca_backoff_counter,
    "harq_count": packet.harq_count,
    "retry_count": packet.retry_count,
    "delay": env.time - packet.arrival_time,
    "deadline_remaining": packet.deadline - env.time,
}
```

---

## 26. 중요한 성능 지표 계산

### Throughput

```python
throughput = total_success_bits / simulation_time
```

### Packet delivery ratio

```python
pdr = num_delivered_packets / num_generated_packets
```

### Mean delay

```python
mean_delay = mean(delivery_time - arrival_time)
```

### Tail delay

```python
p95_delay = percentile(delays, 95)
p99_delay = percentile(delays, 99)
```

### Collision probability

```python
collision_probability = num_collisions / num_tx_attempts
```

### HARQ gain

```python
harq_gain = pdr_harq - pdr_arq
```

또는

```python
harq_retx_success_rate = successful_harq_retx / total_harq_retx
```

### Jain fairness index

```python
jain = (sum(throughputs) ** 2) / (n * sum(t ** 2 for t in throughputs))
```

### Legacy degradation

```python
legacy_degradation = (
    legacy_throughput_without_npca - legacy_throughput_with_npca
) / legacy_throughput_without_npca
```

---

## 27. 구현 시 주의사항

### 27.1 HARQ와 MCS tradeoff

HARQ-CC retransmission은 기존 MCS에 묶인다고 가정한다.

```text
HARQ retransmission: reliability gain
Fresh/ARQ transmission: can exploit higher MCS
```

따라서 policy는 다음 tradeoff를 반영해야 한다.

```text
HARQ combining gain
vs.
higher MCS opportunity
vs.
NPCA access delay reduction
vs.
NPCA switching/backoff overhead
```

### 27.2 NPCA transition은 free가 아님

NPCA transition에는 다음 overhead가 있다.

```python
T_npca_switch
T_npca_backoff
T_icf_icr
T_npca_tx
T_switch_back
```

Delay 계산 시 반드시 반영한다.

### 27.3 NPCA backoff는 primary backoff와 분리

다음 동작을 보장해야 한다.

```text
primary → NPCA:
    save primary state
    initialize NPCA state

NPCA failure:
    increase NPCA CW

NPCA → primary:
    restore primary state

primary → NPCA again:
    initialize NPCA state again
```

### 27.4 Reward와 evaluation metric을 분리

학습 reward가 높다고 반드시 좋은 policy는 아니다. 최종 비교는 raw metric으로 한다.

```text
Do not compare policies only by cumulative reward.
Always compare throughput, delay, PDR, fairness, collision, and legacy degradation.
```

---

## 28. 권장 구현 순서

### Step 1: HARQ 없이 NPCA 동작 검증

- primary backoff 저장/복원
- NPCA backoff 초기화
- NPCA failure 시 NPCA CW 증가
- switch-back 후 primary state 복원

### Step 2: ARQ-only retransmission 구현

- retry counter
- CW 증가
- retry limit
- packet drop

### Step 3: HARQ buffer 추가

- failed PHY attempt 저장
- combining count 증가
- accumulated SNR 계산
- HARQ success probability 계산

### Step 4: NPCA-HARQ action 추가

- HARQ retransmission on primary
- HARQ retransmission on NPCA
- ARQ retransmission
- fresh packet transmission
- flush HARQ

### Step 5: Adaptive CW_npca_init 추가

- rule-based adaptive QSRC
- recent NPCA congestion 기반 조정
- deadline 기반 조정

### Step 6: Reward module 추가

- normalized metric basis
- intent별 reward profile
- constraint penalty

### Step 7: RL policy 또는 LLM reward profile 연결

- LLM은 reward weight만 생성
- reward template은 고정
- profile validator 구현

### Step 8: Baseline comparison

- Legacy
- NPCA only
- HARQ only
- NPCA-HARQ fixed
- NPCA-HARQ adaptive
- LLM reward
- grid-best reward

---

## 29. 최소 구현 목표

최초 버전에서는 다음만 구현해도 충분하다.

```text
1. primary/NPCA backoff 분리
2. NPCA transition condition
3. HARQ buffer
4. HARQ-CC combining model
5. fixed vs adaptive CW_npca_init
6. throughput/delay/PDR/collision/fairness logging
```

초기 버전에서는 다음은 단순화해도 된다.

```text
- capture effect 없음
- collision 시 HARQ soft information 저장 안 함
- AP는 single-radio로 가정
- LLM은 실제 API 호출 없이 predefined reward profiles로 대체
- RL 없이 rule-based policy 먼저 구현
```

---

## 30. 최종적으로 검증해야 할 연구 질문

시뮬레이션은 다음 질문에 답할 수 있어야 한다.

1. NPCA-HARQ는 HARQ-only 또는 NPCA-only보다 throughput/PDR/delay 측면에서 이득이 있는가?
2. primary CW가 큰 상황에서 NPCA transition은 delay를 줄이는가?
3. fixed small `CW_npca_init`은 NPCA collision burst를 유발하는가?
4. adaptive `CW_npca_init`은 NPCA collision을 줄이면서 delay 이득을 유지하는가?
5. HARQ retransmission을 NPCA channel에서 수행하는 것이 primary에서 기다리는 것보다 유리한 조건은 무엇인가?
6. HARQ combining gain과 higher-MCS fresh transmission gain 사이의 tradeoff는 NPCA에서 어떻게 달라지는가?
7. LLM-designed reward profile은 hand-crafted reward 또는 grid-best reward에 비해 어느 정도 성능을 내는가?
8. delay-sensitive, QoS-aware, fairness-aware intent에서 선택되는 NPCA/HARQ policy가 어떻게 달라지는가?

---

## 31. 요약

이 지침의 핵심은 다음과 같다.

```text
NPCA transition, HARQ retransmission, adaptive initial contention control, intent-aware reward design을 분리된 모듈로 구현한다.
```

구현상 가장 중요한 원칙은 다음이다.

```text
1. primary backoff와 NPCA backoff는 분리한다.
2. NPCA transition 시 primary state는 저장하고, NPCA state는 초기화한다.
3. HARQ buffer는 packet 단위로 관리한다.
4. HARQ-CC retransmission은 기존 MCS 제약을 갖는다.
5. adaptive CW_npca_init은 NPCA collision burst를 완화하기 위한 제어 변수다.
6. LLM은 reward code를 생성하지 않고, normalized reward weight와 constraint만 생성한다.
7. 최종 평가는 cumulative reward가 아니라 raw metric으로 수행한다.
```

---

## 32. 구현 모듈 구조 (`harq_sim/`)

본 시뮬레이션은 기존 DRL 프레임워크(`drl_framework/`)와 독립된 `harq_sim/` 모듈로 구현한다.
기존 Semi-MDP 기반 코드는 DRL 실험용으로 유지하고, HARQ 시뮬레이션은 별도 모듈에서 slot-by-slot 방식으로 동작한다.

### 32.1 파일 구조

```
harq_sim/
├── __init__.py          ← 모듈 export
├── enums.py             ← STAMode, ChannelType, TxType, FailureReason, TrafficClass, PacketStatus
├── channel.py           ← Channel 클래스 (OBSS/intra-BSS, obss_remain = NPCA_PPDU_REM_DUR)
├── packet.py            ← Packet, TransmissionAttempt 데이터 클래스
├── sta.py               ← STA 상태 머신 (이중 EDCA state, NPCA_TIMER, state save/restore)
├── simulator.py         ← Slot-based 이벤트 루프, 충돌 해결, CSV 출력
├── configs.py           ← CW, 슬롯 시간, OBSS, 에너지 상수
└── run_step1.py         ← Step 1 실행 스크립트 (CLI)

tests/
└── test_step1_npca.py   ← Step 1 검증 테스트 (7개)
```

### 32.2 객체 간 관계

```text
Simulator
 ├── Channel primary_channel      ← obss_remain = D1.2 NPCA_PPDU_REM_DUR
 ├── Channel npca_channel
 └── STA[] stas
      ├── primary EDCA state      ← primary_cw, primary_backoff_counter, ...
      ├── NPCA EDCA state         ← npca_cw, npca_backoff_counter, npca_initial_qsrc, ...
      ├── saved_primary_state     ← NPCA 전환 시 저장, switch-back 시 복원
      ├── npca_timer              ← D1.2 NPCA_TIMER = obss_remain − switch_back_delay
      ├── packet_queue            ← Deque[Packet]
      └── current_packet          ← 현재 전송 중인 패킷
```

### 32.3 TX 흐름 설계

충돌 감지와 TX 완료 보고를 분리하여 multi-slot PPDU를 정확히 모델링한다.

```text
[Slot T] STA.step() → sta.tx_request 생성 (백오프 카운터 = 0)
         Simulator → per-channel 충돌 판정
           - 충돌/AP-absence: handle_tx_result(False) 즉시 호출 → 다음 슬롯 PRIMARY_BACKOFF
           - 충돌 없음: channel.occupy_intra(T, ppdu_duration) → STA = PRIMARY_TX / NPCA_TX

[Slot T+1 ~ T+D-1] STA._handle_primary_tx() / _handle_npca_tx()
         tx_remaining 카운트다운 (채널은 occupied_remain으로 자동 busy 유지)

[Slot T+D] tx_remaining == 0
         STA.handle_tx_result(True) self-report → STA._completed_tx 기록
         Simulator logger → CSV에 tx_success=True 기록
```

### 32.4 D1.2 §37.18 부합도 (Step 1 기준)

| D1.2 조항 | 구현 여부 | 비고 |
|---|---|---|
| §37.18.3.1.b inter-BSS PPDU 감지 | ✅ | `is_busy_by_obss()` |
| §37.18.3.1.c.i `NPCA_PPDU_REM_DUR ≥ Min Duration Threshold` | ✅ | `npca_min_duration_threshold` |
| §37.18.3.1.d NPCA 채널 비중첩 | ✅ | 항상 별도 Channel 객체 |
| §37.18.3.1.e intra-BSS NAV = 0 | ✅ | `intra_bss_nav` 필드 |
| §37.18.3 Condition 2 (RTS/CTS 트리거) | ❌ | 미구현 |
| §37.18.4 pt 3 동일 EDCA 파라미터 세트 | ✅ | `npca_initial_qsrc` 기반 CW 초기화 |
| §37.18.4 pt 4a `NPCA_TIMER = NPCA_PPDU_REM_DUR − switch_back_delay` | ✅ | `sta.npca_timer` |
| §37.18.4 pt 4b NPCA_TIMER 만료 시 복귀 | ✅ | `_should_switch_back()` |
| CW 저장 / 복원 | ✅ | `_save_primary_state()` / `_restore_primary_state()` |
| TXOP 기반 다수 MPDU 연속 전송 | ❌ | PPDU 1개만 전송 |
| OMP 파라미터 협상 | ❌ | 미구현 |

---

## 33. 실행 방법

### 33.1 환경 준비

```bash
# 프로젝트 루트에서 가상환경 활성화
source .venv/bin/activate
# 또는 직접 실행
.venv/bin/python harq_sim/run_step1.py [OPTIONS]
```

### 33.2 Step 1 시뮬레이션 실행

```bash
# 기본 실행 (STA 3개, 300 슬롯)
python harq_sim/run_step1.py

# 주요 옵션
python harq_sim/run_step1.py \
    --slots    500   \   # 총 슬롯 수 (9 μs/slot)
    --stas     3     \   # STA 수
    --obss-rate 0.05 \   # OBSS 발생 확률 (per slot)
    --obss-min  30   \   # OBSS 최소 지속 슬롯
    --obss-max  80   \   # OBSS 최대 지속 슬롯
    --qsrc     0     \   # NPCA initial CW exponent (0→CW=15, 1→31, 2→63, ...)
    --threshold 0    \   # NPCA min duration threshold (슬롯)
    --ppdu     20    \   # PPDU 전송 슬롯 수
    --seed     42    \   # random seed
    --no-npca        \   # NPCA 비활성화 (비교용)
    --out-dir results/step1  # 출력 디렉토리

# NPCA CW 비교 실험 (qsrc=0 vs qsrc=2)
python harq_sim/run_step1.py --slots 1000 --stas 5 --qsrc 0 --out-dir results/step1_qsrc0
python harq_sim/run_step1.py --slots 1000 --stas 5 --qsrc 2 --out-dir results/step1_qsrc2

# NPCA 비활성화 baseline
python harq_sim/run_step1.py --slots 1000 --stas 5 --no-npca --out-dir results/step1_no_npca
```

### 33.3 Step 1 검증 테스트 실행

```bash
# 단독 실행
python tests/test_step1_npca.py

# pytest 사용
python -m pytest tests/test_step1_npca.py -v
```

검증 항목:

| 테스트 | 내용 |
|---|---|
| T1 | Primary state save/restore — NPCA 전환 전후 CW/backoff/stage/retry 일치 |
| T2 | NPCA/Primary CW 독립성 — NPCA 실패 시 primary_cw 불변, vice versa |
| T3 | Switch-back 후 primary state 복원 — OBSS 종료 후 자동 복귀 |
| T4 | NPCA Min Duration Threshold (D1.2 §37.18.3.1.c.i) 조건 강제 |
| T5 | AP absence 실패 처리 — `AP_ABSENCE_DUE_TO_NPCA` 기록 |
| T6 | `npca_initial_qsrc` → `npca_cw` 변환 (`2^q × 16 − 1`) |
| T7 | NPCA enabled/disabled smoke test — transition/switch-back 카운터 검증 |

### 33.4 출력 파일

```
results/step1/
├── sim_trace.csv    ← 슬롯별 전체 상태 로그 (num_slots × num_stas 행)
└── summary.txt      ← 시뮬레이션 파라미터 + STA별 집계 통계
```

---

## 34. CSV 출력 형식

`sim_trace.csv`는 슬롯 × STA 조합마다 1행을 출력한다. 모든 값은 슬롯 **시작 시점** 기준이다 (step() 호출 전 스냅샷). TX 결과(success/fail)만 해당 슬롯 처리 후 기록된다.

### 34.1 컬럼 목록

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `slot` | int | 슬롯 번호 (0-indexed) |
| `time_us` | float | 경과 시간 (μs, slot × 9.0) |
| `sta_id` | int | STA 식별자 |
| `mode` | str | STA 모드 (슬롯 시작 시): `PRIMARY_BACKOFF` / `PRIMARY_FROZEN` / `PRIMARY_TX` / `NPCA_SWITCHING` / `NPCA_BACKOFF` / `NPCA_FROZEN` / `NPCA_TX` / `SWITCH_BACK` |
| `primary_cw` | int | Primary 채널 현재 CW값 |
| `primary_backoff_counter` | int | Primary 백오프 카운터 |
| `primary_backoff_stage` | int | Primary 백오프 단계 (실패 횟수) |
| `primary_retry_counter` | int | Primary 채널 누적 재전송 횟수 |
| `npca_cw` | int | NPCA 채널 현재 CW값 |
| `npca_backoff_counter` | int | NPCA 백오프 카운터 |
| `npca_backoff_stage` | int | NPCA 백오프 단계 |
| `npca_retry_counter` | int | NPCA 채널 누적 재전송 횟수 |
| `npca_timer` | int | D1.2 NPCA_TIMER 잔여값 (0 = switch-back 조건) |
| `saved_primary_cw` | int\|None | NPCA 체류 중 저장된 primary CW (None = primary 모드) |
| `saved_primary_backoff_counter` | int\|None | NPCA 체류 중 저장된 primary 백오프 카운터 |
| `primary_obss_remain` | int | Primary 채널 OBSS 잔여 슬롯 (= D1.2 `NPCA_PPDU_REM_DUR`) |
| `npca_obss_remain` | int | NPCA 채널 OBSS 잔여 슬롯 |
| `tx_channel` | str\|None | TX 이벤트 채널: `PRIMARY` / `NPCA` / None (TX 없음) |
| `tx_type` | str\|None | TX 종류: `NEW` / `ARQ_RETX` / `HARQ_RETX` |
| `tx_success` | bool\|None | TX 결과: `True`(완료), `False`(실패), None(multi-slot TX 진행 중 시작 슬롯) |
| `failure_reason` | str\|None | 실패 원인: `COLLISION` / `AP_ABSENCE_DUE_TO_NPCA` / `PHY_ERROR` / `NONE` |
| `packet_id` | int\|None | 전송 중인 패킷 ID |
| `retry_count` | int\|None | 해당 패킷의 누적 재전송 횟수 |
| `npca_transition_start` | bool | 이 슬롯에서 NPCA 전환 시작 여부 |
| `switch_back_start` | bool | 이 슬롯에서 Primary 복귀 시작 여부 |

### 34.2 TX 이벤트 해석

```text
tx_success = None  → multi-slot TX 시작 슬롯 (결과 미정)
tx_success = True  → TX 완료 슬롯 (ppdu_duration 슬롯 후 자동 기록)
tx_success = False → 즉시 실패 슬롯 (충돌 또는 AP 부재)
```

multi-slot TX 예시 (ppdu_duration = 20):

```text
slot 53: tx_channel=NPCA, tx_type=NEW, tx_success=None   ← TX 시작
slot 54~72: mode=NPCA_TX (tx_channel 없음)               ← 카운트다운 중
slot 73: tx_channel=NPCA, tx_type=NEW, tx_success=True   ← TX 완료
```

### 34.3 NPCA 관련 이벤트 필터링 예시

```python
import pandas as pd

df = pd.read_csv("results/step1/sim_trace.csv")

# NPCA 전환 이벤트
transitions = df[df["npca_transition_start"] == True]

# 충돌 발생 슬롯
collisions = df[df["failure_reason"] == "COLLISION"]

# TX 완료(성공) 슬롯만
successes = df[df["tx_success"] == True]

# NPCA 체류 중 구간 (saved_primary_cw 있음)
in_npca = df[df["saved_primary_cw"].notna()]

# STA별 primary CW 시계열
df[df["sta_id"] == 0][["slot", "primary_cw", "npca_cw", "mode"]].plot(x="slot")
```

---

## 35. 구현 현황 (Step별 완료 상태)

### Step 1 ✅ 완료

**구현 내용:**
- `STAMode` 8개 상태 정의 (`PRIMARY_BACKOFF` ~ `SWITCH_BACK`)
- Primary / NPCA EDCA state 완전 분리
- NPCA 전환 시 primary state 저장, switch-back 시 복원
- NPCA CW 초기화: `npca_cw = 2^npca_initial_qsrc × (CW_MIN + 1) − 1`
- NPCA 실패 시 NPCA CW 증가, primary CW 불변
- `NPCA_TIMER` 관리 (D1.2 §37.18.4)
- AP absence 실패 처리
- Slot-based 시뮬레이터 + CSV 출력
- 검증 테스트 7개 (all pass)

**검증된 핵심 불변식:**
```text
NPCA 전환 시: saved_primary_state = current primary state
NPCA 실패 시: npca_cw 증가, primary_cw 불변
Switch-back 시: primary_cw = saved_primary_state["cw"] (NPCA CW 반영 안 됨)
재전환 시: npca_cw = 2^qsrc × 16 − 1 (이전 NPCA 실패 CW 리셋)
```

### Step 2 ⬜ 미구현

ARQ-only retransmission:
- PHY error model (SNR 기반 성공 확률)
- retry counter, CW 증가, retry limit 초과 시 packet drop
- `tx_type = ARQ_RETX` 로 재전송 구분

### Step 3 ⬜ 미구현

HARQ buffer:
- `HARQBuffer` 클래스 (`harq_sim/harq_buffer.py`)
- PHY 실패 시 soft information 저장
- Chase Combining: `accumulated_snr_linear += snr_linear`
- `validity_deadline = first_tx_time + channel_coherence_time`

### Step 4 ⬜ 미구현

NPCA-HARQ action:
- `Action` enum (`HARQ_RETX_PRIMARY`, `HARQ_RETX_NPCA`, `FLUSH_HARQ` 등)
- rule-based policy: HARQ 유효 시 NPCA에서 재전송 vs primary 대기 비교

### Step 5 ⬜ 미구현

Adaptive `CW_npca_init`:
- `select_npca_qsrc()` rule-based 구현
- primary CW, NPCA 실패율, packet deadline 기반 qsrc 조정

### Step 6 ⬜ 미구현

Reward module:
- normalized metric basis (`T_hat`, `D_hat`, ...)
- intent별 reward weight profile (throughput / delay / energy / fairness / QoS-aware)
- constraint penalty 추가

### Step 7 ⬜ 미구현

LLM / RL 연결:
- LLM reward weight 생성 → reward template 고정
- profile validator
- DRL policy hook

### Step 8 ⬜ 미구현

Baseline 비교:
1. Legacy EDCA (NPCA 없음, ARQ 없음)
2. ARQ-only NPCA
3. HARQ-only (NPCA 없음)
4. Fixed-CW NPCA-HARQ
5. Adaptive-CW NPCA-HARQ
6. LLM-reward NPCA-HARQ
7. Grid-best reward NPCA-HARQ
