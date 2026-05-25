# Step 8: Baseline 비교 실험 지침

로드 시점: Step 8 작업 세션에서만 로드.
전제: `status.md` 와 `core.md` 를 먼저 로드할 것.

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

## 28. 권장 구현 순서

### Step 1: HARQ 없이 NPCA 동작 검증 ✅

- primary backoff 저장/복원
- NPCA backoff 초기화
- NPCA failure 시 NPCA CW 증가
- switch-back 후 primary state 복원

### Step 2: ARQ-only retransmission 구현 ✅

- retry counter
- CW 증가
- retry limit
- packet drop

### Step 3: HARQ buffer 추가 ✅

- failed PHY attempt 저장
- combining count 증가
- accumulated SNR 계산
- HARQ success probability 계산

### Step 4: NPCA-HARQ action 추가 ✅

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

## 33. 실행 방법

### 환경 준비

```bash
source .venv/bin/activate
# 또는 직접 실행
.venv/bin/python harq_sim/run_step5.py [OPTIONS]
```

### Step 1~4 실행 (참고)

```bash
# Step 1 기본 실행
python harq_sim/run_step1.py --slots 500 --stas 3 --obss-rate 0.05

# Step 2 (ARQ + PHY)
python harq_sim/run_step2.py --snr 25.0 --snr-std 0.0

# Step 3 (HARQ-CC)
python harq_sim/run_step3.py --harq-horizon 200

# Step 4 (Policy)
python harq_sim/run_step4.py --policy rule-based
```

### 검증 테스트 실행

```bash
python -m pytest tests/ -v
# 또는 단계별
python -m pytest tests/test_step1_npca.py -v
python -m pytest tests/test_step2_arq.py -v
python -m pytest tests/test_step3_harq.py -v
python -m pytest tests/test_step4_policy.py -v
```

---

## 34. CSV 출력 형식 요약

`sim_trace.csv`는 슬롯 × STA 조합마다 1행. 모든 값은 슬롯 시작 시점 기준.

### 주요 컬럼

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `slot` | int | 슬롯 번호 (0-indexed) |
| `time_us` | float | 경과 시간 (μs) |
| `sta_id` | int | STA 식별자 |
| `mode` | str | STA 모드 |
| `primary_cw` | int | Primary 채널 현재 CW값 |
| `npca_cw` | int | NPCA 채널 현재 CW값 |
| `tx_channel` | str\|None | TX 이벤트 채널: `PRIMARY` / `NPCA` / None |
| `tx_type` | str\|None | TX 종류: `NEW` / `ARQ_RETX` / `HARQ_RETX` |
| `tx_success` | bool\|None | TX 결과 |
| `failure_reason` | str\|None | `COLLISION` / `AP_ABSENCE_DUE_TO_NPCA` / `PHY_ERROR` |
| `snr_db` | float\|None | 샘플링된 SNR (Step 2+) |
| `effective_snr_db` | float\|None | HARQ combining 후 effective SNR (Step 3+) |
| `harq_combining_count` | int\|None | combining 횟수 (Step 3+) |
| `action_taken` | str\|None | policy 결정 (Step 4+) |

### TX 이벤트 해석

```text
tx_success = None  → multi-slot TX 시작 슬롯 (snr_db 기록됨)
tx_success = True  → TX 완료 슬롯 (PHY 성공)
tx_success = False + COLLISION        → 즉시 실패 (snr_db=None)
tx_success = False + AP_ABSENCE_DUE_TO_NPCA → 즉시 실패
tx_success = False + PHY_ERROR        → TX 완료 후 PHY 실패 (snr_db 기록됨)
```

### 이벤트 필터링 예시

```python
import pandas as pd
df = pd.read_csv("results/step4/sim_trace.csv")

transitions = df[df["npca_transition_start"] == True]
collisions  = df[df["failure_reason"] == "COLLISION"]
phy_errors  = df[df["failure_reason"] == "PHY_ERROR"]
harq_retx   = df[df["tx_type"] == "HARQ_RETX"]
policy_npca = df[df["action_taken"].str.contains("NPCA", na=False)]
```
