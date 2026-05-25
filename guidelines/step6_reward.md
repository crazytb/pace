# Step 6: Reward Module 구현 지침

로드 시점: Step 6 작업 세션에서만 로드.
전제: `status.md` 와 `core.md` 를 먼저 로드할 것.

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

## Step 6 구현 목표

```text
1. compute_metrics() 확장 (simulator.py 또는 별도 metrics.py)
   - aggregate_throughput, mean_access_delay, p95_access_delay
   - packet_delivery_ratio, packet_loss_probability
   - collision_probability_primary, collision_probability_npca
   - jain_fairness_index
   - legacy_throughput_degradation (NPCA 없는 baseline 대비)

2. normalize_metrics(metrics, refs) 함수
   - T_hat, D_hat, D95_hat, loss_hat, collision_hat, fairness_hat, energy_hat, legacy_hat
   - refs: 기준값 딕셔너리 (초기에는 하드코딩, 이후 동적 갱신)

3. compute_reward(normalized_metrics, weights, constraints) 함수
   - reward template 적용
   - constraint penalty 계산
   - 반환: scalar reward

4. INTENT_PROFILES 딕셔너리 (configs.py 또는 reward_profiles.py)
   - "throughput", "delay_sensitive", "qos_aware", "fair_coexistence", "energy_aware"
   - 각 프로파일 = weights dict + constraints dict

5. run_step6.py CLI
   - --intent throughput | delay | qos | fair | energy | custom
   - --reward-weights 'throughput=0.45,...' (custom 오버라이드)
   - summary에 reward, T_hat, D_hat, fairness_hat 추가

6. test_step6_reward.py
   - weights 합 == 1.0 검증
   - throughput intent에서 T_hat 가중치 최대 검증
   - constraint violation 시 reward 감소 검증
```

---

## 검증해야 할 핵심 불변식 (Step 6)

```text
sum(weights.values()) == 1.0 (모든 프로파일)
normalize_metrics() → 모든 값 ∈ [0, 1]
constraint penalty > 0 iff violation 존재
throughput intent: 동일 조건에서 fairness intent보다 T_hat 가중 reward 높음
Step 5 동작 유지 (backward compatible)
```
