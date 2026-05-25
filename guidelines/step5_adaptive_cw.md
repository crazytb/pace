# Step 5: Adaptive CW_npca_init 구현 지침

로드 시점: Step 5 작업 세션에서만 로드.
전제: `status.md` 와 `core.md` 를 먼저 로드할 것.

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

## Step 5 구현 목표

```text
1. STA 또는 Simulator에 npca_failure_rate 추적 추가
   - npca_failure_rate = (최근 N번 NPCA TX 중 실패 수) / N
   - 슬롯마다 또는 switch-back마다 업데이트

2. Simulator에 num_recent_npca_transitions 추적 추가
   - 최근 W 슬롯 내 NPCA 전환 이벤트 count

3. select_npca_qsrc(sta, env) 함수 구현
   - policy.py 또는 별도 adaptive_cw.py에 추가
   - 입력: primary_cw, npca_failure_rate, num_recent_npca_transitions, deadline_remaining
   - 출력: qsrc ∈ {0, 1, 2, 3, 4, 5}

4. NPCAHARQPolicy.select_action()에서 qsrc 적용
   - NPCA action 결정 시 select_npca_qsrc() 호출
   - 결과를 sta의 npca_initial_qsrc에 반영

5. run_step5.py CLI 작성
   - --adaptive-cw / --no-adaptive-cw 옵션
   - summary에 avg_npca_qsrc 컬럼 추가

6. test_step5_adaptive_cw.py 작성
   - 높은 NPCA 실패율 → qsrc 증가 검증
   - 짧은 deadline → qsrc 감소 검증
   - qsrc 범위 클램핑 검증 (min/max 초과 방지)
```

---

## 검증해야 할 핵심 불변식 (Step 5)

```text
npca_failure_rate > 0.3 → qsrc >= default + 1
primary_cw >= 4 * cw_min → qsrc >= default + 1
deadline_remaining < urgent_threshold → qsrc <= default - 1
qsrc 항상 [npca_qsrc_min, npca_qsrc_max] 범위 내
adaptive=False → Step 4 동작 유지 (backward compatible)
```
