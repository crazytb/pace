# Step 7: LLM / RL 연결 구현 지침

로드 시점: Step 7 작업 세션에서만 로드.
전제: `status.md` 와 `core.md` 를 먼저 로드할 것.

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

## Step 7 구현 목표

```text
1. LLMRewardDesigner 클래스 (llm_reward_designer_harq.py 또는 기존 llm_reward_designer.py 확장)
   - 입력: operator intent (자연어 문자열)
   - 출력: validate_reward_profile() 통과하는 JSON
   - use_mock=True: API 호출 없이 predefined profile 반환 (테스트용)
   - use_mock=False: Anthropic API 호출 (ANTHROPIC_API_KEY 환경변수 필요)

2. validate_reward_profile(profile) 함수
   - weights 합 == 1.0
   - 모든 weight >= 0
   - constraints 키 존재
   - 실패 시 ValueError 발생

3. DRL policy hook (선택)
   - LLM 반환 weights를 compute_reward()에 주입
   - RL agent의 reward signal로 사용

4. run_step7.py CLI
   - --intent "delay sensitive XR traffic"  (자연어)
   - --mock  (LLM API 미사용)
   - --llm-interval 50  (에피소드마다 LLM 재호출 주기)

5. test_step7_llm.py
   - mock 모드: 결정론적 profile 반환 검증
   - validate_reward_profile() 통과/실패 케이스 검증
   - weights 합 != 1.0 → ValueError 검증
```

---

## 검증해야 할 핵심 불변식 (Step 7)

```text
validate_reward_profile() 통과한 profile만 compute_reward()에 전달
use_mock=True → API 호출 없음 (네트워크 없는 환경에서도 동작)
intent "delay" → delay/tail_delay weight 합 > 0.5
intent "throughput" → throughput weight > 0.3
Step 6 동작 유지 (backward compatible, LLM 없이도 predefined profile 사용 가능)
```
