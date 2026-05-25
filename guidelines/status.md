# 구현 현황 (Step별 완료 상태)

항상 로드: 현재 어느 Step까지 완료됐는지 파악하기 위한 파일.

---

## Step 1 ✅ 완료

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

---

## Step 2 ✅ 완료

**구현 내용:**
- `harq_sim/phy.py` 신규 작성 — logistic PER 모델 (`success_prob`), MCS 선택 (`select_mcs`), HARQ-CC 준비용 SNR 변환
- `STA` 파라미터 추가: `snr_db_mean`, `snr_db_std` (링크 품질 설정)
- TX 시작 시 SNR 샘플링 + MCS 자동 선택 (`pkt.current_mcs` 갱신)
- TX 완료 시 PHY 판정: `phy.attempt_success(_current_tx_snr_db, mcs)` → PHY_ERROR 또는 성공
- `_phy_failure_tx` dict: PHY 실패 이벤트를 simulator logger가 슬롯 종료 후 수집
- `phy_error_failures` 통계 추가 (`stats` dict + `compute_metrics()`)
- `SlotLog.snr_db` 필드 추가 (PHY 성공/실패 이벤트에 기록, collision은 None)
- `run_step2.py` CLI: `--snr`, `--snr-std` 옵션 포함
- 검증 테스트 8개 (all pass)

**검증된 핵심 불변식:**
```text
PHY 실패 시: npca/primary CW 증가 방향은 Step 1과 동일 (채널 독립성 유지)
ARQ 재시도:  tx_type = ARQ_RETX, retry_count ≥ 1, 새 SNR 샘플링 + MCS 재선택
Retry limit: pkt.retry_count > retry_limit → DROPPED (deadline 체크보다 우선)
Deadline:    current_slot > latency_deadline → DROPPED (XR: 1111슬롯, BEST_EFFORT: 없음)
Collision:   PHY 판정 없음 — simulator가 즉시 handle_tx_result(False, COLLISION) 호출
```

**PHY 모델 파라미터:**
```python
# harq_sim/phy.py — MCS SNR thresholds (dB)
MCS_SNR_THRESHOLDS = {
    0:  5.0,   # BPSK  1/2  → p=0.5 at SNR=5dB
    1:  8.0,   # QPSK  1/2
    2: 11.0,   # QPSK  3/4
    3: 14.0,   # 16-QAM 1/2
    4: 17.0,   # 16-QAM 3/4
    5: 20.0,   # 64-QAM 2/3
    6: 23.0,   # 64-QAM 3/4
    7: 26.0,   # 64-QAM 5/6
}
SIGMOID_STEEPNESS = 1.0  # logistic curve a parameter
```

---

## Step 3 ✅ 완료

**구현 내용:**
- `harq_sim/harq_buffer.py` 신규 작성 — `HARQBuffer` 클래스
  - `store(packet, snr_linear, slot)`: 첫 PHY 실패 시 초기화, 이후 누적 (Chase Combining)
  - `effective_snr_db(new_snr_linear)`: `10·log10(accumulated + new)` 계산
  - `is_valid(slot)`: `first_tx_slot + validity_horizon` 이내 여부 확인
  - `flush()`: 전달/drop 후 버퍼 초기화
- `STA` 파라미터 추가: `harq_enabled`, `harq_validity_horizon` (기본 200 슬롯 ≈ 1.8 ms)
- `STA.harq_buffer` 필드 추가 — 항상 초기화되며 `harq_enabled=True`일 때만 활성
- `_is_harq_retx_applicable(pkt, slot)` 헬퍼: 버퍼 active + 동일 packet_id + 유효 확인; 만료 시 자동 flush
- `_compute_effective_snr(pkt)` 헬퍼: 버퍼 활성 시 accumulated + current SNR → effective_snr_db
- `_handle_primary_backoff()` / `_handle_npca_backoff()` TX 트리거 수정:
  - 버퍼 유효 → `tx_type=HARQ_RETX`, `mcs=harq_buffer.original_mcs` (MCS 제약)
  - 버퍼 없음/만료 → `tx_type=NEW|ARQ_RETX`, 새 MCS 선택
- `handle_tx_result()` 수정: PHY_ERROR 시 `harq_buffer.store()` 호출 + `pkt.harq_count += 1`
- `run_step3.py` CLI: `--harq-horizon`, `--no-harq` 옵션
- 검증 테스트 9개 (all pass)

**검증된 핵심 불변식:**
```text
PHY_ERROR → harq_buffer.store(): combining_count++, accumulated_snr += snr_linear
HARQ_RETX:  tx_type=HARQ_RETX, mcs=original_mcs, eff_snr = acc + current
Collision:  harq_buffer.active=False (soft information 저장 안 함)
Validity:   _is_harq_retx_applicable() 만료 시 flush → ARQ_RETX fallback
PDR gain:   경계 SNR(14dB, MCS3)에서 HARQ PDR ≈ 0.976 vs ARQ PDR ≈ 0.746 (+23%)
```

**HARQ-CC SNR 누적 예시 (SNR=14dB, MCS3, retry_limit=1):**
```text
attempt 1: snr_db=14.0, p_success=0.500 → PHY FAIL
           → buffer.accumulated = snr_linear(14dB) = 25.12, combining_count=1
attempt 2: snr_db=14.0, new_linear=25.12
           eff_snr = 10·log10(25.12 + 25.12) = 10·log10(50.24) = 17.01 dB
           p_success(17.01dB, MCS3) = sigmoid(17.01-14) ≈ 0.953 → PHY SUCCESS
```

---

## Step 4 ✅ 완료

**구현 내용:**
- `harq_sim/enums.py` 추가: `Action` enum (8개 값) + `NPCA_ACTIONS` frozenset
- `harq_sim/policy.py` 신규 작성 — `NPCAHARQPolicy` 클래스
  - `estimate_primary_access_delay(sta)`: `obss_remain + primary_backoff_counter`
  - `estimate_npca_access_delay(sta)`: `switching_delay + npca_cw_init // 2`
  - `select_action(sta, slot)`: HARQ 유효 여부 + delay 비교 → NPCA/primary 중 선택
- `STA` 파라미터 추가: `policy: Optional[NPCAHARQPolicy] = None`
- `STA._decide_npca_or_stay()` 헬퍼:
  - policy 있음 → `policy.select_action()` 호출, NPCA_ACTIONS면 전환, 아니면 PRIMARY_FROZEN 유지
  - policy 없음 → Step 3 backward-compat (항상 NPCA 전환)
- `STA._last_action` 필드: 매 슬롯 초기화, policy 결정 시 기록
- `STA.stats` 추가: `policy_npca_chosen`, `policy_primary_chosen`
- `SlotLog.action_taken` 필드 추가 (Optional[str]): policy 결정 CSV 기록
- `run_step4.py` CLI: `--policy rule-based | none` 옵션
- 검증 테스트 8개 (all pass)

**검증된 핵심 불변식:**
```text
HARQ_RETX_NPCA : HARQ 버퍼 유효 + npca_delay < primary_delay → NPCA 전환
HARQ_RETX_PRIMARY : HARQ 버퍼 유효 + primary_delay ≤ npca_delay → primary 대기
TX_NEW_NPCA : 버퍼 없음 + npca_delay < primary_delay → NPCA 전환
NPCA 불가(disabled/threshold) → 항상 primary (HARQ_RETX_PRIMARY 등)
policy=None → Step 3 동작 유지 (backward compatible)
```

**delay 추정 예시 (npca_qsrc=0, switching_delay=1):**
```text
npca_cw_init = 15 → expected_backoff = 7 → npca_delay = 8 슬롯
short OBSS (6 slots): primary_delay = 6 < 8 = npca_delay → STAY PRIMARY ✓
long  OBSS (60 slots): primary_delay = 60 >> 8 = npca_delay → GO NPCA  ✓
```

**`policy_primary_chosen` 관찰 예시 (obss_min=5, obss_max=20, 1000 slots, 2 STAs):**
```text
STA0: Pol_NPCA=19, Pol_Pri=13  ← 짧은 OBSS에서 primary 선택 다수
STA1: Pol_NPCA=20, Pol_Pri=7
```

---

## Step 5 ✅ 완료

**구현 내용:**
- `harq_sim/configs.py` 추가: `NPCA_QSRC_MIN/MAX`, `NPCA_FAILURE_WINDOW`, `NPCA_TRANSITION_WINDOW/THRESHOLD`, `URGENT_DEADLINE_THRESHOLD`
- `harq_sim/policy.py` 추가: `select_npca_qsrc(sta, slot, default_qsrc)` — 4개 규칙 기반 qsrc 조정
  - `primary_cw ≥ 4×CW_MIN` → +1
  - `num_recent_npca_transitions > THRESHOLD` → +1
  - `npca_failure_rate > 0.3` → +1
  - `deadline_remaining < URGENT_THRESHOLD` → -1
  - 결과 `[NPCA_QSRC_MIN, NPCA_QSRC_MAX]`으로 클램핑
- `NPCAHARQPolicy(adaptive_cw=False)` 파라미터 추가: adaptive=True 시 전환 결정 직전 `sta.npca_initial_qsrc` 업데이트
- `STA` 추가: `_npca_tx_window` 슬라이딩 윈도우 (maxlen=10), `npca_failure_rate` property, `num_recent_npca_transitions` 속성, `_npca_qsrc_history` 리스트
- `Simulator` 추가: `_npca_transition_deque`, `_count_recent_transitions(slot)`, 매 슬롯 STA에 recent count 주입
- `compute_metrics()` 추가: `avg_npca_qsrc`
- `harq_sim/run_step5.py` CLI: `--adaptive-cw` / `--no-adaptive-cw` 옵션, summary에 AvgQsrc 컬럼
- 검증 테스트 9개 (all pass)

**검증된 핵심 불변식:**
```text
npca_failure_rate > 0.3 → qsrc >= default + 1
primary_cw >= 4 * CW_MIN → qsrc >= default + 1
deadline_remaining < URGENT_DEADLINE_THRESHOLD → qsrc <= default - 1
qsrc 항상 [NPCA_QSRC_MIN, NPCA_QSRC_MAX] 범위 내
adaptive=False → Step 4 동작 유지 (backward compatible)
```

## Step 6 ✅ 완료

**구현 내용:**
- `harq_sim/configs.py` 추가: `REWARD_THROUGHPUT_REF`, `REWARD_DELAY_REF`, `REWARD_D95_REF`, `REWARD_ENERGY_REF`, `REWARD_LEGACY_REF`, constraint 상수 6개
- `harq_sim/sta.py` 추가: `total_energy_uj` 필드 (슬롯별 TX/LISTEN 에너지 누적), `_delivered_delays` 리스트 (패킷 전달 지연 기록), `_start_npca_transition()` NPCA 전환 에너지 이벤트, `_peek_head(slot)` slot 기반 arrival_time 추적
- `harq_sim/reward.py` 신규 작성 — 5개 intent 프로파일 + `normalize_metrics()` + `compute_reward()`
  - `INTENT_PROFILES`: throughput / delay_sensitive / qos_aware / fair_coexistence / energy_aware
  - `normalize_metrics(metrics, refs)`: T_hat, D_hat, D95_hat, loss_hat, collision_hat, fairness_hat, energy_hat, legacy_hat → 모두 [0, 1]
  - `compute_reward(normalized, weights, constraints, raw_metrics)`: reward template + constraint penalty (정규화 위반)
- `harq_sim/simulator.py` `compute_metrics()` 확장:
  - 기존 per-STA dict에 `total_energy_uj` 추가
  - `"aggregate"` 키 추가: aggregate_throughput, mean_access_delay, p95/p99_access_delay, PDR/loss, collision_probability (primary/NPCA 분리), jain_fairness_index, total_energy_uj, npca_transition_count/rate
- `harq_sim/run_step6.py` CLI: `--intent`, `--reward-weights` 옵션, 정규화·reward 값 출력
- 검증 테스트 9개 (all pass)

**검증된 핵심 불변식:**
```text
sum(weights.values()) == 1.0 (모든 5개 프로파일)
normalize_metrics() → 모든 값 ∈ [0, 1] (경계 포함)
zero 입력: T_hat=0.0, D_hat=1.0, loss_hat=1.0, fairness_hat=1.0
constraint 위반 시: reward 0.79 → 0.57 (-constraint_penalty)
throughput intent > fair_coexistence intent (처리량 높은 케이스) ✓
energy_hat 계산 근거: 100-slot 시뮬레이션에서 STA별 ≥ 49 μJ (0.495×100)
```

**에너지 모델 (sta.py):**
```python
TX_MODES  = {PRIMARY_TX, NPCA_TX}       → ENERGY_TX_PER_SLOT_UJ  = 2.772 μJ/slot
else                                     → ENERGY_LISTEN_PER_SLOT_UJ = 0.495 μJ/slot
_start_npca_transition()                 → ENERGY_NPCA_TRANSITION_UJ = 0.75 μJ (단발)
```

## Step 7 ✅ 완료

**구현 내용:**
- `harq_sim/llm_reward_designer.py` 신규 작성 — `LLMRewardDesigner` 클래스 + `validate_reward_profile()`
  - `use_mock=True`: intent 키워드 매칭으로 predefined profile 반환 (API 불필요)
  - `use_mock=False`: Anthropic API 호출 (system prompt 캐싱 with `cache_control: ephemeral`)
  - `design_reward(intent_str)` → validate 통과한 profile dict 반환
  - `_normalize_constraints()`: LLM 반환 `p95_delay_max_ms` → slot 단위 변환
  - keyword map: delay/throughput/energy/fair/qos 5종 → INTENT_PROFILES 매핑
- `validate_reward_profile(profile)` 함수:
  - weights 합 == 1.0 (허용 오차 1e-6)
  - 모든 weight >= 0
  - constraints 키 존재
  - 위반 시 ValueError 발생
- `harq_sim/run_step7.py` CLI: `--intent`, `--mock`/`--no-mock`, `--llm-interval` 옵션
  - LLM profile을 `results/step7/llm_profile.json` 저장
- `harq_sim/__init__.py` 업데이트: `LLMRewardDesigner`, `validate_reward_profile`, `INTENT_PROFILES`, `normalize_metrics`, `compute_reward` export 추가
- 검증 테스트 9개 (all pass)

**검증된 핵심 불변식:**
```text
validate_reward_profile() 통과한 profile만 compute_reward()에 전달
use_mock=True → API 호출 없음 (네트워크 없는 환경에서도 동작)
intent "delay" → delay+tail_delay weight 합 = 0.60 > 0.5
intent "throughput" → throughput weight = 0.45 > 0.3
Step 6 동작 유지 (backward compatible, LLM 없이도 predefined profile 사용 가능)
```

**mock 키워드 매핑:**
```python
["delay", "latency", "xr", "rtc", "voice", "video"] → delay_sensitive
["throughput", "speed", "bandwidth", "download"]     → throughput
["energy", "battery", "power", "saving"]             → energy_aware
["fair", "coexist", "legacy"]                        → fair_coexistence
["qos", "quality", "balanced"]                       → qos_aware (default)
```

**constraint 단위 변환:**
```python
p95_delay_max_ms (LLM 출력) → slots = ms × 1000 / SLOT_DURATION_US (9 μs)
# 예: 10 ms → 1,111 slots, 300 ms → 33,333 slots
```

## Step 8 ✅ 완료

**구현 내용:**
- `harq_sim/run_step8.py` 신규 작성 — 7개 baseline 비교 CLI 스크립트
  - `build_and_run(...)` 공통 시뮬레이터 빌더 (Step 7과 동일 시그니처)
  - `grid_best_reward_profile(sim)` — INTENT_PROFILES 5종 중 reward 최고 profile 선택
  - `run_baseline(name, common_kwargs, reward_profile)` — 1개 baseline 실행 + 결과 반환
  - `print_summary()` / `save_comparison_csv()` — 비교 테이블 출력 및 CSV 저장
- 7개 baseline 구성:

| 번호 | 이름 | npca_enabled | harq_enabled | adaptive_cw |
|------|------|:---:|:---:|:---:|
| 1 | legacy_edca | ✗ | ✗ | ✗ |
| 2 | arq_only_npca | ✓ | ✗ | ✗ |
| 3 | harq_only | ✗ | ✓ | ✗ |
| 4 | fixed_cw_npca_harq | ✓ | ✓ | ✗ |
| 5 | adaptive_cw_npca_harq | ✓ | ✓ | ✓ |
| 6 | llm_reward_npca_harq | ✓ | ✓ | ✓ |
| 7 | grid_best_reward_npca_harq | ✓ | ✓ | ✓ |

- Grid-best: adaptive-CW NPCA-HARQ 1회 시뮬레이션 → 5개 INTENT_PROFILES 중 reward 최고 선택
- LLM-reward: mock/API 선택 가능, `--intent` 자연어 → reward profile 설계
- 출력: `results/step8/comparison.csv`, `summary.txt`, `llm_profile.json`
- 검증 테스트 9개 (all pass)

**검증된 핵심 불변식:**
```text
Legacy EDCA:      npca_transitions == 0, harq_tx_success == 0, harq_tx_fail == 0
ARQ-only NPCA:    npca_transitions > 0, harq_tx_success == 0
HARQ-only:        npca_transitions == 0, (harq_tx_success + harq_tx_fail) > 0
Fixed-CW NPCA-HARQ: npca_transitions > 0, (harq_tx_success + harq_tx_fail) > 0
Adaptive-CW:      avg_npca_qsrc != None (adaptive history 기록됨)
grid_best:        반환 profile_name ∈ INTENT_PROFILES 키, 실제 최고 reward 검증됨
LLM-reward:       validate_reward_profile() → True (5개 intent 전부)
전체 7 baselines: aggregate metrics 9개 필수 키 모두 존재, reward는 valid float
```

**테스트 파라미터 (SNR=14dB → HARQ 활성화 조건):**
```python
slots=300, stas=2, obss_rate=0.2, obss_min=20, obss_max=60
snr=14.0 dB  # MCS3 경계 → ~50% PHY 성공 → HARQ combining 활성
seed=42, enable_trace=False
```
