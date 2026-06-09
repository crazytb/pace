# MFG-Optimal NPCA Access Protocol — 알고리즘 상세 설명

**대상 독자**: NPCA 및 MFG 비전공자  
**관련 실험**: `guidelines/step9/fig15.md`, `harq_sim/run_step9_fig15.py`

---

## 1. 문제 설정

### NPCA Visit란?

NPCA(Non-Primary Channel Access)에서 STA는 primary 채널에 OBSS(타 BSS 신호)가 감지되면
NPCA 채널로 전환해 데이터를 전송한다.
이 "전환 후 전송 가능한 시간"을 **NPCA visit**이라 부른다.

```
Primary channel:  ──[OBSS 점유 중]──────────────────────────────────────────►
                         │                                               │
                         ↓ OBSS 감지                                    ↓ OBSS 종료
NPCA channel:          [W_eff 슬롯 동안 N개 STA가 경쟁]
```

**핵심 파라미터**:
- `N` : NPCA 채널에서 경쟁하는 STA 수
- `W_eff` : NPCA visit 내 사용 가능한 슬롯 수 (OBSS 잔여 시간 − switching delay − PPDU duration)
- `t` : visit 시작 이후 경과 슬롯 인덱스 (0부터 시작)
- `remaining(t)` : 슬롯 t 시점에서 아직 전송 성공하지 못한 STA 수

---

## 2. 표준 DCF의 한계 (CW Amnesia + Finite Horizon)

### 표준 DCF 동작

```
STA가 NPCA 채널 진입 시:
  1. CW_0 설정 (보통 15 또는 31)
  2. backoff = U{0, CW_0 - 1} 에서 랜덤 추출
  3. 매 idle 슬롯마다 backoff -= 1
  4. backoff == 0 → 전송 시도
  5. 충돌 시 → BEB: CW *= 2, backoff 재추첨
  6. W_eff 슬롯 소진 → visit 강제 종료
```

### 두 가지 구조적 문제

**문제 1 — CW Amnesia (CW 기억 상실)**:
표준 BEB는 visit 간 CW 상태를 유지한다고 설계됨.
NPCA visit에서는 매 방문마다 CW가 `CW_0`으로 초기화(reset)됨.
→ BEB의 멀티-스테이지 적응 효과가 무의미해짐.
→ `CW_0` 하나만이 유일한 설계 자유도.

**문제 2 — Finite Horizon**:
```
예: N=50, W_eff=20, CW_0=100 (= 2×50)

backoff ~ U{0, 99} → 대부분 STA의 backoff > 20
→ W_eff=20 슬롯이 지나기 전에 backoff가 0에 도달하지 못함
→ 전송 기회 없이 visit 종료

실제 성공 수 ≈ 4.7 / visit  (이론 상한 20 대비 23% 효율)
```

BEB로 CW가 2배 증가하면 re-backoff > 잔여 슬롯 → 전송 기회 상실.
`W_eff`가 작을수록 이 낭비가 심화됨.

---

## 3. 세 가지 프로토콜 비교

| 프로토콜 | CW 설정 | τ(t) | 핵심 아이디어 |
|---------|---------|------|--------------|
| `dcf_qsrc_star` | CW_0 = 2N | 고정 ≈ 1/(2N) | 경험적으로 최적인 qsrc 적용 |
| `mfg_precommit` | CW_0 = N  | 고정 ≈ 1/N   | MFG 단일시도 이론값 적용 |
| **`mfg_adaptive`** | 없음 | **τ*(t) = 1/remaining(t)** | 매 슬롯 잔존 STA 수 기반 갱신 |

---

## 4. MFG 이론 배경

### Mean Field Game (MFG) 공식화

N이 충분히 클 때 개별 STA의 영향이 미미하다는 mean field 근사 적용:

```
State:   (t, n(t))  — 슬롯 인덱스, 잔존 STA 비율 n(t) = remaining(t)/N

Forward equation  (population evolution):
  n(0) = 1
  n(t+1) = n(t) · P(성공 TX 없음 | τ*(t), N·n(t))

Backward equation (individual optimality):
  각 STA는 남은 슬롯 내 성공 확률 최대화
  V(t) = max_τ [ τ·P(solo|τ) + (1-τ·P(solo|τ))·V(t+1) ]

Fixed-point condition:
  τ*(t) consistent with both equations
```

### 해석적 해 (단일 시도 근사)

고정점 조건 `E[성공 TX/슬롯] = 1` 에서:

```
N·n(t)·τ*(t) = 1

n(t) = remaining(t)/N 대입:

  τ*(t) = 1 / (N · n(t)) = 1 / remaining(t)
```

**직관**: 매 슬롯 정확히 기대 성공 TX = 1이 되도록 TX 확률 조정.

n(t)의 선형 감소 가정 하에:
```
n(t) = 1 - t/N    →    τ*(t) = 1/(N - t)

t → N 에 가까워질수록 τ*(t) 급증 (deadline urgency)
```

---

## 5. MFG Adaptive 알고리즘

### 의사코드 (Pseudocode)

```
Algorithm: MFG-Adaptive NPCA Access
Input:  N (contending STAs), W_eff (available slots)
Output: success_count (STAs that successfully transmitted)

─────────────────────────────────────────────────────────
INITIALIZATION
  remaining ← N          // 미성공 STA 수
  success   ← 0          // 성공 카운터
  t         ← 0          // 슬롯 인덱스

─────────────────────────────────────────────────────────
MAIN LOOP  (for each slot t = 0, 1, ..., W_eff - 1)

  IF remaining == 0: BREAK    // 모든 STA 성공 → 조기 종료

  STEP 1. Compute TX probability
    τ*(t) ← 1 / remaining

  STEP 2. Each STA independently attempts TX
    For each STA i in {1, ..., remaining}:
      transmit_i ← Bernoulli(τ*(t))    // 독립 시행

    // 동등하게: n_tx ~ Binomial(remaining, τ*(t))

  STEP 3. Determine outcome
    IF n_tx == 1:   // Solo TX → SUCCESS
      success   ← success + 1
      remaining ← remaining - 1

    IF n_tx > 1:    // Collision → 모두 재시도 (remaining 변동 없음)
      (no state change)

    IF n_tx == 0:   // Idle slot
      (no state change)

  t ← t + 1

─────────────────────────────────────────────────────────
RETURN success
```

### 플로우차트

```
┌─────────────────────────────────┐
│  NPCA Visit 시작                 │
│  remaining = N, success = 0      │
│  t = 0                           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  t < W_eff  AND  remaining > 0 ?│
└──────┬──────────────────┬───────┘
       │ YES              │ NO
       ▼                  ▼
┌──────────────┐    ┌─────────────────┐
│ τ = 1/remain │    │  Visit 종료      │
│ 계산         │    │  return success  │
└──────┬───────┘    └─────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  n_tx ~ Binomial(remaining, τ)       │
│  각 STA 독립 Bernoulli(τ) 시행       │
└──────┬───────────────────────────────┘
       │
       ├─── n_tx == 0 ──► [Idle] t += 1
       │
       ├─── n_tx == 1 ──► [Success]
       │                   success += 1
       │                   remaining -= 1
       │                   t += 1
       │
       └─── n_tx  > 1 ──► [Collision]
                           remaining 변동 없음
                           t += 1
                           (재시도는 다음 슬롯 자동)
```

---

## 6. DCF vs MFG Adaptive 슬롯별 비교 예시

**조건**: N=10, W_eff=15

### DCF (CW_0 = 2N = 20)
```
초기화: 각 STA backoff ~ U{0, 19}

Slot  0: backoff = [17,  3,  9, 11,  0,  5, 14,  2, 18,  7]
          STA4만 ready → SUCCESS (STA4 퇴장)  remaining=9

Slot  1: backoff = [16,  2,  8, 10,  -,  4, 13,  1, 17,  6]  (감소)
Slot  2: backoff = [15,  1,  7,  9,  -,  3, 12,  0, 16,  5]
          STA7만 ready → SUCCESS  remaining=8

Slot  3: backoff = [14,  0,  6,  8,  -,  2, 11,  -, 15,  4]
          STA1만 ready → SUCCESS  remaining=7

...

Slot 14: 일부 STA는 backoff이 아직 남아 있어 W_eff 소진 후 실패
최종 success ≈ 8~9 / visit (운에 따라 편차 큼)
```

### MFG Adaptive
```
Slot  0: remaining=10, τ=0.100
          n_tx ~ Bin(10, 0.10) ≈ 1  → SUCCESS  remaining=9

Slot  1: remaining=9,  τ=0.111
          E[n_tx]=1  → SUCCESS  remaining=8

Slot  2: remaining=8,  τ=0.125
          n_tx=2  → Collision  remaining=8 (unchanged)

Slot  3: remaining=8,  τ=0.125  (remaining 그대로, τ 그대로)
          n_tx=1  → SUCCESS  remaining=7

...

매 슬롯 E[success]=1 유지 → W_eff=15 슬롯에서 기대 success ≈ 10
(단, 충돌이 많으면 일부 STA는 W_eff 안에 못 보낼 수도 있음)
```

**핵심 차이**:
- DCF: backoff counter 사전 추첨 → 늦게 뽑힌 STA는 W_eff 내 도달 못함
- MFG: 매 슬롯 remaining 보고 τ 재계산 → 늦어질수록 더 공격적, 낭비 슬롯 없음

---

## 7. 왜 mfg_precommit은 dcf_qsrc_star와 거의 같은가?

`mfg_precommit` = CW_0=N으로 설정한 DCF (BEB 포함).

```
첫 충돌 전:
  mfg_precommit: backoff ~ U{0, N-1}    (CW_0 = N)
  dcf_qsrc_star: backoff ~ U{0, 2N-1}   (CW_0 = 2N)
  → 분포 다름 (mfg가 더 공격적)

첫 충돌 후 BEB:
  mfg_precommit: 새 CW = 2N-1  →  backoff ~ U{0, 2N-1}
  dcf_qsrc_star: 새 CW = 4N-1  →  backoff ~ U{0, 4N-1}
  → 두 번째 충돌 후에는 또 달라지지만...

실제로 W_eff 안에서 첫 충돌이 빠르게 발생 → 이후 BEB가 초기 CW 차이를 덮어씀
→ 실험적으로 두 기법의 성능 차이 < 5%
```

**결론**: MFG의 기여는 `CW_0=N` 선택이 아니라 **τ*(t)=1/remaining 메커니즘**에 있음.
pre-committed 분포(CW_0)는 retry 환경에서 BEB에 의해 무력화됨.

---

## 8. 이론적 이득 구간

### 이득 발생 조건

```
W_eff >> N : DCF도 모든 STA를 처리할 수 있는 충분한 슬롯 보유
             → MFG 이득 ≈ 0%

W_eff ≈ N  : 경계 구간, 이득 증가

W_eff << N : DCF의 BEB로 re-backoff > W_eff → 많은 STA 기회 상실
             → MFG adaptive 이득 최대 +70%
```

### 실험 결과 요약 heatmap (mfg_adaptive vs dcf_qsrc_star, %)

```
         W_eff=20   W_eff=50   W_eff=100  W_eff=200  W_eff=500
N=5       +26.8%     +1.5%      +0.0%      +0.0%      +0.0%
N=10      +70.4%    +17.8%      +1.7%      +0.0%      +0.0%
N=20      +67.1%    +61.1%     +18.6%      +1.7%      +0.0%
N=30      +68.0%    +68.2%     +44.6%      +5.6%      +0.1%
N=50      +67.4%    +66.8%     +68.0%     +34.0%      +1.4%

[이득 큰 구간: W_eff/N ≤ 4~5]
```

gain ≥ 10% 경계: **W_eff/N ≤ 4~5**  
gain ≥ 50% 구간: **W_eff ≤ N** (tight-window regime)

---

## 9. 현재 모델의 가정 및 한계

### 가정

| 항목 | 가정 | 비고 |
|------|------|------|
| 네트워크 토폴로지 | 완전 그래프 (모든 STA 상호 청취) | 실제는 adjacency 그래프 |
| PPDU 길이 | 1 슬롯 (단순화) | 실제는 ppdu_duration > 1 |
| STA 수 N | 모든 STA에 공통 알려짐 | 실제는 N 추정 필요 |
| OBSS duration | 결정론적 W_eff | 실제는 랜덤 |
| 채널 품질 | 완벽 (PHY 실패 없음) | MAC 경쟁만 측정 |

### 현실적 확장 방향

**이질적 PPDU + 토폴로지 확장**:

```
완전 그래프 + 동질 PPDU (현재 Fig 15):
  n_eff(t) = remaining(t) → τ*(t) = 1/remaining  ← 완전 분산 가능

임의 토폴로지 + 이질 PPDU:
  n_eff_i(t) = |{j ∈ N_i : ppdu_j ≤ rem(t)}| + 1
  τ_i*(t) = 1/n_eff_i(t)
  → 각 STA마다 다른 τ → Graphical MFG 필요
  → 타인의 ppdu_j 정보 없으면 n_eff_i 계산 불가
```

---

## 10. 구현 요약 (Python)

```python
def mfg_adaptive_single_visit(N: int, W_eff: int, rng) -> int:
    """
    단일 NPCA visit 시뮬레이션.
    Returns: 성공 TX 수
    """
    remaining = N
    success = 0

    for t in range(W_eff):
        if remaining == 0:
            break

        # τ*(t) = 1/remaining  (MFG fixed-point)
        tau = 1.0 / remaining

        # 각 STA 독립 Bernoulli — Binomial으로 집계
        n_tx = rng.binomial(remaining, tau)

        if n_tx == 1:       # Solo TX
            success += 1
            remaining -= 1
        # n_tx > 1: 충돌, remaining 변동 없음
        # n_tx == 0: idle, 변동 없음

    return success
```

**벡터화 버전** (1000 visits 동시):
```python
def mfg_adaptive_batch(N: int, W_eff: int, n_visits: int, rng) -> np.ndarray:
    remaining = np.full(n_visits, N, dtype=np.float64)   # (V,)
    success   = np.zeros(n_visits, dtype=np.int32)

    for t in range(W_eff):
        active = remaining > 0
        tau    = np.where(active, 1.0 / np.maximum(remaining, 1.0), 0.0)
        n_rem  = np.round(remaining).astype(np.int32)
        n_tx   = rng.binomial(n_rem, tau.clip(0, 1))     # (V,)

        solo      = (n_tx == 1) & active
        success  += solo.astype(np.int32)
        remaining = np.where(solo, remaining - 1.0, remaining)

    return success
```

---

## 관련 문서

- [Fig 15 실험 계획](step9/fig15.md) — 실험 파라미터, 결과, 수정 이력
- [Fig 14 PPDU-aware threshold](step9/fig14.md) — n_eff 분리의 전제 조건
- [step9 인덱스](step9_index.md) — 전체 Figure 목록
- 구현 스크립트: `harq_sim/run_step9_fig15.py`
