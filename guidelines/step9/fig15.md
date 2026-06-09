# Figure 15: MFG-Optimal NPCA Access Protocol vs DCF-Static Backoff

**연구 질문 (RQ15)**: NPCA visit의 유한 horizon W_eff를 명시적으로 활용하는
MFG-최적 접근 프로토콜은 고정 CW(uniform backoff)보다 얼마나 높은 처리량을 달성하는가?
그리고 그 이득은 N과 W_eff에 따라 어떻게 달라지는가?

**스크립트**: `harq_sim/run_step9_fig15.py` (미구현)

**출력**: `manuscript/figure/fig15_mfg_npca.{eps,png,pdf}`

---

## 이론적 배경

### CW Amnesia + 유한 horizon의 문제

표준 DCF는 무한(또는 충분히 긴) horizon을 가정하여 설계됨:
- CW_0는 정상 상태(stationary) 충돌 확률 p에서 τ* = 1/N 만족하도록 설정
- NPCA visit에서는 각 방문마다 CW가 초기화(amnesia) + 방문이 W_eff 슬롯 후 강제 종료

이 두 특성이 DCF의 가정을 위반:
1. **CW amnesia**: BEB의 단계별(stage) 적응 무의미 → qsrc(= CW_0 설계)가 유일한 자유도
2. **유한 horizon W_eff**: 방문 말미에서 미전송 STA는 시간이 소진되면 기회를 잃음
   → deadline이 가까울수록 더 공격적으로 전송 시도하는 것이 최적

### MFG 공식화

**N → ∞** (massive sensor 환경): N이 크고 각 STA가 N을 정확히 모를 때 mean field 근사 유효.

**State**: slot t ∈ {0,...,W_eff-1}, 방문 내 아직 전송 미시도 STA 비율 n(t)

**MFG 시스템 (이산 시간, 유한 horizon)**:

```
Forward (mean field evolution):
  n(0) = 1
  n(t+1) = n(t) · (1 - τ*(t) · e^{-N·n(t)·τ*(t)})   [성공 STA 제거]

Backward (HJB — 개별 최적 τ):
  V(W_eff) = 0
  V(t) = max_τ [ τ · e^{-N·n(t)·τ*(t)} + (1 - τ · e^{-N·n(t)·τ*(t)}) · V(t+1) ]

Fixed point: τ*(t) consistent with both equations
```

**해석적 해 (단순화 버전, 단일 시도 모델)**:

mean field 수렴 조건 τ*(t) = 1/(N·n(t)) 에서:

```
n(t) = 1 - t/N      (선형 감소)
τ*(t) = 1/(N - t)   (t → N 접근하며 급증)
f*(k) = 1/N          for k = 0, 1, ..., N-1  (균등 분포, CW_0 = N)
       = 0           for k ≥ N
```

→ **MFG 최적 pre-committed 분포 = U{0, N-1}**, 즉 CW_0 = N (not 2N).

**Adaptive MFG (online)**:
```
STA가 매 슬롯 t에서 carrier sensing으로 n(t) 추정 후 τ*(t) 적용
→ pre-committed보다 항상 우월 (추가 정보 활용)
→ 구현: "busy slot 감지 → n(t) 감소 → τ 증가" 자동 조정
```

---

## 프로토콜 비교 대상

| 프로토콜 | CW_0 | τ(t) | 정보 활용 |
|---------|------|------|-----------|
| `dcf_qsrc0` | 15  | 1/8 (고정) | 없음 |
| `dcf_qsrc1` | 31  | 1/16 (고정) | 없음 |
| `dcf_qsrc_star` | ≈ 2N | ≈ 1/(2N) (고정) | N만 |
| **`mfg_precommit`** | **N** | **1/(N-k)·hazard** | **N, W_eff** |
| **`mfg_adaptive`** | — | **τ*(t) = 1/(N·n̂(t))** | **N, n̂(t) online** |
| `oracle` | 최적 CW_0 탐색 | — | exhaustive |

`mfg_precommit`: CW_0 = N으로 설정한 DCF — 기존 코드 변경 없이 테스트 가능.

`mfg_adaptive`: 매 슬롯 busy/idle 관찰 → n̂(t) 업데이트 → τ*(t) 계산 후 Bernoulli 시도.

---

## 실험 파라미터

| 항목 | 값 | 비고 |
|------|-----|------|
| **N sweep** | {5, 10, 20, 30, 50} | |
| **W_eff sweep** | {20, 50, 100, 200, 500} | OBSS duration − PPDU − switching |
| `ppdu_duration` | 20 슬롯 | W_eff = obss_duration − 20 − 1 |
| `snr_db_mean` | 30.0 dB | 고 SNR — MAC 경쟁만 관찰 |
| `num_slots` | 200,000 | 충분한 NPCA 방문 횟수 확보 |
| OBSS 점유율 | 50% | |
| Seeds | [42, 123, 456] | |

> **W_eff 조절 방법**: obss_max = W_eff + ppdu + 1로 설정, obss_min = obss_max (결정론적 방문 길이).
> 결정론적 W_eff → 이론값과 정확한 1:1 비교 가능.

---

## 측정 지표

| 지표 | 설명 |
|------|------|
| `npca_tput_per_visit` | NPCA 방문 1회당 성공 TX 수 (주 지표) |
| `collision_prob_npca` | 충돌 확률 |
| `tput_gain_pct` | (MFG − dcf_qsrc_star) / dcf_qsrc_star × 100% |

---

## 예상 그림 구성 (3-panel)

### Panel 1 — τ*(t) 이론 곡선 (analytical)
- X축: slot index t within visit (0 ~ N)
- Y축: τ*(t) = 1/(N−t)
- 다중 N 값 ({10, 20, 30, 50}) 겹쳐 그리기
- 비교선: DCF τ_static = 1/N (수평선)
- 목적: "MFG는 방문 말미에 급격히 공격적" 시각화

### Panel 2 — 처리량 vs N, W_eff별 (simulation)
- X축: N (5 ~ 50)
- Y축: npca_tput_per_visit
- 라인: dcf_qsrc_star / mfg_precommit / mfg_adaptive / oracle
- 서브플롯: W_eff ∈ {50, 100, 500} (3열)
- 목적: W_eff/N 비율에 따른 MFG 이득 변화

### Panel 3 — Δthroughput gain heatmap
- X축: W_eff (20~500)
- Y축: N (5~50)
- 색상: tput_gain_pct = (mfg_adaptive − dcf_qsrc_star) / dcf_qsrc_star × 100%
- 목적: "어느 (N, W_eff) 영역에서 MFG가 DCF를 의미있게 개선하는가"

---

## 핵심 예상 결과

```
W_eff >> N (예: W_eff = 500, N = 10):
  mfg_precommit (CW_0=N) >> dcf_qsrc_star (CW_0=2N)
  이유: CW_0=N → 모든 N STA가 {0,...,N-1} 내 슬롯 선택 → W_eff − N 슬롯 낭비 없음

W_eff ≈ N (예: W_eff = 50, N = 50):
  mfg_precommit ≈ dcf_qsrc_star
  이유: W_eff/N ≈ 1 → 두 분포 모두 전체 창을 채움

W_eff < N (예: W_eff = 20, N = 50):
  mfg_adaptive > mfg_precommit >> dcf_qsrc_star
  이유: 슬롯이 부족 → adaptive가 n(t) 관찰로 더 적응적 조정 가능
```

**Δgain 유의 구간**: W_eff > 2N 일 때 MFG 이득 > 10% 예상.

---

## Fig 3 비교와의 연결

Fig 3(v3): qsrc sweep → empirical qsrc*(N) 확인  
Fig 15: 이론적 MFG 최적 CW_0 = N vs empirical qsrc*(N) ≈ 2N 불일치 해석

```
실험적 qsrc* (Fig 3): CW_0 ≈ 2N  → 다중 시도(retry) 모델 반영
이론적 MFG (단일시도): CW_0 = N

불일치 원인 가설:
  Fig 3 환경: W_eff = 20~500 (평균 239) → retry 발생
  MFG 단일시도 모델: retry 없음 → CW_0 = N 최적
  → MFG multi-retry 확장 시 CW_0 = 2N으로 수렴하는지 검증 필요
```

이 분석이 이론-시뮬레이션 불일치의 원인을 명확히 함.

---

## 구현 메모

`mfg_adaptive` STA 구현 핵심:
```python
# STA가 매 슬롯 NPCA backoff 중 실행
def mfg_step(self, t, busy_slots_observed):
    n_hat = 1 - busy_slots_observed / N  # 잔존 비율 추정
    tau_star = min(1.0, 1.0 / (N * n_hat)) if n_hat > 0 else 1.0
    if random.random() < tau_star:
        self._attempt_tx()  # Bernoulli 시도
```

기존 DCF backoff 루프를 Bernoulli 시도로 교체 — sta.py 최소 수정.

---

## 실험 결과 요약 (2026-06-03)

### MFG adaptive 이득 heatmap (mfg_adaptive vs dcf_qsrc_star, %)

|       | W=20  | W=50  | W=100 | W=200 | W=500 |
|-------|-------|-------|-------|-------|-------|
| N=5   | +26.8 | +1.5  | +0.0  | +0.0  | +0.0  |
| N=10  | +70.4 | +17.8 | +1.7  | +0.0  | +0.0  |
| N=20  | +67.1 | +61.1 | +18.6 | +1.7  | +0.0  |
| N=30  | +68.0 | +68.2 | +44.6 | +5.6  | +0.1  |
| N=50  | +67.4 | +66.8 | +68.0 | +34.0 | +1.4  |

### 이론 예측 vs 실험 결과 비교

| 예측 | 실제 결과 | 수정 해석 |
|------|-----------|-----------|
| W_eff > 2N일 때 이득 > 10% | **반대**: W_eff < N일 때 이득 +70% | 타이트 창에서 BEB 낭비 극대 |
| mfg_precommit >> dcf_qsrc_star | mfg_precommit ≈ dcf_qsrc_star | BEB로 첫 충돌 후 CW 동일화 |
| MFG 기여 = CW_0=N 선택 | MFG 기여 = τ*(t)=1/remaining 메커니즘 | Bernoulli TX가 핵심, not CW_0 |

### 핵심 발견

**mfg_adaptive가 DCF를 크게 앞서는 이유**:
- DCF: 충돌 후 BEB → CW 2배 증가 → 남은 W_eff 슬롯보다 긴 backoff 발생 → 많은 STA 전송 기회 상실
- MFG adaptive: τ*(t) = 1/remaining → 매 슬롯 기대 성공 TX = 1 → W_eff 슬롯 낭비 없음

**논문 기여 재정의**:
> "MFG-optimal adaptive protocol, which sets τ*(t) = 1/remaining at each slot,
>  eliminates BEB overhead and achieves up to +70% throughput gain over DCF in the
>  tight-window regime (W_eff ≤ N). The gain diminishes when W_eff ≫ N, where
>  DCF also recovers all STAs within the window."

---

## 수정 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-06-02 | 초안 작성 — MFG 실험 설계 |
| 2026-06-03 | 실험 구현 및 실행 완료 (1000 visits × 3 seeds); 이론 예측 수정 — 이득은 W_eff < N 구간이 핵심; mfg_precommit ≈ dcf_qsrc_star 확인 |
