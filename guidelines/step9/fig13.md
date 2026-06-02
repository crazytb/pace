# Figure 13: Frame Delivery Delay — qsrc × HARQ 분석

**연구 질문 (RQ13)**: qsrc 선택(fixed vs adaptive)과 HARQ on/off가 프레임 전달 지연에
어떤 영향을 미치는가? Fig 2의 NPCA-only 지연 분석을 qsrc × HARQ 차원으로 확장하라.

**Fig 2와의 차이점**:

| 측면 | Fig 2 (구) | Fig 13 (신) |
|---|---|---|
| 비교 대상 | Legacy EDCA vs ARQ-only NPCA | Fixed qsrc={0,1,2,3} × HARQ on/off + Adaptive |
| HARQ 분석 | 없음 | HARQ 유/무 × qsrc 교차 분석 |
| qsrc 영향 | 없음 | qsrc-delay 비단조 관계 규명 |
| 지연 해석 | primary CW 증가 → NPCA 우회 | 재전송 횟수(충돌+PHY) vs 백오프 대기 tradeoff |

**스크립트**: `harq_sim/run_step9_fig13.py`

**출력**: `manuscript/figure/fig13_delay_qsrc_harq.{eps,png,pdf}`

---

## 지연 정의

```
delay(packet) = delivery_slot − pkt.arrival_time
```

- `pkt.arrival_time`: STA가 해당 패킷을 처음 큐에 등록한 슬롯 (이전 패킷 전달 직후)
  - lazy generation 모델에서 arrival_time ≈ 첫 TX 시도 직전 준비 완료 시점
- `delivery_slot`: 패킷이 최종 성공 수신된 슬롯
- 재전송(ARQ/HARQ retx)은 별도 패킷으로 카운트하지 않음 — 원본 패킷 단위로만 측정
- 시뮬레이터 기존 `_delivered_delays` 리스트 그대로 활용 (`mean_access_delay`, `p95_access_delay`)

**지연 구성 요소:**
```
delay = backoff_wait          (첫 TX 기회 대기)
      + Σ(TX_duration_k)      (전송 시간 × 시도 횟수)
      + Σ(inter_attempt_wait) (충돌/PHY실패 후 재대기: 재백오프 또는 OBSS 종료 대기)
```

HARQ의 효과: 동일 SNR에서 combining으로 p_success 향상 → 평균 TX 시도 횟수 감소
→ Σ(TX_duration) + Σ(inter_attempt_wait) 모두 감소

---

## 실험 파라미터

### 공통 설정

| 항목 | 값 | 비고 |
|---|---|---|
| `num_slots` | 50,000 | Fig 3/4와 동일 |
| `ppdu_duration` | 20 슬롯 | |
| `harq_horizon` | 200 슬롯 | HARQ 버퍼 유효기간 |
| `npca_threshold` | 0 | |
| `obss_max` | 500 슬롯 | Fig 3/4와 동일 |
| OBSS 점유율 | 50% | |
| `snr_db_mean` | **14.0 dB** | PHY 실패율 50% (MCS3 임계) → HARQ 효과 극대화 |
| `snr_db_std` | **2.0 dB** | Gaussian SNR 변동 → 현실적 채널 모델 |
| Seeds | [42, 123, 456] | |

> **SNR 선택 근거**: snr=14dB는 MCS3 임계값(14dB)에 정확히 위치 →
> p_success ≈ 0.5 (첫 TX 성공 50%). HARQ combining으로 2회 합산 시
> SNR_eff ≈ 17dB → p_success ≈ 0.95. 이 조건에서 HARQ 이득이 가장 크게 나타남.
> Fig 3/4의 snr=20dB 대비 PHY 실패가 많아 HARQ 영향이 지연 분석에서 가시적.

### Sweep 변수

```
NUM_STAS_LIST = [5, 10, 20, 30, 50]
```

---

## 비교 대상 (10가지 조건)

| 기법명 | npca_enabled | harq_enabled | adaptive_cw | npca_qsrc |
|---|---|---|---|---|
| `fixed_q0_arq`  | True | False | False | 0 |
| `fixed_q1_arq`  | True | False | False | 1 |
| `fixed_q2_arq`  | True | False | False | 2 |
| `fixed_q3_arq`  | True | False | False | 3 |
| `adaptive_arq`  | True | False | True  | 0 (초기값) |
| `fixed_q0_harq` | True | True  | False | 0 |
| `fixed_q1_harq` | True | True  | False | 1 |
| `fixed_q2_harq` | True | True  | False | 2 |
| `fixed_q3_harq` | True | True  | False | 3 |
| `adaptive_harq` | True | True  | True  | 0 (초기값) |

---

## 측정 지표

| 지표 | 설명 | 출처 |
|---|---|---|
| `mean_access_delay` | 평균 전달 지연 (슬롯) | `simulator.py` 기존 집계 |
| `p95_access_delay` | 95th percentile 지연 | `simulator.py` 기존 집계 |
| `p99_access_delay` | 99th percentile 지연 | `simulator.py` 기존 집계 |
| `mean_retx_count` | 패킷당 평균 TX 시도 횟수 | 계산: `total_tx / total_delivered` |
| `aggregate_throughput` | 전달 패킷 수 (참고) | 기존 |
| `collision_probability_npca` | NPCA 충돌 확률 (지연 원인 분리) | 기존 |

`mean_retx_count` 계산:
```python
total_tx = (metrics["aggregate"]["collision_probability"] 계산에 사용된 total_tx_all)
# 또는 직접:
total_tx = sum of (npca_tx_success + npca_tx_fail + primary_tx_success + primary_tx_fail)
mean_retx = total_tx / total_delivered
```

---

## Figure 구성

```
Figure 13: 3-panel

Panel (a): mean_access_delay vs N_stas — HARQ=off
           x축: N_stas ∈ {5, 10, 20, 30, 50}
           y축: mean_access_delay (슬롯)
           lines: fixed_q0 / fixed_q1 / fixed_q2 / fixed_q3 / adaptive (5개)
           스타일: fixed = dashed, adaptive = solid thick
           핵심 메시지: qsrc-delay 비단조 관계 (낮은 N에서 high-qsrc 불리,
                        높은 N에서 low-qsrc 불리); adaptive가 envelope 추적

Panel (b): mean_access_delay vs N_stas — HARQ=on
           Panel (a)와 동일 구조 (같은 y축 스케일)
           핵심 메시지: HARQ가 전 조건에서 지연 감소; 감소폭은 qsrc에 따라 다름

Panel (c): HARQ 지연 감소율 (%) vs N_stas
           Δdelay(%) = (ARQ_delay − HARQ_delay) / ARQ_delay × 100
           lines: adaptive / fixed_q0 / fixed_q1 / fixed_q2 / fixed_q3 (5개)
           핵심 메시지: HARQ 이득은 PHY 실패 주도 구간(낮은 N, 낮은 충돌)에서 극대;
                        충돌 주도 구간(높은 N + low qsrc)에서 HARQ 이득 감소
                        (충돌 손실은 combining으로 구제 불가)
```

---

## 총 실험 횟수

```
10 조건 × 5 N_stas × 3 seeds = 150 runs
```

---

## 예상 결과

### Panel (a) — HARQ=off 지연 패턴

**비단조(non-monotone) qsrc-delay 관계**:

| N_stas | 지배 현상 | qsrc 효과 |
|---|---|---|
| 5 | PHY 실패 주도 | qsrc=0 유리 (백오프 짧음, 충돌 적음) |
| 20 | 충돌+PHY 혼합 | qsrc=1~2 최적 (충돌 감소 vs 백오프 증가 균형) |
| 50 | 충돌 주도 | qsrc=0 사용 시 재전송 폭발 → delay 급증 |

예상 교차: qsrc=0 curve는 N=5에서 최저, N=50에서 최고  
qsrc=2 curve는 N=5에서 다소 높으나, N=30~50에서 qsrc=0보다 낮음

### Panel (b) — HARQ=on 지연 패턴

Panel (a) 대비 전반적으로 아래로 이동 (10~25% 감소 예상).  
qsrc-delay 비단조 형태는 동일하게 유지 — qsrc 영향과 HARQ 영향이 독립적으로 작용.

### Panel (c) — HARQ 지연 감소율

| 조건 | N=5 예상 Δ% | N=50 예상 Δ% |
|---|---|---|
| adaptive | ~20% | ~10% |
| fixed_q0 | ~20% | ~5% (충돌 多 → HARQ 효과 약함) |
| fixed_q2 | ~20% | ~15% (충돌 억제로 PHY 실패 비중 유지) |

**핵심 예측**: high-N + low-qsrc 조건에서는 HARQ 이득이 작음
→ 충돌로 인한 지연은 HARQ로 해결 불가; qsrc 최적화가 우선

---

## qsrc × HARQ 상호작용 이론

### 지연 분해 모델

```
E[delay] ≈ E[backoff_wait]      ← qsrc에 의존 (CW ∝ 2^qsrc)
          + E[retx_count] × (ppdu_duration + E[inter_attempt_gap])
                                ← retx_count = f(p_col, p_phy_fail)
```

- `p_col`: 충돌 확률 — qsrc 증가 → CW 증가 → p_col 감소
- `p_phy_fail`: PHY 실패 확률 — SNR 고정, HARQ=off이면 상수; HARQ=on이면 combining으로 감소
- `inter_attempt_gap`: 충돌/PHY실패 후 재대기 슬롯 수 (재백오프 + OBSS 대기)

HARQ 이득이 의미있는 조건: `p_phy_fail`이 총 재전송 횟수의 주요 원인일 때.
→ 충돌이 적은 상황 (qsrc 높거나 N 낮음)에서 HARQ 이득이 가장 큼.

### qsrc* for delay vs qsrc* for throughput

throughput을 최대화하는 qsrc*와 delay를 최소화하는 qsrc*가 일치하는가?

- 원칙적으로 동일해야 함: 두 지표 모두 MAC 효율을 최대화하는 CW 선택에 의존
- 단, delay는 전달 **성공한 패킷**의 평균을 측정하므로, 재전송이 많더라도 최종 성공하는 패킷들의 대기 시간이 반영됨
- 가설: qsrc*(delay) ≈ qsrc*(throughput) — 논문의 이론과 실험이 일치함을 Fig 13으로 확인

---

## CDF 보조 분석 (선택적 panel 또는 부록)

N=20 (중간 부하)에서 10가지 조건의 지연 CDF 비교:
- x축: access delay (슬롯)
- y축: CDF
- `p95_access_delay` 지점에 수직선 표시

핵심 관찰 포인트:
- HARQ=on이 HARQ=off의 CDF를 왼쪽으로 shift (mean 감소) + tail 압축 (p99 감소)
- fixed_q0 vs fixed_q2 tail 비교 (high-N에서 q0의 긴 tail)

---

## 출력 파일

```
manuscript/figure/
  fig13_delay_qsrc_harq.eps / .png / .pdf

results/step9/fig13/
  data.csv    ← (num_stas, method, harq_enabled, adaptive_cw, npca_qsrc, seed,
                  mean_access_delay, p95_access_delay, p99_access_delay,
                  mean_retx_count, aggregate_throughput, collision_probability_npca)
  fig13_delay_qsrc_harq_preview.png
```

---

## 논문 내 위치

```
§ Extension  : Fig 13 (delay 분석)
  "qsrc* minimizes not only throughput loss but also frame delivery delay.
   HARQ reduces delay primarily in the PHY-failure-dominated regime (small N or
   high qsrc), while collision-dominated regimes (large N, small qsrc) require
   qsrc optimization as the primary lever. Adaptive qsrc tracks qsrc*(N) and
   achieves near-optimal delay across all contention levels."
```

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-01 | 초안 작성 — Fig 2 대체/보완 실험으로 qsrc × HARQ 지연 분석 설계 |
