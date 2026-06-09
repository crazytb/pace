# Figure 14: Per-STA PPDU-Aware Threshold — 이질적 PPDU 환경에서의 Adaptive qsrc

**연구 질문 (RQ14)**: STA가 자신의 PPDU 길이와 OBSS duration을 알고 있을 때,
`npca_threshold = own_ppdu`로 설정하면 waste_rate 신호의 모호성이 제거되고
adaptive qsrc 알고리즘의 성능이 향상되는가?

**핵심 문제**:
- `threshold = 0`: 자신의 PPDU가 남은 OBSS에 안 들어가도 경쟁 참여
  → TX 불가 방문도 waste로 카운트 → col_rate/waste_rate 신호 오염
  → adaptive가 qsrc를 잘못 조정 (Case B를 Case A로 오인)
- `threshold = own_ppdu`: TX 가능한 방문에만 참여
  → waste_rate = 순수하게 N 과다 신호 → adaptive가 정확하게 동작

**스크립트**: `harq_sim/run_step9_fig14.py`

**출력**: `manuscript/figure/fig14_ppdu_aware_threshold.{eps,png,pdf}`

---

## 실험 파라미터

### 공통 설정

| 항목 | 값 | 비고 |
|---|---|---|
| `num_slots` | 50,000 | |
| OBSS duration | `U(20, 100)` | 짧은 창 위주 — W_eff 효과 극대화 |
| OBSS occupancy | 50% | |
| `harq_horizon` | 200 | |
| `snr_db_mean` | 20.0 | PHY 실패 최소화 — qsrc 효과만 측정 |
| `snr_db_std` | 0.0 | |
| `harq_enabled` | True | |
| Seeds | [42, 123, 456] | |

> OBSS_MAX = 100 선택 이유: 평균 W_eff ≈ 40슬롯. PPDU=40 STA는 절반의 방문에서
> TX 불가 → threshold 유무의 차이가 가장 크게 나타남.

### Sweep 변수

```python
NUM_STAS_LIST = [10, 20, 30, 50]

# PPDU 구성 (전체 N을 균등 3분할)
PPDU_CONFIGS = {
    "homo_short":  lambda n: [10] * n,                          # 모두 PPDU=10
    "homo_medium": lambda n: [20] * n,                          # 모두 PPDU=20 (baseline)
    "homo_long":   lambda n: [40] * n,                          # 모두 PPDU=40
    "hetero":      lambda n: [10]*(n//3) + [20]*(n//3) + [40]*(n - 2*(n//3)),  # 혼합
}
```

---

## 비교 대상 (8가지 조건)

| 기법명 | threshold | adaptive_cw | 비고 |
|---|---|---|---|
| `fixed_q0_t0`     | 0        | False | 기준선 |
| `fixed_q0_tppdu`  | own_ppdu | False | threshold만 개선 |
| `fixed_q1_t0`     | 0        | False | |
| `fixed_q1_tppdu`  | own_ppdu | False | |
| `adaptive_t0`     | 0        | True  | 현재 adaptive |
| `adaptive_tppdu`  | own_ppdu | True  | **제안 기법** |
| `oracle_t0`       | 0        | False | qsrc* 알고 있음 |
| `oracle_tppdu`    | own_ppdu | False | qsrc* + 제안 threshold |

Oracle qsrc는 PPDU config × N별로 시뮬레이션 사전 탐색으로 결정.

---

## 구현 주의사항

### STA별 PPDU duration 지원

현재 `build_and_run()`은 모든 STA에 동일한 `ppdu_duration`을 사용한다.
Fig 14를 위해 **STA별 ppdu_duration 리스트**를 받을 수 있도록 확장 필요:

```python
# run_step8.py build_and_run() 확장 방안
# 옵션 A: ppdu_duration이 list이면 STA별로 다르게 설정
if isinstance(ppdu_duration, list):
    # ppdu_duration[i] → stas[i].ppdu_duration
else:
    # 기존 동작 (모두 동일)

# 옵션 B: 별도 헬퍼 함수 작성 (build_and_run 수정 최소화)
def build_and_run_hetero(ppdu_list, threshold_mode, ...):
    stas = [STA(..., ppdu_duration=ppdu_list[i],
                npca_min_duration_threshold=(ppdu_list[i] if threshold_mode=="own" else 0))
            for i in range(n)]
```

### threshold 계산

```python
# threshold=own_ppdu: STA 생성 시 자신의 PPDU 길이를 threshold로 설정
STA(npca_min_duration_threshold = own_ppdu_duration)

# threshold=0: 기존 동작 (항상 전환)
STA(npca_min_duration_threshold = 0)
```

---

## 측정 지표

| 지표 | 설명 |
|---|---|
| `aggregate_throughput` | 전체 전달 패킷 수 |
| `per_group_throughput` | PPDU 그룹별 전달 패킷 수 (short/medium/long) |
| `waste_rate` | TX 없이 끝난 NPCA 방문 비율 |
| `col_rate` | NPCA TX 중 충돌 비율 |
| `mean_qsrc` | adaptive 기법의 평균 사용 qsrc |
| `npca_participation_rate` | threshold 조건 통과 → 실제 경쟁 참여한 방문 비율 |

`npca_participation_rate` = (threshold 통과 방문 수) / (OBSS 감지 수)  
→ threshold=own_ppdu에서 PPDU가 길수록 이 값이 낮아짐 (정상 동작 확인용)

---

## Figure 구성

```
Figure 14: 3-panel

Panel (a): Aggregate Throughput vs N — hetero PPDU 환경
           x축: N_stas ∈ {10, 20, 30, 50}
           lines: fixed_q0_t0 / fixed_q0_tppdu / adaptive_t0 / adaptive_tppdu / oracle_tppdu
           핵심: adaptive_tppdu가 adaptive_t0 대비 개선, oracle_tppdu에 근접

Panel (b): Throughput by PPDU group vs N — hetero PPDU, adaptive 비교
           3 sub-lines per method: short(PPDU=10) / medium(PPDU=20) / long(PPDU=40) STA
           비교: adaptive_t0 vs adaptive_tppdu
           핵심: long-PPDU STA의 throughput이 threshold=ppdu에서 개선 (공정성 향상)

Panel (c): mean_qsrc vs N — adaptive_t0 vs adaptive_tppdu (hetero 환경)
           핵심: threshold=ppdu → waste_rate 신호 순수 → qsrc가 oracle에 가깝게 수렴
```

---

## 총 실험 횟수

```
4 PPDU configs × 4 N_stas × 8 방법 × 3 seeds = 384 runs
(oracle 방법은 사전 qsrc 탐색 포함)
```

---

## 예상 결과

### Panel (a)
- `adaptive_tppdu` > `adaptive_t0`: threshold 개선으로 신호 순수화 → qsrc 최적 수렴
- 이득은 hetero > homo_long > homo_medium 순 (신호 오염이 클수록 개선폭 큼)
- N=50에서 차이 가장 뚜렷 (N 과다 + W_eff 짧은 방문 비중 높음)

### Panel (b)
- `adaptive_t0`: long-PPDU STA는 waste rate 높아 qsrc 과도 상승 → short-STA도 피해
- `adaptive_tppdu`: long-PPDU STA가 불참 선언 → N_effective 정확히 반영
  → short STA는 낮은 qsrc 유지 가능 → short-STA throughput 향상
- **공정성 역설**: threshold=ppdu는 long-PPDU STA의 참여 기회를 줄이지만
  전체 효율은 향상 (short-STA 기회 증가가 보상)

### Panel (c)
- `adaptive_t0`: hetero 환경에서 qsrc 진동 또는 과추정 (Case B 오인)
- `adaptive_tppdu`: N_effective 기반 깔끔한 수렴 → oracle_tppdu에 근접

---

## 출력 파일

```
manuscript/figure/
  fig14_ppdu_aware_threshold.eps / .png / .pdf

results/step9/fig14/
  data.csv    ← (ppdu_config, num_stas, method, threshold_mode,
                  seed, aggregate_throughput, waste_rate, col_rate,
                  mean_qsrc, per_group_throughput_short/medium/long)
  fig14_ppdu_aware_threshold_preview.png
```

---

## 논문 내 위치

```
§ Extension 3 : Fig 14 (per-STA PPDU-aware threshold)
  "When STAs can measure OBSS duration from the preamble and know their
   own PPDU length, setting npca_threshold = own_ppdu eliminates the
   ambiguity in the waste_rate signal. Under heterogeneous PPDU lengths,
   this per-STA adaptive threshold reduces qsrc overestimation by XX%
   and improves aggregate throughput by YY% at N=50, confirming that
   the adaptive qsrc framework generalizes beyond uniform PPDU assumptions."
```

---

## 실험 결과 (results/step9/fig14/, hetero PPDU=10/20/40)

### Aggregate Throughput vs N (hetero PPDU config)

| N | fixed_q0_t0 | fixed_q0_tppdu | adaptive_t0 | adaptive_tppdu | oracle_tppdu |
|---|-------------|----------------|-------------|----------------|--------------|
| 10 | 1249 | 1287 (+3.0%) | 1233 | **1317 (+5.4%)** | 1287 |
| 20 | 1173 | 1214 (+3.5%) | 1230 | 1214 (+3.5%) | 1234 (+5.2%) |
| 30 | 1193 | 1271 (+6.5%) | 1204 | 1240 (+4.0%) | 1294 (+8.5%) |
| 50 | 1059 | 1116 (+5.4%) | 1061 | 1111 (+4.9%) | **1194 (+12.7%)** |

### 핵심 발견

**PPDU-aware threshold의 일관된 이득**: 전 N에서 +3~7% (fixed_q0_tppdu vs _t0).

**adaptive_tppdu mean_qsrc 과소적응**: qsrc≈0.02~0.06 (oracle=1~2). 이질 PPDU 환경에서 waste_rate 신호가 흐릿해 adaptive가 qsrc*를 충분히 올리지 못함. threshold 효과 > adaptive qsrc 효과.

**논문 메시지**: "Per-STA PPDU-aware threshold eliminates futile NPCA transitions (W_eff < ppdu_dur), providing consistent +3~7% throughput gain in heterogeneous PPDU environments."

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-01 | 초안 작성 — per-STA PPDU-aware threshold 실험 설계 |
| 2026-06-03 | 실험 결과 추가 — thr=ppdu +3~7% 이득, adaptive 과소적응 확인 |
