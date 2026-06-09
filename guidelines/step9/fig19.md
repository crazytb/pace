# Figure 19: DCF-Benchmark Comparison — Throughput / Delay / τ Trajectory

**연구 질문 (RQ19)**:  
IEEE 802.11 DCF + self-exclusion을 benchmark로 삼을 때,  
(1) EMA 기반 adaptive τ (ema_ad_low)와 연속-idle trigger (consec_L2)가  
    throughput과 delay 양 측면에서 DCF를 얼마나 개선하는가?  
(2) open-loop AND는 DCF 대비 얼마나 열위인가?  
(3) τ trajectory를 통해 세 적응 방식의 내부 동작 차이를 시각화할 수 있는가?

**스크립트**: `harq_sim/run_step9_fig19.py`  
**출력**: `manuscript/figure/fig19_dcf_benchmark.{eps,png,pdf}`

---

## 1. 배경 및 동기

Fig 17은 oracle을 기준점(efficiency=1)으로 모든 방법을 비교했다.
그러나 실용적 관점에서는 **현행 IEEE 802.11 DCF**가 가장 자연스러운 baseline이다:
- DCF는 표준 contention 메커니즘으로 널리 배포됨
- `dcf_self_excl`은 DCF에 PPDU-aware self-exclusion만 추가한 최소 수정 버전
- 이를 benchmark로 삼으면 "DCF 대비 얼마나 개선되는가"를 직접 서술 가능

비교 대상을 4종으로 압축:

| 방법 | 특성 | 역할 |
|---|---|---|
| `pnd` | MIMD solo-copy (cc=1.2/ci=1.2, no CD) | **논문 주요 기여** |
| `ema_ad_low` | EMA idle-rate + α↓=0.10 | EMA 계열 최고 성능 대표 |
| `consec_L2` | 연속 2-idle trigger | 단순 counting 계열 대표 |
| `dcf_self_excl` | BEB + self-excl (IEEE 802.11 표준) | 표준 비교 방법 |
| `and` | Phase-based open-loop | Open-loop lower bound |

---

## 2. 시스템 모델

Fig 17과 동일한 NPCA visit 시뮬레이션 (이질 PPDU 확장).

```
W_eff slots 내 N STAs 경쟁 (완전 그래프)
각 visit: ppdu_i ~ PPDU_DIST, W_rem = W_eff

매 contention round:
  viable_i = (W_rem >= ppdu_i) AND (not yet succeeded)
  solo  → success, W_rem -= ppdu_i
  coll  → W_rem -= 1
  idle  → W_rem -= 1
  종료: W_rem < min(ppdu_i for viable) OR all succeeded
```

### Self-Exclusion (전 방법 공통)
```
τ_i = 0 if ppdu_i > W_rem
```

### DCF 상세
```
초기: CW_i = CW0 = N
TX 결정: backoff_i == 0 → TX (결정론적, 비베르누이)
충돌 후: CW_i = min(CW_i × 2, CW_max=1023); 새 backoff draw
성공 후: CW_i = CW0 리셋
idle:    backoff_i -= 1 (단, not viable이면 동결)
```

### AND 상세
```
Phase i: p_i = 1 / 2^i (고정 전송 확률)
Phase i 지속: ⌈2^i × e × i × ln2⌉ slots
i = 1, 2, 3, ... (비적응, 결과 무관)
```

---

## 3. 측정 지표

Delay 제외. Throughput만 측정.

```python
efficiency = successes / oracle_successes   # Fig 17과 동일 지표
```

---

## 4. 실험 파라미터

```python
# Fig 17 v7과 동일 — 데이터 비교 가능성 유지
N_LIST    = [10, 20, 30, 50]
WEFF_LIST = [20, 50, 100, 200]
PPDU_DIST = {
    "homo":    lambda rng, N: np.full(N, 6),
    "uniform": lambda rng, N: rng.integers(3, 13, size=N),   # U[3,12]
    "bimodal": lambda rng, N: rng.choice([4, 12], size=N),
}
SEEDS  = [42, 123, 456, 789, 1234]
VISITS = 1000

# EMA 파라미터 (ema_ad_low)
ALPHA_UP   = 0.3
ALPHA_DOWN = 0.10   # ema_ad_low
BAND       = 0.05
BETA_BASE  = 0.1

# CONSEC 파라미터
CONSEC_L   = 2      # consec_L2

# DCF 파라미터
CW0        = N      # N에 따라 가변
CW_MAX     = 1023

# AND 파라미터
AND_PHASE_CAP = 60  # 2^60 overflow 방지
```

---

## 5. Figure 패널 구성 (2패널, 1×2 가로 배치)

```
┌──────────────────────────┬──────────────────────────┐
│  (a) Throughput          │  (b) τ Trajectory        │
│  efficiency vs N         │  W_rem → 0 (descending)  │
│  uniform, W_eff=50       │  N=20, W_eff=100, seed=42│
└──────────────────────────┴──────────────────────────┘
```

### Panel (a): Throughput — efficiency vs N

- **X축**: N ∈ {10, 20, 30, 50}
- **Y축**: efficiency = successes / oracle_successes
- **W_eff**: 50, **PPDU_DIST**: uniform U[3,12]
- **방법**: dcf_self_excl (굵은 실선, benchmark), ema_ad_low, consec_L2, and
- **참조선**: oracle 얇은 회색 점선
- Y축 범위: [0.40, 1.10]

### Panel (b): τ Trajectory — 단일 visit 시각화

- **X축**: W_rem (W_eff=100 → 0, 역방향) — fig17 panel c 스타일
- **Y축**: τ 값 (0~1)
- **설정**: N=20, W_eff=100, seed=42, PPDU_DIST=uniform
- **방법**: dcf_self_excl (1/CW_mean 환산), ema_ad_low, consec_L2, and
- **표시**: STA ppdu_i 배제 임계 수직선
- **dcf τ 환산**: `τ_eff = 1/mean(CW_viable)` (viable STA 평균 CW)
- 기대: AND = 계단형(phase), DCF = 충돌마다 하락·solo마다 리셋, EMA/consec = 점진적 조정

---

## 6. 가설

| ID | 가설 | 지표 | 예상 결과 |
|---|---|---|---|
| H1 | pnd > dcf_self_excl | efficiency (N=20, uniform, W=50) | 0.963 vs 0.828 → ✅ |
| H2 | ema_ad_low > dcf_self_excl | efficiency | 0.943 vs 0.828 → ✅ |
| H3 | consec_L2 > dcf_self_excl | efficiency | 0.900 vs 0.828 → ✅ |
| H4 | and < dcf_self_excl | efficiency | 0.541 vs 0.828 → ✅ |
| H5 | pnd > ema_ad_low | efficiency | 0.963 vs 0.943 → ✅ |
| H6 | N 증가 → dcf 열화 가속 | efficiency slope vs N | dcf 기울기 > ema/consec 기울기 |

---

## 7. 출력 파일

```
results/step9/fig19/
└── data.csv          ← 4 methods + oracle (fig17_v7 CSV 재사용 가능)

manuscript/figure/fig19_dcf_benchmark.{eps,png,pdf}
```

### 실행 방법

```bash
# fig17_v7 데이터 재사용 (빠름, 권장)
.venv/bin/python harq_sim/run_step9_fig19.py \
    --base-csv results/step9/fig17_v7/data.csv

# 신규 시뮬레이션 (4 methods + oracle)
.venv/bin/python harq_sim/run_step9_fig19.py

# 빠른 테스트
.venv/bin/python harq_sim/run_step9_fig19.py --fast
```

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-04 | 초안 작성 — dcf_self_excl benchmark, 4종 방법, throughput/delay/τ 3패널 |
