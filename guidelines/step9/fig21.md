# Figure 21: Native vs. Visitor Fairness — NPCA Channel Access Competition

**연구 질문 (RQ21)**:  
NPCA visitor STA(primary 채널 OBSS 감지 후 일시 전환)와  
NPCA native STA(NPCA 채널 상시 운용)가 동일 W_eff 창 내에서 경쟁할 때,  
PND 프로토콜은 visitor STA의 접근 기회를 보장하는가?  
동시에 native STA에게 불공정하게 작용하지 않는가?

**스크립트**: `harq_sim/run_step9_fig21.py`  
**출력**: `manuscript/figure/fig21_native_visitor_fairness.{eps,png,pdf}`

---

## 1. 모델 확장: Mixed Native/Visitor Contention

### 기존 모델 (fig17~20)
```
W_eff window: N STAs 전체 = visitor STA (동일 방법 사용)
```

### fig21 확장 모델
```
W_eff window:
  ├── N_visitor visitor STAs  → 적응형 프로토콜 (pnd, ema_ad_low 등)
  │                              ppdu: bimodal {4, 12}
  └── N_native  native STAs   → 고정 DCF (CW₀=N_total, ppdu=6)
                                 (IEEE 802.11 표준 EDCA 추상화)
```

**핵심 비대칭**:
- Visitor STA: W_eff 내 완료 강제 (시간 제한 있음), 적응형 τ 사용
- Native STA: W_eff 내 기회 경쟁만 (창 이후 계속 운용 가능), 표준 DCF 사용

**W_eff 창 종료**: W_rem=0 또는 모든 viable STA 없을 때 → native STA는 그 이후 계속 채널 사용하지만 본 시뮬레이션 범위 밖

---

## 2. 비교 방법

| 역할 | 방법 | 특성 |
|---|---|---|
| Visitor protocol (6종) | oracle | per-round Aloha-optimal reference |
| | pnd | 논문 주요 기여 |
| | ema_ad_low | EMA 계열 |
| | consec_L2 | counting 계열 |
| | dcf_self_excl | 표준 DCF (visitor도 DCF 사용 — baseline) |
| | and | open-loop lower bound |
| Native STAs | dcf (고정) | 항상 standard DCF, ppdu=6, CW₀=N_total |

---

## 3. 실험 파라미터

```python
N_VISITOR = 10                    # visitor STA 수 (고정)
N_NATIVE_LIST = [0, 5, 10, 20]   # native STA 수 스윕 (0 = fig19/20 baseline)
WEFF_LIST = [50, 100]
PPDU_VISITOR_DIST = "bimodal"     # {4, 12}
PPDU_NATIVE = 6                   # 고정 (homo)
SEEDS = [42, 123, 456, 789, 1234]
VISITS = 1000
```

`N_native=0`은 fig20의 N=10 조건과 동일 → 연속성 확인 가능

---

## 4. Fairness 지표

### 4-1. Proportionality Index

```python
succ_visitor = Σ succeeded[0:N_visitor]
succ_native  = Σ succeeded[N_visitor:]

visitor_share  = succ_visitor / (succ_visitor + succ_native)  # 실제 visitor 비율
ideal_share    = N_visitor / (N_visitor + N_native)            # 이상적 비율 (인원 비례)
proportionality = visitor_share / ideal_share
# 1.0 = 공정, >1 = visitor 과다 획득, <1 = visitor 불이익
```

### 4-2. Per-group W_eff_utilization

```python
useful_visitor = Σ(ppdu_i × succeeded_i for i in visitor_STAs)
useful_native  = Σ(ppdu_i × succeeded_i for i in native_STAs)

weff_util_visitor = useful_visitor / W_eff
weff_util_native  = useful_native  / W_eff
weff_util_total   = (useful_visitor + useful_native) / W_eff
```

### 4-3. Native throughput preservation

```python
# native_only 기준: N_native STAs만 있을 때 native throughput
# native_with_visitor: N_native + N_visitor 있을 때 native throughput
native_preservation = weff_util_native_with_visitor / weff_util_native_only
# 1.0 = visitor가 native에 영향 없음, <1 = visitor가 native 침해
```

---

## 5. Figure 패널 구성 (3패널, 1×3)

```
┌────────────────────┬────────────────────┬────────────────────┐
│ (a) Visitor TP     │ (b) Proportionality│ (c) TP scatter     │
│ vs N_native        │ vs N_native        │ visitor vs native  │
│ W_eff=50           │ W_eff=50           │ N_native=10, W=50  │
└────────────────────┴────────────────────┴────────────────────┘
```

### Panel (a): Visitor W_eff_utilization vs N_native

- **X축**: N_native ∈ {0, 5, 10, 20}
- **Y축**: weff_util_visitor (visitor STA들의 useful_slots / W_eff)
- **W_eff**: 50, **N_visitor**: 10
- **방법**: 6종 visitor protocol
- **기대**: 모든 방법에서 N_native 증가 → visitor TP 감소; PND는 완만하게 감소

### Panel (b): Proportionality Index vs N_native

- **X축**: N_native ∈ {0, 5, 10, 20}
- **Y축**: proportionality = visitor_share / ideal_share
- **참조선**: y=1.0 (공정 기준), y>1 = visitor 과다, y<1 = visitor 불이익
- **기대**:
  - N_native=0: 정의 불가 (N_native=0이면 ideal_share=1, visitor_share=1 → prop=1)
  - N_native 증가 → prop 변화 관찰
  - PND: prop ≥ 1.0 (adaptive τ가 visitor 이익 유지)
  - AND: prop < 1.0 (고정 낮은 τ → visitor 불이익)
  - DCF visitor vs DCF native: prop ≈ 1.0 (동일 메커니즘 → 공정)

### Panel (c): Visitor TP vs Native TP scatter

- **X축**: weff_util_visitor (visitor 효율)
- **Y축**: weff_util_native (native 효율)
- **각 점**: 방법 1개 (N_native=10, W_eff=50)
- **참조점**: "no visitor" 기준 (visitor=0일 때 native TP)
- **기대**: PND → visitor TP 높지만 native TP도 유지 (Pareto-dominant region)

---

## 6. 가설

| ID | 가설 | 지표 | 예상 결과 |
|---|---|---|---|
| H1 | 모든 방법: N_native 증가 → visitor TP 감소 | weff_util_visitor | 단조 감소 |
| H2 | PND proportionality ≥ DCF proportionality | proportionality | PND adaptive τ가 경쟁 우위 유지 |
| H3 | AND proportionality ≤ DCF proportionality | proportionality | 고정 낮은 τ → visitor 불이익 |
| H4 | PND native TP ≈ DCF native TP | weff_util_native | PND가 native를 과도 침해하지 않음 |
| H5 | DCF_visitor vs DCF_native: proportionality ≈ 1.0 | proportionality | 동일 메커니즘 → 인원 비례 공정 |

**H2 근거 (PND visitor 보호 기대)**:  
PND solo-copy 메커니즘: visitor STA_i 성공 후 DW viable STAs가 τ_i 복사  
→ native DCF STA들도 DW 조건이면 τ 동기화됨  
→ 단, native DCF는 CW 기반이라 solo-copy 효과 무시 → 동기화 안 됨  
→ visitor STAs끼리만 solo-copy → visitor 집단이 τ 빠르게 수렴  
→ visitor 집단의 효율적 채널 사용이 native DCF보다 앞서 완료 가능

---

## 7. 시뮬레이션 구조

### `_run_visit_mixed()` (fig21 전용 함수)

```python
def _run_visit_mixed(
    visitor_method: str,
    N_visitor: int, N_native: int,
    W_eff: int,
    ppdus: np.ndarray,    # len = N_visitor + N_native
    rng: np.random.Generator,
    oracle_successes: int,
) -> dict:
    """
    Visitor STAs [0:N_visitor] : visitor_method (pnd, ema_ad_low 등)
    Native STAs  [N_visitor:]  : DCF (CW₀=N_visitor+N_native, ppdu=PPDU_NATIVE=6)
    
    반환:
      weff_util_visitor, weff_util_native, weff_util_total
      succeeded_visitor_mask, succeeded_native_mask
      proportionality
    """
```

**상태 분리**:
- Visitor STA [0:N_visitor]: visitor_method의 τ / EMA / consec / PND / DCF 상태
- Native STA [N_visitor:]: 독립 DCF 상태 (dcf_cw_native, dcf_bo_native)

**TX 결정 (per-STA)**:
```python
# Visitor STAs: visitor_method에 따른 τ 또는 DCF
# Native STAs:  항상 DCF (dcf_bo == 0)

tx_visitor = [visitor_tx_decision(i) for i in range(N_visitor)]
tx_native  = [(dcf_bo[j] == 0) for j in range(N_native)]
tx = np.concatenate([tx_visitor, tx_native])
```

**상태 업데이트**:
- Visitor STAs: visitor_method에 따른 업데이트
- Native STAs: DCF BEB 업데이트 (visitor 상태와 독립)

---

## 8. CSV 스키마

```
method, N_visitor, N_native, W_eff, seed,
weff_util_visitor, weff_util_native, weff_util_total,
proportionality, visitor_share, ideal_share,
native_preservation
```

---

## 9. 출력 파일

```
results/step9/fig21/
└── data.csv

manuscript/figure/fig21_native_visitor_fairness.{eps,png,pdf}
```

### 실행 방법

```bash
.venv/bin/python harq_sim/run_step9_fig21.py
.venv/bin/python harq_sim/run_step9_fig21.py --fast
```

---

## 9. 실험 결과 (full run: 1000 visits × 5 seeds, N_native=[0,5,10,20], W_eff=[50,100])

### Full-run 가설 검증

| 가설 | 결과 | 비고 |
|---|---|---|
| H1: visitor TP 단조 감소 | ✅ PASS | AND도 full run에서 단조 (fast-mode 노이즈였음) |
| H2: PND prop ≥ DCF prop | ✅ PASS | PND=1.51, DCF=1.00 @ N_native=10 |
| H3: AND prop ≤ DCF prop | ❌ FAIL | AND=1.90 > DCF=1.00 @ N_native=10 — 반직관 |
| H4: PND native TP ≈ DCF native TP | ❌ FAIL | PND=0.162, DCF=0.293 (−44.6%) |
| H5: DCF visitor prop ≈ 1.0 | ✅ PASS | 0.985~1.025 ≈ 1 (대칭 메커니즘 → 비례 공정) |

### Full-run 요약 테이블 (N_native=10, W_eff=50)

| Method | util_v | util_n | util_t | prop | nat_pres |
|---|---|---|---|---|---|
| oracle | 0.555 | 0.206 | 0.761 | 1.39 | 0.336 |
| **pnd** | **0.593** | 0.162 | 0.755 | **1.51** | 0.265 |
| ema_ad_low | 0.498 | 0.235 | 0.733 | 1.27 | 0.384 |
| consec_L2 | 0.396 | 0.308 | 0.704 | 1.01 | 0.502 |
| **dcf_self_excl** | 0.354 | **0.293** | 0.646 | **1.00** | 0.478 |
| and | 0.601 | 0.027 | 0.628 | 1.90 | 0.043 |

### 핵심 발견 (full-run 확정)

**H3 반직관 분석 (AND 과점)**:  
AND open-loop schedule은 phase 1에서 τ=0.5로 시작. 초기 rounds에서 visitor AND STA들이 빠르게 전송 → native DCF와 충돌 → native BEB CW 기하급수적 증가 (→ DCF_CW_MAX=1023까지).  
N_native 증가할수록 proportionality 악화: 1.44 → 1.90 → 2.84.  
→ 논문 기여: "open-loop schedules can accidentally exclude native STAs in the early window phase"

**H4 분석 (PND native 침해)**:  
PND solo-copy로 visitor 집단이 빠르게 τ 수렴 → 고효율 선점 → W_eff 조기 소진.  
native_preservation = 0.265 (native 단독 대비 26.5% 수준만 획득).  
→ 논문 한계: "PND's collective convergence inadvertently starves native DCF STAs — an unintended consequence of adaptive coordination"

**consec_L2 & DCF: 가장 공정 (prop ≈ 1.0)**:  
consec_L2: 보수적 τ 조정 → native STAs에 비례적 기회 보장.  
dcf_self_excl: 동일 메커니즘 → 인원 비례 공정 (H5 원리).

**PND proportionality 증가 트렌드**: N_native 5→10→20: 1.26→1.51→1.94  
→ native STAs가 많아질수록 PND visitor들이 오히려 더 많은 비율 독식.  
원인: native DCF가 충돌 후 BEB → τ 감소 → PND visitors의 solo 기회 증가.

---

## 10. Fast-mode 예비 결과 (50 visits × 1 seed, N_native=[0,5,10], W_eff=50, 참고)

### 가설 검증

| 가설 | 결과 | 비고 |
|---|---|---|
| H1: visitor TP 단조 감소 (모든 방법) | ❌ FAIL (AND 예외) | AND phase-based 비적응 → N_native=10에서 비단조 |
| H2: PND prop ≥ DCF prop | ✅ PASS | PND=1.43, DCF=0.92 @ N_native=10 |
| H3: AND prop ≤ DCF prop | ❌ FAIL | AND=1.95 > DCF=0.92 @ N_native=10 — 반직관 |
| H4: PND native TP ≈ DCF native TP | ❌ FAIL | PND=0.187, DCF=0.319 (−41%) |
| H5: DCF visitor prop ≈ 1.0 | ✅ PASS | 0.92~0.98 ≈ 1 |

### 핵심 발견 (fast-mode, 노이즈 있음)

**H3 반직관 분석**: AND의 open-loop 스케줄은 phase 1에서 τ=0.5 (높음)로 시작.  
초기 rounds에서 visitor AND STA들이 적극 전송 → native DCF와 충돌 → native STA BEB 급증.  
visitor AND STA들은 τ를 낮춰가는 schedule이지만 초반 선점으로 proportionality > 1.  
→ 논문 기여: "open-loop schedule can accidentally advantage visitors in early window phase"

**H4 분석 (PND native TP 감소)**: PND solo-copy로 visitor STA들이 빠르게 수렴 → 고효율  
→ W_eff 더 빨리 소진 → native STA들이 전송 기회를 갖기 전에 W_rem 고갈  
→ native_preservation = 0.31 (full DCF 대비 31% 수준으로 저하)  
→ 논문 해석: "PND's aggressive channel utilization inadvertently starves native DCF STAs"

**H1 AND 예외**: N_native=0→5: 0.638→0.598, N_native=5→10: 0.598→0.627 (소폭 상승)  
원인: N_native 증가 → 더 많은 충돌 → native BEB 폭증 → 결국 native가 양보하는 효과.

### 요약 테이블 (N_native=10, W_eff=50)

| Method | util_v | util_n | util_t | prop | nat_pres |
|---|---|---|---|---|---|
| oracle | 0.574 | 0.197 | 0.771 | 1.43 | 0.327 |
| **pnd** | **0.547** | 0.187 | 0.734 | **1.43** | 0.311 |
| ema_ad_low | 0.531 | 0.192 | 0.723 | 1.37 | 0.319 |
| consec_L2 | 0.362 | 0.336 | 0.698 | 0.93 | 0.558 |
| **dcf_self_excl** | 0.317 | **0.319** | 0.636 | **0.92** | 0.530 |
| and | 0.627 | 0.014 | 0.642 | 1.95 | 0.024 |

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-04 | 초안 작성 — native/visitor mixed contention 모델, 3패널 proportionality 분석 |
| 2026-06-04 | 구현 완료 (run_step9_fig21.py); fast-mode 예비 결과 추가 |
| 2026-06-04 | Full run 완료 (1000 visits × 5 seeds); 확정 결과 추가; H2/H5 PASS, H3/H4 FAIL |
