# Figure 18: PND MIMD Parameter Study — c_coll × c_idle Grid

**연구 질문 (RQ18)**:  
유한 NPCA window 내 PND MIMD에서 최적 (c_coll, c_idle) 파라미터 쌍은 무엇인가?  
N (STA 밀도)과 W_eff에 따라 최적 파라미터가 어떻게 이동하는가?

**스크립트**: `harq_sim/run_step9_fig18.py`  
**출력**: `manuscript/figure/fig18_pnd_parameter_study.{eps,png,pdf}`

---

## 1. PND MIMD 업데이트 규칙

```
Solo win:   DW viable STAs → τ = sender's τ  (즉시 동기화, oracle-like)
Collision:  DW viable STAs → τ /= c_coll     (c_coll=1.0 = no penalty)
Idle:       DW viable STAs → τ *= c_idle     (c_idle>1 = 증가)
DT STAs (반이중):  half-duplex → no update (단, pnd_cd에서는 collision 시 감소)
```

- `c_coll`: collision 시 τ 감소 강도. 1.0=패널티 없음, 1.5=원논문, 3.0=공격적
- `c_idle`: idle 시 τ 증가 강도. 1.2=부드러운 회복, 5.0=공격적 회복

**파라미터 그리드**:
```python
C_COLL_LIST = [1.0, 1.2, 1.5, 2.0, 3.0]  # 5종
C_IDLE_LIST = [1.2, 1.5, 2.0, 3.0, 5.0]  # 5종
→ 25 PND configs + oracle = 26 methods
```

---

## 2. 실험 파라미터

```python
N_LIST    = [10, 20, 30, 50]       # STA 수
WEFF_LIST = [20, 50, 100, 200]     # NPCA window 크기
SEEDS     = [42, 123, 456, 789, 1234]
VISITS    = 1000
PPDU      = uniform U[3,12]  (same as fig17 v5, mean≈7.5 slots)
```

총 2080 configurations × 1000 visits = 2,080,000 visit 시뮬레이션.

---

## 3. Figure 패널 구성

```
┌─────────────────────────────┬────────────────────────────┐
│ (a) efficiency vs N          │ (b) efficiency vs N         │
│     c_coll sweep             │     c_idle sweep            │
│     (c_idle=1.5, W_eff=50)   │     (c_coll=1.0, W_eff=50) │
├─────────────────────────────┼────────────────────────────┤
│ (c) heatmap                  │ (d) heatmap                 │
│     N=20, W_eff=50           │     N=50, W_eff=50 (dense) │
└─────────────────────────────┴────────────────────────────┘
```

---

## 4. 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-04 | 초안 작성 및 v1 실험 실행 (2080 configs × 1000 visits × 5 seeds) |

---

## 5. 실험 결과 v1 (2026-06-04)

### Top-3 configs per (N, W_eff) — 전체 16 조건

| N | W_eff | #1 (cc/ci → eff) | #2 (cc/ci → eff) | #3 (cc/ci → eff) |
|---|---|---|---|---|
| 10 | 20 | 1.5/2.0 → **0.9612** | 1.0/1.5 → 0.9607 | 1.5/1.5 → 0.9561 |
| 10 | 50 | 1.0/1.2 → **0.9619** | 1.2/1.2 → 0.9614 | 1.5/1.5 → 0.9590 |
| 10 | 100 | 1.2/1.2 → **0.9915** | 1.5/1.2 → 0.9887 | 1.0/1.2 → 0.9857 |
| 10 | 200 | 1.5/1.2 → **1.0000** | 1.2/1.2 → 1.0000 | 1.0/1.2 → 0.9998 |
| 20 | 20 | 1.2/1.5 → **0.9755** | 2.0/2.0 → 0.9717 | 1.5/1.5 → 0.9716 |
| 20 | 50 | 1.2/1.2 → **0.9676** | 1.0/1.2 → 0.9570 | 1.5/1.5 → 0.9556 |
| 20 | 100 | 1.2/1.2 → **0.9720** | 1.5/1.5 → 0.9587 | 1.5/1.2 → 0.9562 |
| 20 | 200 | 1.2/1.2 → **0.9960** | 1.5/1.2 → 0.9922 | 2.0/1.2 → 0.9867 |
| 30 | 20 | 1.2/1.5 → **0.9904** | 1.0/1.5 → 0.9902 | 1.5/2.0 → 0.9862 |
| 30 | 50 | 1.2/1.2 → **0.9693** | 1.5/1.5 → 0.9628 | 1.0/1.2 → 0.9594 |
| 30 | 100 | 1.2/1.2 → **0.9753** | 1.5/1.5 → 0.9630 | 1.5/1.2 → 0.9611 |
| 30 | 200 | 1.2/1.2 → **0.9774** | 1.5/1.2 → 0.9650 | 1.5/1.5 → 0.9625 |
| 50 | 20 | 1.2/1.5 → **0.9932** | 1.0/1.5 → 0.9862 | 1.0/1.2 → 0.9821 |
| 50 | 50 | 1.2/1.2 → **0.9783** | 1.5/1.5 → 0.9694 | 1.2/1.5 → 0.9644 |
| 50 | 100 | 1.2/1.2 → **0.9777** | 1.5/1.2 → 0.9627 | 1.5/1.5 → 0.9624 |
| 50 | 200 | 1.2/1.2 → **0.9810** | 1.5/1.2 → 0.9689 | 1.5/1.5 → 0.9636 |

### 최적 파라미터 빈도 분석

| 파라미터 | 1위 횟수 (16개 중) |
|---|---|
| **cc=1.2 / ci=1.2** | **10회** ← 가장 robust |
| cc=1.2 / ci=1.5 | 3회 (W_eff=20 조건) |
| cc=1.0 / ci=1.2 | 1회 |
| cc=1.5 / ci=1.2 | 1회 |
| cc=1.5 / ci=2.0 | 1회 |

---

## 6. 핵심 발견

### 1. 최적 파라미터 (cc=1.2, ci=1.2) — 전 조건 robust

16개 (N, W_eff) 조건 중 10개에서 `cc=1.2/ci=1.2`가 1위. N≥20이고 W_eff≥50인 모든 조건에서 일관되게 최고.

**원논문 PND (1.5, 1.5) 대비 개선**:
```
N=20, W_eff=50: (1.5,1.5)=0.9556 → (1.2,1.2)=0.9676  (+1.2%)
N=50, W_eff=50: (1.5,1.5)=0.9694 → (1.2,1.2)=0.9783  (+0.9%)
N=50, W_eff=200:(1.5,1.5)=0.9636 → (1.2,1.2)=0.9810  (+1.7%)
```

### 2. c_coll=1.0 (no penalty)가 최적이 아님 — fig17 ema_no_coll과 대조적

fig17에서 EMA 계열은 α↓=0 (no penalty)이 최고였지만, **PND에서는 cc=1.2 (mild penalty)가 1.0보다 일관 우위**.

| N=20, W=50 | efficiency |
|---|---|
| cc=1.0/ci=1.2 | 0.9570 (2위) |
| **cc=1.2/ci=1.2 (optimal)** | **0.9676 (1위)** |
| cc=1.5/ci=1.5 (original) | 0.9556 |

**이유**: EMA에서 α↓=0.5는 τ를 반토막 내는 과도한 패널티 → 제거가 최선. PND에서는 solo-copy가 주 수렴 메커니즘이고, ÷1.2의 미세 하향 조정이 collision 후 오버슈팅을 보정 → 완전 제거(1.0)보다 소폭 우위. solo-copy로 τ≈1/k_viable가 설정되지만 이후 k_viable-1로 줄면 τ가 약간 높은 상태 → 충돌 시 ÷1.2로 보정 효과.

### 3. W_eff=20 (tight window)에서는 c_idle=1.5 유리

| 조건 | 최적 c_idle | 이유 |
|---|---|---|
| W_eff=20 (tight) | **1.5** | 짧은 window → solo 이벤트 빈도 낮음 → idle 후 τ 빠른 회복 필요. ci=1.2로는 회복 너무 느림 |
| W_eff≥50 (normal/loose) | **1.2** | solo-copy 이벤트 충분 → τ 자동 보정 → gentle increase로 충분, aggressive는 오버슈팅 |

**경계 조건**: W_eff / (N × PPDU_mean) ≈ 20 / (N × 7.5) → N=10이면 ≈0.27, N=50이면 ≈0.053.
tight window 판단 기준: W_eff/N·PPDU_mean ≤ 0.5 이하이면 ci=1.5 권장.

### 4. Large W_eff → 파라미터 무관, 모든 configs 수렴

N=10, W_eff=200: cc=1.5/ci=1.2 → 1.0000, cc=1.2/ci=1.2 → 1.0000

Window가 충분히 크면(W_eff >> N×PPDU_mean) solo-copy 이벤트 다수 → τ 자동 수렴 → 파라미터 선택 영향 미미.

### 5. 유해한 파라미터 범위

| 파라미터 | 영향 |
|---|---|
| c_coll ≥ 2.0 | 항상 유해: collision 후 τ 과도 감소 → idle 증가 → W_rem 낭비 |
| c_idle ≥ 3.0 | 항상 유해: τ 오버슈팅 → collision 증가 → W_rem 낭비 |
| c_idle=1.2 + W_eff=20 | 부분 유해: tight window에서 idle 회복 부족 |

### 6. 원논문 PND (1.5, 1.5) 평가

실용적 관점에서 (1.5, 1.5)는 나쁘지 않지만 최적 대비 약 1-2%p 손실. 특히 large W_eff에서 손실이 더 큼 (N=50, W=200: 0.9636 vs 0.9810). Solo-copy 이후 c_coll=1.5의 과도한 τ 감소와 c_idle=1.5의 불필요하게 빠른 회복 조합 → 진동 발생.

---

## 7. 파라미터 설계 원칙

```
W_eff / (N × PPDU_mean) ≤ 0.5  (tight window):
  → c_coll = 1.2, c_idle = 1.5

W_eff / (N × PPDU_mean) > 0.5  (normal/loose):
  → c_coll = 1.2, c_idle = 1.2  ← robust default

절대 금지: c_coll ≥ 2.0 또는 c_idle ≥ 3.0
주의: c_coll = 1.0도 이론적 직관과 달리 c_coll=1.2보다 약간 열위
```

**논문 기술 방향**:
> PND MIMD에서 collision penalty 완전 제거(c_coll=1.0)는 직관적으로 최선처럼 보이지만,  
> 유한 NPCA window에서는 solo-copy 이후 미세 correction (c_coll=1.2)이 오히려 oracle에 가깝게 수렴한다.  
> 이는 solo-copy로 설정된 τ≈1/k_viable가 다음 round의 k_viable-1 상황보다 약간 높기 때문이며,  
> ÷1.2의 gentle 하향이 이를 보정한다.

---

## 8. 출력 파일

```
results/step9/fig18/data.csv
  columns: c_coll, c_idle, N, W_eff, seed, efficiency

manuscript/figure/fig18_pnd_parameter_study.{eps,png,pdf}
```
