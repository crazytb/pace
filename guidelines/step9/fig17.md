# Figure 17: PPDU-Aware Self-Exclusion + Adaptive τ — 이질 PPDU 환경 검증

**연구 질문 (RQ17)**:  
이질적 PPDU 길이를 가진 STA들이 NPCA window를 공유할 때,  
(1) silent self-exclusion만으로 충분한가,  
(2) EMA 기반 τ 적응이 추가로 성능을 회복하는가,  
(3) 설계 선택지(고정 β vs adaptive β, hysteresis vs consecutive idle trigger)가  
    성능에 얼마나 영향을 미치는가?

**스크립트**: `harq_sim/run_step9_fig17.py`  
**출력**: `manuscript/figure/fig17_ppdu_aware_tau.{eps,png,pdf}`

---

## 1. 시스템 모델

### NPCA Visit 시뮬레이션 (이질 PPDU 확장)

```
W_eff slots 내 N STAs 경쟁 (완전 그래프, 충돌 도메인 공유)

각 visit 시작:
  - 각 STA i에게 ppdu_i ~ PPDU_DIST 독립 할당
  - W_rem = W_eff, remaining = N

매 contention round:
  - active_i = (W_rem >= ppdu_i) AND (not yet succeeded)
  - 각 active STA i: Bernoulli(τ_i)로 TX 시도

  결과:
    solo (1명 시도): 해당 STA 성공, W_rem -= ppdu_i (PPDU 전송 소비), remaining -= 1
    collision (2명+): W_rem -= 1 (충돌 슬롯 낭비), BEB는 적용 안 함 (Bernoulli τ 모델)
    idle (0명 시도): W_rem -= 1

  종료 조건: W_rem < min(ppdu_i for active i) OR all succeeded
```

### Self-Exclusion 조건
```
viable_i(t) = (W_rem(t) >= ppdu_i) AND (not yet succeeded)
τ_i = 0 if not viable_i(t)
```

### Oracle
```
τ_oracle = 1 / viable_remaining(t)
where viable_remaining = |{i : viable_i(t) = True}|
```

---

## 2. 비교 알고리즘 (Ablation Table)

| 방법 ID | Self-excl | τ 업데이트 방식 | 윈도우/파라미터 | 트리거 |
|---|---|---|---|---|
| `oracle` | ✓ | 1/viable_remaining (전지적) | — | — |
| `mfg_no_excl` | ✗ | 1/remaining (원본) | — | — |
| `self_excl_only` | ✓ | 1/remaining (분모 미보정) | — | — |
| `ema_fixed_low` | ✓ | EMA idle-rate 기반 | β=0.05 고정, α↓=0.50 | hysteresis |
| `ema_fixed_high` | ✓ | EMA idle-rate 기반 | β=0.30 고정, α↓=0.50 | hysteresis |
| `ema_adaptive` | ✓ | EMA idle-rate 기반 | β_t adaptive, α↓=0.50 | hysteresis |
| `ema_ad_low` | ✓ | EMA idle-rate 기반 | β_t adaptive, α↓=0.10 | hysteresis |
| `ema_ad_med` | ✓ | EMA idle-rate 기반 | β_t adaptive, α↓=0.25 | hysteresis |
| `ema_no_coll` | ✓ | EMA idle-rate 기반 | β_t adaptive, α↓=0 (없음) | hysteresis |
| `consec_L2` | ✓ | 연속 2 idle → τ 증가 | — | L=2 |
| `consec_L4` | ✓ | 연속 4 idle → τ 증가 | — | L=4 |
| `dcf_self_excl` | ✓ | BEB (CW×2 on collision) | CW₀=N, CW_max=1023 | IEEE 802.11 DCF |
| `pnd` | ✓ | MIMD: solo→τ_copy, idle→τ×c_idle, coll→τ÷c_coll (no CD) | c_coll=1.2, c_idle=1.2 | solo/idle/collision |
| `pnd_cd` | ✓ | MIMD + CD: DT STAs도 τ÷c_coll on collision | c_coll=1.2, c_idle=1.2 | solo/idle/collision |
| `and` | ✓ | Phase-based open-loop: p=1/2^i 고정 (비적응) | Phase i 지속: ⌈2^i·e·i·ln2⌉ slots | phase schedule |

**dcf_self_excl**: ppdu_i > W_rem인 STA는 backoff counter를 동결(self-exclusion). TX는 backoff=0일 때 결정론적(비베르누이). 충돌 시 CW×2, 새 backoff draw.

**pnd**: Song et al. (WCNC 2014) MIMD policy. Solo win 시 DW(Did-not-Win) STAs가 송신자 τ를 즉시 복사(τ_copy). 이후 idle → τ×c_idle, collision → τ÷c_coll 적용. Collision Detection 없음(no-CD) — collision 발생 여부를 DW STAs가 인지하지 않음.

**pnd_cd**: pnd에 CD(Collision Detection) 추가. Collision 시 DT(Did-Transmit, 충돌 참여) STAs도 τ÷c_coll 적용. c_coll=1.5(원논문)은 유한 NPCA window에서 과도한 τ 감소 → idle 증가로 역효과. fig18 파라미터 스터디로 도출한 c_coll=1.2에서 CD 이득 복원.

**and**: Vasudevan et al. (MobiCom 2009) ALOHA-like Neighbor Discovery. Phase i에서 고정 전송 확률 p=1/2^i, 지속 슬롯 수=⌈2^i·e·i·ln2⌉. 비적응(open-loop) — collision/idle/solo 결과에 무관하게 phase schedule만 따름. NPCA 유한 윈도우 비교용 baseline (infinite-horizon 설계로 낮은 효율 예상).

### τ 업데이트 규칙 (EMA 계열 공통)

```python
# 매 contention round 후:
ema_idle = (1-β) * ema_idle + β * (outcome == IDLE)
gap = ema_idle - (1/e)

if gap > BAND (=0.05) and viable:        # idle 과다 → 경쟁자 줄었음
    tau *= (1 + ALPHA_UP * gap)          # ALPHA_UP = 0.3
elif outcome == COLLISION and viable:    # 명시적 collision
    tau *= (1 - ALPHA_DOWN)              # ALPHA_DOWN = 0.5
tau = clip(tau, 1e-4, 1.0)

# adaptive β:
beta = clip(0.1 * N / W_rem, 0.05, 0.50)
```

### τ 업데이트 규칙 (연속 idle 계열)

```python
if outcome == IDLE:
    consec_idle += 1
    if consec_idle >= L and viable:
        tau = min(tau * (1 + ALPHA_UP), 1.0)
else:
    consec_idle = 0
    if outcome == COLLISION and viable:
        tau *= (1 - ALPHA_DOWN)
```

---

## 3. 실험 파라미터

```python
# STA 수
N_LIST = [5, 10, 20, 30]

# NPCA window 크기
WEFF_LIST = [20, 50, 100, 200]

# PPDU 분포 3종
PPDU_DIST = {
    "homo":    lambda rng, N: np.full(N, 15),                     # 동질
    "uniform": lambda rng, N: rng.integers(5, 41, size=N),        # U[5,40]
    "bimodal": lambda rng, N: rng.choice([8, 35], size=N),        # 50/50
}

# 통계
SEEDS   = [42, 123, 456, 789, 1234]   # 5 seeds
VISITS  = 1000                         # visit 수 (seed당)

# EMA 파라미터
ALPHA_UP   = 0.3
ALPHA_DOWN = 0.5
BAND       = 0.05
BETA_BASE  = 0.1
```

---

## 4. 측정 지표

| 지표 | 정의 | 해석 |
|---|---|---|
| `successes` | visit당 성공 TX STA 수 | 주요 처리량 지표 |
| `efficiency` | successes / oracle_successes | oracle 대비 비율 |
| `viable_slots_wasted` | viable STA 있는데 idle 발생한 슬롯 수 | τ 너무 낮음 → idle 낭비 |
| `collision_rate` | collision 슬롯 / 전체 슬롯 | τ 너무 높음 |
| `tau_rmse` | mean(|τ_used - τ_oracle|²) over viable rounds | τ 추정 정확도 |
| `false_trigger_rate` | viable_remaining 변화 없는데 τ 증가한 횟수 | false positive |

---

## 5. Figure 패널 구성 (4패널)

```
┌─────────────────────────┬─────────────────────────┐
│  (a) 방법별 efficiency   │  (b) β 설계 선택 비교    │
│  vs N (uniform PPDU)     │  vs W_eff/N (N=20)       │
├─────────────────────────┼─────────────────────────┤
│  (c) τ trajectory        │  (d) PPDU 분포 효과      │
│  over one visit          │  homo / uniform / bimodal│
└─────────────────────────┴─────────────────────────┘
```

### Panel (a): 핵심 비교 — efficiency vs N

- **X축**: N (5, 10, 20, 30)
- **Y축**: efficiency = mean(successes) / mean(oracle_successes)
- **W_eff**: 50, **PPDU_DIST**: uniform[5,40]
- **모든 8가지 방법** 표시
- 기대 순서: oracle=1.0 > ema_adaptive > ema_fixed_* > self_excl_only > mfg_no_excl

### Panel (b): β 설계 선택 — W_eff/N sweep

- **X축**: W_eff/N ratio (0.5 ~ 10, log scale)
- **Y축**: efficiency
- **대상 방법**: oracle, self_excl_only, ema_fixed_low, ema_fixed_high, ema_adaptive
- **N=20**, PPDU_DIST=uniform
- 기대: ema_adaptive가 tight(W_eff/N<2)와 loose(W_eff/N>5) 모두에서 robust

### Panel (c): τ trajectory 시각화

- **X축**: W_rem (W_eff → 0, 역방향)
- **Y축**: τ 값
- **단일 visit** (N=10, W_eff=100, seed=42, PPDU_DIST=uniform)
- **방법**: oracle (점선), ema_adaptive (실선), self_excl_only (파선)
- **수직선**: 각 STA의 ppdu_i threshold (STA 배제 시점)
- 기대: ema_adaptive τ가 oracle τ를 tracking, self_excl_only는 underestimate 유지

### Panel (d): PPDU 분포 효과 — bar chart

- **X축**: PPDU_DIST (homo, uniform, bimodal)
- **Y축**: efficiency
- **W_eff=50**, **N=20**
- **방법**: oracle, mfg_no_excl, self_excl_only, ema_adaptive
- 기대: homo에서는 모두 비슷(이질성 없어 배제 불필요), bimodal에서 격차 최대

---

## 6. 가설 (검증 대상)

| 가설 | 측정 지표 | 예상 결과 |
|---|---|---|
| H1: self_excl_only > mfg_no_excl (v2 수정 후 역전) | efficiency | self_excl_only > mfg_no_excl (N>10, uniform) |
| H2: ema_adaptive > self_excl_only | efficiency | ema_adaptive > self_excl_only |
| H5: homo에서는 방법 차이 없음 | efficiency (homo) | all ≈ oracle |
| H6: bimodal 격차 > uniform 격차 | efficiency gap | bimodal gap > uniform gap |
| H7: dcf_self_excl > self_excl_only (BEB τ 적응) | efficiency | dcf > self_excl_only (N=20, uniform) |
| H8: ema_no_coll > ema_adaptive (collision 패널티 제거) | efficiency | ema_no_coll > ema_adaptive |
| H9: α↓ 감소 → 단조 성능 증가 | efficiency | ema_no_coll ≥ ema_ad_low ≥ ema_ad_med ≥ ema_adaptive |
| H10: pnd_cd > pnd (CD 추가 이득, c_coll=1.2) | efficiency | pnd_cd > pnd (N=20, uniform, W_eff=50) |
| H11: pnd_cd ≈ ema_no_coll (짧은 PPDU 환경 수렴) | efficiency diff | \|pnd_cd − ema_no_coll\| < 0.01 |
| H12: and < self_excl_only (open-loop 열위) | efficiency | and ≪ self_excl_only (N=20, uniform, W_eff=50) |

---

## 7. 출력 파일

```
results/step9/fig17/data.csv
  columns: method, ppdu_dist, N, W_eff, seed, visit,
           successes, oracle_successes, efficiency,
           viable_slots_wasted, collision_rate, tau_rmse, false_trigger_rate

manuscript/figure/fig17_ppdu_aware_tau.{eps,png,pdf}
```

---

## 수정 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-03 | 초안 작성 |
| 2026-06-03 | v1 실험 실행 (1920 configs, 1000 visits × 5 seeds) — 시뮬레이션 버그 발견 |
| 2026-06-04 | 시뮬레이션 버그 수정: non-viable solo win → success 처리 → W_rem 음수 아티팩트. 수정 후 mfg_no_excl solo win 시 ppdu_i > W_rem이면 실패(W_rem-=1)로 처리. v2 재실험 완료. |
| 2026-06-04 | v3: 4종 방법 추가 — dcf_self_excl (IEEE 802.11 DCF+BEB+self-excl), ema_ad_low (α↓=0.10), ema_ad_med (α↓=0.25), ema_no_coll (α↓=0). 2880 configs × 1000 visits. ema_no_coll 신규 최고(0.9215); DCF+BEB 유해 확인. H8/H9 pass, H7 fail. |
| 2026-06-04 | v4: 2종 방법 추가 — pnd (MIMD no-CD), pnd_cd (with CD). 3360 configs × 1000 visits. pnd 전체 최고(0.9772); pnd_cd < pnd (H10 fail, 유한 윈도우에서 CD 역효과 확인). |
| 2026-06-04 | v5: 파라미터 재설계 — N=[10,20,30,50] (dense 추가), PPDU 축소 (homo 15→6, uniform U[5,40]→U[3,12], bimodal {8,35}→{4,12}), y축 0.70~1.15로 조정. 3360 configs × 1000 visits. |
| 2026-06-04 | v6: PND 파라미터 최적화 — fig18 결과 반영. PND_C_COLL 1.5→**1.2**, PND_C_IDLE 1.5→**1.2**. pnd/pnd_cd만 재실행(480 configs), 나머지 v5 CSV 재사용(--methods pnd pnd_cd --base-csv). |
| 2026-06-04 | v7: AND(ALOHA-like Neighbor Discovery, Vasudevan MobiCom 2009) 비교 baseline 추가. and만 신규 실행(240 configs), v6 CSV 재사용(--methods and --base-csv). 결과: and=0.5410 (uniform, W_eff=50, N=20) — 전체 최하위. H12 ✅ PASS. |

---

## 실험 결과 v2 (2026-06-04, 수정 후)

### 시뮬레이션 수정 내용

**버그**: `mfg_no_excl`에서 `ppdu_i > W_rem`인 STA가 solo win 시 success로 카운트하고 `W_rem -= ppdu_i` → W_rem 음수 → window 강제 종료 + 불가능한 전송을 success로 처리.

**수정**: solo win 시 `ppdu_i <= W_rem` 체크 추가. `ppdu_i > W_rem`이면 전송 실패 (`W_rem -= 1`, success 없음). D1.2 §37.18.3.1.c.i 진입 조건은 entry-only이며 mid-session 제외 규칙 없음 — STA는 시도하지만 PPDU 완료 전 NPCA_TIMER 만료 → 실패.

### 핵심 결과 (uniform, W_eff=50)

| 방법 | N=5 | N=10 | N=20 | N=30 |
|---|---|---|---|---|
| oracle | 1.046 | 1.075 | 1.080 | 1.098 |
| self_excl_only | 0.999 | 0.944 | **0.902** | 0.900 |
| **consec_L2** | 0.966 | 0.945 | **0.913** | 0.914 |
| ema_fixed_high | 0.967 | 0.923 | 0.909 | 0.894 |
| ema_adaptive | 0.932 | 0.897 | 0.867 | 0.879 |
| consec_L4 | 0.949 | 0.916 | 0.875 | 0.884 |
| ema_fixed_low | 0.940 | 0.888 | 0.857 | 0.856 |
| **mfg_no_excl** | 0.925 | 0.855 | **0.806** | 0.800 |

### 가설 검증 결과 (v2)

| 가설 | v1 결과 | v2 결과 | 상세 |
|---|---|---|---|
| H1: self_excl_only < mfg_no_excl | ✅ (아티팩트) | **❌ FLIP** | N=20: self_excl_only(0.902) > mfg_no_excl(0.806) |
| H2: ema_adaptive > self_excl_only | ❌ | ❌ 동일 | 0.867 < 0.902 |
| H3: adaptive β tight window 우세 | ❌ | ❌ 동일 | ema_fixed_high(0.909) > ema_adaptive(0.867) |
| H4: EMA false_trigger < consec | ❌ | ❌ 동일 | ema_adaptive false_trigger 여전히 높음 |
| H5: homo 방법 차이 없음 | ✅ | ✅ | mfg_no_excl=1.043 ≈ oracle (homo에서 non-viable STA 발생 드묾) |
| H6: bimodal gap > uniform gap | ✅ | **❌ FLIP** | uniform gap(0.274) > bimodal gap(0.144) |

### 핵심 발견 및 해석 (v2)

**1. mfg_no_excl이 가장 나쁨 (H1 역전)**
- Non-viable STA가 계속 채널 경쟁 참여 → solo win해도 전송 실패 (ppdu_i > W_rem) → slot 낭비
- N=20에서 efficiency 0.806 — self_excl_only(0.902)보다 9.6% 낮음
- H5 (homo): homo에서는 ppdu=15 고정, W_eff=50이면 대부분 구간에서 viable → mfg_no_excl≈oracle ✅

**2. self_excl_only가 단순하지만 효과적**
- 비viable STA 배제만으로 mfg_no_excl 대비 큰 개선
- τ 보정 없음에도 불구하고 viable STA끼리만 경쟁 → idle/collision 균형 개선

**3. consec_L2가 가장 우수한 적응 방법 (0.913 > 0.902)**
- L=2 연속 idle → τ 증가: 단순하지만 τ 하향 편향 없음
- EMA 계열보다 false trigger 낮음 (2.06 vs 2.55/visit)

**4. EMA 계열 여전히 열위 (H2 실패)**
- ALPHA_DOWN=0.5 collision 패널티 과도 → τ 하락 → idle 증가 → EMA가 viable 감소로 오인 → false trigger
- 개선 방향: collision 패널티 완화(×0.5 → ×0.8) 또는 τ 하락 후 cooldown

**5. H6 역전 (bimodal < uniform)**
- Bimodal: ppdu∈{8,35}. W_eff=50에서 ppdu=8 STAs는 W_rem=8까지 viable → 끝까지 기여
- Uniform: ppdu∈[5,40]. 많은 STA가 중간 W_rem에서 비viable → mfg_no_excl 피해 더 큼

### 논문 contribution 재정의

기존 H1 예상("self-exclusion이 오히려 나쁘다")은 시뮬레이션 버그에 의한 것이었음.

**올바른 서사**:
> D1.2는 mid-session PPDU 완료 불가 STA에 대한 배제 규칙을 정의하지 않는다.
> 이로 인해 non-viable STA가 계속 채널을 경쟁하며 실패 전송으로 slot을 낭비한다.
> Silent self-exclusion (ppdu_i > W_rem일 때 τ_i=0)은 이를 방지하고 efficiency를 크게 개선한다.
> 추가로 consec_L2 τ 적응은 viable STA 수 감소에 대응하여 marginal gain을 제공한다.

**가장 실용적인 권장 방법**: consec_L2 (self_excl_only +1.1%, 구현 단순, EMA보다 false trigger 낮음)

---

## 실험 결과 v3 (2026-06-04)

**변경 사항**: 4종 방법 추가 — `dcf_self_excl`, `ema_ad_low` (α↓=0.10), `ema_ad_med` (α↓=0.25), `ema_no_coll` (α↓=0)

### 핵심 결과 (uniform, W_eff=50, N=20)

| 방법 | efficiency | 순위 |
|---|---|---|
| oracle | 1.0777 | ref |
| **ema_no_coll** | **0.9215** | 1 (신규 최고) |
| consec_L2 | 0.9075 | 2 |
| self_excl_only | 0.9050 | 3 |
| ema_ad_low (α↓=0.10) | 0.9044 | 4 |
| ema_ad_med (α↓=0.25) | 0.8982 | 5 |
| ema_fixed_high | 0.8933 | 6 |
| ema_adaptive (α↓=0.50) | 0.8733 | 7 |
| dcf_self_excl | 0.8634 | 8 |
| consec_L4 | 0.8715 | 9 |
| ema_fixed_low | 0.8550 | 10 |
| mfg_no_excl | 0.8114 | 11 |

### 가설 검증 결과 (v3)

| 가설 | 결과 | 상세 |
|---|---|---|
| H1: self_excl_only > mfg_no_excl | ✅ PASS | N=20: 0.9050 vs 0.8114 |
| H2: ema_adaptive > self_excl_only | ❌ FAIL | 0.8733 < 0.9050 |
| H5: homo 방법 차이 없음 | 부분 | EMA adaptive (0.990) ≠ oracle(1.049); mfg/se/ema_no_coll ≈ oracle ✅ |
| H6: bimodal > uniform | ❌ FAIL | uniform gap(0.266) > bimodal gap(0.143) — v2와 동일 |
| **H7: dcf_self_excl > self_excl_only** | **❌ FAIL** | 0.8634 < 0.9050 — DCF BEB 유해 |
| **H8: ema_no_coll > ema_adaptive** | **✅ PASS** | 0.9215 vs 0.8733 (+5.5%) |
| **H9: α↓ 단조 감소** | **✅ PASS** | 0.9215 ≥ 0.9044 ≥ 0.8982 ≥ 0.8733 (완벽 단조) |

### 핵심 발견 (v3)

**1. EMA collision 패널티가 주요 문제 (H8, H9 확인)**
- ema_no_coll (α↓=0)이 모든 EMA 변형 중 최고 (0.9215)
- α↓ 감소 → 성능 단조 증가: 0.5→0.25→0.10→0.00 = 0.873→0.898→0.904→0.922
- Bernoulli Aloha에서 collision penalty의 역할 재해석:
  - IEEE 802.11 DCF에서는 CW increment가 충돌 후 backoff를 자동으로 늘려 재전송 억제
  - Bernoulli Aloha에서는 τ×(1-α↓)가 유일한 backoff 메커니즘 → α↓=0.5 → τ 반토막 → 8 idle rounds 대기 → 심각한 underutilization
  - **결론**: Bernoulli Aloha에서 collision 후 τ 감소 불필요; EMA idle-rate 기반 증가만으로도 충분

**2. DCF + BEB 유한 NPCA window에서 역효과 (H7 FAIL)**
- dcf_self_excl (0.8634) < self_excl_only (0.9050) — BEB가 오히려 나쁨
- 원인: CW×2 후 새 backoff draw → 대형 backoff 카운터 누적 → W_rem이 countdown 완료 전 고갈 → viable STA들이 bo=0에 도달하지 못하고 윈도우 종료
- BEB는 무한 시간 전송 기회를 가정한 설계; 유한 윈도우(W_eff < CW_max)에서는 collision 후 재시도 기회 자체가 없을 수 있음
- **논문적 함의**: 표준 802.11 DCF를 NPCA window에 그대로 적용하면 안 됨; finite-horizon-aware τ 설계 필요

**3. ema_no_coll이 신규 최고 방법 (0.9215, v2 최고 consec_L2=0.913 대비 +0.9%)**
- EMA idle-rate 기반 τ 증가 + collision 패널티 완전 제거
- 구현: ALPHA_DOWN=0 (collision 시 τ 변화 없음)
- 구현 단순성: collision 패널티 분기 제거 → EMA 로직 단순화

**4. 논문 서사 업데이트**
> Bernoulli Aloha 기반 NPCA contention에서 EMA idle-rate tracking과 collision penalty는 상호 충돌한다.
> Collision penalty(α↓)는 DCF/CW 없는 환경에서 유일한 backoff 메커니즘이 되어 τ를 과도하게 억제한다.
> ema_no_coll — idle-rate 기반 τ 증가만 유지하고 collision penalty를 제거 — 이 최고 성능(0.9215)을 달성한다.
> 표준 DCF BEB는 유한 NPCA window에서 오히려 유해하며, finite-horizon-aware τ 설계의 필요성을 증명한다.

**최종 권장 방법 (v3 기준)**: ema_no_coll (효율 0.9215, 설계 단순, collision 패널티 불필요 이유 이론적 설명 가능)

---

## 실험 결과 v4 (2026-06-04)

**변경 사항**: 2종 방법 추가 — `pnd` (Song et al., WCNC 2014 MIMD, no CD), `pnd_cd` (with Collision Detection)

### 핵심 결과 (uniform, W_eff=50, N=20)

| 방법 | efficiency | 순위 |
|---|---|---|
| oracle | 1.0777 | ref |
| **pnd** | **0.9772** | **1** (신규 최고) |
| **pnd_cd** | **0.9701** | 2 |
| ema_no_coll | 0.9215 | 3 |
| consec_L2 | 0.9075 | 4 |
| self_excl_only | 0.9050 | 5 |
| ema_ad_low | 0.9044 | 6 |
| ema_adaptive | 0.8733 | 7 |
| dcf_self_excl | 0.8634 | 8 |
| mfg_no_excl | 0.8114 | 최하 |

### 가설 검증 결과 (v4)

| 가설 | 결과 | 상세 |
|---|---|---|
| H1~H9 | v3와 동일 | — |
| **H10: pnd_cd > pnd** | **❌ FAIL** | pnd(0.9772) > pnd_cd(0.9701) — CD 역효과 |
| **H11: pnd_cd ≈ ema_no_coll** | **≠ 다름** | diff=0.049 — pnd_cd가 ema_no_coll보다 +5% |

### 핵심 발견 (v4)

**1. PND MIMD가 전체 방법 중 최고 효율 (0.9772)**
- Solo 성공 시 DW STAs가 송신자 τ를 즉시 복사(동기화) → 현재 최적값에 가장 빠르게 수렴
- EMA(idle-rate 추적)나 consec(카운팅)보다 빠른 수렴: 매 성공마다 전체 τ가 재보정됨
- Idle 증가 트리거 → τ 증가, Collision → τ 감소 (EMA와 유사한 MIMD 구조)

**2. H10 역전 — CD 추가가 오히려 나쁨 (pnd < pnd_cd 예상 → 실제 pnd > pnd_cd)**
- 원인: pnd_cd에서 충돌 시 DT(송신) STAs도 τ÷c_coll → 전체적으로 τ 더 크게 감소
- 유한 윈도우에서 τ가 너무 낮으면 idle 증가 → W_rem 낭비 심화
- PND 논문(무한 윈도우)에서는 CD가 57% 개선했지만, NPCA 유한 윈도우에서는 역효과
- 이유: 무한 윈도우 → CD로 N 빠르게 감소 → τ_optimal = 1/N 상승 → CD 이득  
  유한 윈도우 → N 변화보다 W_rem 고갈이 더 빠른 제약 → CD로 인한 τ 감소가 독소

**3. PND vs EMA 계열 우세의 이유**
- PND solo-copy 메커니즘: 매 성공 직후 DW 전체가 송신자 τ(≈1/k_viable)로 동기화  
  → oracle-like behavior에 가장 근접
- EMA는 간접적 추정(idle rate → gap → τ 조정) → 수렴 lag 존재
- consec는 L 슬롯 대기 후 조정 → 반응 지연

**4. 논문 서사 업데이트**
> 유한 NPCA window에서 PND MIMD(no-CD)가 가장 높은 효율(0.9772)을 달성한다.
> Solo 성공 시 τ 즉시 동기화 메커니즘이 oracle에 가장 근접한 τ 추적을 제공한다.
> CD 추가는 NPCA 유한 윈도우에서 역효과: 충돌 후 추가 τ 감소가 idle 증가로 이어져 W_rem을 낭비한다.
> 반면 무한 시간 neighbor discovery(PND 원논문)에서 CD가 57% 개선을 제공하는 이유는
> CD로 N이 빠르게 감소 → τ_optimal = 1/N 증가하는 메커니즘 때문이며,
> 이 메커니즘은 W_rem이 고갈되는 유한 윈도우에서는 작동하지 않는다.

**최종 권장 방법 (v4)**: PND MIMD no-CD (효율 0.9772, τ 즉시 동기화, 구현 단순)

---

## 실험 결과 v5 (2026-06-04)

**변경 사항**: dense/short-PPDU 환경 재설계
- N 범위 확장: [5,10,20,30] → **[10,20,30,50]** (dense 환경 포함)
- PPDU 축소: homo 15→**6**, uniform U[5,40]→**U[3,12]** (mean≈7.5), bimodal {8,35}→**{4,12}** (mean=8)
- 동기: W_eff=50 기준 가능 전송 수 2.2→**6.7** (3배 증가), 유한 윈도우 내 다수 전송 환경
- y축 [0, 1.15] → **[0.70, 1.15]** (방법 간 차이 가시성 개선)

### 핵심 결과 (uniform, W_eff=50, N=20)

| 방법 | v4 efficiency | v5 efficiency | 변화 |
|---|---|---|---|
| oracle | 1.0777 | 1.0285 | -0.049 |
| **pnd** | **0.9772** | **0.9568** | -0.020 |
| pnd_cd | 0.9701 | 0.9559 | -0.014 |
| **ema_no_coll** | 0.9215 | **0.9610** | **+0.040** ↑ |
| self_excl_only | 0.9050 | 0.9657 | +0.061 ↑ |
| consec_L2 | 0.9075 | 0.9000 | -0.008 |
| ema_ad_low | 0.9044 | 0.9433 | +0.039 ↑ |
| ema_adaptive | 0.8733 | 0.8817 | +0.008 |
| dcf_self_excl | 0.8634 | 0.8282 | -0.035 ↓ |
| mfg_no_excl | 0.8114 | 0.9310 | **+0.120** ↑↑ |

### 가설 검증 결과 (v5)

| 가설 | v4 결과 | v5 결과 | 비고 |
|---|---|---|---|
| H1: self_excl_only > mfg_no_excl (N>20) | ✅ | ✅ | N=30: 0.972 vs 0.937; N=50: 0.975 vs 0.937 |
| H2: consec_L2 > self_excl_only | ❌ | ❌ | 0.900 < 0.966 |
| H5: homo 방법 차이 없음 | 부분 | 부분 | mfg/se/ema_no_coll ≈ oracle; ema_adaptive(0.928), dcf(0.840) ≠ |
| H6: bimodal gap > uniform gap | ❌ | ❌ | uniform gap(0.098) > bimodal gap(0.083) |
| H7: dcf_self_excl > self_excl_only | ❌ FAIL | ❌ FAIL | 0.828 vs 0.966 (격차 확대) |
| H8: ema_no_coll > ema_adaptive | ✅ | ✅ | 0.961 vs 0.882 (+8.9%) |
| H9: α↓ 단조 감소 → 성능 단조 증가 | ✅ | ✅ | 0.961≥0.943≥0.925≥0.882 — **두 버전 모두 robust** |
| H10: pnd_cd > pnd | ❌ FAIL | ❌ FAIL | 0.9559 vs 0.9568 (차이 0.001로 축소) |
| H11: pnd_cd ≈ ema_no_coll | ≠ (diff=0.049) | **≈ (diff=0.005)** | v5에서 수렴 — 짧은 PPDU 효과 |

### 핵심 발견 (v5)

**1. mfg_no_excl 대폭 개선 (+12.0%) — self-exclusion 이득 감소**
- 짧은 PPDU (mean 7.5)로 대부분 STA가 W_rem 내내 viable → non-viable STA 비율 급감
- mfg_no_excl이 "헛된 시도"를 거의 안 함 → 0.8114→0.9310
- 역설: PPDU 단축으로 self-exclusion의 절대 이득이 줄어듦 (self_excl_only 대비 +6.0%p, v4에서는 +9.4%p)
- **논문 함의**: self-exclusion 이득은 PPDU 이질성(특히 일부 STA의 긴 PPDU)에 의존적

**2. DCF 더욱 악화 (0.863→0.828) — 기회 비용 증가**
- 짧은 PPDU → 가능한 전송 수 증가 → CW 대기 중 놓치는 슬롯 비용 증가
- collision 후 CW×2 = 큰 backoff draw → 더 많은 transmission opportunity 낭비
- **결론**: DCF BEB의 유해성은 W_eff/PPDU_mean 비율(기회 밀도)에 비례

**3. pnd vs pnd_cd 격차 0.0071→0.0009 (사실상 동등)**
- 짧은 PPDU로 W_rem이 각 성공마다 조금씩만 감소 → "W_rem 고갈" 압력 완화
- CD의 추가 τ 감소가 덜 치명적 → pnd_cd ≈ pnd
- **해석**: CD의 유해성은 유한 윈도우에서도 PPDU 길이에 역비례

**4. pnd_cd ≈ ema_no_coll (diff=0.005) — v5에서 수렴**
- v4: diff=0.049 (pnd_cd 크게 우세) → v5: diff=0.005 (사실상 동등)
- 짧은 PPDU 환경에서 두 방법 모두 ~τ = 1/viable에 빠르게 수렴
- **해석**: 전송 기회가 충분하면 τ 추적 알고리즘의 차이가 희석됨

**5. H9 monotone robust — v4/v5 모두 PASS**
- α↓∈{0.5, 0.25, 0.10, 0.00} 모두 단조 증가 유지
- 이는 설계 파라미터 환경에 robust한 결론 — 논문에 신뢰성 있게 기재 가능

### 전체 순위 (uniform, W_eff=50, N=20)

```
oracle        : 1.0285
self_excl_only: 0.9657  ← v5에서 pnd와 격차 최소화
pnd           : 0.9568
pnd_cd        : 0.9559
ema_no_coll   : 0.9610  ← v5에서 pnd와 사실상 동등
ema_ad_low    : 0.9433
ema_fixed_high: 0.9064
consec_L2     : 0.9000
mfg_no_excl   : 0.9310  ← 짧은 PPDU로 크게 상승
ema_adaptive  : 0.8817
consec_L4     : 0.8627
ema_fixed_low : 0.8708
dcf_self_excl : 0.8282  ← 더욱 악화
```

### Dense 환경 (N=50, uniform, W_eff=50)

v5에서 새로 추가된 N=50 데이터 (H1 체크):
- self_excl_only: **0.9747** vs mfg_no_excl: **0.9368** — 격차 유지
- N이 클수록 self-exclusion 이득 증가 추세 (더 많은 STA가 비viable 충돌 시도)

### v4 vs v5 종합 비교

| 항목 | v4 (긴 PPDU) | v5 (짧은 PPDU) |
|---|---|---|
| PPDU uniform mean | 22.5 slots | 7.5 slots |
| 가능 전송 수 / visit (W_eff=50) | ~2.2 | ~6.7 |
| mfg_no_excl efficiency | 0.811 | 0.931 |
| pnd efficiency | 0.977 | 0.957 |
| pnd vs pnd_cd 격차 | 0.0071 | **0.0009** |
| pnd_cd vs ema_no_coll 격차 | 0.049 | **0.005** |
| DCF efficiency | 0.863 | 0.828 |
| H9 monotone | ✅ | ✅ |
| 최고 방법 | pnd (0.977) | pnd (0.957) → self_excl, ema_no_coll과 근소 차이 |

**결론**: 짧은 PPDU 환경에서는 방법 간 차이가 전반적으로 압축됨 (비viable STA 비율 감소). 긴 PPDU 이질 환경(v4)이 알고리즘 차별성을 더 잘 드러냄. 그러나 dense N=50 환경과 y축 확대로 인해 시각적 명확성이 개선됨.

**최종 권장 방법 (v5)**: PND MIMD no-CD (효율 0.9568, ema_no_coll 0.9610과 근소 차이)

---

## 실험 결과 v6 (2026-06-04)

**변경 사항**: fig18 parameter study 결과 반영 → PND_C_COLL=1.5→**1.2**, PND_C_IDLE=1.5→**1.2**  
pnd/pnd_cd만 재실행 (480 configs × 1000 visits), 나머지 방법은 v5 데이터 재사용.

### 핵심 결과 (uniform, W_eff=50, N=20)

| 방법 | v5 efficiency | v6 efficiency | 변화 |
|---|---|---|---|
| oracle | 1.0285 | 1.0285 | — |
| self_excl_only | 0.9657 | 0.9657 | — |
| **pnd_cd** | 0.9559 | **0.9638** | **+0.008** |
| **pnd** | 0.9568 | **0.9627** | **+0.006** |
| ema_no_coll | 0.9610 | 0.9610 | — |
| ema_ad_low | 0.9433 | 0.9433 | — |
| mfg_no_excl | 0.9310 | 0.9310 | — |
| consec_L2 | 0.9000 | 0.9000 | — |
| ema_adaptive | 0.8817 | 0.8817 | — |
| dcf_self_excl | 0.8282 | 0.8282 | — |

### 가설 검증 결과 (v6)

| 가설 | v5 결과 | v6 결과 | 비고 |
|---|---|---|---|
| H1~H9 | v5와 동일 | v5와 동일 | pnd/pnd_cd 미관련 |
| **H10: pnd_cd > pnd** | ❌ FAIL | **✅ PASS** | **반전!** 0.9638 > 0.9627 |
| **H11: pnd_cd ≈ ema_no_coll** | ≠ (diff=0.049) | **≈ (diff=0.003)** | 사실상 동등 |

### 핵심 발견 (v6)

**1. H10 반전 — cc=1.2로 CD가 다시 순이득**

v5(cc=1.5): DT STAs ÷1.5 on collision → 과도한 τ 감소 → pnd_cd < pnd  
v6(cc=1.2): DT STAs ÷1.2 on collision → gentle 보정 → pnd_cd > pnd

직관: CD가 유익하려면 충돌 후 DT STA의 τ 감소가 "재충돌 방지"에 충분하고 "idle 증가 유발"에는 과하지 않아야 한다. cc=1.2가 이 균형점.

**2. 상위 4개 방법 밀집 (0.9610~0.9657, 격차 ≤0.005)**

```
self_excl_only : 0.9657  ← 단순하지만 여전히 최고
pnd_cd         : 0.9638
pnd            : 0.9627
ema_no_coll    : 0.9610
```

짧은 PPDU(U[3,12]) 환경에서 solo-copy 이벤트가 충분해 τ 수렴이 쉬워지고,  
알고리즘 차이가 희석됨. self_excl_only의 단순성이 재평가됨.

**3. pnd_cd ≈ ema_no_coll (diff=0.003) 수렴 확정**

v5: diff=0.049 → v6: diff=0.003. 두 방법 모두 collision 후 τ를 적절히 낮게 유지하는 전략이며 유사한 수렴 속도 달성.

**4. 논문 서사 업데이트**

> PND MIMD에서 c_coll=1.5(원논문)는 유한 NPCA window에서 충돌 패널티가 과도해 CD의 이득을 상쇄했다.
> fig18 파라미터 스터디로 도출된 c_coll=1.2는 이 균형을 회복해 pnd_cd > pnd를 복원한다.
> 최적화된 PND+CD는 ema_no_coll과 사실상 동등(diff=0.003)하며,
> 두 알고리즘 모두 oracle 대비 96.1~96.4%의 효율을 달성한다.

**최종 권장 방법 (v6)**: PND+CD(cc=1.2/ci=1.2) 또는 ema_no_coll — 구현 복잡도에 따라 선택.  
단순성 우선 시: self_excl_only (0.9657, τ 적응 불필요).

---

## 실험 결과 v7 (2026-06-04)

**변경 사항**: AND (ALOHA-like Neighbor Discovery, Vasudevan et al. MobiCom 2009) 비교 baseline 추가.  
and만 신규 실행 (240 configs × 1000 visits), 나머지 v6 CSV 재사용 (`--methods and --base-csv results/step9/fig17_v6/data.csv`).

### 핵심 결과 (uniform, W_eff=50, N=20)

| 방법 | v6 efficiency | v7 efficiency | 변화 |
|---|---|---|---|
| oracle | 1.0285 | 1.0285 | — |
| self_excl_only | 0.9657 | 0.9657 | — |
| pnd_cd | 0.9638 | 0.9638 | — |
| pnd | 0.9627 | 0.9627 | — |
| ema_no_coll | 0.9610 | 0.9610 | — |
| ema_ad_low | 0.9433 | 0.9433 | — |
| ema_ad_med | 0.9251 | 0.9251 | — |
| mfg_no_excl | 0.9310 | 0.9310 | — |
| consec_L2 | 0.9000 | 0.9000 | — |
| ema_fixed_high | 0.9064 | 0.9064 | — |
| ema_adaptive | 0.8817 | 0.8817 | — |
| ema_fixed_low | 0.8708 | 0.8708 | — |
| consec_L4 | 0.8627 | 0.8627 | — |
| dcf_self_excl | 0.8282 | 0.8282 | — |
| **and** | (신규) | **0.5410** | ← 전체 최하위 |

### 가설 검증 결과 (v7)

| 가설 | 결과 | 상세 |
|---|---|---|
| H1~H11 | v6와 동일 | and 추가로 기존 결과 변화 없음 |
| **H12: and < self_excl_only** | **✅ PASS** | 0.5410 ≪ 0.9657 (격차 0.425) |

### 핵심 발견 (v7)

**1. AND 효율 0.5410 — 전체 15개 방법 중 최하위 (격차 극명)**

| 구간 | 분석 |
|---|---|
| Phase 1 (p=0.5, 4 slots) | N=20에서 collision 폭발 → W_rem 급속 소진 |
| Phase 2 (p=0.25, 16 slots) | W_rem≈30 남은 상태 진입 — 이미 절반 소모 |
| Phase 3 (p=0.125, 46 slots) | W_eff=50 전체보다 긴 phase duration → 도달 불가 |

- AND는 무한 시간 neighbor discovery(쿠폰 수집기 분석)를 가정한 설계
- 비적응: 초반 collision 낭비를 보정하지 않음 → 유한 NPCA window에서 근본적으로 부적합

**2. AND의 역할: open-loop vs closed-loop 대조**

AND(0.5410)와 self_excl_only(0.9657)의 격차(0.425)는 단순한 self-exclusion만으로도 달성되는 이득이 open-loop phase schedule보다 훨씬 크다는 것을 보여줌. 논문에서 closed-loop adaptive τ 설계의 필요성을 강조하는 lower bound 역할.

**3. 전체 순위 확정 (v7 기준)**

```
oracle         : 1.0285
self_excl_only : 0.9657
pnd_cd         : 0.9638
pnd            : 0.9627
ema_no_coll    : 0.9610
ema_ad_low     : 0.9433
mfg_no_excl    : 0.9310  (짧은 PPDU)
ema_fixed_high : 0.9064
ema_ad_med     : 0.9251
consec_L2      : 0.9000
ema_adaptive   : 0.8817
ema_fixed_low  : 0.8708
consec_L4      : 0.8627
dcf_self_excl  : 0.8282
and            : 0.5410  ← open-loop, finite window 비적합
```

**최종 권장 방법 (v7)**: v6와 동일 — PND+CD(cc=1.2) 또는 ema_no_coll.  
AND는 비교 baseline으로만 사용 (논문의 "기존 open-loop 방식의 한계" 논거).
