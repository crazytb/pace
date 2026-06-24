# PACE Contribution Positioning (Native vs Visitor 논지)

논문 Introduction / Performance Evaluation(Fig 21) / Discussion 작성 시 활용.
핵심: PACE는 깨끗한 상위호환이 **아니다**. native airtime을 trade한다. 이걸 정직하게 분리·정량화해서 reviewer의 "그냥 native 뺏는 거 아니냐" 공격을 방어한다.

---

## 1. 경쟁 구도 (Fig 21(c), W_eff=50, N_visitor=10, N_native=10, full 1000 visits)

| 기법 | visitor util | native util | **total Σ** | native 보존율 |
|---|---|---|---|---|
| native-only (visitor 없음) | 0 | **0.612** | 0.612 | 1.00 |
| **pnd (PACE)** | 0.593 | 0.162 | **0.755** | 0.265 |
| oracle | 0.555 | 0.206 | 0.761 | 0.336 |
| ema_ad_low | 0.498 | 0.235 | 0.733 | 0.384 |
| consec_L2 | 0.396 | 0.308 | 0.704 | 0.502 |
| dcf_self_excl (표준 baseline) | 0.354 | 0.293 | 0.646 | 0.478 |
| and (open-loop NPCA) | 0.601 | 0.027 | 0.628 | 0.043 |

native-only baseline은 fig21 data.csv에 `method=native_only`로 저장됨 (N_native별, W_eff별).

---

## 2. 두 비교 축으로 분리 (핵심)

### 축 A — vs AND (phase-based open-loop NPCA) → **명확한 dominance**
PACE visitor 0.593 ≈ AND 0.601 (동급)인데 native 0.162 ≫ 0.027, total 0.755 ≫ 0.628.
→ visitor·native·total **전부 우위/동급**. "기존 open-loop NPCA 접근보다 확실히 낫다" = 참.

### 축 B — vs DCF-visitor (표준 baseline) → **Pareto-incomparable (trade-off)**
PACE: visitor↑, total↑, **native↓** (0.293→0.162).
→ 상위호환 아님. frontier 위 다른 점 (visitor 쪽으로 치우침).

---

## 3. 결정적 reframe 2가지

### (A) native 침식은 PACE가 아니라 NPCA가 한다
표준 baseline(DCF-visitor)도 native를 0.612→0.293 (**−52%**) 침식.
visitor가 native 채널 쓰는 순간 발생하는 **NPCA 고유 비용**. PACE가 도입한 게 아님.
PACE는 그 위에서 **visitor 경쟁 효율만** 바꾼다.

### (B) PACE가 total 채널 활용을 가장 많이 올린다
native-only 0.612 → **PACE 0.755 (+23%)**. DCF-visitor는 0.646 (+6%)뿐.
→ 놀던 airtime을 **PACE가 가장 많이 회수.** 스펙트럼 효율 최고.
문제는 회수분의 **배분**이 visitor 쏠림 (fairness가 아니라 allocation 문제).

---

## 4. Contribution 포지셔닝 (옵션 2+3 결합 권장)

1. **핵심 기여 = finite-window 최적 visitor contention.**
   선행연구 누구도 depleting window + frame viability에 τ를 적응시키지 않음 (novel·미해결).
   + total airtime 최대 회수 (스펙트럼 효율).

2. **native 침식을 숨기지 말고** Fig 21로 정직하게 제시 + "NPCA 고유 비용" reframe (3-A).
   Fig 21 = trade-off 정량화로 포지셔닝.

3. **native-aware PACE 변형 추가 (★ 가장 강한 보강).**
   현재 약점 근원 = τ*=1/|V_visitor|로 **visitor만 셈**, native 무시 → 과공격.
   수정: native 활동까지 관측 → **τ* = 1/(N_v + N̂_native)** → 덜 공격적 → native 보존 회복하면서 total 유지.
   → frontier에서 fairness 쪽으로 이동하는 knob. "efficient AND coexistence-aware NPCA" = 결정적 차별성.
   → reviewer의 native-unfairness 공격 선제 방어.
   → **TODO: 실제 구현·시뮬해서 Fig 21에 점 추가.** 되면 contribution 결정적으로 강해짐.

---

## 5. 한 줄 메시지

> "확실한 우위"는 (a) open-loop NPCA(AND) 대비 = 명확, (b) 스펙트럼 total 효율 = 명확.
> (c) native fairness 대비 DCF = trade이나, native-aware 변형으로 메울 수 있음.

이 셋을 정직하게 분리·정량화하면 "그냥 native 뺏는 것" 비판으로 죽이기 어렵다.

---

## 6. 관련 파일
- 데이터: `results/step9/fig21/data.csv` (native_only row 포함)
- 그림: `manuscript/figure/fig21_native_visitor_fairness.{eps,png,pdf}` (panel c zoom inset + native-only reference line)
- 스크립트: `harq_sim/run_step9_fig21.py`
- 기존 한계 명시: `CLAUDE.md` Fig 21 절 (native_preservation=0.265)
