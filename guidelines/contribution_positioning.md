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

---

## 7. Topology / 공간 overlap 논점 (limitation + 방어)

현 모델 = **single collision domain** (전 visitor가 전 native와 상호 carrier-sense). 부분 overlap이면 깨짐.

visitor가 BSS-2(native) overlap 여부로 갈림:

| | Class A (overlap 안) | Class B (BSS-1 assoc, BSS-2 coverage 밖) |
|---|---|---|
| native 감지/경쟁 | ✓ | ✗ (경쟁 상대 없음) |
| native 간섭 | ✓ | ✗ (reciprocity) |
| NPCA primary 체감 | 혼잡 | 빈 채널 (**spatial reuse**) |

함의:
- **Fig 21(full overlap) = worst case.** 현실 native 침식은 B 비율만큼 희석.
- Class B = native 무비용 reuse → "그냥 native 뺏는다" 비판 **완화 논리**.
- **local-sensing native-aware PACE가 두 부류 자동 처리**: 들리는 native만 셈 → B는 0(공격적·무해), A는 $N_n^{local}$(양보). geometry-emergent 정확성.

방어 framing: "보인 침식은 worst case(완전 overlap); 부분 overlap에선 non-overlap visitor가 interference-free reuse. local-sensing 확장이 자연히 활용." → future work.

부작용(hidden terminal): B가 native RX 범위엔 있고 TX는 안 들리면 → hidden 충돌 + 피드백 오염(false idle→τ↑). geometry 의존.

→ 본문 반영: System Model A에 "single collision domain / full-overlap = worst case" 가정 **명시**, partial overlap은 future work.

---

## 8. τ* 유도 — classical임, 과장 주의 (★ reviewer 리스크)

**τ*=1/N은 classical slotted ALOHA** (Abramson 1970; throughput $N\tau(1-\tau)^{N-1}$ argmax). **유도하면 textbook 재유도** → "이거 classical 아니냐" 공격.

→ **유도 X, 인용·STATE.** 기여는 공식이 아니라:
1. 유한창에서 **유효 경쟁자 수가 시변** ($\mathcal{V}(t)$가 deadline 소진 + viability 탈락 두 경로로 감소)
2. 따라서 최적이 **고정 1/N 아니라 시변 $1/|\mathcal{V}(t)|$** → BEB(역방향)·AND(open-loop) 못 따라감
3. $|\mathcal{V}(t)|$ **모른 채 분산 추적**

주의: $1/|\mathcal{V}(t)|$ = **per-slot greedy 최적**, 전역 window-optimal **증명 아님** (이질 $L_i$ 스케줄링은 별개; sim oracle은 탐색으로 구해 미세하게 다름). → "window-optimal 유도" 주장 **금지**. "per-slot success-maximizing (classical), near-optimal는 sim 검증."

`subsec C(viability $\mathcal{V}(t)$)는 삭제 불가` — τ*·self-exclusion 토대이자 **시변성 통찰의 핵심 재료**.

분석 깊이: **레벨 2** (closed-form 최적 인용 + fixed-point 1단락[PND 인용] + 시뮬 수렴 검증 + pseudocode). **full 수렴증명 안 함** (finite-horizon 시변 MIMD = 험악, hand-wavy 위험). PND도 algorithm-form + 시뮬이었음.

---

## 9. "track" 과장 금지 — 실시간 추적 불가 (★)

$|\mathcal{V}(t)|$는 **실시간 측정·카운트 불가** (몇 명 viable인지 못 셈). 따라서 "track 1/|V(t)| in real time"은 **과장**.

PACE 실제 = **암묵적 근사 추적**. 전역 미지 → 국소 신호 변환:
- MIMD가 **idle/collision 통계로 operating point 추론** (카운트 아님; Cali et al. dynamic-tuning)
- **solo-copy = 빠른 consensus** (성공자 τ 복사 → 한 관측에 점프)
- **idle-increase가 population 감소 자동 추종** (탈락→idle↑→τ×)
- **self-exclusion은 순수 로컬** (내 $L_i$ vs $W_\mathrm{rem}$, 남 카운트 불필요)

정당한 한계: **transient lag** — MIMD 수 슬롯 적응 → 급변 못 맞춤. **tight window는 부분 수렴(근사)**. → 그래서 oracle > PACE 약간, gain<100%.

표현 수정:
- ~~"track the time-varying optimum in real time"~~ → **"drive toward / approximately track using only local feedback, without knowing the contender count"**
- $|\mathcal{V}(t)|$ 명시 추정 안 함 = **강점**으로 (Contribution의 "without knowledge of total station count"와 일관)
- 추적 충실도 = **시뮬 검증** (Fig 15 adaptive≈oracle), 이론적 완전추적 주장 X

| 주장 | 가능? |
|---|---|
| $\tau^*=1/n$ per-slot 최적 | ✓ classical 인용 |
| 유한창 시변 $1/\|\mathcal{V}(t)\|$ | ✓ viability 통찰 |
| 정확·실시간 추적 | ✗ 과장 |
| 국소 피드백 근사 추적, near-oracle | ✓ 시뮬 |

---

## TODO (본문 반영 대기)
- [ ] Introduction Contribution #1: "derive optimal τ" → "시변성·viability 통찰 + 카운트 없는 분산 근사 추적"으로 reframe
- [ ] Sec 4 "Throughput-Optimal Target": τ* classical 인용(STATE), window-optimal 주장 제거, "per-slot greedy + sim near-optimal"
- [ ] "track" → "drive toward / approximately track" 전역 수정
- [ ] System Model: single collision domain / full-overlap=worst-case 가정 명시
- [ ] slotted model 배치: 관측부(idle/success/collision)=System Model, τ-attempt 규칙=PACE Algorithm (τ-per-slot은 ALOHA식, 802.11 메커니즘 아님)
- [ ] subsec C 유지, fixed-point 단락 + pseudocode 작성
- [ ] native-aware PACE (local-sensing) 변형 = future work 또는 추가 실험
