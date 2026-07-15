# Figure 24: τ Persistence (Warm-Start) Across Consecutive NPCA Transitions — Mixed Native/Visitor (v2)

**연구 질문 (RQ24)**:
모든 실험이 매 NPCA transition마다 visitor τ₀를 cold 리셋. fig21 mixed 모델은
심지어 τ₀=1/N_total — **non-primary 채널의 native 수까지 아는 genie**.
실제 visitor BSS는 기껏해야 자기 그룹 크기(N_visitor)만 알고, native 부하는
겪어봐야 안다. 동일 집단이 NPCA 채널을 반복 방문하므로, PACE의 τ는
실효 경쟁 수준(visitor+native)의 암묵 추정치 → **기억(warm-start carry)이
두 genie를 동시에 대체할 수 있는가?**

> v1(visitor-only)은 폐기됨 — native 없는 NPCA 채널은 비현실적이라는 판단.
> v2 = fig21 mixed 모델 위에 warm-start 실험 재구축.

**스크립트**: `harq_sim/run_step9_fig24.py`
**출력**: `manuscript/figure/fig24_warm_start.{eps,png,pdf}`
**데이터**: `results/step9/fig24/data.csv`

---

## 1. 모델 (fig21 상속)

- Visitor [0:10]: PACE (PND MIMD no-CD, cc=ci=1.2), PPDU U[3,12] 방문마다 재샘플
- Native [10:]: 표준 DCF (CW₀=N_total, ppdu=6), 매 방문 존재, DCF 상태는
  방문마다 재추출 (정상 배경 트래픽 근사). **τ carry는 visitor만.**
- Native가 solo 승리 → visitor τ 무갱신 (외부 이벤트, fig21 규칙)
- Collision: nocd (max L_i 슬롯), W_eff=50 고정
- Sequence = 50 연속 transition, 20 reps × 5 seeds = 100 sequences/config
- Efficiency = Σ visitor 성공 / Σ oracle visitor 성공 (ratio-of-means)

## 2. 비교 방법 (visitor가 방문 시작에 아는 것)

| method | τ₀ | 아는 것 |
|---|---|---|
| oracle | 1/\|viable(t)\| 매 슬롯 | 전지 (공동 공평 기준선) |
| cold_genie | 1/N_total 리셋 | visitor+**native 수** (fig21 현행) |
| cold_nv | 1/N_visitor 리셋 | 자기 그룹만 (현실적 cold 최선) |
| cold_high | 0.5 리셋 | 무지식 + 무기억 |
| warm_nv | 1/N_visitor → carry | 자기 그룹 + 기억 |
| warm_high | 0.5 → carry | 무지식 + 기억 |
| dcf_conv | (τ 없음) CSMA/CA, CW_min=16 BEB | 무지식 — **conventional NPCA (802.11bn 표준)** |

dcf_conv: visitor가 표준 DCF로 동작 (BEB CW 16→1023, 성공 시 방문 내 완료,
Min Duration Threshold = viable mask self-exclusion). native와 대칭 프로토콜.

Carry: 성공 visitor = solo 성공 직전 τ, 미성공 = 종료 τ. Churn ρ: 매 방문
visitor를 확률 ρ로 교체(carry τ를 init에서 재추출).

## 3. 결과 (full: 5 seeds × 20 reps × 50 transitions, 2026-07-15)

Steady-state visitor efficiency (transition ≥ 25, churn=0):

| method | nat0 | nat5 | nat10 | nat20 |
|---|---|---|---|---|
| oracle | 1.026 | 1.034 | 1.007 | 0.983 |
| cold_genie | 0.952 | 0.948 | 0.947 | 0.971 |
| cold_nv | 0.954 | 0.896 | 0.947 | **1.099** |
| cold_high | 0.132 | 0.150 | 0.162 | 0.201 |
| warm_nv | 0.755 | 0.786 | 0.861 | 1.025 |
| warm_high | 0.736 | 0.794 | 0.866 | 1.033 |
| dcf_conv | 0.992 | 0.652 | 0.597 | 0.609 |

util (N_nat=10): oracle util_n=0.151, cold_genie 0.117, cold_nv 0.052, warm 0.036.

### 판정

- **H2 PASS (핵심)**: warm_high 0.866 vs cold_high 0.162 = **×5.3**. 무지식
  init도 기억으로 근-genie 회복, ~5 transition 내 수렴. warm_nv ≈ warm_high
  전 구간 (기억이 init 지움 — init-free).
- **H1 부분**: warm_nv는 cold_nv를 역전 못함. 단 격차는 native 부하와 함께
  단조 축소: Δ = -0.20(nat0) → -0.07(nat20). 방향은 가설대로.
- **H3 방향 PASS, 크기 미달**: warm τ₀ 고정점 0.167(nat0) → 0.128(nat20) —
  native 부하를 감지해 하강하나 1/N_total(0.10→0.033) 대비 크게 인플레이션.
  원인 (v1보다 심함): ① 방문 종료 τ는 잔존 viable≪N 반영, ② native 승리
  슬롯은 visitor에게 외부 이벤트라 solo-copy 하향 보정 기회 없음.
- **H4 비대칭 재확인**: warm_nv는 churn↑ 시 개선(0.861→0.955; 교체=1/N_v
  재주입=인플레이션 리셋), warm_high는 ρ=0.1에서 0.50으로 급락(신규 τ=0.5
  오염).

### 해석 주의

- **efficiency > 1 가능**: oracle은 visitor-최대가 아니라 전 STA 공평 τ=1/k
  기준선. cold_nv nat20의 1.099는 공격적 τ₀=0.1이 native를 억압해 공평 몫
  이상을 탈취한 것 (util_n: oracle 0.15 vs cold_nv 0.05). visitor efficiency
  단독으론 공정성 은폐 — fig21 fairness 지표와 병행 해석 필요.
- cold_nv가 native 부하에도 잘 버티는 이유: DCF native(CW₀=30)는 약한
  경쟁자라 visitor 공격성이 페널티 대신 점유로 귀결. 공정성 희생의 결과.

### 채널 효율 & airtime 공정성 (panel e/f 추가, N_nat=10 steady)

| 지식 수준 | method | util_v | util_n | **total** | v_share | **prop** |
|---|---|---|---|---|---|---|
| (ref) | oracle | 0.392 | 0.151 | 0.543 | 0.72 | 1.44 |
| 전부 | cold_genie | 0.381 | 0.117 | **0.498** | 0.77 | **1.53** |
| 그룹만 | cold_nv | 0.380 | 0.052 | 0.431 | 0.88 | 1.76 |
| 무지식 | cold_high | 0.048 | 0.000 | **0.049** | 1.00 | (무의미) |
| 무지식+기억 | warm | 0.345 | 0.036 | 0.381 | 0.91 | 1.81 |
| 무지식 (표준) | dcf_conv | 0.242 | **0.155** | 0.397 | 0.61 | **1.22** |

prop = airtime visitor share / population share (fig21 proportionality의 airtime 판).

- **native-count 지식의 진짜 수혜자는 채널**: cold_genie vs cold_nv에서 visitor
  효율은 동일(0.947=0.947)하나 total +16%(0.50 vs 0.43), prop 1.53 vs 1.76 —
  지식이 없으면 그 대가를 native가 지불.
- **기억의 회복 한계**: warm은 "그룹 지식" 수준까지만 회복 (prop 1.81 ≈
  cold_nv 1.76, total 0.38 ≲ 0.43). carry 인플레이션 탓에 cold_nv만큼 공격적
  — native-count 지식까지는 대체 못함.
- **oracle조차 prop 1.44 > 1**: 구조적 편향 — visitor PPDU 길고(7.5 vs 6),
  native DCF는 idle 슬롯에만 backoff 감소라 W=50 창에서 본질적으로 느림.
  이 설정에서 prop=1은 달성 불가; oracle 1.44가 실질 공정 기준선.
- **cold_high는 채널 전체 파괴**: total 0.049, util_n≈0 — visitor 폭주 충돌이
  native까지 질식. 무지식+무기억의 피해는 자신에 국한되지 않음.
- **conventional NPCA (dcf_conv)**: native 없으면 0.99 (near-oracle) —
  BEB는 여유 창에서 충분. native 있으면 visitor 효율 0.60로 붕괴 (BEB
  sawtooth가 유한 창에서 슬롯 낭비, fig19 재현). 대신 **가장 공정**:
  prop 1.01(nat5)~1.65(nat20) — oracle(1.25~1.75)보다도 낮음, native 몫 최대
  (util_n 0.155 ≈ oracle 0.151). 즉 표준 NPCA는 fairness-우선, PACE는
  visitor-throughput-우선: warm_high가 dcf_conv 대비 visitor +45%
  (0.866 vs 0.597), 대가는 native 몫 1/4 (0.036 vs 0.155)과 prop 1.81 vs 1.22.
  → 논문 핵심 대비축: PACE-vs-conventional은 "효율 vs 공정" trade-off이며,
  PACE의 이득은 native 억압분을 포함 — fairness 보정(τ cap 등)이 future work.

### 논문 함의

1. **기억 = 지식 대체**: cold의 성능은 τ₀에 담긴 지식량에 비례
   (genie 0.95 / own-group 0.95 / 무지식 0.16). warm은 무지식에서 출발해도
   0.87 — cold 스펙트럼의 지식 축을 기억으로 대체.
2. carry 인플레이션은 잔존 한계 (-7~-9%p vs cold_nv at 현실 부하).
   종료 τ 대신 shrink/방문 초반 τ carry 보정 = future work.
3. 신규 진입 visitor의 init 품질에 민감 (churn 비대칭) — 한계 명시.
