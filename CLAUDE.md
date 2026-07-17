# PACE: NPCA 논문 프로젝트

## 프로젝트 개요

IEEE 802.11bn **NPCA(Non-Primary Channel Access)** 저널 논문용 시뮬레이션 코드.

**논문**: `manuscript/pace.tex`  
**제목**: PACE: Probabilistic Adaptive Contention Control for Non-Primary Channel Access in IEEE 802.11bn  
**기반 알고리즘**: PND (Song et al., IEEE 2014) → NPCA 전환 결정에 적용

> LLM-DRL 방향(vNPCA 저널)은 별도 논문으로 분리됨.

---

## 핵심 알고리즘: PACE

### 원래 PND (Song et al., 2014)

무선 애드혹 이웃 발견: N개 디바이스가 각자 광고 메시지 전송확률 τ를 MIMD로 조정 → τ → 1/N 수렴.

| 이벤트 | 규칙 |
|---|---|
| 자신이 TX | τ 유지 (half-duplex, 수신 불가) |
| Solo RX (`\|DT\|=1`) | 송신자의 τ 복사 (solo-copy) |
| Collision RX (`\|DT\|≥2`) | τ /= c_coll (과추정 → 감소) |
| Idle (`\|DT\|=0`) | τ *= c_idle (과추정 → 증가) |

CD 변형: 성공 TX 후 디바이스 탈퇴 → 경쟁 디바이스 감소 → τ* 상향 추적.

### NPCA 적응: PACE

**컨텍스트 차이**: 이웃 발견(무한 시간) → NPCA 방문(유한 창 W_eff 슬롯).

MIMD 규칙은 동일하게 적용. 핵심 추가:

**PPDU-aware self-exclusion**: `ppdu_i > W_rem` 이면 τ_i = 0 (비실행 STA 침묵)
- 원래 PND의 CD(성공 후 탈퇴)와 동일한 역할
- W_eff 창에서 실패 확정 STA가 슬롯 낭비 방지

**파라미터**: c_coll = 1.2 (fig17 최적 튜닝), c_idle = 1.5

### 이론적 최적해

NPCA 방문에서의 throughput-optimal τ*(t) (classical slotted ALOHA 확장):

```
τ*(t) = 1 / remaining(t)
```

- `remaining(t)` = 현재 슬롯부터 W_eff 끝까지 남은 실행 가능 STA 수 추정
- DCF BEB 문제: 충돌 → CW×2 → backoff 카운트가 W_rem 초과 → 슬롯 낭비
- PACE는 solo-copy를 통해 τ → τ*(t) 근사 수렴 (중앙 조정 불필요)

---

## 시뮬레이션 결과 요약

> **⚠️ Collision-cost 모델 변경 (base = no-CD)**: fig17/19/20/21은 collision 비용을 **최장 충돌 프레임 길이**(`max L_i`, 표준 802.11 CSMA/CA)로 청구하도록 재생성됨. 이전엔 collision=1 slot(ideal CD). 코드: `run_step9_fig17.py`의 `COLLISION_MODE`(`nocd`/`cd`/`cn`) + `collision_cost()`. 재생성본 파일명 = trailing `_` (원본 보존). fig15는 unit-frame 모델이라 무관, fig22는 `cd` pin, fig23은 cd/cn/nocd 비교 전용.
>
> **핵심**: no-CD에서 절대 util ~40%↓지만 **PACE 상대 우위 전부 보존** — fig17 pnd>dcf 순서, fig20 Pareto-dominant(TP+Jain), fig21 total +18%. 즉 PACE 이득은 cheap-collision 가정의 artifact 아님(robustness). 아래 표는 **no-CD 재생성 수치**(괄호 안 = 이전 CD 값).

### Fig 15: throughput-optimal vs DCF (tight window regime)

| 조건 | adaptive-optimal 대비 DCF |
|---|---|
| N=10, W_eff=20 | **+70.4%** |
| N=20, W_eff=20 | **+67.1%** |
| W_eff >> N | ≈ 0% (DCF 회복) |

핵심: W_eff ≤ N 구간에서 τ*(t)=1/remaining이 BEB 압도.

### Fig 17: PPDU 이질 환경 효율성 순위 (U[3,12], W_eff=50, N=20)

| 방법 | W_eff 효율 |
|---|---|
| oracle | 1.029 |
| self_excl_only | 0.966 |
| pnd_cd (c_coll=1.2) | **0.964** |
| pnd (no-CD) | 0.963 |
| ema_no_coll | 0.961 |
| dcf_self_excl | 0.828 |
| and (open-loop) | 0.541 |

### Fig 19: DCF benchmark 비교 (N=20)

- PACE(pnd) vs DCF: **+16%** 효율
- AND(phase-based open-loop): 유한 창에서 붕괴 (0.541 vs 0.828)
- τ trajectory: DCF = sawtooth (CW reset), PACE = oracle 추적

### Fig 20: Throughput–Fairness (bimodal {4,12}, W_eff=50, N=20)

no-CD 재생성 (괄호=이전 CD):

| 방법 | Throughput | Jain's J |
|---|---|---|
| pnd | **0.457** (0.772) | **0.178** (0.263) |
| dcf_self_excl | 0.430 (0.669) | 0.165 (0.228) |
| and | 0.000 (0.409) | — (0.155) |

PACE = **Pareto-dominant 유지** (H4 PASS): TP 0.457>0.430 AND Jain 0.178>0.165. AND는 nocd에서 붕괴(TP≈0).

### Fig 21: Native vs Visitor Fairness (bimodal {4,12}, W_eff=50, N_visitor=10, N_native=10)

no-CD 재생성 (괄호=이전 CD):

| 방법 | util_v | util_n | util_t | prop |
|---|---|---|---|---|
| oracle | 0.383 (0.555) | 0.146 (0.206) | 0.530 | 1.38 (1.39) |
| pnd | **0.377** (0.593) | 0.118 (0.162) | **0.496** | **1.44** (1.51) |
| dcf_self_excl | 0.228 (0.354) | **0.192** (0.293) | 0.420 | 0.99 (1.00) |

- **PACE total +18%** (0.496 vs dcf 0.420) — CD(+17%) 대비 유지. visitor util pnd +65% (0.377 vs 0.228).
- native tradeoff 동일: pnd nat_pres=0.271, dcf=0.439 (dcf가 native에 공정). 논문 한계로 명시.
- AND: nocd에서 붕괴 (util_v≈0).
- pnd_cd는 METHODS_21에서 제외됨(plotted set = oracle/pnd/dcf).

### Fig 22: Initial τ₀ sensitivity (PND Fig 2 대응, uniform PPDU U[3,12])

RQ: PND은 무한 horizon에서 init τ₀ 무관 (randomized≈optimal, 빠른 수렴). PACE 유한 창 W_eff에서도 성립?

| init τ₀ | W20(tight) | W100 | W200(loose) | 판정 |
|---|---|---|---|---|
| optimal 1/N | 0.955 | 0.972 | 0.996 | ✅ near-oracle |
| rand U(0,2/N) | 0.976 | 0.965 | 0.995 | ✅ near-oracle |
| low 1/W_eff | 0.952 | 0.932 | 0.985 | ✅ near-oracle |
| high 0.5 | 0.551 | 0.837 | 0.970 | ⚠️ 붕괴→창 늘면 회복 |
| **rand U(0,1)** | 0.453 | 0.381 | 0.493 | ❌ 전 구간 붕괴 |

핵심: **init 무관성은 scale~1/N일 때만 성립** (randomization 자체는 무해 — PND Fig 2 재현). init mean≫1/N이면 붕괴.
- fixed high(0.5): 전 STA 동시 c_coll 하강 → coordinated recovery (창 크면 회복)
- rand U(0,1): high-τ tail이 solo success 차단 → solo-copy consensus 실패 → mean τ 고착 → 회복 불가
- 유한 창 = 붕괴 증폭 (τ 낮출 시간 없음, W20 최악)

→ PND의 "init 무관" 주장은 scale-matched 조건부. PACE 유한창+이질 PPDU가 infinite-horizon 분석이 숨긴 regime 노출.

### Fig 23: Collision-cost model — ideal CD vs no-CD (bimodal {4,12}, W_eff=50)

RQ: 시뮬레이터는 collision을 1 slot으로 청구 (ideal CD / RTS-CTS식 짧은 경쟁 전제). CD 없으면 collision = colliding STA들의 max L_i (success와 유사). CD 가정이 결과에 얼마나 영향? PACE 이득 유지되나?

| N | CD PACE/DCF | no-CD PACE/DCF |
|---|---|---|
| 10 | 1.161 | 1.104 |
| 20 | 1.154 | 1.064 |
| 30 | 1.156 | 1.041 |
| 50 | 1.162 | 1.037 |

- **PACE는 두 regime 모두 DCF 이김** (역전 아님 — 초기 1-seed 반전은 노이즈였음).
- CD면 +15~16% 안정, no-CD면 +4~10%로 축소 (N↑ 하면 1로 수렴).
- collision rate (no-CD): pnd 0.24~0.27 > dcf 0.16~0.17, oracle 0.28, **and ~1.0(붕괴)**. PACE(τ*=1/viable)는 success 확률 최대화하려 충돌 감수 → 비싼 충돌에 더 손해 → 이득 축소.
- 전 기법 util 급락: oracle/pnd 0.77→0.47, dcf 0.66→0.44, and→0.

핵심: **1-slot collision 가정은 PACE에 보수적이 아니라 유리(advantage 증폭).** τ*=1/|V|가 optimal인 건 충돌 쌀 때만. **PACE는 RTS/CTS식 짧은 경쟁(충돌≈1slot)을 본질적으로 전제** — 논문에 assumption 명시 필요. 단 PACE 우위는 pessimistic no-CD에서도 살아남음(robustness).

### Fig 24: τ warm-start across NPCA transitions — mixed native/visitor (v2, nocd)

RQ: 매 transition마다 visitor τ₀ cold 리셋은 지식 가정 내포 — fig21은 τ₀=1/N_total(**native 수까지 아는 genie**). 실제 visitor는 자기 그룹(N_v)만 앎. 동일 집단 반복 방문 → 이전 방문 τ carry(warm)가 genie 대체하나? 모델 = fig21 mixed(visitor 10 PACE + native DCF ppdu=6, N_nat∈{0,5,10,20}), V=50 연속 transition, visitor만 carry(성공=성공 직전 τ, 미성공=종료 τ), native solo 승리는 visitor τ 무갱신(외부 이벤트). 효율 = Σvisitor succ/Σoracle succ (ratio-of-means). v1(visitor-only)은 비현실적이라 폐기.

Steady-state visitor efficiency (transition≥25, churn=0):

| method | nat0 | nat10 | nat20 |
|---|---|---|---|
| cold_genie (1/N_total) | 0.952 | 0.947 | 0.971 |
| cold_nv (1/N_visitor) | 0.954 | 0.947 | 1.099 |
| cold_high (0.5, 무기억) | 0.132 | 0.162 | 0.201 |
| warm_nv / warm_high | ~0.75 | ~0.86 | ~1.03 |

- **핵심 (H2)**: warm_high 0.866 vs cold_high 0.162 = **×5.3** — 무지식 init도 기억으로 ~5 transition 내 근-genie 회복. warm_nv ≈ warm_high 전 구간 (기억이 init 지움 = init-free).
- **H1 부분**: warm은 cold_nv 역전 못함, 단 격차 native 부하와 단조 축소 (Δ -0.20 nat0 → -0.07 nat20).
- **carry 인플레이션 (v1보다 심함)**: warm τ₀ 고정점 0.128~0.167 (1/N_total 0.033~0.10 대비 상방). 원인: 종료 τ = 잔존 viable≪N 반영 + native 승리 슬롯은 solo-copy 보정 기회 없음. shrink 보정 = future work.
- **churn 비대칭 (H4)**: warm_nv는 churn↑ 개선(교체=1/N_v 재주입=인플레이션 리셋), warm_high는 ρ=0.1에서 0.50 급락(신규 τ=0.5 오염 — fig22 메커니즘).
- **주의**: efficiency>1 가능 — oracle은 전 STA 공평 기준선. cold_nv nat20의 1.099는 native 억압에 의한 탈취 (util_n: oracle 0.15 vs cold_nv 0.05). fairness는 fig21 지표 병행.
- **채널 효율/공정성 (panel e/f, nat10)**: total util = oracle 0.54 / genie 0.50 / nv 0.43 / warm 0.38 / high 0.05, airtime prop = 1.44/1.53/1.76/1.81/무의미. ① native-count 지식의 수혜자는 visitor가 아니라 **채널**(visitor 효율 동일, total +16%, native 몫 2배) ② 기억은 그룹-지식 수준까지만 회복(prop≈cold_nv) ③ oracle도 prop 1.44>1 = 구조적(visitor PPDU 길고 native DCF는 idle에만 backoff 감소) — prop=1 달성 불가, oracle이 실질 공정 기준 ④ cold_high는 채널 전체 파괴(util_n≈0).
- **conventional NPCA 대비 (dcf_conv = 표준 CSMA/CA visitor, CW_min=16 BEB)**: visitor 효율 nat0 0.99 → nat10 0.60 (BEB sawtooth 유한창 붕괴, fig19 재현). 대신 **최공정**: prop 1.01~1.65 (oracle보다 낮음), util_n 0.155 ≈ oracle. **PACE warm vs 표준 = 효율-공정 trade-off**: visitor +45% (0.866 vs 0.597), 대가 native 몫 1/4 + prop 1.81 vs 1.22. PACE 이득에 native 억압분 포함 — 논문에 명시, fairness 보정(τ cap) = future work.
- **airtime 분해 (nat10, nocd)**: dcf 낭비 = 충돌 42%+idle 15%(frozen backoff), PACE 낭비 = 충돌 53%+idle 5%. Jain 이중 구조: 그룹간 dcf 승(J_all 0.77 vs 0.53), **visitor 내부는 PACE 승**(J_vis 0.89 vs 0.82 — solo-copy 균등화 vs BEB 로터리).

### Fig 25: Collision-cost 민감도 + 의무 RTS/CTS — **표준 단위** (mixed, N_v=10+N_nat=10)

RQ: 충돌 비용↑ 시 PACE 열화? RTS/CTS 의무화 영향? **모든 시간 = 802.11 표준 파라미터 유도** (σ=aSlotTime 9µs, SIFS 16, DIFS 34, RTS/CTS@24Mbps 28µs → 성공 OH 88µs=10σ, RTS 충돌 78µs=9σ; @6Mbps OH 136µs=15σ/충돌 102µs=11σ). 프레임 현실화: visitor PPDU 225–900µs(E[L]=562µs), native 450µs, W_eff=3.78ms(TXOP급). v1(추상 슬롯 OH=2/3)은 폐기.

- **충돌비용 스윕 (basic)**: PACE 단조 감소(C=81µs: 0.74 → C=1.13ms: 0.25), visitor 우위 ×1.54→×1.21 축소되나 **전 구간 생존**. **채널 효율은 전 구간 dcf ≥ pace** (최저 실현가능 충돌비용에서 동률) — v1의 "C≈E[L] 교차" 주장은 표준 단위에서 성립 안 함(정정).
- **의무 RTS/CTS @24Mbps**: PACE visitor +80%(0.36→0.64)/채널 +78% vs dcf +47%/+49% — 비대칭(×1.7) 유지, 단 dcf도 대폭 이득(basic 충돌=max Lᵢ≈700µs라 절감 큼). **PACE 채널 열세 해소**(0.697≈0.691 동률), visitor 우위 ×1.25→×1.53. @6Mbps도 순이득(PACE +61%/dcf +35%).
- 논문: "PACE의 낭비(충돌)는 RTS/CTS로 치유 가능, conventional의 낭비(frozen backoff)는 불치" — 단 v1보다 온건하게(채널 효율은 동률까지). 한계: native도 RTS/CTS, fairness 축 불변, ACK 미모델. 코드: `run_step9_fig25.py`, 상세: `guidelines/step9/fig25.md`.

→ 논문 포지셔닝: cold 성능은 τ₀의 지식량에 비례(genie 0.95/무지식 0.16) — **warm-start는 지식 축을 기억으로 대체** (무지식→0.87). 코드: `run_step9_fig24.py`, 상세: `guidelines/step9/fig24.md`.

---

## 파일 구조

```
pace/
├── CLAUDE.md                        ← 이 파일
├── manuscript/
│   ├── pace.tex                     ← 논문 본문 (PACE)
│   ├── pace.bib                     ← 참고문헌
│   ├── figure/                      ← **논문에 포함할 figure만** (선별 복사)
│   └── ref/
│       ├── A_Probabilistic_Neighbor_Discovery_Algorithm_in_Wireless_Ad_Hoc_Networks.pdf  ← PND 원본
│       └── Draft P802.11bn_D1.2_NPCA.pdf   ← 표준 문서
├── harq_sim/
│   ├── run_step9_fig{N}.py          ← Fig 1–21 생성 스크립트
│   └── mfg_npca_sim.py              ← 핵심 시뮬레이터
├── guidelines/
│   ├── step9_index.md               ← Figure 인덱스 및 상태
│   ├── step9/fig{N}.md              ← 각 Figure 상세 계획
│   └── mfg_algorithm.md            ← MFG 알고리즘 상세
└── results/
    ├── figure/                      ← 스크립트 생성 figure 전체 (구 manuscript/figure)
    └── step9/fig{N}/data.csv        ← 실험 데이터
```

---

## Figure → 논문 섹션 매핑

| Figure | 논문 역할 | 상태 |
|---|---|---|
| Fig 3 | Fixed τ의 한계 — τ*(N, W_eff) 의존성 | ✅ |
| Fig 4 | Adaptive τ vs Fixed vs Oracle (핵심 기여) | ✅ |
| Fig 12 | K = W_eff/PPDU multi-round collapse 경계 | ✅ |
| Fig 15 | MFG-optimal: tight window +70% gain | ✅ |
| Fig 17 | PPDU-aware self-excl + PACE 효율 순위 | ✅ |
| Fig 18 | PND 파라미터 (c_coll × c_idle) 민감도 | ✅ |
| Fig 19 | DCF benchmark — τ trajectory 비교 | 🔄 |
| Fig 20 | Throughput–Fairness Pareto | ✅ |
| Fig 21 | Native vs Visitor 공정성 | ✅ |

---

## Figure 생성 규칙

모든 스크립트는 **`results/figure/`**에 3포맷 동시 저장 (manuscript/figure 직접 저장 금지):

```python
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figure")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.eps"), format="eps", bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.png"), format="png", dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.pdf"), format="pdf", bbox_inches="tight")
```

`manuscript/figure/`에는 논문에 실을 figure만 `results/figure/`에서 선별 복사.

---

## IEEE 802.11bn D1.2 표준 부합도

내 시뮬레이터 vs D1.2 ~45%:

✅ OBSS 감지 → NPCA 결정 트리거  
✅ NPCA_PPDU_REM_DUR = `obss_remain`  
✅ Min Duration Threshold (rule_based: `obss_remain ≥ ppdu_duration`)  
✅ Switching delay (1 slot)  
✅ Exponential backoff (DCF baseline)  
❌ TXOP 다수 MPDU 전송 미구현  
❌ RTS/CTS 트리거 미구현  
❌ CW 저장/복원 미구현  

**논문 포지셔닝**: 표준의 "전환 결정 정책(switching policy)" 추상화에 집중.  
PACE = intent별 최적 NPCA Min Duration Threshold를 τ 수렴으로 달성.

---

## 실행 방법

```bash
source .venv/bin/activate

# Step 9 Figure 생성
python harq_sim/run_step9_fig15.py
python harq_sim/run_step9_fig17.py
python harq_sim/run_step9_fig18.py
python harq_sim/run_step9_fig19.py
python harq_sim/run_step9_fig20.py
python harq_sim/run_step9_fig21.py

# 논문 컴파일
cd manuscript && pdflatex pace.tex && bibtex pace && pdflatex pace.tex
```

---

## 향후 작업

- [x] Fig 15–21 시뮬레이션 완료
- [x] PACE 논문 제목 확정 (pace.tex)
- [x] Abstract 초안 작성
- [ ] Introduction 작성
- [ ] Related Work 작성 (PND, NPCA 선행연구)
- [ ] System Model 작성
- [ ] PACE Algorithm 섹션 작성 (알고리즘 pseudocode 포함)
- [ ] Performance Evaluation 섹션 작성 (Fig 삽입)
- [ ] Conclusion 작성
- [ ] Fig 19 완료 (🔄 진행 중)
