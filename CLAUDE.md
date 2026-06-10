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

| 방법 | Throughput | Jain's J |
|---|---|---|
| pnd | **0.772** | **0.263** |
| dcf_self_excl | 0.669 | 0.228 |
| and | 0.409 | 0.155 |

PACE = Pareto-dominant: 처리량·공정성 동시 최고.

### Fig 21: Native vs Visitor Fairness (bimodal {4,12}, W_eff=50, N_visitor=10, N_native=10)

| 방법 | util_v | util_n | prop |
|---|---|---|---|
| oracle | 0.555 | 0.206 | 1.39 |
| pnd | **0.593** | 0.162 | **1.51** |
| pnd_cd | 0.586 | 0.166 | 1.49 |
| dcf_self_excl | 0.354 | **0.293** | 1.00 |
| and | 0.601 | 0.027 | 1.90 |

- pnd_cd ≈ pnd: NPCA finite window에서 ACK 기반 CD의 marginal 효과 미미 (PPDU-aware self-exclusion이 CD 역할 대체)
- AND: prop=1.90 > pnd=1.51 — open-loop 초기 high-τ 선점으로 native BEB 폭증
- PND native 침해: native_preservation=0.265 (native 단독 대비 26.5%) — 논문 한계로 명시

---

## 파일 구조

```
vNPCA/
├── CLAUDE.md                        ← 이 파일
├── manuscript/
│   ├── pace.tex                     ← 논문 본문 (PACE)
│   ├── pace.bib                     ← 참고문헌
│   ├── figure/                      ← figN_*.{eps,png,pdf}
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
└── results/step9/fig{N}/data.csv   ← 실험 데이터
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

모든 스크립트는 3포맷 동시 저장:

```python
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "manuscript", "figure")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.eps"), format="eps", bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.png"), format="png", dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.pdf"), format="pdf", bbox_inches="tight")
```

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
