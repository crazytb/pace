# 지침 파일 인덱스

원본 `npca_harq_simulation_guidelines.md` (1873줄)을 Step별로 분리.
각 세션 시작 시 필요한 파일만 로드하여 컨텍스트 절약.

---

## 파일 목록 및 로드 전략

| 파일 | 내용 | 크기 | 로드 시점 |
|---|---|---|---|
| `core.md` | §1–3 시뮬레이션 목적, ARQ/HARQ 개념, 시스템 구조 | ~110줄 | **항상** |
| `status.md` | §35 Step별 완료 현황 (불변식, 검증 결과) | ~160줄 | **항상** |
| `step5_adaptive_cw.md` | §13 Adaptive CW_npca_init, §27 주의사항 | ~120줄 | Step 5 작업 시 |
| `step6_reward.md` | §14 Deadline, §19–20 Reward 설계, Intent 프로파일 | ~180줄 | Step 6 작업 시 |
| `step7_llm.md` | §21 LLM as reward designer, §24 Grid-best baseline | ~80줄 | Step 7 작업 시 |
| `step8_experiments.md` | §22–26, §28–34 실험, baseline, 로깅, 실행 방법 | ~280줄 | Step 8 작업 시 |
| `reference_mechanics.md` | §4–12, §15–18 STA 메커니즘 상세 (Step 1–4 참고용) | ~350줄 | 필요할 때만 |
| `contribution_positioning.md` | PACE contribution 논지 (native vs visitor trade-off, native-aware 변형 TODO) | ~90줄 | 논문 작성 시 |

---

## 세션별 로드 가이드

### Step 5 세션 (Adaptive CW)

```
읽어야 할 파일:
  guidelines/core.md
  guidelines/status.md
  guidelines/step5_adaptive_cw.md

합계: ~390줄 (원본 1873줄의 21%)
```

### Step 6 세션 (Reward Module)

```
읽어야 할 파일:
  guidelines/core.md
  guidelines/status.md
  guidelines/step6_reward.md

합계: ~450줄 (원본의 24%)
```

### Step 7 세션 (LLM 연결)

```
읽어야 할 파일:
  guidelines/core.md
  guidelines/status.md
  guidelines/step7_llm.md

합계: ~350줄 (원본의 19%)
```

### Step 8 세션 (Baseline 비교)

```
읽어야 할 파일:
  guidelines/core.md
  guidelines/status.md
  guidelines/step8_experiments.md

합계: ~550줄 (원본의 29%)
```

---

## 세션 시작 프롬프트 예시

```
guidelines/core.md, guidelines/status.md, guidelines/step5_adaptive_cw.md 읽고
Step 5 (Adaptive CW_npca_init) 구현 시작해줘.
현재 harq_sim/ 폴더에 Step 4까지 구현되어 있음.
```
